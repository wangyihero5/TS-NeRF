import os

import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from torch import nn
from opt import get_opts

import glob
import imageio
import numpy as np
import cv2
from einops import rearrange

# data
from torch.utils.data import DataLoader
from datasets import dataset_dict
from datasets.ray_utils import axisangle_to_R, get_rays

# models
from kornia.utils.grid import create_meshgrid3d
from models.networks import NGP
from models.rendering import render, MAX_SAMPLES

# optimizer, losses
from apex.optimizers import FusedAdam
from torch.optim.lr_scheduler import CosineAnnealingLR
from losses import NeRFLoss

# metrics
from torchmetrics import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure
)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# pytorch-lightning
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.distributed import all_gather_ddp_if_available

# custom
from utils import slim_ckpt, load_ckpt
from models_ts.global_val import Global_instance
from models_ts import RAIN
from models_ts.RAIN import Net as RAIN_net
import MyUtil.utils as my_utils
import gc

# torch.set_float32_matmul_precision('medium')
# TF_ENABLE_ONEDNN_OPTS=0

# Constrain all sources of randomness
seed = 29
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# random.seed(seed)
np.random.seed(seed)


def depth2img(depth):
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    depth_img = cv2.applyColorMap((depth * 255).astype(np.uint8),
                                  cv2.COLORMAP_TURBO)

    return depth_img


class NeRFSystem(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.warmup_steps = 256
        self.update_interval = 16

        self.loss = NeRFLoss(lambda_distortion=self.hparams.distortion_loss_w)
        self.train_psnr = PeakSignalNoiseRatio(data_range=1)
        self.val_psnr = PeakSignalNoiseRatio(data_range=1)
        self.val_ssim = StructuralSimilarityIndexMeasure(data_range=1)
        if self.hparams.eval_lpips:
            self.val_lpips = LearnedPerceptualImagePatchSimilarity('vgg')
            for p in self.val_lpips.net.parameters():
                p.requires_grad = False

        rgb_act = 'None' if self.hparams.use_exposure else 'Sigmoid'

        self.model = NGP(scale=self.hparams.scale, rgb_act=rgb_act, stage=self.hparams.stage)

        G = self.model.grid_size
        self.model.register_buffer('density_grid',
                                   torch.zeros(self.model.cascades, G ** 3))
        self.model.register_buffer('grid_coords',
                                   create_meshgrid3d(G, G, G, False, dtype=torch.int32).reshape(-1, 3))

        """custom initialization"""
        if hparams.stage == "second_stage":
            # Setting VGG
            for param in self.model.xyz_encoder.parameters():
                param.requires_grad = False

            # Create vgg and fc_encoder in RAIN_net
            vgg = RAIN.vgg
            fc_encoder = RAIN.fc_encoder
            # Load pretrained weights of vgg and fc_encoder
            vgg.load_state_dict(torch.load(hparams.vgg_pretrained_path))
            fc_encoder.load_state_dict(torch.load(hparams.fc_encoder_pretrained_path))
            vgg = nn.Sequential(*list(vgg.children())[:31])
            self.RAIN_net = RAIN_net(vgg, fc_encoder).to(device)

            # Fixed RAIN_net
            for param in self.RAIN_net.parameters():
                param.requires_grad = False

        # Whether to turn on the nearest neighbor finder
        Global_instance.clip_loss.set_ArtBench_search(self.hparams.enable_ArtBench_search)

    def forward(self, batch, split):
        if split == 'train':
            poses = self.poses[batch['img_idxs']]
            directions = self.directions[batch['pix_idxs']]
        else:
            poses = batch['pose']
            directions = self.directions

        if self.hparams.optimize_ext:
            dR = axisangle_to_R(self.dR[batch['img_idxs']])
            poses[..., :3] = dR @ poses[..., :3]
            poses[..., 3] += self.dT[batch['img_idxs']]

        rays_o, rays_d = get_rays(directions, poses)

        kwargs = {'test_time': split != 'train',
                  'random_bg': self.hparams.random_bg}
        if self.hparams.scale > 0.5:
            kwargs['exp_step_factor'] = 1 / 256
        if self.hparams.use_exposure:
            kwargs['exposure'] = batch['exposure']

        return render(self.model, rays_o, rays_d, **kwargs)

    def setup(self, stage):
        dataset = dataset_dict[self.hparams.dataset_name]
        kwargs = {'root_dir': self.hparams.root_dir,
                  'downsample': self.hparams.downsample}
        self.train_dataset = dataset(split=self.hparams.split, **kwargs)
        self.train_dataset.batch_size = self.hparams.batch_size
        self.train_dataset.ray_sampling_strategy = self.hparams.ray_sampling_strategy

        self.test_dataset = dataset(split='test', **kwargs)

    def configure_optimizers(self):
        # define additional parameters
        self.register_buffer('directions', self.train_dataset.directions.to(self.device))
        self.register_buffer('poses', self.train_dataset.poses.to(self.device))

        if self.hparams.optimize_ext:
            N = len(self.train_dataset.poses)
            self.register_parameter('dR',
                                    nn.Parameter(torch.zeros(N, 3, device=self.device)))
            self.register_parameter('dT',
                                    nn.Parameter(torch.zeros(N, 3, device=self.device)))

        load_ckpt(self.model, self.hparams.weight_path)

        net_params = []
        for n, p in self.named_parameters():
            if n not in ['dR', 'dT']: net_params += [p]

        opts = []
        self.net_opt = FusedAdam(net_params, self.hparams.lr, eps=1e-15)
        opts += [self.net_opt]
        if self.hparams.optimize_ext:
            opts += [FusedAdam([self.dR, self.dT], 1e-6)]  # learning rate is hard-coded
        net_sch = CosineAnnealingLR(self.net_opt,
                                    self.hparams.num_epochs,
                                    self.hparams.lr / 30)

        return opts, [net_sch]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          num_workers=16,
                          persistent_workers=True,
                          batch_size=None,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset,
                          num_workers=8,
                          batch_size=None,
                          pin_memory=True)

    def on_train_start(self):
        self.model.mark_invisible_cells(self.train_dataset.K.to(self.device),
                                        self.poses,
                                        self.train_dataset.img_wh)

    def training_step(self, batch, batch_nb, *args):
        # self.hparams.is_valid = False  # 这个地方千万不能动

        if self.hparams.is_valid:
            loss = torch.zeros(1, requires_grad=True).to(device)
            return loss

        if self.global_step % self.update_interval == 0:
            self.model.update_density_grid(0.01 * MAX_SAMPLES / 3 ** 0.5,
                                           warmup=self.global_step < self.warmup_steps,
                                           erode=self.hparams.dataset_name == 'colmap')

        if self.hparams.stage == "second_stage":

            W, H = batch["W"], batch["H"]

            if self.hparams.enable_random_sampling:
                with torch.no_grad():
                    results = self(batch, split='train')  # 560 283
                """
                n = 4  # Splitting factor, suggested to be Nth power of 2
                w, h = int(batch["W"] / n), int(batch["H"] / n)
                idxs = np.random.choice(n, 2)  # Randomly select n indexes
                i, j = idxs[0], idxs[1]
                grad_idxs = np.arange(0, H * W).reshape(H, W)[i * h:(i + 1) * h, j * w:(j + 1) * w]
                grad_idxs = grad_idxs.flatten()
                """
                n = 7
                grad_idxs = np.random.choice(H * W, n * 8192)  # Sampling n * 8192 samples to calculate the gradien
                batch_grad = batch.copy()
                batch_grad["pix_idxs"] = batch_grad["pix_idxs"][grad_idxs]
                batch_grad["rgb"] = batch_grad["rgb"][grad_idxs]

            if self.hparams.is_valid:
                with torch.no_grad():
                    if self.hparams.enable_random_sampling:
                        results_grad = self(batch_grad, split='train')
                    else:
                        results = self(batch, split='train')
            else:
                if self.hparams.enable_random_sampling:
                    results_grad = self(batch_grad, split='train')
                else:
                    results = self(batch, split='train')

            if self.hparams.enable_random_sampling:
                results["rgb"][grad_idxs] = results_grad["rgb"]

            rgb_gt = batch["rgb"].reshape(H, W, 3)
            rgb_result = results["rgb"].reshape(H, W, 3).permute(2, 0, 1).unsqueeze(0)

            # VGG loss
            content_feat_pred = self.RAIN_net.get_content_feat(rgb_result)
            content_feat_gt = self.RAIN_net.get_content_feat(rgb_gt.permute(2, 0, 1).unsqueeze(0))
            content_loss = my_utils.get_content_loss(content_feat_gt, content_feat_pred)

            # CLIP dirction loss
            clip_loss = Global_instance.clip_loss(rgb_gt.permute(2, 0, 1).unsqueeze(0), "photo", rgb_result,
                                                  self.hparams.style_target)
            # HCL loss
            hcl_loss = Global_instance.clip_loss_nce(rgb_gt.permute(2, 0, 1).unsqueeze(0), "photo", rgb_result,
                                                     self.hparams.style_target, 0)  # lamda=0

            if self.hparams.enable_NeRF_loss:
                loss_d = self.loss(results_grad, batch_grad, first_stage=False)  # NeRF loss
            else:
                loss_d = {}
            loss_d["content_loss"] = content_loss * 0.01
            loss_d["hcl_loss"] = hcl_loss * 2
            loss_d["clip_loss"] = clip_loss * 20
            loss = sum(lo.mean() for lo in loss_d.values())
        else:
            results = self(batch, split='train')
            loss_d = self.loss(results, batch, first_stage=True)
            if self.hparams.use_exposure:
                zero_radiance = torch.zeros(1, 3, device=self.device)
                unit_exposure_rgb = self.model.log_radiance_to_rgb(zero_radiance,
                                                                   **{'exposure': torch.ones(1, 1, device=self.device)})
                loss_d['unit_exposure'] = \
                    0.5 * (unit_exposure_rgb - self.train_dataset.unit_exposure_rgb) ** 2
            loss = sum(lo.mean() for lo in loss_d.values())

        torch.cuda.empty_cache()
        gc.collect()

        with torch.no_grad():
            self.train_psnr(results['rgb'], batch['rgb'])
        self.log('lr', self.net_opt.param_groups[0]['lr'])
        self.log('train/loss', loss)
        # ray marching samples per ray (occupied space on the ray)
        self.log('train/rm_s', results['rm_samples'] / len(batch['rgb']), True)
        # volume rendering samples per ray (stops marching when transmittance drops below 1e-4)
        self.log('train/vr_s', results['vr_samples'] / len(batch['rgb']), True)
        self.log('train/psnr', self.train_psnr, True)

        return loss

    def on_validation_start(self):
        torch.cuda.empty_cache()
        if not self.hparams.no_save_test:
            self.val_dir = f'results/{self.hparams.dataset_name}/{self.hparams.exp_name}'
            os.makedirs(self.val_dir, exist_ok=True)

    def validation_step(self, batch, batch_nb):
        rgb_gt = batch['rgb']
        results = self(batch, split='test')

        logs = {}
        # compute each metric per image
        self.val_psnr(results['rgb'], rgb_gt)
        logs['psnr'] = self.val_psnr.compute()
        self.val_psnr.reset()

        w, h = self.train_dataset.img_wh
        rgb_pred = rearrange(results['rgb'], '(h w) c -> 1 c h w', h=h)
        rgb_gt = rearrange(rgb_gt, '(h w) c -> 1 c h w', h=h)
        self.val_ssim(rgb_pred, rgb_gt)
        logs['ssim'] = self.val_ssim.compute()
        self.val_ssim.reset()
        if self.hparams.eval_lpips:
            self.val_lpips(torch.clip(rgb_pred * 2 - 1, -1, 1),
                           torch.clip(rgb_gt * 2 - 1, -1, 1))
            logs['lpips'] = self.val_lpips.compute()
            self.val_lpips.reset()

        if not self.hparams.no_save_test:  # save test image to disk
            idx = batch['img_idxs']
            rgb_pred = rearrange(results['rgb'].cpu().numpy(), '(h w) c -> h w c', h=h)
            rgb_pred = (rgb_pred * 255).astype(np.uint8)
            depth = depth2img(rearrange(results['depth'].cpu().numpy(), '(h w) -> h w', h=h))
            imageio.imsave(os.path.join(self.val_dir, f'{idx:03d}.png'), rgb_pred)
            imageio.imsave(os.path.join(self.val_dir, f'{idx:03d}_d.png'), depth)

        return logs

    def validation_epoch_end(self, outputs):
        psnrs = torch.stack([x['psnr'] for x in outputs])
        mean_psnr = all_gather_ddp_if_available(psnrs).mean()
        self.log('test/psnr', mean_psnr, True)

        ssims = torch.stack([x['ssim'] for x in outputs])
        mean_ssim = all_gather_ddp_if_available(ssims).mean()
        self.log('test/ssim', mean_ssim)

        if self.hparams.eval_lpips:
            lpipss = torch.stack([x['lpips'] for x in outputs])
            mean_lpips = all_gather_ddp_if_available(lpipss).mean()
            self.log('test/lpips_vgg', mean_lpips)

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items


if __name__ == '__main__':
    hparams = get_opts()
    if hparams.val_only and (not hparams.ckpt_path):
        raise ValueError('You need to provide a @ckpt_path for validation!')
    system = NeRFSystem(hparams)

    # Set the current training stage
    Global_instance.set_current_stage(hparams.stage)
    ckpt_num_epochs = 50
    ckpt_cb = ModelCheckpoint(dirpath=f'ckpts/{hparams.dataset_name}/{hparams.exp_name}',
                              filename='{epoch:d}',
                              save_weights_only=True,
                              every_n_epochs=ckpt_num_epochs,
                              save_on_train_epoch_end=True,
                              save_top_k=-1, save_last=True)
    callbacks = [ckpt_cb, TQDMProgressBar(refresh_rate=1)]

    logger = TensorBoardLogger(save_dir=f"logs/{hparams.dataset_name}",
                               name=hparams.exp_name,
                               default_hp_metric=False)

    trainer = Trainer(max_epochs=hparams.num_epochs,
                      check_val_every_n_epoch=hparams.num_epochs,
                      callbacks=callbacks,
                      logger=logger,
                      enable_model_summary=False,
                      accelerator='gpu',
                      devices=hparams.num_gpus,
                      strategy=DDPPlugin(find_unused_parameters=False)
                      if hparams.num_gpus > 1 else None,
                      num_sanity_val_steps=-1 if hparams.val_only else 0,
                      precision=16, amp_backend="apex", amp_level="O1")

    trainer.fit(system, ckpt_path=hparams.ckpt_path)

    if not hparams.val_only:  # save slimmed ckpt for the last epoch

        ckpt_ = \
            slim_ckpt(f'ckpts/{hparams.dataset_name}/{hparams.exp_name}/last.ckpt',
                      save_poses=hparams.optimize_ext)

        torch.save(ckpt_, f'ckpts/{hparams.dataset_name}/{hparams.exp_name}/epoch={hparams.num_epochs - 1}_slim.ckpt')

    if (not hparams.no_save_test) and hparams.dataset_name == 'nsvf' and 'Synthetic' in hparams.root_dir:  # save video
        imgs = sorted(glob.glob(os.path.join(system.val_dir, '*.png')))
        imageio.mimsave(os.path.join(system.val_dir, 'rgb.mp4'),
                        [imageio.imread(img) for img in imgs[::2]],
                        fps=30, macro_block_size=1)
