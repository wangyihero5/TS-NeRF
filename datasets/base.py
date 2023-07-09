from torch.utils.data import Dataset
import numpy as np
from models_ts.global_val import Global_instance


class BaseDataset(Dataset):
    """
    Define length and sampling method
    """

    def __init__(self, root_dir, split='train', downsample=1.0):
        self.root_dir = root_dir
        self.split = split
        self.downsample = downsample
        self.stage = None

    def read_intrinsics(self):
        raise NotImplementedError

    def __len__(self):
        if self.split.startswith('train'):
            return 1000
        return len(self.poses)

    def __getitem__(self, idx):
        stage = Global_instance.get_current_stage()
        # random sampling
        if self.split.startswith('train') and stage == "first_stage":
            # training pose is retrieved in train.py
            if self.ray_sampling_strategy == 'all_images':  # randomly select images
                img_idxs = np.random.choice(len(self.poses), self.batch_size)
            elif self.ray_sampling_strategy == 'same_image':  # randomly select ONE image
                img_idxs = np.random.choice(len(self.poses), 1)[0]
            # randomly select pixels
            pix_idxs = np.random.choice(self.img_wh[0] * self.img_wh[1], self.batch_size)
            rays = self.rays[img_idxs, pix_idxs]
            sample = {'img_idxs': img_idxs, 'pix_idxs': pix_idxs,
                      'rgb': rays[:, :3]}
            if self.rays.shape[-1] == 4:  # HDR-NeRF data
                sample['exposure'] = rays[:, 3:]

        # Full size sampling
        elif self.split.startswith('train') and stage == "second_stage":
            if self.ray_sampling_strategy == 'all_images':  # randomly select images
                img_idxs = np.random.choice(len(self.poses), self.batch_size)
            elif self.ray_sampling_strategy == 'same_image':  # randomly select ONE image
                img_idxs = np.random.choice(len(self.poses), 1)[0]
            # randomly select pixels
            pix_idxs = np.arange(0, self.img_wh[0] * self.img_wh[1])
            rays = self.rays[img_idxs, pix_idxs]
            # rays = self.rays[idx]
            sample = {'img_idxs': img_idxs, 'pix_idxs': pix_idxs,
                      'rgb': rays[:, :3], "W": self.img_wh[0], "H": self.img_wh[1]}
            if self.rays.shape[-1] == 4:  # HDR-NeRF data
                sample['exposure'] = rays[:, 3:]
        else:
            sample = {'pose': self.poses[idx], 'img_idxs': idx}
            if len(self.rays) > 0:  # if ground truth available
                rays = self.rays[idx]
                sample['rgb'] = rays[:, :3]
                if rays.shape[1] == 4:  # HDR-NeRF data
                    sample['exposure'] = rays[0, 3]  # same exposure for all rays

        return sample
