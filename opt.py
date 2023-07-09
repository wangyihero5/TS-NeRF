import argparse


def get_opts():
    parser = argparse.ArgumentParser()

    # dataset parameters
    parser.add_argument('--root_dir', type=str, required=True,
                        help='root directory of dataset')
    parser.add_argument('--dataset_name', type=str, default='colmap',
                        choices=['nerf', 'nsvf', 'colmap', 'nerfpp', 'rtmv'],
                        help='which dataset to train/test')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'trainval', 'trainvaltest'],
                        help='use which split to train')
    parser.add_argument('--downsample', type=float, default=0.1488,  # 0.08
                        help='downsample factor (<=1.0) for the images')

    # model parameters
    parser.add_argument('--scale', type=float, default=2.0,
                        help='scene scale (whole scene must lie in [-scale, scale]^3')

    parser.add_argument('--use_exposure', action='store_true', default=False,
                        help='whether to train in HDR-NeRF setting')

    # loss parameters
    parser.add_argument('--distortion_loss_w', type=float, default=1e-3,
                        help='''weight of distortion loss (see losses.py),
                        0 to disable (default), to enable,
                        a good value is 1e-3 for real scene and 1e-2 for synthetic scene
                        ''')

    # training options
    parser.add_argument('--batch_size', type=int, default=8192 * 2,
                        help='number of rays in a batch')
    parser.add_argument('--ray_sampling_strategy', type=str, default='same_image',
                        choices=['all_images', 'same_image'],
                        help='''
                        all_images: uniformly from all pixels of ALL images
                        same_image: uniformly from all pixels of a SAME image
                        ''')
    parser.add_argument('--num_epochs', type=int, default=1,
                        help='number of training epochs')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='learning rate')
    # experimental training options
    parser.add_argument('--optimize_ext', action='store_true', default=False,
                        help='whether to optimize extrinsics')
    parser.add_argument('--random_bg', action='store_true', default=False,
                        help='''whether to train with random bg color (real scene only)
                        to avoid objects with black color to be predicted as transparent
                        ''')

    # validation options
    parser.add_argument('--eval_lpips', action='store_true', default=False,
                        help='evaluate lpips metric (consumes more VRAM)')
    parser.add_argument('--val_only', action='store_true', default=False,
                        help='run only validation (need to provide ckpt_path)')
    parser.add_argument('--no_save_test', action='store_true', default=False,
                        help='whether to save test image and video')

    # misc
    parser.add_argument('--exp_name', type=str, default='exp',
                        help='experiment name')
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='pretrained checkpoint to load (including optimizers, etc)')
    parser.add_argument('--weight_path', type=str, default=None,
                        help='pretrained checkpoint to load (excluding optimizers, etc)')

    # custom options
    parser.add_argument('--stage', type=str, default='first_stage',
                        help='experiment stage')
    parser.add_argument('--is_valid', action='store_true', default=False,
                        help='is valid')
    parser.add_argument('--vgg_pretrained_path', type=str, default="pretrained_StyleVAE/vgg_normalised.pth",
                        help='VGG pretrained path ')
    parser.add_argument('--fc_encoder_pretrained_path', type=str, default="pretrained_StyleVAE/fc_encoder_iter_160000.pth",
                        help='fc encoder pretrained path')
    parser.add_argument('--style_target', type=str, default="a painting in the style of Pablo Picasso's The Mandolinist",
                        help='Stylized target text')
    parser.add_argument('--enable_random_sampling', action='store_true', default=False,
                        help='whether to enable random sampling')
    parser.add_argument('--enable_NeRF_loss', action='store_true', default=False,
                        help='whether to use Nerf loss')
    parser.add_argument('--enable_ArtBench_search', action='store_true', default=False,
                        help='whether to enable ArtBench searcher')

    return parser.parse_args()
