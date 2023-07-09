import os
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from models_ts import DirectionLoss
from criteria.nce_loss import NCELoss


class Global_instance(object):
    TAG = "first_stage"  # current stage
    clip_loss = DirectionLoss.CLIPLoss(device)
    clip_loss_nce = NCELoss(device)

    def __init__(self, name):
        pass

    @classmethod
    def get_clip_loss(cls):
        return Global_instance.clip_loss

    @classmethod
    def get_current_stage(cls):
        return Global_instance.TAG

    @classmethod
    def set_current_stage(cls, stage=None):
        Global_instance.TAG = stage
