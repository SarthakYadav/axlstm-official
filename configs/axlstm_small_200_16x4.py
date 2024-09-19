import ml_collections
import os
from .common import get_opt_config, get_data_config


def get_config():
    config = ml_collections.ConfigDict()

    config.model = ml_collections.ConfigDict()
    config.model.arch = "xlstm_ssast_small"
    config.model.type = "ssast"
    config.model.num_classes = 1000     # needed for dataset parsing. Is not used.
    # config.model.img_size = 40000
    # config.model.patch_size = 16
    # config.model.in_chans = 1
    # config.model.embed_dim = 768
    config.model.model_args = {
        "mask_ratio": 0.8,
        "img_size": (200, 80),
        "patch_size": (4, 16),
        "mask_ratio": 0.5,
        "frequency_first": False,
    }
    config.model.patch_embed_args = ml_collections.ConfigDict()
    config.opt = get_opt_config()

    config.log_every_steps = 100
    config.num_train_steps = -1
    config.steps_per_eval = -1

    config.data = get_data_config()

    config.batch_size = 256     # per gpu
    config.shuffle_buffer_multiplier = 250
    config.half_precision = False
    config.num_epochs = 100

    config.wandb = ml_collections.ConfigDict()
    config.wandb.project = "ssast-pytorch-xlstm"

    return config
