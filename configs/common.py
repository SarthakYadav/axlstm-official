import ml_collections
import os
DATASET_DIR = os.environ.get("DATASETS_BASE_DIR", "")


def get_opt_config():
    opt = ml_collections.ConfigDict()
    opt.optimizer = "Adamw"
    opt.learning_rate = 1.5e-4
    opt.weight_decay = 0.05
    opt.schedule = "warmupcosine"
    opt.warmup_epochs = 10
    opt.momentum = 0.9
    opt.norm_pix_loss = False
    return opt


def get_data_config():
    data = ml_collections.ConfigDict()
    data.train_dirs = [
        os.path.join(DATASET_DIR, "audioset_logmelspec_webdataset/balanced_train"),
        os.path.join(DATASET_DIR, "audioset_logmelspec_webdataset/unbalanced_train"),
        os.path.join(DATASET_DIR, "audioset_logmelspec_webdataset/unbalanced_train_2"),
    ]
    data.train_samples = [
        18988,
        1766912,
        265408
    ]
    data.val_dir = os.path.join(DATASET_DIR, "audioset_logmelspec_webdataset/eval")
    data.val_samples = 17408
    data.clip_duration = 2.
    data.num_frames = 200
    data.dataset_name = "audioset"
    return data
