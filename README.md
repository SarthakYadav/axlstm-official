# axlstm-official
This is the official repository for our paper ["Audio xLSTMs: Learning Self-Supervised Audio Representations with xLSTMs"](https://arxiv.org/abs/2408.16568).

# Contents
* [Pre-trained weights for the default AxLSTM configurations](https://drive.google.com/drive/folders/1SyNvA7a6jrWmRwYYj85ggADiKBpDpGLK?usp=sharing)
* Our local copy of [hear-eval-kit](external_sources/hear-eval-kit) for easy downstream reproducibility. Original can be found [here](https://github.com/hearbenchmark/hear-eval-kit)
* [Feature extraction API](hear_api) compatible with the [hear-eval-kit](https://github.com/hearbenchmark/hear-eval-kit) format for extracting features.
* Code used to train the AxLSTM models.
* Helper code to [extract features](extract_features.sh) and [run downstream experiments](downstream_experiments.sh) on provided pre-trained models

---

# Setup

## Environment
* Required: `cuda 11.x` or newer, `cudnn 8.2` or newer.
* Create a new conda environment with `python 3.10` or later.
* Requires `torch 2.1.2` or newer.

Follow these steps
```shell
conda create -n axlstm-env python=3.10 -y
conda activate axlstm-env

pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt

# install hear-eval-kit specific requirements
pip install -r external_sources/hear-eval-kit/requirements.txt

# install hear-eval-kit, WITHOUT AUTO DEPS
cd external_sources/hear-eval-kit && pip install --no-deps . && cd -
```

## Get 16000 Hz data from hear
* Follow https://hearbenchmark.com/hear-tasks.html to get data. By default, data on HEAR's zenodo page is 48000 Hz.
* We recommend downloading data directly from HEAR's [GCS bucket](gs://hear2021-archive/tasks/), where you can find preprocessed 16000 Hz data.
* Extract all the files to a folder `$TASKS_DIR`

## Get pretrained weights

* Pre-trained weights can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1SyNvA7a6jrWmRwYYj85ggADiKBpDpGLK?usp=sharing)
* Download the entire folder and export that folder as `$PT_AXLSTM_MODEL_DIR`

## Extract features

```shell
export PT_AXLSTM_MODEL_DIR=/path/to/pretrained_weights
./extract_features.sh $TASKS_DIR $OUTPUT_DIR
```
where TASKS_DIR is the directory where you extracted tasks from HEAR-2021 to, and OUTPUT_DIR is the base directory where output features will be stored. The given script will extract features from SSAST and AxLSTM Tiny configurations, you can change it as you need.
This also prepares a `todo_audioset` directory in OUTPUT_DIR, which is setting up for downstream classification on 10 seeds.

## Run downstream experiments

After extracting features, to run downstream experiment on a specific config, use the following command:
```shell
./downstream_experiments.sh axlstm_tiny_200_16x4 $OUTPUT_DIR/todo_audioset
```

This will run downstream experiments on all the extracted features for the tiny AxLSTM configuration on 10 random seeds.

## Get results
Finally, you can run the following script to get results of downstream experiments of the two models

```shell
python stats_aggregation_v2.py --base_dir ${OUTPUT_DIR}/todo_audioset --output_dir ${OUTPUT_DIR}/parsed_results
```

---

# Extracting features on your own audio file
The [hear_api](hear_api) can be used to extract features from your own audio files.

```python
import torchaudio

from hear_api import RuntimeSSAST
from importlib import import_module
config = import_module("configs.axlstm_tiny_200_16x4").get_config()
axlstm = RuntimeSSAST(config, "path/to/pretrained_dir").cuda()

# alternatively just use the following if you have the paths setup right
# axlstm = import_module("hear_configs.axlstm_tiny_200_16x4").load_model().cuda()

x, sr = torchaudio.load("path/to/audio.wav")
x = x.cuda()
o = axlstm.get_scene_embeddings(x)

```

---

# Pretraining
Pretraining code is included in the release. Any model configuration (for instance, `axlstm_tiny_200_16x4`) was trained with the following command:
```shell
torchrun --standalone --nnodes=1 --nproc-per-node=4 train.py --config configs.axlstm_tiny_200_16x4 --workdir $EXP_DIR/axlstm_tiny_200_16x4_4x256_fp16_r1 --precision float16 --print_freq 50 --num_workers 16 --no_wandb
```
We use a `torchdata` based datapipe for data loading, operating on precomputed log melspectrogram features stored in webdataset archive(s). You can adapt the data loading for your own use case.

---

# Citation
If you find this work useful, please consider citing our paper:
```
@misc{yadav2024audioxlstmslearningselfsupervised,
      title={Audio xLSTMs: Learning Self-Supervised Audio Representations with xLSTMs}, 
      author={Sarthak Yadav and Sergios Theodoridis and Zheng-Hua Tan},
      year={2024},
      eprint={2408.16568},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2408.16568}, 
}
```

We would like to thank the authors of the original xLSTM and Vision-LSTM papers for their work and codebase. So, please consider citing them as well:
```
@article{xlstm,
  title={xLSTM: Extended Long Short-Term Memory},
  author={Beck, Maximilian and P{\"o}ppel, Korbinian and Spanring, Markus and Auer, Andreas and Prudnikova, Oleksandra and Kopp, Michael and Klambauer, G{\"u}nter and Brandstetter, Johannes and Hochreiter, Sepp},
  journal={arXiv preprint arXiv:2405.04517},
  year={2024}
}


@article{alkin2024visionlstm,
  title={{Vision-LSTM}: {xLSTM} as Generic Vision Backbone},
  author={Benedikt Alkin and Maximilian Beck and Korbinian P{\"o}ppel and Sepp Hochreiter and Johannes Brandstetter},
  journal={arXiv preprint arXiv:2406.04303},
  year={2024}
}
```

---