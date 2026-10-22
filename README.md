# Latent WAM: Latent World Action Modeling for End-to-End Autonomous Driving

**Accepted to ACM MM 2026 (Poster).**

[Paper](https://arxiv.org/abs/2603.24581) ·
[ModelScope Weights](https://modelscope.cn/models/Jinhsi555/Latent-WAM) ·
[中文说明](README_zh.md)

Linbo Wang, Yupeng Zheng, Qiang Chen, Shiwei Li, Yichen Zhang, Zebin Xing, Qichao Zhang, Xiang Li,
Deheng Qian, Pengxuan Yang, Yihang Dong, Ce Hao, Xiaoqing Ye, Junyu Han, Yifeng Pan, Dongbin Zhao

Latent WAM learns spatial structure and future dynamics in a compact latent space for end-to-end autonomous
driving. At inference, it predicts a four-second trajectory from the current three camera views and ego status,
without WorldMirror or future-world rollout.

## Method

![Latent WAM architecture (Figure 2 of the paper)](assets/method-overview.png)

The **Spatial-Aware Compressive World Encoder (SCWE)** distills geometry knowledge into compact scene tokens.
The **Dynamic Latent World Model (DLWM)** learns future scene and ego dynamics during training, improving the
representations used by the trajectory decoder.

## Results

Results below are reproduced from the paper: **89.3 EPDMS** on NAVSIM v2 and **45.9 RC / 28.9 HD-Score** on HUGSIM.
This repository provides the NAVSIM implementation and evaluation workflow.

### NAVSIM v2

![NAVSIM v2 results (Table 1 of the paper)](assets/navsim-results.png)

### HUGSIM

![HUGSIM results (Table 2 of the paper)](assets/hugsim-results.png)

## Quick Start

### 1. Install

Supported environment: Linux, Python 3.10, PyTorch 2.4.1, and CUDA 12.4.

```bash
git clone --recurse-submodules https://github.com/Jinhsi555/Latent-WAM.git latent-wam
cd latent-wam
conda env create -f environment.yml
conda activate latent-wam
pip install -e .

export NAVSIM_DEVKIT_ROOT="$PWD"
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OPENSCENE_DATA_ROOT="/path/to/dataset"
export NAVSIM_EXP_ROOT="/path/to/experiments"
export NUPLAN_MAPS_ROOT="$OPENSCENE_DATA_ROOT/maps"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
```

### 2. Prepare data

Download the nuPlan maps and NAVSIM test split using the scripts in [download/](download/).
Arrange the evaluation data as follows:

```text
dataset/
├── maps/
├── navsim_logs/test/
└── sensor_blobs/test/
```

See [data preparation](docs/latent_wam/usage.md#prepare-navsim-data) for download commands and training data.

### 3. Download weights

```bash
pip install modelscope
python - <<'PY'
from modelscope import snapshot_download
snapshot_download("Jinhsi555/Latent-WAM", local_dir="models/latent-wam")
PY
export LATENT_WAM_CHECKPOINT="$NAVSIM_DEVKIT_ROOT/models/latent-wam/model.safetensors"
```

DINOv2-Base is downloaded from Hugging Face automatically. For offline use, set
`export DINOV2_PATH=/path/to/dinov2-base`. WorldMirror is not required for evaluation.

### 4. Evaluate

Generate the metric cache once, then run GPU-batched inference and CPU-parallel scoring:

```bash
python navsim/planning/script/run_metric_caching.py \
  train_test_split=navtest \
  metric_cache_path="$NAVSIM_EXP_ROOT/metric_cache"

EVALUATION_DEVICES=0 PDM_WORKERS=8 INFERENCE_BATCH_SIZE=32 \
  bash scripts/evaluation/run_latent_wam_epdms_parallel.sh
```

For multiple GPUs, set `EVALUATION_DEVICES=0,1`; tune batch size to available memory.
Add `--max-scenarios 100` for a smoke test. The human-penalty filter is enabled by default.
The output is `$NAVSIM_EXP_ROOT/latent_wam_eval/navtest.csv`; EPDMS is the
`average_all_frames` row's `score × 100`.

## Training

Training requires NAVSIM trainval, DINOv2-Base, and WorldMirror for geometry-cache generation.
The default recipe trains on train + val for 100 epochs with 8 nodes × 8 GPUs, batch 8 per GPU,
no gradient accumulation (global batch 512), BF16, global cache shuffle, and block-causal temporal attention. Learning rate peaks at
`2e-4` and decays to `1e-6`; five validation batches are monitored and a checkpoint is saved every five epochs.
Start with [cache generation and training commands](docs/latent_wam/usage.md#train-latent-wam-from-scratch).

Further documentation:
[Installation](docs/latent_wam/installation.md) ·
[Usage guide](docs/latent_wam/usage.md) ·
[Training details](docs/latent_wam/training.md) ·
[Cache format](docs/latent_wam/cache-format.md)

## Citation

```bibtex
@article{wang2026latentwam,
  title={Latent-WAM: Latent World Action Modeling for End-to-End Autonomous Driving},
  author={Wang, Linbo and Zheng, Yupeng and Chen, Qiang and Li, Shiwei and Zhang, Yichen and
          Xing, Zebin and Zhang, Qichao and Li, Xiang and Qian, Deheng and Yang, Pengxuan and
          Dong, Yihang and Hao, Ce and Ye, Xiaoqing and Han, Junyu and Pan, Yifeng and Zhao, Dongbin},
  journal={arXiv preprint arXiv:2603.24581},
  year={2026}
}
```

## Acknowledgements and License

We thank [NAVSIM](https://github.com/autonomousvision/navsim),
[nuPlan](https://github.com/motional/nuplan-devkit),
[DINOv2](https://github.com/facebookresearch/dinov2), and
[WorldMirror](https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror).

NAVSIM-derived code is licensed under Apache-2.0. WorldMirror is separately licensed under the
Tencent HunyuanWorld-Mirror Community License Agreement and has additional restrictions.
Read [NOTICE](NOTICE), [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md), and
[WorldMirror's license](worldmirror/License.txt) before use.
