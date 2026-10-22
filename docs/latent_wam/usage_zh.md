# 完整使用指南

包含环境配置、数据准备、评测与训练的详细操作。快速上手见[仓库首页](../../README_zh.md)。

## 环境要求

正式支持环境为：

- Linux x86-64；
- Python 3.10；
- PyTorch 2.4.1 和 torchvision 0.19.1；
- CUDA 12.4；模型推理、WorldMirror 提取和训练需要 NVIDIA GPU；
- PDM 仿真和 ePDMS 评分阶段使用 CPU 计算。

Latent WAM 前向推理在可用时运行于 CUDA。NAVSIM PDM 仿真、几何指标、交通参与者回放以及最终
pandas 聚合目前仍是 CPU 实现。

## 工作区目录

下文命令假设使用以下目录布局：

```text
~/latent_wam_workspace/
├── latent-wam/                     # 当前仓库
├── dataset/
│   ├── maps/
│   ├── navsim_logs/
│   │   ├── trainval/
│   │   └── test/
│   └── sensor_blobs/
│       ├── trainval/
│       └── test/
├── exp/
│   ├── metric_cache/
│   └── latent_wam_cache/
└── models/                         # 可选的本地模型快照
    ├── dinov2-base/
    ├── HunyuanWorld-Mirror/
    └── latent-wam/                 # ModelScope 下载的模型快照
        ├── model.safetensors
        ├── config.json
        └── conversion_report.json
```

创建工作区，并设置 Hydra 使用的路径：

```bash
mkdir -p "$HOME/latent_wam_workspace"/{dataset,exp,models}

export NAVSIM_DEVKIT_ROOT="$HOME/latent_wam_workspace/latent-wam"
export NAVSIM_EXP_ROOT="$HOME/latent_wam_workspace/exp"
export OPENSCENE_DATA_ROOT="$HOME/latent_wam_workspace/dataset"
export NUPLAN_MAPS_ROOT="$OPENSCENE_DATA_ROOT/maps"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
```

每个运行 Latent WAM 的 shell 都需要设置这些变量。机器相关路径应通过 Hydra 或环境变量传入，不要
写死在源代码中。

## 克隆与安装

克隆仓库并同时初始化 WorldMirror 子模块：

```bash
cd "$HOME/latent_wam_workspace"
git clone --recurse-submodules https://github.com/Jinhsi555/Latent-WAM.git latent-wam
cd latent-wam

conda env create -f environment.yml
conda activate latent-wam
pip install -e .
```

如果克隆时没有包含子模块，单独执行：

```bash
git submodule update --init --recursive
```

无需下载模型权重即可验证安装：

```bash
python -c "from navsim.agents.latent_wam.config import LatentWAMConfig; print(LatentWAMConfig())"
```

命令成功打印配置，说明 Python 包及基础配置依赖安装正确。完整依赖范围见
[安装说明](installation.md)。

## 准备 NAVSIM 数据

下载数据前请阅读 nuPlan/OpenScene 的数据许可条款。仓库提供的脚本会把文件下载到当前目录，因此应
在数据根目录运行，再把产物整理成上文目录结构。

### 地图

```bash
cd "$OPENSCENE_DATA_ROOT"
bash "$NAVSIM_DEVKIT_ROOT/download/download_maps.sh"
```

脚本生成 `$OPENSCENE_DATA_ROOT/maps`。

### 训练数据

Hugging Face 脚本下载 NAVSIM `navtrain` 子集，包括当前帧和历史相机数据：

```bash
cd "$OPENSCENE_DATA_ROOT"
bash "$NAVSIM_DEVKIT_ROOT/download/download_navtrain_hf.sh"

mkdir -p navsim_logs sensor_blobs
mv trainval_navsim_logs navsim_logs/trainval
mv trainval_sensor_blobs/trainval sensor_blobs/trainval
rmdir trainval_sensor_blobs
```

如果希望使用四分片 AWS 镜像，可改用 `download/download_navtrain_aws.sh`。完整 trainval 数据可通过
`download/download_trainval.sh` 下载，但标准 `navtrain` 训练 split 不要求完整 trainval 数据。

### 测试数据

论文的一阶段评估使用原始 NAVSIM test 场景：

```bash
cd "$OPENSCENE_DATA_ROOT"
bash "$NAVSIM_DEVKIT_ROOT/download/download_test.sh"

mkdir -p navsim_logs sensor_blobs
mv test_navsim_logs navsim_logs/test
mv test_sensor_blobs sensor_blobs/test
```

完成后检查必要目录：

```bash
test -d "$OPENSCENE_DATA_ROOT/navsim_logs/trainval"
test -d "$OPENSCENE_DATA_ROOT/sensor_blobs/trainval"
test -d "$OPENSCENE_DATA_ROOT/navsim_logs/test"
test -d "$OPENSCENE_DATA_ROOT/sensor_blobs/test"
test -d "$NUPLAN_MAPS_ROOT"
```

## 模型资源

Latent WAM 默认通过 Hugging Face 解析模型标识，也可以通过 Hydra 参数传入本地快照目录。

| 资源 | 默认标识 | 用途 | 状态 |
| --- | --- | --- | --- |
| DINOv2-Base | `facebook/dinov2-base` | 推理、缓存生成、训练、评估 | 可公开获取 |
| WorldMirror | `tencent/HunyuanWorld-Mirror` | 仅用于几何缓存生成 | 使用独立许可证 |
| Latent WAM | [`Jinhsi555/Latent-WAM`](https://modelscope.cn/models/Jinhsi555/Latent-WAM) | 推理与评估 | 已发布 |

### 下载发布权重

在已激活的环境中安装下载客户端，并获取公开模型快照：

```bash
pip install modelscope
python - <<'PY'
import os
from modelscope import snapshot_download

snapshot_download(
    "Jinhsi555/Latent-WAM",
    local_dir=os.path.expanduser("~/latent_wam_workspace/models/latent-wam"),
)
PY

export LATENT_WAM_CHECKPOINT="$HOME/latent_wam_workspace/models/latent-wam/model.safetensors"
```

`model.safetensors` 的 SHA-256 应为：

```text
b25a52e3d96b7d92faf99e6d9a61eccabb07137ca93c10d83bc1422ef0c17bb1
```

使用 `sha256sum "$LATENT_WAM_CHECKPOINT"` 校验。严格加载并执行一次单样本推理 smoke test：

```bash
cd "$NAVSIM_DEVKIT_ROOT"
python scripts/validate_latent_wam_checkpoint.py "$LATENT_WAM_CHECKPOINT"
```

公开 safetensors 包含模型权重，不包含 optimizer 状态，适合推理与评测；恢复中断训练需要对应的
Lightning `.ckpt`。附带的 `config.json` 是 Latent WAM 专用配置，不能直接交给 Hugging Face `AutoModel`。

离线运行时可以设置本地模型快照：

```bash
export DINOV2_PATH="$HOME/latent_wam_workspace/models/dinov2-base"
export WORLD_MIRROR_PATH="$HOME/latent_wam_workspace/models/HunyuanWorld-Mirror"
export LATENT_WAM_CHECKPOINT="$HOME/latent_wam_workspace/models/latent-wam/model.safetensors"
```

如果没有设置 `DINOV2_PATH` 或 `WORLD_MIRROR_PATH`，请在后续命令中删除对应的 Hydra override，使用
默认 Hugging Face 标识。推理和评估不需要 WorldMirror。

## NAVSIM 一阶段评估

这是 Latent WAM 使用的论文评估协议，需要 test split、nuPlan 地图、兼容的 Latent WAM checkpoint
以及 DINOv2-Base。发布权重可按上文步骤从 ModelScope 下载。

### 1. 生成 NAVSIM metric cache

metric cache 与 Latent WAM checkpoint 无关，只需要生成一次：

```bash
cd "$NAVSIM_DEVKIT_ROOT"
python navsim/planning/script/run_metric_caching.py \
  train_test_split=navtest \
  metric_cache_path="$NAVSIM_EXP_ROOT/metric_cache"
```

缓存写入 `$NAVSIM_EXP_ROOT/metric_cache`。相同数据和指标配置的后续评估应复用该缓存。

### 2. 评估 Latent WAM

```bash
cd "$NAVSIM_DEVKIT_ROOT"
python scripts/evaluate_latent_wam_epdms.py \
  --checkpoint "$LATENT_WAM_CHECKPOINT" \
  --navsim-log-path "$OPENSCENE_DATA_ROOT/navsim_logs/test" \
  --sensor-path "$OPENSCENE_DATA_ROOT/sensor_blobs/test" \
  --vision-model "$DINOV2_PATH" \
  --metric-cache "$NAVSIM_EXP_ROOT/metric_cache" \
  --output "$NAVSIM_EXP_ROOT/latent_wam_eval/navtest.csv"
```

如果需要从 Hugging Face 自动获取 DINOv2，请删除 `--vision-model` 选项。脚本复用第 1 步生成的
metric cache，在可用 GPU 上完成模型推理，并写出标准 NAVSIM score CSV。生成 CSV 的
`average_all_frames` 行中，`score` 列即为最终一阶段 ePDMS。短 smoke test 可以添加
`--max-scenarios 100`；正式评测时省略该参数。默认开启 human-penalty filter，与报告中的 NAVSIM
ePDMS 协议一致。

多加速卡评测可使用从已验证并行评测流程中正式化的启动脚本。该脚本会把推理任务分配到
指定设备，按 split 顺序合并轨迹，并将 PDM 评分调度到多个 CPU worker：

```bash
cd "$NAVSIM_DEVKIT_ROOT"
LATENT_WAM_CHECKPOINT="$LATENT_WAM_CHECKPOINT" \
DINOV2_PATH="$DINOV2_PATH" \
EVALUATION_DEVICES=0,1 PDM_WORKERS=8 \
INFERENCE_BATCH_SIZE=32 \
bash scripts/evaluation/run_latent_wam_epdms_parallel.sh
```

启动脚本会根据自身位置解析仓库路径，并要求设置 `NAVSIM_EXP_ROOT`、`OPENSCENE_DATA_ROOT` 和
`LATENT_WAM_CHECKPOINT`。如不使用 `NAVSIM_EXP_ROOT` 下的默认位置，可覆盖 `METRIC_CACHE_PATH` 或
`EVALUATION_OUTPUT`。`INFERENCE_WORKERS_PER_DEVICE`、`INFERENCE_BATCH_SIZE`、`PDM_CHUNK_SIZE` 和
`PDM_THREADS_PER_WORKER` 用于控制主要并行参数。脚本末尾的额外参数会传递给
`scripts/evaluate_latent_wam_epdms_parallel.py`；添加 `--max-scenarios 100` 可执行短 smoke test。

`EVALUATION_DEVICES` 控制模型推理设备，`PDM_WORKERS` 独立控制 CPU 评分进程数，可以超过 GPU 数量。
推理 batch size 应按显存调整。比较分数时需保持 split、metric cache 和 human-penalty 设置一致；
CSV 聚合行中的 `score` 乘以 100 后可与仓库首页的论文结果比较。

## 从头训练 Latent WAM

从头训练不需要正式 Latent WAM checkpoint，但需要 NAVSIM `trainval` 数据、DINOv2-Base、
WorldMirror 和 Latent WAM 分片缓存。训练采用 `[-1.5, 0.0, 2.0, 4.0]` 秒四帧 teacher forcing。

WorldMirror 只在提取几何监督时加载，训练循环和正式推理路径都不会实例化 WorldMirror。

### 1. 单 GPU 生成缓存

```bash
cd "$NAVSIM_DEVKIT_ROOT"
python navsim/planning/script/run_latent_wam_caching.py \
  agent=latent_wam_agent \
  agent.config.vision.model_id="$DINOV2_PATH" \
  agent.config.geometry.model_id="$WORLD_MIRROR_PATH" \
  train_test_split=trainval \
  cache_path="$NAVSIM_EXP_ROOT/cache"
```

可以删除 DINOv2 override 以使用 Hugging Face 标识。cache 生成会从配置的模型目录加载 WorldMirror
并提取 geometry。唯一 NAVSIM frame 被写入
边界对齐的 image/geometry safetensors shard，sample 通过 Parquet 索引引用四个 frame token。只有
`$NAVSIM_EXP_ROOT/cache/manifest.json` 发布后，缓存才可用于训练。

中断后重启会按 frame token、tensor 元数据和文件大小复用已完成 shard；只有需要重建全部 shard 时才添加
`force_cache_computation=true`。旧 sample-level 缓存没有兼容 reader，必须重新生成。详见
[缓存格式说明](cache-format.md)。

### 2. 可选的分布式缓存生成

首先在不加载 WorldMirror 的情况下，检查每个节点的 source 可见性和本地 plan 摘要：

```bash
cd "$NAVSIM_DEVKIT_ROOT"
NNODES=2 NPROC_PER_NODE=8 TRAIN_TEST_SPLIT=trainval CACHE_PATH="$NAVSIM_EXP_ROOT/cache" \
NODE_RANK="$NODE_RANK" MASTER_ADDR="$MASTER_ADDR" MASTER_PORT="$MASTER_PORT" \
DINOV2_PATH="$DINOV2_PATH" WORLD_MIRROR_PATH="$WORLD_MIRROR_PATH" \
bash scripts/run_dataset_caching_multi_node.sh distributed_dry_run=true
```

检查通过后，删除 `distributed_dry_run=true` 并再次运行。每个 immutable shard 只有一个 owner rank；
rank 0 校验、合并 Parquet fragment 并发布最终 manifest。同一缓存根目录同一时间只能运行一个
`torchrun` 作业。

### 3. 训练

在每个节点各执行一次启动脚本，并使用相同的共享缓存和输出路径。
默认入口为 `all_navtrain_training`，将 train 和 val 日志合并训练，保留 7 月复现的
100 个 epoch、BF16、学习率峰值 `2e-4`、余弦衰减终值 `1e-6`、每个 DataLoader 进程
12 个 worker，以及每轮监控 5 个验证 batch。这里的验证 loss 来自训练数据子集。
缓存应覆盖 `trainval`，如上面的生成命令所示；已有匹配的 trainval 缓存可直接复用。

默认拓扑为 8 节点 × 8 卡，每卡 batch 8，梯度累积 1：
`NNODES * NPROC_PER_NODE * BATCH_SIZE * ACCUMULATE_GRAD_BATCHES = 512`。
采样使用 `global`，时序 attention 默认使用块下三角 `block_causal`，每 5 个 epoch 保存完整 `.ckpt`。

```bash
cd "$NAVSIM_DEVKIT_ROOT"
NNODES=8 NPROC_PER_NODE=8 NODE_RANK="$NODE_RANK" \
MASTER_ADDR="$MASTER_ADDR" MASTER_PORT="$MASTER_PORT" \
bash scripts/training/run_latent_wam_training_multi_node.sh
```

该脚本兼容 `MLP_WORKER_NUM`、`MLP_ROLE_INDEX`、`MLP_WORKER_0_HOST` 和
`MLP_WORKER_0_PORT` 集群变量。单卡运行可设置 `NNODES=1 NPROC_PER_NODE=1`，此时有效 batch 为 8。
`BATCH_SIZE`、`MAX_EPOCHS`、`ACCUMULATE_GRAD_BATCHES`、`LIMIT_VAL_BATCHES`、`LEARNING_RATE`
和 `CACHE_SAMPLER_MODE` 也可通过环境变量覆盖。

需要独立的 train/val 划分时，设置 `TRAINING_CONFIG=default_training`；完整验证设置
`LIMIT_VAL_BATCHES=1.0`。默认 experiment UID 为 `navtrain_bs512_epoch100_lr2e-4_global`，
新实验可设置 `NAVSIM_EXPERIMENT_UID`。所有节点必须一致地解析 `NAVSIM_EXP_ROOT`、`CACHE_PATH`
和模型路径。脚本末尾的额外参数会作为 Hydra override 传递；例如添加
`resume_ckpt_path=/shared/checkpoints/last.ckpt` 可恢复兼容的 Lightning checkpoint。

删除本地 DINOv2 override 即可使用 `facebook/dinov2-base`。训练产物、Lightning 日志和周期性
checkpoint 保存在 `$NAVSIM_EXP_ROOT/training/<experiment_uid>/`。target DINO encoder 从
online encoder 初始化，并在训练中通过 EMA 更新。loss 定义和分布式日志行为见
[训练说明](training.md)。

cache 生成和训练应使用相同的 `agent.config.geometry.model_id` 值，使显式 cache 语义保持一致。
训练不会加载 WorldMirror，也不要求本地存在其权重文件。

训练默认跳过重复的全量 safetensors shard header 扫描，同时保留
[缓存格式说明](cache-format.md)中的快速启动检查。缓存被手动复制或修改后，
可设置 `validate_cache_headers=true`，逐个检查 shard 的 tensor key、shape 和 dtype。这个可选审计
会在每个 rank 上分别为 train 和 validation dataset 执行，在远程存储上可能较慢。

训练默认使用 global sample shuffle 和每 worker 8-entry shard LRU。设置
`cache_staging_path=/local/nvme/latent_wam_cache` 可在训练前将共享缓存复制到节点本地 NVMe；设置
`cache_sampler_mode=locality` 可切换到 shard-local block shuffle。
