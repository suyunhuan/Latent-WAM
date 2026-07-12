# Latent WAM

Latent WAM 是基于 NAVSIM v2.1.2 的相机自动驾驶研究实现。推理只使用当前时刻的左、前、右三路
相机图像和当前自车状态，预测未来 4 秒的自车轨迹。WorldMirror 几何特征仅用于生成训练缓存时的
辅助监督。

[English guide](README.md)

## 发布状态

当前版本包含推理、缓存生成、训练和 NAVSIM 一阶段评估代码。正式 Latent WAM checkpoint 尚未包含在
仓库中，将在后续单独发布。在此之前：

- 用户可以安装代码并从头训练；
- 评估需要用户自行提供兼容的 `model.safetensors`；
- 公开 agent 配置为 `agent=latent_wam_agent`；
- 公开正式评测入口为 `scripts/evaluate_latent_wam_epdms.py`；
- 推理输出为 `trajectory`、`all_trajectories` 和 `mode_logits`；
- `navsim/agents/transfuser/` 是独立保留的上游 NAVSIM baseline。

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
    └── model.safetensors           # 正式 checkpoint 尚未发布
```

创建工作区，并设置 Hydra 使用的路径：

```bash
mkdir -p "$HOME/latent_wam_workspace"/{dataset,exp,models}

export NAVSIM_DEVKIT_ROOT="$HOME/latent_wam_workspace/latent-wam"
export NAVSIM_EXP_ROOT="$HOME/latent_wam_workspace/exp"
export OPENSCENE_DATA_ROOT="$HOME/latent_wam_workspace/dataset"
export NUPLAN_MAPS_ROOT="$OPENSCENE_DATA_ROOT/maps"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
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
[安装说明](docs/latent_wam/installation.md)。

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
| Latent WAM | `model.safetensors` | 推理与评估 | 正式 checkpoint 尚未发布 |

离线运行时可以设置本地模型快照：

```bash
export DINOV2_PATH="$HOME/latent_wam_workspace/models/dinov2-base"
export WORLD_MIRROR_PATH="$HOME/latent_wam_workspace/models/HunyuanWorld-Mirror"
export LATENT_WAM_CHECKPOINT="$HOME/latent_wam_workspace/models/model.safetensors"
```

如果没有设置 `DINOV2_PATH` 或 `WORLD_MIRROR_PATH`，请在后续命令中删除对应的 Hydra override，使用
默认 Hugging Face 标识。推理和评估不需要 WorldMirror。

## NAVSIM 一阶段评估

这是 Latent WAM 使用的论文评估协议，需要 test split、nuPlan 地图、兼容的 Latent WAM checkpoint
以及 DINOv2-Base。当前纯代码版本尚不包含正式 checkpoint。

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

## 从头训练 Latent WAM

从头训练不需要正式 Latent WAM checkpoint，但需要 NAVSIM `navtrain` 数据、DINOv2-Base、
WorldMirror 和 Latent WAM 分片缓存。训练采用 `[-1.5, 0.0, 2.0, 4.0]` 秒四帧 teacher forcing。

WorldMirror 只在提取几何监督时加载，训练循环和正式推理路径都不会实例化 WorldMirror。

### 1. 单 GPU 生成缓存

```bash
cd "$NAVSIM_DEVKIT_ROOT"
python navsim/planning/script/run_latent_wam_caching.py \
  agent=latent_wam_agent \
  agent.config.vision.model_id="$DINOV2_PATH" \
  agent.config.geometry.model_id="$WORLD_MIRROR_PATH" \
  train_test_split=navtrain \
  cache_path="$NAVSIM_EXP_ROOT/latent_wam_cache"
```

可以删除 DINOv2 override 以使用 Hugging Face 标识。WorldMirror 必须解析到包含
`model.safetensors` 的本地目录，因为权重 checksum 属于严格 cache identity。唯一 NAVSIM frame 被写入
边界对齐的 image/geometry safetensors shard，sample 通过 Parquet 索引引用四个 frame token。只有
`$NAVSIM_EXP_ROOT/latent_wam_cache/manifest.json` 发布后，缓存才可用于训练。

中断后重启会校验并复用已完成 shard；只有需要重建全部 shard 时才添加
`force_cache_computation=true`。旧 sample-level 缓存没有兼容 reader，必须重新生成。详见
[缓存格式说明](docs/latent_wam/cache-format.md)。

### 2. 可选的分布式缓存生成

首先在不加载 WorldMirror 的情况下，检查所有节点是否看到相同 token 集合：

```bash
cd "$NAVSIM_DEVKIT_ROOT"
NNODES=2 NPROC_PER_NODE=8 NODE_RANK="$NODE_RANK" \
MASTER_ADDR="$MASTER_ADDR" MASTER_PORT="$MASTER_PORT" \
DINOV2_PATH="$DINOV2_PATH" WORLD_MIRROR_PATH="$WORLD_MIRROR_PATH" \
bash scripts/run_dataset_caching_multi_node.sh distributed_dry_run=true
```

检查通过后，删除 `distributed_dry_run=true` 并再次运行。每个 immutable shard 只有一个 owner rank；
rank 0 校验、合并 Parquet fragment 并发布最终 manifest。同一缓存根目录同一时间只能运行一个
`torchrun` 作业。

### 3. 训练

```bash
cd "$NAVSIM_DEVKIT_ROOT"
python navsim/planning/script/run_training.py \
  agent=latent_wam_agent \
  agent.build_target_encoder=true \
  agent.config.vision.model_id="$DINOV2_PATH" \
  agent.config.geometry.model_id="$WORLD_MIRROR_PATH" \
  use_cache_without_dataset=true \
  force_cache_computation=false \
  cache_path="$NAVSIM_EXP_ROOT/latent_wam_cache" \
  train_test_split=navtrain \
  experiment_name=latent_wam_train
```

删除本地 DINOv2 override 即可使用 `facebook/dinov2-base`。训练产物、Lightning 日志和周期性
checkpoint 保存在 `$NAVSIM_EXP_ROOT/latent_wam_train/<experiment_uid>/`。target DINO encoder 从
online encoder 初始化，并在训练中通过 EMA 更新。loss 定义和分布式日志行为见
[训练说明](docs/latent_wam/training.md)。

训练配置中应保留 `agent.config.geometry.model_id="$WORLD_MIRROR_PATH"`。训练不会加载 WorldMirror，
但启动时会对本地 `model.safetensors` 计算一次 checksum，以拒绝由不同几何权重生成的缓存。

训练默认使用 locality-aware shard shuffle 和每 worker 8-entry shard LRU。设置
`cache_staging_path=/local/nvme/latent_wam_cache` 可在训练前将共享缓存经 checksum 校验复制到节点本地
NVMe；设置 `cache_sampler_mode=global` 可运行全局随机吞吐对照。

## 许可证

NAVSIM 衍生 Python 代码使用 Apache-2.0。`worldmirror` 子模块受 **Tencent HunyuanWorld-Mirror
Community License Agreement** 约束，该协议包含地域、用途、输出和再分发限制，不属于本仓库
Apache-2.0 Python 包许可证的覆盖范围。

下载 WorldMirror 权重、生成几何监督、训练或重新分发衍生产物前，请阅读 [NOTICE](NOTICE)、
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) 和 `worldmirror/License.txt`。
