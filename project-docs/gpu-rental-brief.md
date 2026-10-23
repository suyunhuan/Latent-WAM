# GPU rental brief — Latent WAM NAVSIM ePDMS

Goal: rent one Linux box so Project **latent-wam** can finish **unmodified** official NAVSIM ePDMS (`navtest`). This document does **not** rent a GPU for you — pick a provider, provision, then connect a Cursor self-hosted worker.

Verified baseline: fork `suyunhuan/Latent-WAM` tip `504508a`; weights SHA `b25a52e3d96b7d92faf99e6d9a61eccabb07137ca93c10d83bc1422ef0c17bb1`; CPU checkpoint smoke already passed. See [`runbook-navisim-inference.md`](runbook-navisim-inference.md) and [`autodl-bootstrap-status.md`](autodl-bootstrap-status.md).

---

## Recommended SKU

| Spec | Minimum | Preferred (rent this) |
| --- | --- | --- |
| GPU | 16 GB VRAM (batch ↓) | **NVIDIA RTX 4090 24 GB** (or A5000/A6000 24–48 GB) |
| Driver / CUDA | Driver supporting **CUDA 12.4** wheels | Same; confirm `nvidia-smi` shows driver ≥ 550-class typical for 12.4 |
| CPU | 8 cores | **16+ cores** (PDM / ePDMS scoring is CPU) |
| RAM | 32 GB | **64 GB** |
| Disk | **≥ 400 GB** free | **500 GB+** data disk (navtest + maps + cache + conda) |
| OS | Linux x86-64 | Ubuntu 22.04 |
| Network | Outbound HTTPS | ModelScope + Hugging Face + OpenScene/AWS reachable |

**Why 24 GB:** released `model.safetensors` ≈ 384 MB plus DINOv2-Base at eval; official launcher default `INFERENCE_BATCH_SIZE=8` (usage examples use up to 32 on multi-GPU). 24 GB is the comfortable single-GPU class; if OOM, keep `INFERENCE_BATCH_SIZE=8` or lower to 4 — do not change model code.

**Not needed for this phase:** multi-node, WorldMirror GPU weights, trainval/navtrain (training-only).

---

## Providers (Asia / Shanghai–friendly)

| Provider | Notes |
| --- | --- |
| **AutoDL** | Common CN academic/hobby GPU rental; RTX 4090 instances; data disk attachable; SSH + Jupyter. Prefer a **data disk ≥ 500 GB** on create. Good default if you already have an AutoDL account. |
| **恒源云 (Hengyuan)** | Similar CN marketplace; watch disk SKU separately from GPU. Use Ubuntu + preinstalled NVIDIA driver images when available. |
| **阿里云** | ECS / PAI with A10/A100/4090-class; more enterprise billing; strong CN network to OSS mirrors. Use if you already have Aliyun credit. |
| **RunPod / Vast.ai** | Overseas; often cheaper $/hr for 4090; from Shanghai expect higher latency to US regions — still fine for multi-hour downloads if disk is large. Prefer EU/Asia pods when listed. |

**Pick order for Shanghai user:** AutoDL or 恒源云 (4090 + 500 GB disk) → else 阿里云 → else RunPod/Vast if CN inventory is empty.

### 火山引擎（火山云）— 可用，但不卖 4090

公有云 ECS **没有 RTX 4090**（消费级卡一般不在大厂目录里）。个人账号可买的推理档是 **GPU 计算型 gni2 = NVIDIA A10 24GB**，对本项目 ePDMS 够用，但时价通常比 AutoDL 4090 **贵数倍**。

| 项 | 选这个 |
| --- | --- |
| 下单 | [ECS 控制台创建实例](https://console.volcengine.com/ecs)（登录后走「向导购买」） |
| 规格文档 | [GPU 实例规格介绍](https://docs.volcengine.com/docs/gpu/Introduction-to-instance-types) |
| 规格 | **`ecs.gni2.3xlarge`**：14 vCPU / 56 GiB / **A10 × 1（24GB）** |
| 地域 | **华北2（北京）** 或 **华东2（上海）**（gni2 主要在这两区） |
| 计费 | 按量计费 |
| 镜像 | Ubuntu 22.04 + 勾选安装 **NVIDIA Tesla 驱动**（或创建后按官方装驱动） |
| 磁盘 | 系统盘 40–100 GB + **数据盘 ≥ 500 GB**（极速型 SSD） |
| 网络 | 默认 VPC；安全组放行 **22**（SSH）；需要公网 IP 才能下数据/起 Cursor worker |

参考价（机器学习平台同档 A10 公开表曾约 **¥10/时** 量级，以控制台实时报价为准）——同任务周期往往远贵于 AutoDL ~¥1.9/时的 4090。

**结论：** 已有火山账号/代金券 → 用上面 gni2.3xlarge；否则 **仍优先 AutoDL 4090**。

Avoid: CPU-only, Apple Silicon remote, Windows GPU boxes, images pinned to CUDA ≪ 12 without upgrade path.

---

## Estimated disk layout

Assume workspace root `$HOME/latent_wam_workspace` (matches official usage.md):

```text
~/latent_wam_workspace/
├── latent-wam/              # git clone (~few GB with submodule)
├── dataset/
│   ├── maps/                # ~1.4 GB unpacked (zip ~971 MB)
│   ├── navsim_logs/test/    # from openscene_metadata_test
│   └── sensor_blobs/test/   # 32 cam + 32 lidar shards
├── exp/
│   ├── metric_cache/        # once; tens of GB class depending on split
│   └── latent_wam_eval/     # CSV outputs
├── models/
│   └── latent-wam/          # ModelScope snapshot (~0.4 GB)
└── (conda envs elsewhere)   # pytorch+CUDA env ~10–20 GB
```

| Item | Size (order of magnitude) | Source |
| --- | --- | --- |
| Official `download_test.sh` | **~202 GB compressed** (meta + 32 cam + 32 lidar; probed) | OpenScene; lidar unused by Latent WAM but script fetches it |
| Maps | ~1–1.5 GB | `download_maps.sh` |
| Weights | ~0.4 GB | ModelScope `Jinhsi555/Latent-WAM` |
| DINOv2 (optional local) | ~0.3 GB | HF `facebook/dinov2-base` (or resolve online) |
| Metric cache + env + headroom | tens of GB | leave slack |

**Budget ≥ 400 GB free** before starting `download_test.sh`. Prefer **500 GB+** so decompress + metric cache do not fill the volume mid-run.

### What to download (inference-only)

1. Repo: `https://github.com/suyunhuan/Latent-WAM` with submodules (WorldMirror not used for eval).
2. Conda env from official `environment.yml` (PyTorch 2.4.1 + `pytorch-cuda=12.4`).
3. `model.safetensors` via ModelScope; verify SHA above.
4. Maps + full `download_test.sh` outputs, rearranged to `navsim_logs/test` and `sensor_blobs/test`.
5. Metric cache via `run_metric_caching.py` (once).
6. DINOv2 from HF at first eval unless `DINOV2_PATH` is set.

Do **not** download navtrain/trainval or HunyuanWorld-Mirror for this ePDMS pass.

---

## Connect Cursor (self-hosted My Machines)

So the Project can drive the rented box: [My Machines](https://cursor.com/docs/cloud-agent/self-hosted/my-machines).

On the GPU box (after SSH):

```bash
# 1. Install Cursor CLI
curl https://cursor.com/install -fsS | bash
agent --version

# 2. Sign in (browser / device flow — do not put API keys on shared argv logs)
agent login

# 3. Start a long-lived worker from the Latent-WAM checkout
cd "$HOME/latent_wam_workspace/latent-wam"   # after clone
agent worker start --name "latent-wam-gpu" --worker-dir "$PWD"
```

Keep `agent worker start` running. Then at [cursor.com/agents](https://cursor.com/agents), select this machine under **My Machines** and continue ePDMS (bootstrap script / runbook).

Notes:

- Worker opens **outbound** HTTPS only; no inbound ports required.
- Prefer `agent login` over pasting `--api-key` into shell history on shared rental images.
- Register the **git remote** of `suyunhuan/Latent-WAM` so routing matches the Project repo.

---

## Chosen order (2026-10-22)

**Platform:** [AutoDL 算力市场](https://www.autodl.com/market/list)（首页标价 RTX 4090 约 ¥1.88/时，以市场实时为准）

**User cart (approved):** RTX **4090D** 24GB ×1 · PyTorch **2.5.1** / Python 3.12 / Ubuntu 22.04 / **CUDA 12.4** · 16 vCPU · 62GB RAM · 系统盘 30GB + **付费数据盘 500GB** · 按量 ~¥1.88–1.98/时. Put all Latent-WAM data under the data disk (usually `/root/autodl-tmp`).

| 项 | 选这个 |
| --- | --- |
| GPU | **首选 1× RTX 4090 24GB**；无货依次：**4090D → 3090 → A5000 → V100 32GB**。V100 与官方 2.4.1/CUDA 12.4 兼容性好，但比 40 系慢。**V100 16GB / 4080 / A4000**：可跑，需把 `INFERENCE_BATCH_SIZE` 降到 4/2。**5090 32GB**：显存够，但 Blackwell 可能不兼容官方 12.4，仅无货时用，镜像选 CUDA **12.8** 并准备升级 torch。单卡即可，勿囤多卡 |
| 地区 | 有货优先：**西北** 或 **内蒙**；缺货就换区 / 换卡型，不要死等北京 4090 |
| 计费 | **按量计费**（先跑通 ePDMS；下载数据时可关机只留数据盘） |
| 镜像 | 见下方「AutoDL 镜像怎么选」——**没有 2.4 是正常的** |
| 数据盘 | **扩到 ≥ 500 GB**（navtest ~202 GB 压缩包 + 解压/cache/conda；系统盘默认不够） |
| CPU/内存 | 随 1 卡分配即可（通常够用）；若同价有更高内存档优先选 |

下单路径：打开市场 → 筛选 **RTX 4090** → 选空闲主机 → **创建实例** → 数据盘 **500GB+** → 开机 → SSH。

### AutoDL 镜像怎么选（无 PyTorch 2.4 / 看不到 CUDA 时）

AutoDL **基础镜像列表里本来就没有 PyTorch 2.4**（官方表是 2.3 / **2.5.1** / 2.7 / 2.8…）。CUDA 也不是单独一级菜单：先选框架，版本会带上对应 CUDA。说明见 [基础环境配置](https://www.autodl.com/docs/base_config/)。

**推荐（二选一）：**

1. **首选 — 最接近官方 CUDA 12.4**  
   创建实例 → 镜像选 **PyTorch**（不要只翻「Miniconda」）→ 版本选 **2.5.1** → 应显示 **Python 3.12 / CUDA 12.4 / Ubuntu 22.04**。  
   进机后仍用仓库 `conda env create -f environment.yml` 装出官方 **latent-wam（PyTorch 2.4.1 + cuda 12.4）** 环境；镜像自带 torch 可忽略。

2. **备选 — 干净 Miniconda（专为 environment.yml）**  
   镜像选 **Miniconda** → CUDA 尽量选 **11.8 或更高**（有 12.x 更好）→ 进机后只靠 `environment.yml` 装齐，不依赖镜像里的 torch。

不要为了「必须看到 2.4」卡住下单；**2.5.1+CUDA12.4** 或 **Miniconda** 都能继续，真正环境以 `environment.yml` 为准。

备用： [恒源云控制台](https://gpushare.com/) 云市场同样选 1×4090 + 大磁盘（若 `/hy-tmp` 默认小，立刻扩容到 ≥500GB）。

---

## One-sentence next step for the human

**打开 [AutoDL 算力市场](https://www.autodl.com/market/list) 租 1×RTX 4090 + 数据盘 ≥500GB → SSH → `agent login` + `agent worker start --name latent-wam-gpu` → 跑 [`gpu-bootstrap.md`](gpu-bootstrap.md)。**
