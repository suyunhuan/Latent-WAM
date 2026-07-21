# Latent WAM

Latent WAM is a camera-based autonomous-driving research implementation built on NAVSIM v2.1.2. At
inference time it uses only the current left/front/right camera images and current ego status to predict a
four-second ego trajectory. WorldMirror geometry is used only as auxiliary supervision while building training
caches.

[中文使用指南](README_zh.md)

## Release status

This release contains the inference, cache-generation, training, and single-stage NAVSIM evaluation code. The
official Latent WAM checkpoint is not included yet and will be released separately. Until then:

- installation and from-scratch training are available;
- evaluation requires a compatible `model.safetensors` supplied by the user;
- the public agent is `agent=latent_wam_agent`;
- the public evaluation entry point is `scripts/evaluate_latent_wam_epdms.py`;
- inference returns `trajectory`, `all_trajectories`, and `mode_logits`;
- `navsim/agents/transfuser/` remains an independent upstream NAVSIM baseline.

## Requirements

The supported environment is:

- Linux x86-64;
- Python 3.10;
- PyTorch 2.4.1 and torchvision 0.19.1;
- CUDA 12.4 and an NVIDIA GPU for model inference, WorldMirror extraction, and training;
- CPU execution for the PDM simulation and ePDMS scoring stages of evaluation.

The Latent WAM forward pass runs on CUDA when available. NAVSIM PDM simulation, geometric metrics, traffic
agent replay, and final pandas aggregation are currently CPU implementations.

## Workspace layout

The commands below assume this layout:

```text
~/latent_wam_workspace/
├── latent-wam/                     # this repository
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
└── models/                         # optional local model snapshots
    ├── dinov2-base/
    ├── HunyuanWorld-Mirror/
    └── model.safetensors           # official checkpoint: not released yet
```

Create the workspace and define the paths used by Hydra:

```bash
mkdir -p "$HOME/latent_wam_workspace"/{dataset,exp,models}

export NAVSIM_DEVKIT_ROOT="$HOME/latent_wam_workspace/latent-wam"
export NAVSIM_EXP_ROOT="$HOME/latent_wam_workspace/exp"
export OPENSCENE_DATA_ROOT="$HOME/latent_wam_workspace/dataset"
export NUPLAN_MAPS_ROOT="$OPENSCENE_DATA_ROOT/maps"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
```

Set these variables in every shell used for Latent WAM. Paths are supplied through Hydra or environment
variables; do not hard-code machine-specific paths in the source tree.

## Clone and install

Clone the repository with the WorldMirror submodule:

```bash
cd "$HOME/latent_wam_workspace"
git clone --recurse-submodules https://github.com/Jinhsi555/Latent-WAM.git latent-wam
cd latent-wam

conda env create -f environment.yml
conda activate latent-wam
pip install -e .
```

If the repository was cloned without submodules, initialize WorldMirror separately:

```bash
git submodule update --init --recursive
```

Verify the installation without downloading model weights:

```bash
python -c "from navsim.agents.latent_wam.config import LatentWAMConfig; print(LatentWAMConfig())"
```

The command succeeds when the package and its configuration dependencies are installed correctly. See
[installation details](docs/latent_wam/installation.md) for the exact dependency scope.

## Prepare NAVSIM data

Read the nuPlan/OpenScene dataset terms before downloading the data. The supplied scripts download into the
current directory, so run them from the dataset root and then move their outputs into the layout above.

### Maps

```bash
cd "$OPENSCENE_DATA_ROOT"
bash "$NAVSIM_DEVKIT_ROOT/download/download_maps.sh"
```

This creates `$OPENSCENE_DATA_ROOT/maps`.

### Training split

The Hugging Face script downloads the NAVSIM `navtrain` subset, including current and history camera blobs:

```bash
cd "$OPENSCENE_DATA_ROOT"
bash "$NAVSIM_DEVKIT_ROOT/download/download_navtrain_hf.sh"

mkdir -p navsim_logs sensor_blobs
mv trainval_navsim_logs navsim_logs/trainval
mv trainval_sensor_blobs/trainval sensor_blobs/trainval
rmdir trainval_sensor_blobs
```

Use `download/download_navtrain_aws.sh` instead when the four-part AWS mirror is preferred. Full trainval data
can be downloaded with `download/download_trainval.sh`, but it is not required for the standard `navtrain`
training split.

### Test split

The single-stage paper evaluation uses the original NAVSIM test scenes:

```bash
cd "$OPENSCENE_DATA_ROOT"
bash "$NAVSIM_DEVKIT_ROOT/download/download_test.sh"

mkdir -p navsim_logs sensor_blobs
mv test_navsim_logs navsim_logs/test
mv test_sensor_blobs sensor_blobs/test
```

After preparation, verify the required directories:

```bash
test -d "$OPENSCENE_DATA_ROOT/navsim_logs/trainval"
test -d "$OPENSCENE_DATA_ROOT/sensor_blobs/trainval"
test -d "$OPENSCENE_DATA_ROOT/navsim_logs/test"
test -d "$OPENSCENE_DATA_ROOT/sensor_blobs/test"
test -d "$NUPLAN_MAPS_ROOT"
```

## Model resources

Latent WAM resolves model identifiers through Hugging Face by default. Each identifier may instead be
overridden with a local snapshot directory.

| Resource | Default identifier | Required for | Status |
| --- | --- | --- | --- |
| DINOv2-Base | `facebook/dinov2-base` | inference, cache generation, training, evaluation | publicly resolvable |
| WorldMirror | `tencent/HunyuanWorld-Mirror` | geometry-cache generation only | separately licensed |
| Latent WAM | `model.safetensors` | inference and evaluation | official checkpoint not released yet |

For offline use, define local snapshots:

```bash
export DINOV2_PATH="$HOME/latent_wam_workspace/models/dinov2-base"
export WORLD_MIRROR_PATH="$HOME/latent_wam_workspace/models/HunyuanWorld-Mirror"
export LATENT_WAM_CHECKPOINT="$HOME/latent_wam_workspace/models/model.safetensors"
```

If `DINOV2_PATH` or `WORLD_MIRROR_PATH` is not set, omit the corresponding Hydra override in later commands.
WorldMirror is not needed for inference or evaluation.

## Single-stage NAVSIM evaluation

This is the evaluation protocol reported by Latent WAM. It requires the test split, nuPlan maps, a compatible
Latent WAM checkpoint, and DINOv2-Base. The official checkpoint is not part of the current code-only release.

### 1. Build the NAVSIM metric cache

The metric cache is independent of the Latent WAM checkpoint and is generated once:

```bash
cd "$NAVSIM_DEVKIT_ROOT"
python navsim/planning/script/run_metric_caching.py \
  train_test_split=navtest \
  metric_cache_path="$NAVSIM_EXP_ROOT/metric_cache"
```

The cache is written to `$NAVSIM_EXP_ROOT/metric_cache`. Reuse it for subsequent evaluations of the same data
and metric configuration.

### 2. Evaluate Latent WAM

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

When DINOv2 should be resolved from Hugging Face, remove the `--vision-model` option. The script reuses the
metric cache from step 1, runs model inference on the available GPU, and writes a standard NAVSIM score CSV.
The final single-stage ePDMS is the `score` value in the `average_all_frames` row. For a short smoke test, add
`--max-scenarios 100`; omit it for the complete split. The human-penalty filter is enabled by default to match
the reported NAVSIM ePDMS protocol.

For multi-accelerator evaluation, use the formal launcher extracted from the verified parallel evaluation
workflow. It partitions inference across the selected devices, merges trajectories in split order, and schedules
PDM scoring across CPU workers:

```bash
cd "$NAVSIM_DEVKIT_ROOT"
LATENT_WAM_CHECKPOINT="$LATENT_WAM_CHECKPOINT" \
DINOV2_PATH="$DINOV2_PATH" \
EVALUATION_DEVICES=0,1 PDM_WORKERS=8 \
bash scripts/evaluation/run_latent_wam_epdms_parallel.sh
```

The launcher derives the repository path from its own location and requires `NAVSIM_EXP_ROOT`,
`OPENSCENE_DATA_ROOT`, and `LATENT_WAM_CHECKPOINT`. Override `METRIC_CACHE_PATH` or `EVALUATION_OUTPUT` when the
default locations under `NAVSIM_EXP_ROOT` are not suitable. `INFERENCE_WORKERS_PER_DEVICE`,
`INFERENCE_BATCH_SIZE`, `PDM_CHUNK_SIZE`, and `PDM_THREADS_PER_WORKER` expose the main parallelism controls. Extra
arguments are forwarded to `scripts/evaluate_latent_wam_epdms_parallel.py`; append `--max-scenarios 100` for a
short smoke test.

## Train Latent WAM from scratch

Training does not require the official Latent WAM checkpoint. It requires the NAVSIM `navtrain` data,
DINOv2-Base, WorldMirror, and a Latent WAM sharded cache. The training path is four-frame teacher forcing at
`[-1.5, 0.0, 2.0, 4.0]` seconds.

WorldMirror is loaded only while extracting geometry supervision. It is not instantiated by the training loop
or by the released inference path.

### 1. Generate the cache on one GPU

```bash
cd "$NAVSIM_DEVKIT_ROOT"
python navsim/planning/script/run_latent_wam_caching.py \
  agent=latent_wam_agent \
  agent.config.vision.model_id="$DINOV2_PATH" \
  agent.config.geometry.model_id="$WORLD_MIRROR_PATH" \
  train_test_split=navtrain \
  cache_path="$NAVSIM_EXP_ROOT/latent_wam_cache"
```

The DINOv2 override may be removed to use its Hugging Face identifier. WorldMirror must resolve to a local
directory containing `model.safetensors`, because its checksum is part of the strict cache identity. Unique
NAVSIM frames are packed into aligned image/geometry safetensors shards; samples reference four frame tokens through Parquet indices.
The cache becomes trainable only when `$NAVSIM_EXP_ROOT/latent_wam_cache/manifest.json` is published.

Completed shards are checksum-validated and reused after an interrupted run. Add
`force_cache_computation=true` only to rebuild every shard. Older sample-level caches have no compatibility
reader and must be regenerated. See [the cache-format specification](docs/latent_wam/cache-format.md).

### 2. Optional distributed cache generation

First verify that all nodes see the same token set without loading WorldMirror:

```bash
cd "$NAVSIM_DEVKIT_ROOT"
NNODES=2 NPROC_PER_NODE=8 NODE_RANK="$NODE_RANK" \
MASTER_ADDR="$MASTER_ADDR" MASTER_PORT="$MASTER_PORT" \
DINOV2_PATH="$DINOV2_PATH" WORLD_MIRROR_PATH="$WORLD_MIRROR_PATH" \
bash scripts/run_dataset_caching_multi_node.sh distributed_dry_run=true
```

Then run the same command without `distributed_dry_run=true` to generate the cache. Each immutable shard has
exactly one owner rank; rank zero validates and merges the Parquet fragments and publishes the final manifest.
Run only one `torchrun` job against a cache root at a time.

### 3. Train

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

For multi-node training, run the launcher once on every node with the same shared cache/output paths. The launcher
defaults match the verified local run: BF16, batch size 8 per device, 12 DataLoader workers, 25 epochs, learning rate
`2e-4`, and cache path `/mnt/oss_data/latent-wam/cache`. `BATCH_SIZE` is per device, so the global batch size is
`NNODES * NPROC_PER_NODE * BATCH_SIZE * ACCUMULATE_GRAD_BATCHES`.

```bash
cd "$NAVSIM_DEVKIT_ROOT"
NNODES=2 NPROC_PER_NODE=8 NODE_RANK="$NODE_RANK" \
MASTER_ADDR="$MASTER_ADDR" MASTER_PORT="$MASTER_PORT" \
bash scripts/training/run_latent_wam_training_multi_node.sh
```

The launcher accepts the same `MLP_WORKER_NUM`, `MLP_ROLE_INDEX`, `MLP_WORKER_0_HOST`, and `MLP_WORKER_0_PORT`
fallbacks as distributed cache generation. It uses the same default experiment UID as the local run; set
`NAVSIM_EXPERIMENT_UID` explicitly when starting a new experiment. All nodes must resolve `NAVSIM_EXP_ROOT`,
`CACHE_PATH`, and model paths consistently. Additional arguments are forwarded as Hydra overrides; for example,
append `resume_ckpt_path=/shared/checkpoints/last.ckpt` to resume a Lightning checkpoint.

Remove the local DINOv2 override to use `facebook/dinov2-base`. Training outputs, Lightning logs, and periodic
checkpoints are stored under `$NAVSIM_EXP_ROOT/latent_wam_train/<experiment_uid>/`. The target DINO encoder is
initialized from the online encoder and updated with EMA during training. See
[training details](docs/latent_wam/training.md) for the loss definitions and distributed logging behavior.

Keep `agent.config.geometry.model_id="$WORLD_MIRROR_PATH"` in the training configuration. Training does not load
WorldMirror, but startup hashes its local `model.safetensors` once to reject caches generated with different geometry weights.

Training skips the redundant all-shard safetensors header scan by default while retaining the fast startup checks
described in [the cache-format specification](docs/latent_wam/cache-format.md). Set `validate_cache_headers=true`
to audit every shard's tensor keys, shapes, and dtypes after a manual copy or modification. The opt-in audit runs
for both train and validation datasets on every rank, does not recompute payload checksums, and can be slow on
remote storage.

Training uses locality-aware shard shuffle and an eight-entry per-worker shard LRU by default. Set
`cache_staging_path=/local/nvme/latent_wam_cache` to checksum-copy a shared cache to node-local NVMe before
training, or `cache_sampler_mode=global` for a global-random throughput comparison.

## License

The NAVSIM-derived Python source is distributed under Apache-2.0. The `worldmirror` submodule is governed by
the **Tencent HunyuanWorld-Mirror Community License Agreement**, which includes geographic, use, output, and
redistribution restrictions. It is not covered by this repository's Apache-2.0 package license.

Before downloading WorldMirror weights, generating geometry supervision, training, or redistributing derived
artifacts, read [NOTICE](NOTICE), [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md), and
`worldmirror/License.txt`.
