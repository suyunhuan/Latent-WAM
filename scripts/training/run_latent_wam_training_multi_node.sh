#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${NAVSIM_DEVKIT_ROOT:-/mnt/workspace/wlb/world4drive}"
RUN_ROOT="${NAVSIM_EXP_ROOT:-/mnt/oss_data/latent-wam}"
DATA_ROOT="${OPENSCENE_DATA_ROOT:-/mnt/workspace/wlb/Egonav/openscene-v1.1}"
PYTHON="${PYTHON_BIN:-$WORKSPACE/.venv-ppu-smoke/bin/python}"
WORLD_MIRROR_PATH="${WORLD_MIRROR_PATH:-$WORKSPACE/checkpoints/HunyuanWorld-Mirror}"
DINOV2_PATH="${DINOV2_PATH:-$WORKSPACE/checkpoints/dinov2-base}"

export NAVSIM_DEVKIT_ROOT="$WORKSPACE"
export NAVSIM_EXP_ROOT="$RUN_ROOT"
export OPENSCENE_DATA_ROOT="$DATA_ROOT"
export PYTHONPATH="$WORKSPACE${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

NNODES="${NNODES:-${MLP_WORKER_NUM:-1}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NODE_RANK="${NODE_RANK:-${MLP_ROLE_INDEX:-0}}"
MASTER_ADDR="${MASTER_ADDR:-${MLP_WORKER_0_HOST:-127.0.0.1}}"
MASTER_PORT="${MASTER_PORT:-${MLP_WORKER_0_PORT:-29500}}"
TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtrain}"
CACHE_PATH="${CACHE_PATH:-$RUN_ROOT/cache}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-training}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-12}"
PRECISION="${PRECISION:-bf16-mixed}"
MAX_EPOCHS="${MAX_EPOCHS:-25}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-1}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
NAVSIM_EXPERIMENT_UID="${NAVSIM_EXPERIMENT_UID:-navtrain_per_device_bs8_e25_lr2e-4}"
TORCHRUN_BIN="${TORCHRUN_BIN:-}"

if [[ -n "$TORCHRUN_BIN" ]]; then
  TORCHRUN_COMMAND=("$TORCHRUN_BIN")
else
  TORCHRUN_COMMAND=("$PYTHON" -m torch.distributed.run)
fi

MODEL_OVERRIDES=()
MODEL_OVERRIDES+=("agent.config.geometry.model_id=$WORLD_MIRROR_PATH")
MODEL_OVERRIDES+=("agent.config.vision.model_id=$DINOV2_PATH")

CACHE_OVERRIDES=()
if [[ -n "${CACHE_STAGING_PATH:-}" ]]; then
  CACHE_OVERRIDES+=("cache_staging_path=$CACHE_STAGING_PATH")
fi

"${TORCHRUN_COMMAND[@]}" \
  --nnodes="$NNODES" \
  --nproc_per_node="$NPROC_PER_NODE" \
  --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  "$WORKSPACE/navsim/planning/script/run_training.py" \
  agent=latent_wam_agent \
  agent.build_target_encoder=true \
  agent.config.training.learning_rate="$LEARNING_RATE" \
  use_cache_without_dataset=true \
  force_cache_computation=false \
  validate_cache_headers=false \
  cache_path="$CACHE_PATH" \
  cache_sampler_mode=locality \
  train_test_split="$TRAIN_TEST_SPLIT" \
  dataloader.params.batch_size="$BATCH_SIZE" \
  dataloader.params.num_workers="$NUM_WORKERS" \
  trainer.params.devices="$NPROC_PER_NODE" \
  trainer.params.num_nodes="$NNODES" \
  trainer.params.precision="$PRECISION" \
  trainer.params.max_epochs="$MAX_EPOCHS" \
  trainer.params.accumulate_grad_batches="$ACCUMULATE_GRAD_BATCHES" \
  experiment_name="$EXPERIMENT_NAME" \
  experiment_uid="$NAVSIM_EXPERIMENT_UID" \
  "${MODEL_OVERRIDES[@]}" \
  "${CACHE_OVERRIDES[@]}" \
  "$@"
