#!/usr/bin/env bash
set -Eeuo pipefail

WORKSPACE=/mnt/workspace/wlb/world4drive
RUN_ROOT=/mnt/oss_data/latent-wam
CACHE_PATH="$RUN_ROOT/cache"
DATA_ROOT=/mnt/workspace/wlb/Egonav/openscene-v1.1
WORLD_MIRROR_PATH="$WORKSPACE/checkpoints/HunyuanWorld-Mirror"
DINOV2_PATH="$WORKSPACE/checkpoints/dinov2-base"
PYTHON="$WORKSPACE/.venv-ppu-smoke/bin/python"
LOG_PATH=/tmp/latent-wam-training-2ppu.log
STATUS_PATH=/tmp/latent-wam-training-2ppu.status

on_exit() {
  local rc=$?
  printf 'training_exit=%s\nfinished_at=%s\n' "$rc" "$(date --iso-8601=seconds)" \
    | tee -a "$LOG_PATH" > "$STATUS_PATH"
}
trap on_exit EXIT

exec > >(tee -a "$LOG_PATH") 2>&1
cd "$WORKSPACE"

export NAVSIM_DEVKIT_ROOT="$WORKSPACE"
export NAVSIM_EXP_ROOT="$RUN_ROOT"
export OPENSCENE_DATA_ROOT="$DATA_ROOT"
export PYTHONPATH="$WORKSPACE"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=4

echo "[$(date --iso-8601=seconds)] Starting full 2-PPU BF16 training"

set +e
"$PYTHON" navsim/planning/script/run_training.py \
  agent=latent_wam_agent \
  agent.build_target_encoder=true \
  agent.config.vision.model_id="$DINOV2_PATH" \
  agent.config.geometry.model_id="$WORLD_MIRROR_PATH" \
  agent.config.training.learning_rate=2e-4 \
  use_cache_without_dataset=true \
  force_cache_computation=false \
  validate_cache_headers=false \
  cache_path="$CACHE_PATH" \
  train_test_split=navtrain \
  dataloader.params.batch_size=8 \
  dataloader.params.num_workers=12 \
  trainer.params.devices=2 \
  trainer.params.precision=bf16-mixed \
  trainer.params.num_nodes=1 \
  trainer.params.max_epochs=25 \
  trainer.params.accumulate_grad_batches=1 \
  experiment_name=training \
  experiment_uid=navtrain_per_device_bs8_e25_lr2e-4
rc=$?
set -e

echo "[$(date --iso-8601=seconds)] Full training exited with code $rc"
exit "$rc"
