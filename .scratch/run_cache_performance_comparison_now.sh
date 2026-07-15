#!/usr/bin/env bash
set -Eeuo pipefail

WORKSPACE=/mnt/workspace/wlb/world4drive
BASELINE=/tmp/world4drive-baseline-head
OUTPUT_ROOT=/mnt/oss_data/latent-wam/benchmarks/single_ppu_bs16_repeated_sample_concurrent
PYTHON="$WORKSPACE/.venv-ppu-smoke/bin/python"
OLD_CACHE=/tmp/latent-wam-current-cache
NEW_CACHE=/tmp/latent-wam-sharded-final-smoke
VISION_MODEL="$WORKSPACE/checkpoints/dinov2-base"
SAMPLE_TOKEN=a6a284584b19599f

mkdir -p "$OUTPUT_ROOT"
exec > >(tee -a "$OUTPUT_ROOT/benchmark.log") 2>&1
cd "$WORKSPACE"
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1

COMMON_ARGS=(
  --mode train
  --vision-model "$VISION_MODEL"
  --batch-size 16
  --num-workers 12
  --warmup-steps 50
  --measure-steps 500
)

echo "[$(date --iso-8601=seconds)] Starting concurrent pre-refactor end-to-end benchmark on physical PPU 1"
PYTHONPATH="$BASELINE:$WORKSPACE/.scratch" "$PYTHON" .scratch/run_baseline_training_benchmark.py \
  --cache-path "$OLD_CACHE" \
  --checkpoint /tmp/unused-latent-wam-checkpoint.ckpt \
  --output "$OUTPUT_ROOT/sample-gzip.json" \
  "${COMMON_ARGS[@]}"

echo "[$(date --iso-8601=seconds)] Starting concurrent current end-to-end benchmark on physical PPU 1"
PYTHONPATH="$WORKSPACE" "$PYTHON" scripts/benchmark_latent_wam_training.py \
  --cache-path "$NEW_CACHE" \
  --output "$OUTPUT_ROOT/sharded.json" \
  "${COMMON_ARGS[@]}" \
  --sample-token "$SAMPLE_TOKEN"

echo "[$(date --iso-8601=seconds)] Concurrent cache performance comparison complete"
