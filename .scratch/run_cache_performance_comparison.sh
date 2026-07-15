#!/usr/bin/env bash
set -Eeuo pipefail

WORKSPACE=/mnt/workspace/wlb/world4drive
RUN_ROOT=/mnt/oss_data/latent-wam
STATUS_FILE="$RUN_ROOT/status.env"
OUTPUT_ROOT="$RUN_ROOT/benchmarks/single_ppu_bs16_repeated_sample"
PYTHON="$WORKSPACE/.venv-ppu-smoke/bin/python"
OLD_CACHE=/tmp/latent-wam-current-cache
NEW_CACHE="$RUN_ROOT/cache"
VISION_MODEL="$WORKSPACE/checkpoints/dinov2-base"
SAMPLE_TOKEN=a6a284584b19599f

while true; do
  status=$(sed -n 's/^status=//p' "$STATUS_FILE" 2>/dev/null || true)
  if [[ "$status" == complete ]]; then
    break
  fi
  if [[ "$status" == failed ]]; then
    echo "Current cache/training task failed; benchmark will not start." >&2
    exit 1
  fi
  sleep 60
done

mkdir -p "$OUTPUT_ROOT"
exec > >(tee -a "$OUTPUT_ROOT/benchmark.log") 2>&1
cd "$WORKSPACE"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export PYTHONPATH="$WORKSPACE/.scratch:$WORKSPACE"
export PYTHONUNBUFFERED=1

COMMON_ARGS=(
  --mode train
  --vision-model "$VISION_MODEL"
  --batch-size 16
  --num-workers 12
  --warmup-steps 50
  --measure-steps 500
  --sample-token "$SAMPLE_TOKEN"
)

echo "[$(date --iso-8601=seconds)] Starting legacy sample-gzip benchmark"
"$PYTHON" .scratch/run_legacy_cache_benchmark.py \
  --cache-path "$OLD_CACHE" \
  --output "$OUTPUT_ROOT/sample-gzip.json" \
  "${COMMON_ARGS[@]}"

echo "[$(date --iso-8601=seconds)] Starting sharded cache benchmark"
"$PYTHON" scripts/benchmark_latent_wam_training.py \
  --cache-path "$NEW_CACHE" \
  --output "$OUTPUT_ROOT/sharded.json" \
  "${COMMON_ARGS[@]}"

echo "[$(date --iso-8601=seconds)] Cache performance comparison complete"
