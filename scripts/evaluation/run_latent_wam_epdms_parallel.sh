#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_WORKSPACE="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE="${NAVSIM_DEVKIT_ROOT:-$DEFAULT_WORKSPACE}"
PYTHON="${PYTHON_BIN:-python}"

: "${NAVSIM_EXP_ROOT:?Set NAVSIM_EXP_ROOT to the evaluation output root}"
: "${OPENSCENE_DATA_ROOT:?Set OPENSCENE_DATA_ROOT to the NAVSIM dataset root}"
: "${LATENT_WAM_CHECKPOINT:?Set LATENT_WAM_CHECKPOINT to a compatible checkpoint}"

METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-$NAVSIM_EXP_ROOT/metric_cache}"
EVALUATION_OUTPUT="${EVALUATION_OUTPUT:-$NAVSIM_EXP_ROOT/latent_wam_eval/navtest.csv}"
EVALUATION_DEVICES="${EVALUATION_DEVICES:-0}"
INFERENCE_WORKERS_PER_DEVICE="${INFERENCE_WORKERS_PER_DEVICE:-1}"
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-8}"
PDM_WORKERS="${PDM_WORKERS:-0}"
PDM_CHUNK_SIZE="${PDM_CHUNK_SIZE:-64}"
PDM_THREADS_PER_WORKER="${PDM_THREADS_PER_WORKER:-1}"

export NAVSIM_DEVKIT_ROOT="$WORKSPACE"
export PYTHONPATH="$WORKSPACE${PYTHONPATH:+:$PYTHONPATH}"

command=(
  "$PYTHON"
  "$WORKSPACE/scripts/evaluate_latent_wam_epdms_parallel.py"
  --checkpoint "$LATENT_WAM_CHECKPOINT"
  --navsim-log-path "$OPENSCENE_DATA_ROOT/navsim_logs/test"
  --sensor-path "$OPENSCENE_DATA_ROOT/sensor_blobs/test"
  --metric-cache "$METRIC_CACHE_PATH"
  --output "$EVALUATION_OUTPUT"
  --devices "$EVALUATION_DEVICES"
  --inference-workers-per-device "$INFERENCE_WORKERS_PER_DEVICE"
  --inference-batch-size "$INFERENCE_BATCH_SIZE"
  --pdm-workers "$PDM_WORKERS"
  --pdm-chunk-size "$PDM_CHUNK_SIZE"
  --pdm-threads-per-worker "$PDM_THREADS_PER_WORKER"
)

if [[ -n "${DINOV2_PATH:-}" ]]; then
  command+=(--vision-model "$DINOV2_PATH")
fi

exec "${command[@]}" "$@"
