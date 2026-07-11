#!/usr/bin/env bash
set -euo pipefail

: "${NAVSIM_DEVKIT_ROOT:?Set NAVSIM_DEVKIT_ROOT first}"
: "${NAVSIM_EXP_ROOT:?Set NAVSIM_EXP_ROOT first}"
: "${OPENSCENE_DATA_ROOT:?Set OPENSCENE_DATA_ROOT first}"

NNODES="${NNODES:-${MLP_WORKER_NUM:-1}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NODE_RANK="${NODE_RANK:-${MLP_ROLE_INDEX:-0}}"
MASTER_ADDR="${MASTER_ADDR:-${MLP_WORKER_0_HOST:-127.0.0.1}}"
MASTER_PORT="${MASTER_PORT:-${MLP_WORKER_0_PORT:-29500}}"
TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtrain}"
CACHE_PATH="${CACHE_PATH:-$NAVSIM_EXP_ROOT/latent_wam_cache}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"

MODEL_OVERRIDES=()
if [[ -n "${WORLD_MIRROR_PATH:-}" ]]; then
  MODEL_OVERRIDES+=("agent.config.geometry.model_id=$WORLD_MIRROR_PATH")
fi
if [[ -n "${DINOV2_PATH:-}" ]]; then
  MODEL_OVERRIDES+=("agent.config.vision.model_id=$DINOV2_PATH")
fi

"$TORCHRUN_BIN" \
  --nnodes="$NNODES" \
  --nproc_per_node="$NPROC_PER_NODE" \
  --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_latent_wam_caching.py" \
  agent=latent_wam_agent \
  train_test_split="$TRAIN_TEST_SPLIT" \
  cache_path="$CACHE_PATH" \
  "${MODEL_OVERRIDES[@]}" \
  "$@"
