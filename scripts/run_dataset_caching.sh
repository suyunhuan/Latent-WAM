#!/usr/bin/env bash
set -euo pipefail

: "${NAVSIM_DEVKIT_ROOT:?Set NAVSIM_DEVKIT_ROOT first}"
: "${NAVSIM_EXP_ROOT:?Set NAVSIM_EXP_ROOT first}"

TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtrain}"
CACHE_PATH="${CACHE_PATH:-$NAVSIM_EXP_ROOT/latent_wam_cache}"

python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_latent_wam_caching.py" \
  agent=latent_wam_agent \
  train_test_split="$TRAIN_TEST_SPLIT" \
  cache_path="$CACHE_PATH" \
  "$@"
