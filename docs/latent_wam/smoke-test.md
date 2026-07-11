# End-to-end smoke test

This is the authoritative minimum-cost release check. It uses one trainval
sample and one test scene; it is not a benchmark run and must not be used to
train on test data.

## Acceptance path

1. Load one real trainval scene with four temporal frames and three cameras.
2. Extract camera-conditioned WorldMirror visual-geometry tokens and round-trip
   the version 2 cache through `LatentWAMCacheDataset`.
3. Run one forward/backward/optimizer step, verify all losses and gradients,
   update the target DINO encoder with EMA, save inference weights, and strictly
   reload them.
4. Generate a metric cache for one test token and run one-stage PDM scoring with
   the released epoch-84 inference weights.

For OpenScene layouts that store annotations under `meta_datas` rather than
`navsim_logs`, override `navsim_log_path` explicitly. Token and log filters are
important: `max_scenes=1` alone may select a different scene.

## Verified fixture

The July 11, 2026 release smoke used:

- data root: `/mnt/oss_data/navsim/Egonav/openscene-v1.1`;
- trainval token: `a6a284584b19599f`;
- test token: `8051c60f20705909`;
- test log: `2021.05.25.17.54.41_veh-35_04111_04288`.

Observed cache tensors were images `[4, 3, 1080, 1920, 3]`, geometry
`[4, 3, 519, 2048]`, ego status `[4, 8]`, and trajectory target `[8, 3]`.
The legacy WorldMirror regression reached mean cosine `0.9998027682304382`
against the supplied reference geometry (minimum `0.999`). The four training
losses were finite, EMA changed the target encoder, and the
saved weights strictly reloaded. The single-scene evaluation was valid and
produced PDM score `0.9180014612343158`; all multiplicative compliance metrics
were `1.0`.

The PDM value must still be compared with the historical result produced from
the same token and evaluator version. A large difference is a failed migration,
even when the process exits successfully.

The machine-readable result is committed as
[`smoke-result.json`](smoke-result.json).

The WorldMirror comparison details and reference file hashes are in
[`worldmirror-cache-regression.json`](worldmirror-cache-regression.json).

Distributed coordination was checked with two local Gloo processes. The test
covered process-group lifecycle, token agreement, balanced sharding including
an empty rank, object gathering, and manifest generation. Real extraction was
also checked with two PPU devices and the default NCCL backend: two ranks each
generated one WorldMirror cache, then a second run skipped both valid samples
without loading the deliberately invalid model paths. Multi-node extraction
was not executed on this single-node machine.

## Eight-log dual-PPU reproduction

A larger smoke used eight trainval scenes from eight logs. Two NCCL ranks each
generated four camera-conditioned WorldMirror caches in 40.771 seconds. All
eight feature/target pairs passed generation, dtype, shape, and finite-value
validation; a second run skipped all eight while configured with deliberately
invalid model paths.

The same cache then ran through the official `run_training.py` entrypoint with
two-device Lightning DDP, mixed precision, four global training steps, and two
validation batches. All losses were finite; epoch train loss was
`10.745153427124023` and validation loss was `14.169358253479004`. This proves
the multi-log cache-to-training path runs, but is not a statistical accuracy
reproduction. Full details are in
[`repro-8logs-result.json`](repro-8logs-result.json).

The experiment initially exposed independent Hydra timestamp resolution in
Lightning child ranks. The training runtime now pins the parent UID before
launch: a two-PPU regression produced one experiment directory, a rank-zero
`run_training.log`, and an isolated `rank_1.log`. Multi-node execution remains
untested.

## Reproduction commands

The commands below are the token-scoped recipe used for the verified fixture.
Set the two local model paths before running it.

```bash
export DATA_ROOT=/mnt/oss_data/navsim/Egonav/openscene-v1.1
export NAVSIM_DEVKIT_ROOT="$PWD"
export NAVSIM_EXP_ROOT="$PWD/artifacts/smoke"
export OPENSCENE_DATA_ROOT="$DATA_ROOT"
export NUPLAN_MAPS_ROOT="$DATA_ROOT/maps"
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export WORLD_MIRROR_PATH=/path/to/HunyuanWorld-Mirror
export DINOV2_PATH=/path/to/dinov2-base
export LATENT_WAM_WEIGHTS=/path/to/model.safetensors
```

Generate and reload one trainval cache:

```bash
python navsim/planning/script/run_latent_wam_caching.py \
  'train_test_split.scene_filter.max_scenes=1' \
  'train_test_split.scene_filter.log_names=[2021.05.12.19.36.12_veh-35_00005_00204]' \
  'train_test_split.scene_filter.tokens=[a6a284584b19599f]' \
  navsim_log_path="$DATA_ROOT/meta_datas/trainval" \
  original_sensor_path="$DATA_ROOT/sensor_blobs/trainval" \
  cache_path="$NAVSIM_EXP_ROOT/latent_wam_cache" \
  agent.config.geometry.model_id="$WORLD_MIRROR_PATH" \
  agent.config.vision.model_id="$DINOV2_PATH"

python scripts/smoke_train_latent_wam.py \
  --cache-path "$NAVSIM_EXP_ROOT/latent_wam_cache" \
  --checkpoint "$LATENT_WAM_WEIGHTS" \
  --vision-model "$DINOV2_PATH" \
  --output "$NAVSIM_EXP_ROOT/train_step_report.json"
```

Release maintainers with the trusted legacy frame can reproduce the
WorldMirror regression separately:

```bash
python scripts/validate_worldmirror_cache.py \
  --legacy-frame-cache /path/to/frame.gz \
  --legacy-geometry /path/to/geometry_feature.npy \
  --data-root "$DATA_ROOT" \
  --calibration-log 2021.05.12.19.36.12_veh-35_00005_00204 \
  --calibration-token 0a678d2136b35b56 \
  --worldmirror-model "$WORLD_MIRROR_PATH" \
  --vision-model "$DINOV2_PATH" \
  --output worldmirror-cache-regression.json
```

Build and score the single test scene. Local `/tmp` storage avoids sending the
serialized metric-cache buffer through a network filesystem.

```bash
python navsim/planning/script/run_metric_caching.py \
  worker=sequential train_test_split=navtest \
  'train_test_split.scene_filter.max_scenes=1' \
  'train_test_split.scene_filter.log_names=[2021.05.25.17.54.41_veh-35_04111_04288]' \
  'train_test_split.scene_filter.tokens=[8051c60f20705909]' \
  navsim_log_path="$DATA_ROOT/meta_datas/test" \
  synthetic_scenes_path="$DATA_ROOT/meta_datas/test" \
  metric_cache_path=/tmp/latent_wam_metric_cache_8051

python navsim/planning/script/run_pdm_score_one_stage.py \
  experiment_name=latent_wam_8051_smoke worker=sequential \
  train_test_split=navtest agent=latent_wam_agent \
  'train_test_split.scene_filter.max_scenes=1' \
  'train_test_split.scene_filter.log_names=[2021.05.25.17.54.41_veh-35_04111_04288]' \
  'train_test_split.scene_filter.tokens=[8051c60f20705909]' \
  navsim_log_path="$DATA_ROOT/meta_datas/test" \
  original_sensor_path="$DATA_ROOT/sensor_blobs/test" \
  metric_cache_path=/tmp/latent_wam_metric_cache_8051 \
  agent.checkpoint_path="$LATENT_WAM_WEIGHTS" \
  agent.config.vision.model_id="$DINOV2_PATH"
```

## Local metric-cache writes

Metric caches can contain large serialized buffers. Generate the temporary
single-scene cache on local storage such as `/tmp`, then keep only the small
CSV/JSON report in the experiment directory. `MetricCache.dump` performs an
atomic local write so interrupted jobs cannot leave a valid-looking partial
cache.
