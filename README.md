# Latent WAM

Latent WAM is a research implementation of latent world-action modeling for
camera-based autonomous driving. It predicts a four-second ego trajectory from
the current left/front/right camera images and the current ego status. During
training, a temporal latent world model and WorldMirror geometry features act as
auxiliary supervision.

[中文说明](README_zh.md)

## What is released

- A single canonical `LatentWAMModel`; experimental W4D/LAW/RFT variants have
  been removed.
- Current-frame inference with three explicit outputs: `trajectory`,
  `all_trajectories`, and `mode_logits`.
- The teacher-forcing training path used by the epoch-84 model. The unused
  curriculum, autoregressive, refinement, and staged-loss branches are gone.
- A versioned cache format plus a read-only adapter for original
  `transfuser_feature.gz` two-level caches.
- Explicit conversion of the original Lightning checkpoint to self-contained
  `safetensors` weights.
- The official NAVSIM v2.1.2 TransFuser baseline, retained separately from
  Latent WAM.

## Architecture

The released checkpoint uses DINOv2-Base (768 dimensions), three camera views,
16 scene tokens per view, a 256-dimensional trajectory/world transformer, four
command-conditioned trajectory modes, and eight poses at 0.5-second intervals.
Training uses frames at `[-1.5, 0.0, +2.0, +4.0]` seconds. Public inference only
uses the frame at `0.0` seconds.

Source code lives in [`navsim/agents/latent_wam`](navsim/agents/latent_wam).
The Hydra entry is `agent=latent_wam_agent`.

## Installation

The supported release environment is Linux, Python 3.10, NVIDIA CUDA 12.4, and
a single GPU or more. CPU is supported for configuration, cache migration,
checkpoint conversion, and unit tests; model inference is intended for CUDA.

```bash
git clone --recurse-submodules <repository-url> latent-wam
cd latent-wam
conda env create -f environment.yml
conda activate latent-wam
pip install -e .
```

See [installation details](docs/latent_wam/installation.md) for dataset paths,
environment variables, and WorldMirror setup.

## Checkpoint conversion

The original Lightning checkpoint remains usable through the compatibility
loader. For publication, convert it once:

```bash
python scripts/convert_latent_wam_checkpoint.py \
  /path/to/epoch84.ckpt \
  artifacts/latent-wam/model.safetensors
```

The converter removes optimizer state, duplicate encoder aliases, the
training-only target encoder, refinement modules, and unused geometry decoder
parameters. It writes `model.safetensors`, `config.json`, and a conversion
report with a SHA-256 digest. This preserves inference weights, but it is not an
exact optimizer-state training resume. See
[checkpoint migration](docs/latent_wam/checkpoint-migration.md).

## NAVSIM evaluation

The paper reports the **single-stage ePDMS**. Run the single-stage evaluator on
the original test scenes:

```bash
python navsim/planning/script/run_pdm_score_one_stage.py \
  agent=latent_wam_agent \
  agent.checkpoint_path=/path/to/model.safetensors \
  train_test_split=navtest \
  experiment_name=latent_wam_eval
```

The final single-stage ePDMS is the `score` in the `average_all_frames` row of
the generated CSV. This evaluator includes two-frame extended comfort, but it
does not evaluate the synthetic second-stage scenes.

For the NAVSIM v2 two-stage ePDMS instead, use:

```bash
python navsim/planning/script/run_pdm_score.py \
  agent=latent_wam_agent \
  agent.checkpoint_path=/path/to/model.safetensors \
  train_test_split=navhard_two_stage \
  experiment_name=latent_wam_two_stage_eval
```

In the two-stage output, the final aggregate is the `score` in the
`extended_pdm_score_combined` row.

## Cache generation and training

Camera-conditioned WorldMirror supervision is computed once and stored in the
version 2 cache. Camera calibration is read from NAVSIM during extraction and
is not stored in the cache:

```bash
python navsim/planning/script/run_latent_wam_caching.py \
  agent=latent_wam_agent \
  train_test_split=navtrain \
  cache_path=$NAVSIM_EXP_ROOT/latent_wam_cache
```

```bash
python navsim/planning/script/run_training.py \
  agent=latent_wam_agent \
  agent.build_target_encoder=true \
  use_cache_without_dataset=true \
  force_cache_computation=false \
  cache_path=$NAVSIM_EXP_ROOT/latent_wam_cache \
  train_test_split=navtrain \
  experiment_name=latent_wam_train
```

Details and the legacy layout are documented in
[cache format](docs/latent_wam/cache-format.md) and
[training](docs/latent_wam/training.md).

The minimal real-data release check is documented in
[end-to-end smoke testing](docs/latent_wam/smoke-test.md). It covers one
WorldMirror cache sample, one optimizer/EMA step, checkpoint reload, and one
single-scene PDM evaluation.

## Tests

```bash
python -m pytest tests/latent_wam -q
pre-commit run --all-files
```

## Upstream projects and licenses

The NAVSIM-derived source is distributed under Apache-2.0. WorldMirror is a
separate git submodule governed by the **Tencent HunyuanWorld-Mirror Community
License Agreement**, not Apache-2.0. That license includes geographic, use, and
redistribution restrictions. Moreover, Latent WAM weights were trained using
WorldMirror feature supervision; obtain legal approval before publishing those
weights.

Read [NOTICE](NOTICE), [third-party licenses](THIRD_PARTY_LICENSES.md), and
`worldmirror/License.txt` before use or redistribution. This repository does not
imply affiliation with or endorsement by Tencent.

## Acknowledgements

Latent WAM builds on [NAVSIM](https://github.com/autonomousvision/navsim),
[DINOv2](https://github.com/facebookresearch/dinov2), and
[HunyuanWorld-Mirror](https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror).
Please cite those projects and the Latent WAM paper when using this code. The
final Latent WAM citation will be added with the public paper release.
