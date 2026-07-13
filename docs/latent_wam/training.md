# Training

Latent WAM has one training path: four-frame teacher forcing. The epoch-84 run
called the old `forward_train_curriculum` function with autoregressive ratio
zero; the public implementation names this behavior `forward_train` and removes
the unused scheduler/branch.

Losses are preserved from the final implementation:

- winner-take-all L1 trajectory loss;
- MSE latent world-model loss;
- cosine geometry alignment loss;
- command cross-entropy plus velocity/acceleration MSE.

Generate the sharded cache first, then train with `build_target_encoder=true`
and `use_cache_without_dataset=true`; commands are in the root README. The
target DINO encoder is initialized from the online encoder when loading public
safetensors and is updated with EMA momentum 0.999.

The cache stores preprocessed `uint8 [3,224,448,3]` images once per unique
frame. The model moves them to the accelerator and applies only DINO rescaling
and normalization. Locality-aware block shuffle is the default; use
`cache_sampler_mode=global` only for throughput comparison. A node-local copy
can be enabled with `cache_staging_path=/local/nvme/latent_wam_cache`.

## Cache startup validation

Cache generation and resume validate shard checksums and structure before
publishing a complete manifest. Training therefore defaults to
`validate_cache_headers=false` and uses the fast startup checks described in
[the cache-format specification](cache-format.md).

Set `validate_cache_headers=true` when a cache needs an additional structural
audit, such as after a manual copy or filesystem incident. The audit checks the
tensor keys, shapes, and dtypes in every shard header. It runs once for each of
the train and validation datasets on every distributed rank, may be slow on
shared storage, and does not recompute payload checksums. `cache_staging_path`
performs checksum-verified copying when a full payload-integrity check is
required before training.

The public safetensors file intentionally omits optimizer and target-encoder
state. It can initialize fine-tuning, but cannot reproduce an exact Lightning
resume from epoch 84.

For a one-sample optimizer and EMA check after generating a cache:

```bash
python scripts/smoke_train_latent_wam.py \
  --cache-path "$NAVSIM_EXP_ROOT/latent_wam_cache" \
  --checkpoint /path/to/model.safetensors \
  --vision-model /path/to/dinov2-base \
  --output "$NAVSIM_EXP_ROOT/latent_wam_smoke/train_step.json"
```

The command fails unless all losses and gradients are finite, the online DINO
encoder receives a non-zero gradient, EMA updates the target encoder, and the
saved inference weights strictly reload to produce a finite `[1, 8, 3]`
trajectory.

## Distributed logging

The parent training process pins one `experiment_uid` in
`NAVSIM_EXPERIMENT_UID` before Lightning launches child ranks. Every rank
therefore resolves the same `output_dir`. Global rank zero owns the main Hydra
log, Lightning metrics, and checkpoints; each non-zero rank writes diagnostics
to `rank_<global_rank>.log` in the same experiment directory without attaching
to the main console or log file.

An explicit `experiment_uid=my_run` command-line override remains useful when a
human-readable or externally coordinated run identifier is required, but is no
longer necessary merely to keep local DDP ranks on one path.
