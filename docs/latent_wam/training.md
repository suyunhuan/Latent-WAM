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

## Temporal attention modes

`agent.config.training.temporal_attention_mask` selects the mask used by both
world-model self-attention and cross-attention:

- `block_diagonal` (default): each prediction block reads only its paired source
  block. Configurations that omit the field retain the released behavior.
- `block_causal`: each prediction block reads its paired source and all earlier
  source blocks; prediction queries can also attend to earlier query blocks.
  Attention within each block remains unrestricted.

For a new causal experiment, pass these overrides to the existing training launcher:

```text
agent.config.training.temporal_attention_mask=block_causal
experiment_uid=navtrain_block_causal
```

The token order and 3D RoPE coordinates remain unchanged: each block contains one
ego token and 16 scene tokens for each of three views. Query time indices are
`[4, 8, 12]`, while source time indices are `[1, 4, 8]`. The three coordinates
are time, view, and learned scene-token slot, rather than a pixel-space grid.
The causal mask permits
`source_block <= prediction_block`, not `source_time <= prediction_time`: a real
observation at the prediction's own timestamp would reveal its target.

The training timestamps remain `[-1.5, 0, 2, 4]` seconds. In causal mode the 0s
prediction reads -1.5s; the 2s prediction reads -1.5s and 0s; the 4s prediction
reads -1.5s, 0s, and 2s. Reading the real 2s observation to predict 4s is teacher
forcing, not an open-loop 4s forecast made at 0s. Neither mode adds a future
rollout to current-frame trajectory inference.

Lightning checkpoints record `latent_wam_temporal_attention_mask`. Exact resume
requires a matching mode; checkpoints without the field are treated as
`block_diagonal`. To change modes, start a new run initialized from weights,
without restoring the old optimizer or scheduler. Parameter names and shapes
are unchanged, and released safetensors remain strictly loadable. The mask mode
is present in the saved Hydra configuration and logged at module construction.
Both modes use the same data cache and cache fingerprint; no regeneration is needed.

### Initialize a new run from old weights

Set a new experiment UID, a `.ckpt` or `.safetensors` weights file, and no
resume checkpoint when changing attention modes:

```text
agent.config.training.temporal_attention_mask=block_causal
agent.checkpoint_path=/path/to/old_run.ckpt
resume_ckpt_path=null
experiment_uid=navtrain_block_causal_from_old_weights
```

The training entry point calls `agent.initialize(device=cpu)` before creating the
training module. It strictly validates model keys and shapes, then Lightning
places the model on each rank's device. The requested mask mode stays in effect;
weights have no mask-mode-specific parameters. If the weights omit the target
DINO encoder, it is initialized from the loaded online encoder. Provided target
encoder weights are preserved.

This starts at global step zero with a newly created optimizer and warmup/cosine
scheduler from the new run's settings. No old optimizer, scheduler, epoch, or
global-step state is loaded. `agent.checkpoint_path` accepts `.safetensors` with
bare model keys, or `.ckpt` with a tensor `state_dict`. Checkpoint keys must be
uniformly bare model names or uniformly prefixed with `agent.model.`; that
prefix is stripped explicitly. Mixed namespaces and unknown keys are rejected
under the default strict loading policy. The tensor-only checkpoint loader does
not deserialize arbitrary Python objects or restore training state.

For exact resume, set `agent.checkpoint_path=null` and
`resume_ckpt_path=/path/to/run.ckpt`; the checkpoint's attention mode must match.
Setting both paths is an error rather than silently preferring either one.

### Default checkpoint output and manual conversion

Training defaults to full `.ckpt` output only:
`agent.config.training.export_safetensors=false`. Model weights can be exported
later without starting training or constructing a vision model:

```bash
python scripts/convert_latent_wam_checkpoint.py /path/to/run.ckpt \
  --output /path/to/model.safetensors
```

Without `--output`, the result is saved beside the source with a `.safetensors`
suffix. The source checkpoint is never modified. Existing outputs are rejected;
use `--overwrite` explicitly to replace an output. Default no-overwrite publication
uses an atomic hard link, so the output filesystem must support hard links;
explicit overwrite uses atomic rename instead.

All model tensors, including EMA when present, are retained by default. Add
`--inference-only` to omit `target_dino_encoder.*` for inference-only consumers.
Only the uniform `agent.model.` prefix is removed; bare model names are retained
and mixed namespaces are rejected. Optimizer and scheduler state are excluded.
Metadata records the source filename, mask mode, and epoch/global step when
available. A missing legacy mask field is recorded as `block_diagonal`.

### Optional paired safetensors export

Full Lightning `.ckpt` files remain the training checkpoint format. Enable a
paired model-weight export at the same checkpoint cadence with:

```text
agent.config.training.export_safetensors=true
```

The default is `false`. Each successful scheduled ModelCheckpoint save then
writes a sibling with the same stem and `.safetensors` suffix, on rank zero only.
This export contains all model tensors, including the EMA target encoder, plus
text metadata for the mask mode, epoch, global step, and source checkpoint name.
It contains no optimizer or scheduler state. Complete EMA weights are retained
for training initialization; inference explicitly ignores valid EMA tensors.
Partial or incompatible EMA weights are rejected.

The exporter copies detached tensors to independent contiguous CPU storage and
publishes the file with an atomic rename on the local or mounted filesystem.
Export failure leaves the full `.ckpt` intact and is reported as an error. When
the checkpoint callback prunes a checkpoint, it also removes its paired export.
The callback retains the original ModelCheckpoint state key for exact resume.
Direct calls to `trainer.save_checkpoint()` save only the full checkpoint;
paired exports are managed by the training checkpoint callback.

## Cache preparation

Generate the sharded cache first, then train with `build_target_encoder=true`
and `use_cache_without_dataset=true`; commands are in the [usage guide](usage.md#train-latent-wam-from-scratch). The
target DINO encoder is initialized from the online encoder when loading public
safetensors and is updated with EMA momentum 0.999.

The cache stores preprocessed `uint8 [3,224,448,3]` images once per unique
frame. The model moves them to the accelerator and applies only DINO rescaling
and normalization. Locality-aware block shuffle is the default; use
`cache_sampler_mode=global` only for throughput comparison. A node-local copy
can be enabled with `cache_staging_path=/local/nvme/latent_wam_cache`.

## Cache startup validation

Cache generation and resume validate shard metadata and structure before
publishing a complete manifest. Training therefore defaults to
`validate_cache_headers=false` and uses the fast startup checks described in
[the cache-format specification](cache-format.md).

Set `validate_cache_headers=true` when a cache needs an additional structural
audit, such as after a manual copy or filesystem incident. The audit checks the
tensor keys, shapes, and dtypes in every shard header. It runs once for each of
the train and validation datasets on every distributed rank and may be slow on
shared storage. `cache_staging_path` does the same presence and size checks
while copying; malformed tensors fail when loaded.

The public safetensors file intentionally omits optimizer and target-encoder
state. It can initialize fine-tuning, but cannot reproduce an exact Lightning
resume from epoch 84.

## Multi-node launcher

Run `scripts/training/run_latent_wam_training_multi_node.sh` once on every node.
The launcher maps `NNODES`, `NPROC_PER_NODE`, `NODE_RANK`, `MASTER_ADDR`, and
`MASTER_PORT` to `torchrun`, with the same `MLP_*` fallbacks as distributed
cache generation. `BATCH_SIZE` is per device; gradient accumulation also
multiplies the effective global batch size.

The launcher defaults to the experiment UID and concrete cache/model paths used
by the verified local run; override `NAVSIM_EXPERIMENT_UID` when starting a new
experiment. The cache, experiment root, checkpoint path, and local model paths
must resolve consistently across nodes. Extra command-line arguments are
passed through as Hydra overrides, including `resume_ckpt_path` for exact
Lightning resume.

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
