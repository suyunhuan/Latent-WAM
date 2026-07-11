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

Generate versioned caches first, then train with `build_target_encoder=true` and
`use_cache_without_dataset=true`; commands are in the root README. The target
DINO encoder is initialized from the online encoder when loading public
safetensors and is updated with EMA momentum 0.999.

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
