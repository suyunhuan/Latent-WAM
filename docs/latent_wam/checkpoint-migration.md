# Checkpoint migration

The original model is a PyTorch Lightning 2.2.1 checkpoint with 1,063 state
keys under `agent._transfuser_model.*`. It contains duplicate encoder aliases,
EMA target weights, dead refinement/geometry modules, optimizer state, and
training metadata.

The converter maps the effective model to 444 public state keys and writes a
self-contained safetensors file. The main mappings are:

| Legacy prefix | Public prefix |
| --- | --- |
| `lora_dino_encoder` | `vision_encoder` |
| `_tf_decoder` | `trajectory_decoder` |
| `_trajectory_head` | `trajectory_head` |
| `_status_encoding` | `status_encoder` |
| `_query_embedding` | `trajectory_queries` |
| `_wm_query_embedding` | `world_queries` |

The original temporal decoder registers one shared layer four times. Since
safetensors does not represent storage aliases, conversion materializes four
identical tensors. Loading remains numerically equivalent.

```bash
python scripts/convert_latent_wam_checkpoint.py source.ckpt output/model.safetensors
```

The source checkpoint must be trusted because PyTorch checkpoints may contain
pickle data. Safetensors is the distribution format. Conversion preserves
inference and fine-tuning weights, but not exact optimizer/scheduler resume.
