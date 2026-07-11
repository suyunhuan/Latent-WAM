# Cache format

## Version 2

Each sample directory contains:

- `latent_wam_features_v2.gz`
- `latent_wam_targets_v2.gz`

The feature file stores:

| Field | Shape | Meaning |
| --- | --- | --- |
| `images` | `[4, 3, H, W, 3]` | uint8 RGB frames |
| `geometry` | `[4, 3, 519, 2048]` | WorldMirror tokens |
| `ego_status` | `[4, 8]` | command, velocity, acceleration |
| `schema_version` | scalar | currently `2` |

The temporal offsets are `[-1.5, 0.0, 2.0, 4.0]` seconds. Geometry patch
tokens start at index 7; the remaining 512 tokens align with DINO patches.

WorldMirror extraction uses the original camera-conditioned path. Images are
resized by their shortest edge to 256 pixels, center-cropped to `448x224`, and
rescaled without ImageNet normalization. Camera intrinsics are adjusted for the
same resize/crop, `sensor2lidar` poses are inverted to world-to-camera matrices,
and pose/ray priors are enabled with condition flags `[1, 0, 1]`. Calibration is
required while generating a cache but is not stored in the cache.

The earlier public version 1 used image-only WorldMirror extraction and is not
supported. Regenerate it with the current caching command.

## Legacy adapter

The adapter reads the original `transfuser_feature.gz`, resolves its four frame
tokens through `<cache>/feature_cache/<token>/<token>.gz`, and loads each
`geometry_feature.npy`. Legacy files use gzip-pickle and must only be loaded
from trusted sources. Exact scene-token matching is not required by the
structural compatibility test, but tensor ranks and temporal field names are.
The legacy adapter does not recompute geometry and therefore preserves the
original cached supervision exactly.

## Distributed generation

`run_latent_wam_caching.py` uses a single process when invoked with `python` and
automatically enables distributed sharding under `torchrun`. All ranks sort the
same token set, compare its count and SHA-256 fingerprint, and then take
contiguous balanced shards. No DDP model synchronization is needed: every rank
loads one independent WorldMirror encoder only when it has uncached work.

Valid feature/target pairs are fully loaded and validated before being skipped.
New pairs are written to rank/PID-specific temporary files, reloaded for
validation, and atomically published. A failed token is recorded while the rank
continues its remaining shard; the overall command exits non-zero after all
ranks finish if any token failed.

Hydra logs are isolated under `metadata/rank_<rank>/`. Rank zero writes
`metadata/cache_manifest.json` with per-rank counts and failed token details.
Use `distributed_dry_run=true` to verify data visibility, token agreement, and
sharding without loading model weights. `LATENT_WAM_DIST_BACKEND` overrides the
default NCCL backend (Gloo when CUDA is unavailable).

Run only one `torchrun` job against a cache root at a time. Independent jobs
sharing the same root are unsupported because rank sharding and the final
manifest are coordinated only within one process group.
