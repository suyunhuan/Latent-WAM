# Cache format

## Version 1

Each sample directory contains:

- `latent_wam_features_v1.gz`
- `latent_wam_targets_v1.gz`

The feature file stores:

| Field | Shape | Meaning |
| --- | --- | --- |
| `images` | `[4, 3, H, W, 3]` | uint8 RGB frames |
| `geometry` | `[4, 3, 519, 2048]` | WorldMirror tokens |
| `ego_status` | `[4, 8]` | command, velocity, acceleration |
| `schema_version` | scalar | currently `1` |

The temporal offsets are `[-1.5, 0.0, 2.0, 4.0]` seconds. Geometry patch
tokens start at index 7; the remaining 512 tokens align with DINO patches.

## Legacy adapter

The adapter reads the original `transfuser_feature.gz`, resolves its four frame
tokens through `<cache>/feature_cache/<token>/<token>.gz`, and loads each
`geometry_feature.npy`. Legacy files use gzip-pickle and must only be loaded
from trusted sources. Exact scene-token matching is not required by the
structural compatibility test, but tensor ranks and temporal field names are.
