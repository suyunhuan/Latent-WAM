# Cache format

Latent WAM uses one frame-deduplicated, immutable sharded cache format. There
is no compatibility reader or converter for older sample-level caches;
regenerate them with the current caching command.

## Layout

```text
cache/
├── manifest.json
├── schema.json
├── meta/
│   ├── frames.parquet
│   ├── samples.parquet
│   ├── partial/
│   └── shards/
├── images/
│   └── shard-000000.safetensors
└── geometry/
    └── shard-000000.safetensors
```

`manifest.json` is published last. A cache without a complete final manifest
is never accepted for training. Its stable format identifier is
`latent_wam_sharded_cache`. Readers validate the explicit structure they use
and ignore unknown metadata, so existing tensor and index payloads remain
directly usable.

## Frame store

Every unique NAVSIM frame token occurs exactly once in `frames.parquet`. A row
maps the token to a shard ID and row index and retains its log name and
timestamp. Image and geometry shards use identical frame membership and row
ordering.

Each image shard contains:

| Tensor | Shape | Dtype | Meaning |
| --- | --- | --- | --- |
| `images` | `[N, 3, 224, 448, 3]` | `uint8` | left/front/right RGB images after resize and center crop |

Each geometry shard contains:

| Tensor | Shape | Dtype | Meaning |
| --- | --- | --- | --- |
| `geometry` | `[N, 3, 519, 2048]` | `float32` | WorldMirror camera-conditioned tokens |
| `ego_status` | `[N, 8]` | `float32` | command, velocity, and acceleration |

Shards contain at most 64 frames by default. Frames are ordered by log,
timestamp, and token before packing. This preserves strict global frame
deduplication while making the four frames of most samples fall in one shard
or two adjacent shards.

Images are resized to `455x256` with antialiased bicubic interpolation and
center-cropped to `448x224` before being stored. Training therefore performs
only channel-layout conversion, rescaling, and DINO normalization on the
accelerator. Online inference uses the same resize/crop function.

WorldMirror geometry uses adjusted intrinsics, inverted `sensor2lidar`
extrinsics, and condition flags `[1, 0, 1]`. Geometry patch tokens start at
index 7; the remaining 512 tokens align with DINO patches.

## Sample index

`samples.parquet` stores one row per training sample:

- sample token and log name;
- four frame tokens at `[-1.5, 0.0, 2.0, 4.0]` seconds;
- the float32 `[8, 3]` future trajectory.

The dataset resolves the four tokens through `frames.parquet` and reconstructs
the existing model input API. Samples do not duplicate image, geometry, or ego
status tensors.

## Integrity and identity

The manifest records paths, sizes, shapes, and dtypes for every Parquet and
safetensors file. It also records an explicit semantic fingerprint covering the
NAVSIM split, sample count, logs, temporal/camera configuration, image
preprocessing revision, WorldMirror model identifier, patch layout, and
geometry dtype/shape.

Cache generation/resume, ordinary training startup, and node-local staging
check the complete manifest plus every recorded file's presence and size.
Training also reads the Parquet indices and compares the explicit semantic
fields with the model and configured split. Tensor contents are validated when
they are loaded.

Set `validate_cache_headers=true` on the training command to additionally open
every safetensors header and verify tensor keys, shapes, and dtypes against the
manifest. This is useful after manually copying or modifying a cache, but is an
O(number of shards) audit. Dataset construction performs it once for the train
dataset and once for the validation dataset on every distributed rank, so it
can be especially slow on remote storage. It reads headers only, not tensor
payloads.

## Distributed generation and resume

All ranks independently build the deterministic global frame plan. Each
immutable shard has exactly one owner rank. Ranks write their own Parquet index
fragments; rank zero validates and merges all fragments, then atomically
publishes the final manifest.

Every completed shard has a sidecar under `meta/shards/`. It records the exact
ordered frame-token list, tensor metadata, paths, and sizes. A restarted job
reuses matching shards and rebuilds missing or structurally invalid shards. Use
`force_cache_computation=true` after changing preprocessing or WorldMirror.
`extraction_batch_frames` controls how many frames are passed to WorldMirror at
once and defaults to the conservative value `1`; it is independent of the
64-frame storage shard size.

Use `distributed_dry_run=true` to inspect source visibility and each rank's
deterministic plan summary without loading WorldMirror. Run only one coordinated
`torchrun` job against a cache root at a time.

## Training locality and optional staging

Training uses locality-aware block shuffle by default: shard-local sample
buckets and their contents are shuffled deterministically before balanced DDP
distribution. Set `cache_sampler_mode=global` for the comparison mode. Each
DataLoader worker keeps a small shard LRU; configure it with
`cache_shard_lru`.

The shared cache is always directly usable. Set `cache_staging_path` during
training to copy the complete cache to node-local NVMe. Staging is protected by
a local file lock, validates file presence and sizes, and publishes the local
manifest last. Malformed tensors fail when loaded.
