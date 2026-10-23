# Runbook: Latent WAM inference on NAVSIM (unmodified)

Paper evaluation is **NAVSIM** single-stage ePDMS (`navtest`). No model/training code changes.

**Legend:** ✅ verified on this cloud VM · ⬜ assumed / blocked here · ▶ next on a GPU box

---

## 0. Machine prerequisites

| Need | Notes |
| --- | --- |
| Linux x86-64, NVIDIA GPU, CUDA **12.4**-compatible driver | ✅ CPU-only smoke worked; ⬜ full ePDMS needs GPU |
| Disk | Maps ~1.4 GB; **test split ~200 GB compressed** if using official `download_test.sh`; plus metric cache + env |
| Network | ModelScope, Hugging Face, AWS/OpenScene (✅ reachable from this VM) |

---

## 1. Clone (user fork)

```bash
git clone --recurse-submodules https://github.com/suyunhuan/Latent-WAM.git /workspace/Latent-WAM
cd /workspace/Latent-WAM
# If cloned without submodules:
git submodule update --init --recursive
```

✅ Verified: clone of `suyunhuan/Latent-WAM` at `504508a`; submodule init OK.  
(Docs still show upstream `Jinhsi555/Latent-WAM`; tip SHA matched on probe day.)

---

## 2. Environment (official)

```bash
cd /workspace/Latent-WAM
conda env create -f environment.yml
conda activate latent-wam
pip install -e .

export NAVSIM_DEVKIT_ROOT="$PWD"
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OPENSCENE_DATA_ROOT="/path/to/dataset"   # e.g. $HOME/latent_wam_workspace/dataset
export NAVSIM_EXP_ROOT="/path/to/experiments"   # e.g. $HOME/latent_wam_workspace/exp
export NUPLAN_MAPS_ROOT="$OPENSCENE_DATA_ROOT/maps"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
```

Install smoke (no weights):

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "from navsim.agents.latent_wam.config import LatentWAMConfig; print(LatentWAMConfig())"
```

⬜ Official conda+CUDA env **not** created here (no conda on image; no GPU).  
✅ Alternate: micromamba Py3.10 + `torch==2.4.1+cpu` + partial deps; `LatentWAMConfig` import OK.  
▶ On GPU box: use `environment.yml` as written; confirm `torch.cuda.is_available() == True`.

---

## 3. Download Latent WAM weights

```bash
pip install modelscope
python - <<'PY'
from modelscope import snapshot_download
snapshot_download("Jinhsi555/Latent-WAM", local_dir="models/latent-wam")
PY
export LATENT_WAM_CHECKPOINT="$NAVSIM_DEVKIT_ROOT/models/latent-wam/model.safetensors"
sha256sum "$LATENT_WAM_CHECKPOINT"
# expect: b25a52e3d96b7d92faf99e6d9a61eccabb07137ca93c10d83bc1422ef0c17bb1
```

✅ Verified download + SHA. WorldMirror **not** required for eval.

---

## 4. Checkpoint-only inference smoke (no NAVSIM scenes)

```bash
cd "$NAVSIM_DEVKIT_ROOT"
python scripts/validate_latent_wam_checkpoint.py "$LATENT_WAM_CHECKPOINT"
# expect: checkpoint and inference smoke test passed
```

✅ **Actually run** on this VM (CPU). Loads safetensors strictly and runs `predict` on synthetic tensors → trajectory `[1,8,3]`.  
This is **not** NAVSIM ePDMS; it only proves weight compatibility.

---

## 5. Prepare NAVSIM evaluation data

Accept nuPlan / OpenScene terms first. Scripts download into **cwd**.

```bash
mkdir -p "$OPENSCENE_DATA_ROOT" "$NAVSIM_EXP_ROOT"
cd "$OPENSCENE_DATA_ROOT"

# Maps (~971 MB zip)
bash "$NAVSIM_DEVKIT_ROOT/download/download_maps.sh"
# → $OPENSCENE_DATA_ROOT/maps

# Test split (large: ~200 GB compressed with cameras+lidar)
bash "$NAVSIM_DEVKIT_ROOT/download/download_test.sh"
mkdir -p navsim_logs sensor_blobs
mv test_navsim_logs navsim_logs/test
mv test_sensor_blobs sensor_blobs/test
```

Verify:

```bash
test -d "$OPENSCENE_DATA_ROOT/maps"
test -d "$OPENSCENE_DATA_ROOT/navsim_logs/test"
test -d "$OPENSCENE_DATA_ROOT/sensor_blobs/test"
```

✅ Maps download verified.  
⬜ Test logs/sensors **not** downloaded (~202 GB; disk/time; lidar unused by Latent WAM but fetched by script).

Optional offline DINOv2:

```bash
export DINOV2_PATH=/path/to/dinov2-base
```

Otherwise evaluation resolves `facebook/dinov2-base` from Hugging Face, then loads Latent WAM weights on top.

---

## 6. Metric cache (once)

```bash
cd "$NAVSIM_DEVKIT_ROOT"
# On AutoDL-like hosts that advertise many CPUs, limit Ray workers (null → hang at 128).
python navsim/planning/script/run_metric_caching.py \
  train_test_split=navtest \
  metric_cache_path="$NAVSIM_EXP_ROOT/metric_cache" \
  worker.threads_per_node=16
```

✅ Verified on AutoDL RTX 4090D (2026-10-22): with `worker.threads_per_node=16`, cache finished (~3.6 GB). Default `null` hung after Ray init.

---

## 7. Run NAVSIM ePDMS inference (official)

### 7a. Parallel launcher (README Quick Start)

```bash
cd "$NAVSIM_DEVKIT_ROOT"
export LATENT_WAM_CHECKPOINT=...   # as above
# CN GPU boxes: HF often blocked — mirror + local DINOv2
# export HF_ENDPOINT=https://hf-mirror.com
# export DINOV2_PATH=$LW_ROOT/models/dinov2-base
# Also pin: pip install 'huggingface-hub>=0.23.2,<1.0'  # transformers 4.45.x

EVALUATION_DEVICES=0 PDM_WORKERS=8 INFERENCE_BATCH_SIZE=8 \
  bash scripts/evaluation/run_latent_wam_epdms_parallel.sh --max-scenarios 100
```

Omit `--max-scenarios 100` for full `navtest`. Multi-GPU: `EVALUATION_DEVICES=0,1`.

### 7b. Single-process CLI

```bash
python scripts/evaluate_latent_wam_epdms.py \
  --checkpoint "$LATENT_WAM_CHECKPOINT" \
  --navsim-log-path "$OPENSCENE_DATA_ROOT/navsim_logs/test" \
  --sensor-path "$OPENSCENE_DATA_ROOT/sensor_blobs/test" \
  --metric-cache "$NAVSIM_EXP_ROOT/metric_cache" \
  --output "$NAVSIM_EXP_ROOT/latent_wam_eval/navtest.csv" \
  --max-scenarios 100
```

Drop `--vision-model` / `DINOV2_PATH` to use HF DINOv2 ID.

**Score:** CSV row `average_all_frames`, column `score` × 100 ≈ paper EPDMS (human-penalty filter on by default).

✅ AutoDL smoke (100 scenarios, unmodified Latent WAM): **ePDMS ×100 ≈ 89.08**  
CSV: `/root/autodl-tmp/latent_wam_workspace/exp/latent_wam_eval/navtest_smoke100.csv`  
✅ AutoDL **full navtest** (12146 scenarios): **ePDMS ×100 ≈ 89.34**  
CSV: `/root/autodl-tmp/latent_wam_workspace/exp/latent_wam_eval/navtest_full.csv`  
Details: [`autodl-bootstrap-status.md`](autodl-bootstrap-status.md).

---

## 8. Exact next commands on a GPU box

Assuming empty machine with NVIDIA GPU and conda:

```bash
git clone --recurse-submodules https://github.com/suyunhuan/Latent-WAM.git latent-wam && cd latent-wam
conda env create -f environment.yml && conda activate latent-wam && pip install -e .
export NAVSIM_DEVKIT_ROOT="$PWD" PYTHONPATH="$PWD"
export OPENSCENE_DATA_ROOT="$HOME/latent_wam_workspace/dataset"
export NAVSIM_EXP_ROOT="$HOME/latent_wam_workspace/exp"
export NUPLAN_MAPS_ROOT="$OPENSCENE_DATA_ROOT/maps" NUPLAN_MAP_VERSION=nuplan-maps-v1.0
mkdir -p "$OPENSCENE_DATA_ROOT" "$NAVSIM_EXP_ROOT" models
pip install modelscope && python -c "from modelscope import snapshot_download; snapshot_download('Jinhsi555/Latent-WAM', local_dir='models/latent-wam')"
export LATENT_WAM_CHECKPOINT="$PWD/models/latent-wam/model.safetensors"
python scripts/validate_latent_wam_checkpoint.py "$LATENT_WAM_CHECKPOINT"
cd "$OPENSCENE_DATA_ROOT" && bash "$NAVSIM_DEVKIT_ROOT/download/download_maps.sh"
bash "$NAVSIM_DEVKIT_ROOT/download/download_test.sh"
mkdir -p navsim_logs sensor_blobs && mv test_navsim_logs navsim_logs/test && mv test_sensor_blobs sensor_blobs/test
cd "$NAVSIM_DEVKIT_ROOT"
python navsim/planning/script/run_metric_caching.py train_test_split=navtest \
  metric_cache_path="$NAVSIM_EXP_ROOT/metric_cache" worker.threads_per_node=16
# export DINOV2_PATH=... if huggingface.co unreachable
EVALUATION_DEVICES=0 PDM_WORKERS=8 INFERENCE_BATCH_SIZE=8 \
  bash scripts/evaluation/run_latent_wam_epdms_parallel.sh --max-scenarios 100
```

**CPU cloud VM:** first hard blocker was no NVIDIA GPU. **AutoDL:** smoke path above completed (ePDMS≈89.08 on 100 scenarios).

---

## Artifacts left on this VM (not in git)

| Path | Content |
| --- | --- |
| `/workspace/Latent-WAM` | Fork checkout + worldmirror submodule |
| `/workspace/latent_wam_workspace/models/latent-wam/` | ModelScope weights (SHA OK) |
| `/workspace/latent_wam_workspace/dataset/maps/` | nuPlan maps |
| `/workspace/mamba` | Micromamba root (CPU torch env) |

No fork commits / PRs made in phase 1.
