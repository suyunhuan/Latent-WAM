# GPU box bootstrap — official Latent WAM ePDMS

Ordered commands for a rented Linux NVIDIA box. Aligns with [`runbook-navisim-inference.md`](runbook-navisim-inference.md). **No Latent-WAM model/training code edits.** Project store 内曾有自动化脚本 `internal/gpu-bootstrap.sh`；本公开导出未附带该脚本，请按下文命令逐步执行。

**Pin:** clone `https://github.com/suyunhuan/Latent-WAM` (tip verified `504508a`); checkpoint SHA-256 `b25a52e3d96b7d92faf99e6d9a61eccabb07137ca93c10d83bc1422ef0c17bb1`.

**Legend:** ▶ run in order · ⏭ optional · ✅ already verified on CPU cloud VM (re-do on GPU box).

---

## 0. Preflight (GPU + disk)

```bash
nvidia-smi
df -h "$HOME"
# Need: NVIDIA GPU, CUDA 12.4-compatible driver, ≥400 GB free (prefer 500+)
```

Accept nuPlan / OpenScene dataset terms before step 5.

---

## 1. Workspace + clone

```bash
export LW_ROOT="${LW_ROOT:-$HOME/latent_wam_workspace}"
mkdir -p "$LW_ROOT"/{dataset,exp,models}
cd "$LW_ROOT"

git clone --recurse-submodules https://github.com/suyunhuan/Latent-WAM.git latent-wam
cd latent-wam
# If clone lacked submodules:
# git submodule update --init --recursive

git rev-parse --short HEAD   # expect 504508a or newer tip; do not patch sources
```

---

## 2. Official conda env (`environment.yml`)

```bash
cd "$LW_ROOT/latent-wam"
conda env create -f environment.yml    # skip if env latent-wam already exists
conda activate latent-wam
pip install -e .

export NAVSIM_DEVKIT_ROOT="$LW_ROOT/latent-wam"
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OPENSCENE_DATA_ROOT="$LW_ROOT/dataset"
export NAVSIM_EXP_ROOT="$LW_ROOT/exp"
export NUPLAN_MAPS_ROOT="$OPENSCENE_DATA_ROOT/maps"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
```

Install smoke:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); assert torch.cuda.is_available()"
python -c "from navsim.agents.latent_wam.config import LatentWAMConfig; print(LatentWAMConfig())"
```

Must print `True` for CUDA. If conda is missing, install Miniconda/Mambaforge first — do not invent a non-official PyTorch pin.

---

## 3. Weights (ModelScope)

```bash
conda activate latent-wam
pip install modelscope
python - <<'PY'
import os
from modelscope import snapshot_download
root = os.path.expanduser(os.environ.get("LW_ROOT", "~/latent_wam_workspace"))
snapshot_download("Jinhsi555/Latent-WAM", local_dir=os.path.join(root, "models/latent-wam"))
PY

export LATENT_WAM_CHECKPOINT="$LW_ROOT/models/latent-wam/model.safetensors"
sha256sum "$LATENT_WAM_CHECKPOINT"
# must contain: b25a52e3d96b7d92faf99e6d9a61eccabb07137ca93c10d83bc1422ef0c17bb1
```

WorldMirror is **not** required for eval.

---

## 4. Checkpoint smoke (no NAVSIM scenes)

```bash
cd "$NAVSIM_DEVKIT_ROOT"
python scripts/validate_latent_wam_checkpoint.py "$LATENT_WAM_CHECKPOINT"
# expect: checkpoint and inference smoke test passed
```

✅ Same script already passed on CPU; re-run here to confirm CUDA path.

---

## 5. Maps + navtest (official download scripts)

Scripts write into **cwd**. Run from the dataset root:

```bash
mkdir -p "$OPENSCENE_DATA_ROOT" "$NAVSIM_EXP_ROOT"
cd "$OPENSCENE_DATA_ROOT"

bash "$NAVSIM_DEVKIT_ROOT/download/download_maps.sh"
# → $OPENSCENE_DATA_ROOT/maps

bash "$NAVSIM_DEVKIT_ROOT/download/download_test.sh"
# ~202 GB compressed; hours on typical links

mkdir -p navsim_logs sensor_blobs
mv test_navsim_logs navsim_logs/test
mv test_sensor_blobs sensor_blobs/test

test -d "$OPENSCENE_DATA_ROOT/maps"
test -d "$OPENSCENE_DATA_ROOT/navsim_logs/test"
test -d "$OPENSCENE_DATA_ROOT/sensor_blobs/test"
```

Optional offline DINOv2 (else HF resolves `facebook/dinov2-base` at eval):

```bash
# export DINOV2_PATH="$LW_ROOT/models/dinov2-base"
```

---

## 6. Metric cache (once per split)

```bash
cd "$NAVSIM_DEVKIT_ROOT"
# Prefer threads_per_node=16 on AutoDL (null → Ray uses all advertised CPUs and can hang).
python navsim/planning/script/run_metric_caching.py \
  train_test_split=navtest \
  metric_cache_path="$NAVSIM_EXP_ROOT/metric_cache" \
  worker.threads_per_node=16
```

---

## 7. ePDMS smoke (100 scenarios), then full

### 7a. Smoke

```bash
cd "$NAVSIM_DEVKIT_ROOT"
export LATENT_WAM_CHECKPOINT="$LW_ROOT/models/latent-wam/model.safetensors"
# optional: export DINOV2_PATH=...

EVALUATION_DEVICES=0 PDM_WORKERS=8 INFERENCE_BATCH_SIZE=8 \
  bash scripts/evaluation/run_latent_wam_epdms_parallel.sh --max-scenarios 100
```

Default output: `$NAVSIM_EXP_ROOT/latent_wam_eval/navtest.csv`.  
Score: CSV row `average_all_frames`, column `score` × 100 ≈ paper ePDMS (human-penalty on by default).

If OOM: `INFERENCE_BATCH_SIZE=4`. Multi-GPU: `EVALUATION_DEVICES=0,1`.

### 7b. Full navtest (only if smoke OK)

```bash
EVALUATION_DEVICES=0 PDM_WORKERS=8 INFERENCE_BATCH_SIZE=8 \
  bash scripts/evaluation/run_latent_wam_epdms_parallel.sh
# omit --max-scenarios
```

### Alternate single-process CLI

```bash
python scripts/evaluate_latent_wam_epdms.py \
  --checkpoint "$LATENT_WAM_CHECKPOINT" \
  --navsim-log-path "$OPENSCENE_DATA_ROOT/navsim_logs/test" \
  --sensor-path "$OPENSCENE_DATA_ROOT/sensor_blobs/test" \
  --metric-cache "$NAVSIM_EXP_ROOT/metric_cache" \
  --output "$NAVSIM_EXP_ROOT/latent_wam_eval/navtest.csv" \
  --max-scenarios 100
```

---

## 8. Cursor worker (so the Project can continue)

See [`gpu-rental-brief.md`](gpu-rental-brief.md) and [My Machines](https://cursor.com/docs/cloud-agent/self-hosted/my-machines):

```bash
curl https://cursor.com/install -fsS | bash
agent login
cd "$NAVSIM_DEVKIT_ROOT"
agent worker start --name "latent-wam-gpu" --worker-dir "$PWD"
```

---

## Idempotent script（可选）

Project store 曾有 `gpu-bootstrap.sh` 封装上述步骤；**本 `project-docs/` 导出未包含该脚本**。没有脚本时，按本文各节命令顺序手工执行即可。若你本地仍有该脚本：

```bash
bash gpu-bootstrap.sh                  # env + weights + checkpoint smoke
bash gpu-bootstrap.sh --with-data      # also maps + download_test + rearrange
bash gpu-bootstrap.sh --with-data --with-metric-cache
bash gpu-bootstrap.sh --smoke-epdms    # requires data + metric cache already present
```

Do not put API keys or tokens on the script argv.
