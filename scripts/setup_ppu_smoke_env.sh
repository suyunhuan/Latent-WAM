#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${1:-$REPO_ROOT/.venv-ppu-smoke}"

# The Aliyun PPU image supplies its hardware-matched torch/torchvision wheels
# at system level. Keep that binary stack and isolate the project overlay.
python - <<'PY'
import sys
import torch
import torchvision

if sys.version_info[:2] != (3, 12):
    raise SystemExit(f"expected the Aliyun PPU Python 3.12 image, got {sys.version.split()[0]}")
if not torch.cuda.is_available():
    raise SystemExit("the host PPU torch build cannot see a CUDA-compatible device")
print(f"host stack: torch={torch.__version__}, torchvision={torchvision.__version__}")
PY

python -m venv --system-site-packages "$VENV_PATH"

unset ALL_PROXY all_proxy HTTPS_PROXY https_proxy HTTP_PROXY http_proxy
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://aiext-pypi.mirrors.aliyuncs.com/pg1-pip/ubuntu_cu128/simple/}"
"$VENV_PATH/bin/python" -m pip install -r "$REPO_ROOT/requirements-ppu-smoke.txt"
"$VENV_PATH/bin/python" -m pip install --no-deps --ignore-requires-python -e "$REPO_ROOT"

MPLCONFIGDIR="$REPO_ROOT/.cache/matplotlib" "$VENV_PATH/bin/python" - <<'PY'
import cv2
import torch
from navsim.common.dataloader import SceneLoader
from navsim.agents.latent_wam.config import LatentWAMConfig

assert torch.cuda.is_available()
print(f"PPU smoke environment ready: cv2={cv2.__version__}, config={LatentWAMConfig.__name__}")
PY
