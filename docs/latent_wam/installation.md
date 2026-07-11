# Installation

This document is authoritative for the released environment.

## Supported platform

- Linux x86-64
- Python 3.10
- PyTorch 2.4.1, torchvision 0.19.1
- CUDA 12.4 and an NVIDIA GPU for inference/cache extraction

Clone the repository with WorldMirror and create the unified environment:

```bash
git clone --recurse-submodules <repository-url> latent-wam
cd latent-wam
conda env create -f environment.yml
conda activate latent-wam
pip install -e .
```

If the repository was cloned without submodules:

```bash
git submodule update --init --recursive
```

The OpenCV dependency is the headless wheel so server installations do not
require `libGL.so.1`.

## NAVSIM paths

```bash
export NAVSIM_DEVKIT_ROOT=/path/to/latent-wam
export NAVSIM_EXP_ROOT=/path/to/experiments
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export NUPLAN_MAPS_ROOT=/path/to/dataset/maps
export OPENSCENE_DATA_ROOT=/path/to/dataset
```

Use `nvidia-smi` and the following smoke check before running extraction:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "from navsim.agents.latent_wam.config import LatentWAMConfig; print(LatentWAMConfig())"
```

## Scope of the unified environment

The environment supports Latent WAM training/inference and WorldMirror visual
geometry extraction. It does not install or promise reproducibility of the
full upstream WorldMirror renderer or training stack. WorldMirror is separately licensed; read
`worldmirror/License.txt` before downloading its weights or running it.

Latent WAM cache extraction loads only WorldMirror's visual geometry encoder.
It does not instantiate the camera/depth/Gaussian heads or execute gsplat.
Users who run the upstream model outside the Latent WAM cache path must follow
WorldMirror's own environment instructions, including a platform-compatible
gsplat build.

Hardware-vendor PyTorch builds must use the wheel matrix supplied by that
vendor. Such local environments are outside the supported Python 3.10/NVIDIA
release environment.
