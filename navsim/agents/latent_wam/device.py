"""Device selection helpers for Latent WAM inference and short training.

Auto selection keeps NVIDIA CUDA as the default when it is available (AutoDL /
official recipes). On Apple Silicon without CUDA, MPS is preferred over CPU.
Override with ``LATENT_WAM_DEVICE=mps|cuda|cpu`` or an explicit preference.
"""

from __future__ import annotations

import os
from typing import Literal, Optional, Union

# DINOv2 bicubic pos-embed upsample is missing on MPS; must be set before torch loads.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch

DevicePreference = Literal["auto", "cuda", "mps", "cpu"]
_ENV_VAR = "LATENT_WAM_DEVICE"


def _normalize_preference(value: Optional[str]) -> DevicePreference:
    if value is None or value.strip() == "":
        return "auto"
    normalized = value.strip().lower()
    if normalized not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError(
            f"unsupported device preference {value!r}; expected auto|cuda|mps|cpu"
        )
    return normalized  # type: ignore[return-value]


def mps_is_available() -> bool:
    """Return True when PyTorch can use Apple Metal Performance Shaders."""
    return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())


def enable_mps_cpu_fallback() -> None:
    """Allow unimplemented MPS ops (e.g. bicubic upsample in DINOv2) to fall back to CPU.

    Must be set before the first MPS kernel launch. Safe no-op when already configured.
    """
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def resolve_device_preference(preference: Optional[str] = None) -> DevicePreference:
    """Resolve preference from an argument or ``LATENT_WAM_DEVICE`` (default auto)."""
    if preference is not None and str(preference).strip() != "":
        return _normalize_preference(str(preference))
    return _normalize_preference(os.environ.get(_ENV_VAR))


def resolve_torch_device(
    preference: Optional[str] = None,
    *,
    local_rank: int = 0,
    require_accelerator: bool = False,
) -> torch.device:
    """Pick a ``torch.device`` for Latent WAM.

    Order when preference is ``auto``:
    1. CUDA if available (preserves the official NVIDIA path)
    2. MPS if available (Apple Silicon)
    3. CPU

    Args:
        preference: Explicit ``auto|cuda|mps|cpu``, or ``None`` to read the env.
        local_rank: CUDA device index for multi-GPU / torchrun.
        require_accelerator: If True, refuse to fall back to CPU.

    Returns:
        Selected torch device.

    Raises:
        RuntimeError: When a requested accelerator is missing, or when
            ``require_accelerator`` is set and only CPU is available.
    """
    choice = resolve_device_preference(preference)

    def _cuda() -> torch.device:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
        return torch.device("cuda", local_rank)

    def _mps() -> torch.device:
        if not mps_is_available():
            raise RuntimeError("MPS was requested but torch.backends.mps.is_available() is False")
        enable_mps_cpu_fallback()
        return torch.device("mps")

    if choice == "cuda":
        return _cuda()
    if choice == "mps":
        return _mps()
    if choice == "cpu":
        return torch.device("cpu")

    # auto: CUDA first (official NVIDIA path), then Apple MPS, then CPU
    if torch.cuda.is_available():
        return torch.device("cuda", local_rank)
    if mps_is_available():
        enable_mps_cpu_fallback()
        return torch.device("mps")
    if require_accelerator:
        raise RuntimeError(
            "Latent WAM requires CUDA or Apple MPS; neither is available "
            f"(set {_ENV_VAR}=cpu to force a CPU-only smoke path)"
        )
    return torch.device("cpu")


def describe_device(device: Optional[torch.device] = None) -> str:
    """Human-readable device label for logs."""
    target = device if device is not None else resolve_torch_device()
    if target.type == "cuda":
        index = target.index if target.index is not None else torch.cuda.current_device()
        name = torch.cuda.get_device_name(index)
        return f"cuda:{index} ({name})"
    if target.type == "mps":
        return "mps (Apple Silicon)"
    return "cpu"


def lightning_accelerator(device: Optional[Union[torch.device, str]] = None) -> str:
    """Map a torch device to a Lightning ``accelerator`` string."""
    if device is None:
        device = resolve_torch_device()
    kind = device.type if isinstance(device, torch.device) else str(device).split(":", 1)[0]
    if kind == "cuda":
        return "gpu"
    if kind == "mps":
        return "mps"
    return "cpu"


def training_precision(default: Union[str, int] = "bf16-mixed") -> Union[str, int]:
    """Return a Lightning precision setting safe for the selected device.

    Apple MPS commonly requires float32; mixed bf16/fp16 paths are disabled.
    """
    device = resolve_torch_device()
    if device.type == "mps":
        return "32-true"
    if device.type == "cpu":
        return "32-true"
    return default


def pin_memory_enabled(device: Optional[torch.device] = None) -> bool:
    """Pin host memory only when transferring to CUDA."""
    target = device if device is not None else resolve_torch_device()
    return target.type == "cuda"


def synchronize(device: Optional[torch.device] = None) -> None:
    """Device-side sync used by benchmarks (no-op on CPU)."""
    target = device if device is not None else resolve_torch_device()
    if target.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(target)
    elif target.type == "mps" and mps_is_available():
        torch.mps.synchronize()


def empty_cache(device: Optional[torch.device] = None) -> None:
    """Release cached allocator blocks when the backend supports it."""
    target = device if device is not None else resolve_torch_device()
    if target.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif target.type == "mps" and mps_is_available():
        torch.mps.empty_cache()


def max_memory_allocated_gib(device: Optional[torch.device] = None) -> Optional[float]:
    """Peak allocated memory in GiB when the backend exposes stats."""
    target = device if device is not None else resolve_torch_device()
    if target.type == "cuda" and torch.cuda.is_available():
        index = target.index if target.index is not None else 0
        return torch.cuda.max_memory_allocated(index) / 1024**3
    return None
