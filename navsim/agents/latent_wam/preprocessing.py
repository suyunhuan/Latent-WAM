"""Deterministic image preprocessing shared by cache generation and inference."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

DINO_RESIZE_SIZE: Tuple[int, int] = (256, 455)
DINO_CROP_SIZE: Tuple[int, int] = (224, 448)
DINO_IMAGE_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
DINO_IMAGE_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)
PREPROCESSING_REVISION = 1


def preprocess_camera_images(
    images: torch.Tensor,
    resize_size: Tuple[int, int] = DINO_RESIZE_SIZE,
    crop_size: Tuple[int, int] = DINO_CROP_SIZE,
) -> torch.Tensor:
    """Resize and center-crop channel-last uint8 camera images.

    Args:
        images: Images shaped ``[..., H, W, 3]`` with dtype ``uint8``.
        resize_size: Intermediate ``(height, width)``.
        crop_size: Final ``(height, width)``.

    Returns:
        Contiguous channel-last uint8 images with the original leading dimensions.
    """
    if images.dtype != torch.uint8 or images.ndim < 3 or images.shape[-1] != 3:
        raise ValueError("camera images must be channel-last uint8 tensors")
    resize_height, resize_width = resize_size
    crop_height, crop_width = crop_size
    if min(resize_height, resize_width, crop_height, crop_width) < 1:
        raise ValueError("resize and crop dimensions must be positive")
    if crop_height > resize_height or crop_width > resize_width:
        raise ValueError("crop size must not exceed resize size")

    leading_shape = images.shape[:-3]
    flat = images.reshape(-1, *images.shape[-3:]).permute(0, 3, 1, 2).float()
    resized = F.interpolate(
        flat,
        size=resize_size,
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    top = (resize_height - crop_height) // 2
    left = (resize_width - crop_width) // 2
    cropped = resized[:, :, top : top + crop_height, left : left + crop_width]
    result = cropped.round().clamp_(0, 255).to(torch.uint8).permute(0, 2, 3, 1)
    return result.reshape(*leading_shape, crop_height, crop_width, 3).contiguous()


def normalize_dino_images(images: torch.Tensor) -> torch.Tensor:
    """Convert preprocessed uint8 NHWC images to normalized float NCHW tensors."""
    if images.dtype != torch.uint8 or images.ndim < 3 or images.shape[-1] != 3:
        raise ValueError("DINO images must be channel-last uint8 tensors")
    leading_shape = images.shape[:-3]
    height, width = images.shape[-3:-1]
    flat = images.reshape(-1, height, width, 3).permute(0, 3, 1, 2).float().div_(255.0)
    mean = flat.new_tensor(DINO_IMAGE_MEAN).view(1, 3, 1, 1)
    std = flat.new_tensor(DINO_IMAGE_STD).view(1, 3, 1, 1)
    normalized = flat.sub_(mean).div_(std)
    return normalized.reshape(*leading_shape, 3, height, width)
