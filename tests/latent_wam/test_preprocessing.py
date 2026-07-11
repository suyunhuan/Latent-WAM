import torch
from transformers import BitImageProcessor

from navsim.agents.latent_wam.preprocessing import normalize_dino_images, preprocess_camera_images


def test_preprocess_camera_images_produces_fixed_uint8_cache_tensor() -> None:
    images = torch.arange(2 * 3 * 32 * 64 * 3, dtype=torch.int64).reshape(2, 3, 32, 64, 3).remainder(256).byte()

    processed = preprocess_camera_images(images, resize_size=(30, 60), crop_size=(24, 48))

    assert processed.shape == (2, 3, 24, 48, 3)
    assert processed.dtype == torch.uint8
    assert processed.is_contiguous()


def test_normalize_dino_images_only_changes_layout_scale_and_normalization() -> None:
    images = torch.zeros((1, 1, 1, 2, 2, 3), dtype=torch.uint8)
    images[..., 0] = 255

    normalized = normalize_dino_images(images)

    expected_red = (1.0 - 0.485) / 0.229
    expected_green = (0.0 - 0.456) / 0.224
    expected_blue = (0.0 - 0.406) / 0.225
    assert normalized.shape == (1, 1, 1, 3, 2, 2)
    assert torch.allclose(normalized[0, 0, 0, :, 0, 0], torch.tensor([expected_red, expected_green, expected_blue]))


def test_cached_preprocessing_matches_previous_dino_processor_within_one_uint8_level() -> None:
    torch.manual_seed(0)
    images = torch.randint(0, 256, (1, 1080, 1920, 3), dtype=torch.uint8)
    processor = BitImageProcessor(
        do_resize=True,
        size={"shortest_edge": 256},
        resample=3,
        do_center_crop=True,
        crop_size={"height": 224, "width": 448},
        do_rescale=True,
        do_normalize=True,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
    )

    previous = processor(images=images, return_tensors="pt").pixel_values
    cached = normalize_dino_images(preprocess_camera_images(images))

    assert previous.shape == cached.shape
    assert torch.max(torch.abs(previous - cached)) < 0.018
