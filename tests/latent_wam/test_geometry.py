import pytest
import torch

from navsim.agents.latent_wam.geometry import WorldMirrorFeatureExtractor


class _Processor:
    def __call__(self, images: torch.Tensor, return_tensors: str) -> object:
        del return_tensors
        pixel_values = images.permute(0, 3, 1, 2).float() / 255
        return type("ProcessorOutput", (), {"pixel_values": pixel_values})()


class _VisualEncoder(torch.nn.Module):
    def forward(self, images: torch.Tensor) -> tuple[list[torch.Tensor], int]:
        batch_size, views = images.shape[:2]
        tokens = torch.zeros(batch_size, views, 519, 2048)
        return [tokens], 6


def test_geometry_extractor_rejects_wrong_patch_start_index() -> None:
    extractor = WorldMirrorFeatureExtractor(
        model=_VisualEncoder(),
        processor=_Processor(),
        device=torch.device("cpu"),
        expected_patch_start_index=7,
    )

    with pytest.raises(ValueError, match="expected patch start index 7, got 6"):
        extractor(torch.zeros((3, 8, 12, 3), dtype=torch.uint8))
