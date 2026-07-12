import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from navsim.agents.latent_wam.geometry import WorldMirrorFeatureExtractor, WorldMirrorInputs
from navsim.planning.script.run_latent_wam_caching import build_world_mirror_inputs


def _worldmirror_camera_utils_available() -> bool:
    try:
        return importlib.util.find_spec("worldmirror.src.models.utils.camera_utils") is not None
    except ModuleNotFoundError:
        return False


requires_worldmirror = pytest.mark.skipif(
    not _worldmirror_camera_utils_available(),
    reason="requires the checked-out WorldMirror submodule",
)


class _Processor:
    def __call__(self, images: torch.Tensor, return_tensors: str) -> object:
        del return_tensors
        pixel_values = images.permute(0, 3, 1, 2).float() / 255
        return type("ProcessorOutput", (), {"pixel_values": pixel_values})()


class _VisualEncoder(torch.nn.Module):
    def __init__(self, patch_start_index: int = 7) -> None:
        super().__init__()
        self.patch_start_index = patch_start_index

    def forward(
        self,
        images: torch.Tensor,
        priors: tuple[torch.Tensor | None, torch.Tensor, torch.Tensor],
        cond_flags: list[int],
    ) -> tuple[list[torch.Tensor], int]:
        assert cond_flags == [1, 0, 1]
        assert priors[0] is None
        assert priors[1].shape == (1, 3, 4)
        assert priors[2].shape == (1, 3, 7)
        batch_size, views = images.shape[:2]
        tokens = torch.zeros(batch_size, views, 519, 2048)
        return [tokens], self.patch_start_index


def _inputs() -> WorldMirrorInputs:
    return WorldMirrorInputs(
        images=torch.zeros((1, 3, 8, 12, 3), dtype=torch.uint8),
        intrinsics=torch.eye(3).reshape(1, 1, 3, 3).repeat(1, 3, 1, 1),
        extrinsics=torch.eye(4).reshape(1, 1, 4, 4).repeat(1, 3, 1, 1),
    )


@requires_worldmirror
def test_geometry_extractor_uses_pose_and_ray_conditions() -> None:
    extractor = WorldMirrorFeatureExtractor(
        model=_VisualEncoder(),
        processor=_Processor(),
        device=torch.device("cpu"),
    )

    assert extractor(_inputs()).shape == (1, 3, 519, 2048)


@requires_worldmirror
def test_geometry_extractor_rejects_wrong_patch_start_index() -> None:
    extractor = WorldMirrorFeatureExtractor(
        model=_VisualEncoder(patch_start_index=6),
        processor=_Processor(),
        device=torch.device("cpu"),
        expected_patch_start_index=7,
    )

    with pytest.raises(ValueError, match="expected patch start index 7, got 6"):
        extractor(_inputs())


def test_geometry_inputs_require_camera_calibration() -> None:
    with pytest.raises(ValueError, match="intrinsics must have shape"):
        WorldMirrorInputs(
            images=torch.zeros((1, 3, 8, 12, 3), dtype=torch.uint8),
            intrinsics=torch.empty(0),
            extrinsics=torch.eye(4).reshape(1, 1, 4, 4).repeat(1, 3, 1, 1),
        ).validate()


def _camera(translation: tuple[float, float, float]) -> SimpleNamespace:
    return SimpleNamespace(
        image=np.zeros((1080, 1920, 3), dtype=np.uint8),
        intrinsics=np.array(
            [
                [1000.0, 0.0, 960.0],
                [0.0, 800.0, 540.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        sensor2lidar_rotation=np.eye(3, dtype=np.float32),
        sensor2lidar_translation=np.array(translation, dtype=np.float32),
    )


def _scene() -> SimpleNamespace:
    def cameras() -> SimpleNamespace:
        return SimpleNamespace(
            cam_l0=_camera((1.0, 2.0, 3.0)),
            cam_f0=_camera((0.0, 0.0, 0.0)),
            cam_r0=_camera((-1.0, -2.0, -3.0)),
        )

    return SimpleNamespace(frames=[SimpleNamespace(cameras=cameras()) for _ in range(12)])


def test_navsim_cameras_build_legacy_world_mirror_inputs() -> None:
    inputs = build_world_mirror_inputs(_scene())

    assert inputs.images.shape == (4, 3, 1080, 1920, 3)
    torch.testing.assert_close(
        inputs.intrinsics[0, 0],
        torch.tensor(
            [
                [236.9791717529297, 0.0, 224.0],
                [0.0, 189.62962341308594, 112.0],
                [0.0, 0.0, 1.0],
            ]
        ),
    )
    torch.testing.assert_close(inputs.extrinsics[0, 0, :3, 3], torch.tensor([-1.0, -2.0, -3.0]))


def test_navsim_cameras_reject_missing_calibration() -> None:
    scene = _scene()
    scene.frames[3].cameras.cam_f0.intrinsics = None

    with pytest.raises(ValueError, match="frame 3 camera cam_f0 is missing intrinsics"):
        build_world_mirror_inputs(scene)
