from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from spatial_ingestion.reconstruction.export import (
    _dense_points_xyz_rgb,
    _patch_tsdf_cuda_hardcode,
    export_scene_to_mesh,
)


class _FakeScene:
    """Minimal stand-in for a MASt3R scene exposing dense points and confidence."""

    def __init__(
        self,
        imgs: list[np.ndarray],
        pts3d: list[np.ndarray],
        confs: list[np.ndarray],
    ) -> None:
        self.imgs = imgs
        self._pts3d = pts3d
        self._confs = confs

    def get_dense_pts3d(self, clean_depth: bool = True) -> tuple[Any, Any, Any]:
        return self._pts3d, None, self._confs


def test_dense_points_xyz_rgb_masks_low_confidence_points() -> None:
    rng = np.random.default_rng(0)
    imgs = [rng.random((4, 4, 3))]
    pts3d = [rng.random((4, 4, 3))]
    confs = [np.full((4, 4), 2.0)]
    confs[0][0, 0] = 0.5

    xyz, rgb = _dense_points_xyz_rgb(imgs, pts3d, confs, min_conf_thr=1.5)

    assert xyz.shape == (15, 3)
    assert rgb.shape == (15, 3)


def test_export_scene_to_mesh_writes_mesh_and_point_cloud(tmp_path: Path) -> None:
    try:
        import dust3r  # noqa: F401
        import mast3r  # noqa: F401
    except ImportError:
        pytest.skip("MASt3R is not installed")

    rng = np.random.default_rng(1)
    imgs = [rng.random((4, 4, 3))]
    pts3d = [rng.random((4, 4, 3))]
    confs = [np.full((4, 4), 2.0)]
    scene = _FakeScene(imgs, pts3d, confs)

    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True)
    output = output_dir / "mesh.glb"

    fell_back = export_scene_to_mesh(scene, output, output_dir, min_conf_thr=1.5)

    assert output.exists() and output.stat().st_size > 0
    assert (output_dir / "point_cloud.ply").exists()
    assert fell_back is False


def test_write_cameras_records_each_image_pose(tmp_path: Path) -> None:
    pytest.importorskip("dust3r")
    import json

    from spatial_ingestion.reconstruction.export import write_cameras

    poses = np.repeat(np.eye(4)[None], 2, axis=0)
    poses[1, :3, 3] = [0.1, 0.2, 0.3]

    class _Posed:
        img_paths = ["/x/rect_001_3_r5000.png", "/x/rect_002_3_r5000.png"]

        def get_im_poses(self) -> np.ndarray:
            return poses

        def get_focals(self) -> np.ndarray:
            return np.array([800.0, 810.0])

    write_cameras(_Posed(), tmp_path / "cameras.json")
    cameras = json.loads((tmp_path / "cameras.json").read_text())["cameras"]
    assert [c["image"] for c in cameras] == _Posed.img_paths
    assert np.allclose(cameras[1]["cam2world"], poses[1])
    assert cameras[1]["focal_px"] == [810.0]


def test_patch_tsdf_cuda_hardcode_makes_cuda_a_noop_without_cuda() -> None:
    """Regression test for a vendored MASt3R bug: TSDFPostProcess calls
    `tensor.cuda()` unconditionally, which raises AssertionError on a
    CPU-only torch build and crashed reconstruction outright."""
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        pytest.skip("only meaningful on a CPU-only torch build")

    tensor = torch.zeros(3)
    with pytest.raises(AssertionError):
        tensor.cuda()

    _patch_tsdf_cuda_hardcode()
    result = tensor.cuda()
    assert result is tensor

    _patch_tsdf_cuda_hardcode()  # idempotent: re-patching must not wrap twice
    assert tensor.cuda() is tensor
