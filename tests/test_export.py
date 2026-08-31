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
    assert np.all(xyz == pts3d[0].reshape(-1, 3)[1:])


def test_export_scene_to_mesh_writes_ply_and_glb(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    imgs = [rng.random((8, 8, 3))]
    pts3d = [rng.random((8, 8, 3))]
    confs = [np.full((8, 8), 2.0)]
    scene = _FakeScene(imgs=imgs, pts3d=pts3d, confs=confs)

    output = tmp_path / "out" / "mesh.glb"
    result = export_scene_to_mesh(
        scene,
        output_path=output,
        output_dir=output.parent,
        tsdf_thresh=0.0,
        min_conf_thr=1.5,
    )

    assert result is False
    assert output.exists() and output.stat().st_size > 0
    ply_path = output.parent / "point_cloud.ply"
    assert ply_path.exists() and ply_path.stat().st_size > 0


def test_patch_tsdf_cuda_hardcode_makes_cuda_a_noop_without_cuda() -> None:
    """Regression test for a vendored MASt3R bug: TSDFPostProcess calls
    `tensor.cuda()` unconditionally, which raises AssertionError on a
    CPU-only torch build (or RuntimeError if no driver is present) and crashed
    reconstruction outright."""
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        pytest.skip("only meaningful on a CPU-only torch build")

    tensor = torch.zeros(3)
    with pytest.raises((AssertionError, RuntimeError)):
        tensor.cuda()

    _patch_tsdf_cuda_hardcode()
    result = tensor.cuda()
    assert result is tensor

    _patch_tsdf_cuda_hardcode()  # idempotent: re-patching must not wrap twice
    assert tensor.cuda() is tensor
