import json
from pathlib import Path
from typing import Any

import pytest

from spatial_ingestion.metadata.schema import CameraIntrinsics
from spatial_ingestion.reconstruction._io import write_json
from spatial_ingestion.reconstruction.device import resolve_device
from spatial_ingestion.reconstruction.export import build_run_manifest
from spatial_ingestion.reconstruction.models import HandoffFrame, SyncViewGroup
from spatial_ingestion.reconstruction.pairing import build_sync_pairs


@pytest.fixture
def sync_test_data(tmp_path: Path):
    cam_a_0 = tmp_path / "cam_a_0.jpg"
    cam_b_0 = tmp_path / "cam_b_0.jpg"
    cam_a_1 = tmp_path / "cam_a_1.jpg"
    cam_b_1 = tmp_path / "cam_b_1.jpg"
    for p in [cam_a_0, cam_b_0, cam_a_1, cam_b_1]:
        p.write_bytes(b"x")

    image_paths = [cam_a_0, cam_b_0, cam_a_1, cam_b_1]

    sync_groups = [
        SyncViewGroup(
            anchor_timestamp_ms=0.0,
            frames_by_source={
                "cam_a": HandoffFrame(frame_id="cam_a_0", uri=cam_a_0.as_uri(), index=0, source_id="cam_a"),
                "cam_b": HandoffFrame(frame_id="cam_b_0", uri=cam_b_0.as_uri(), index=0, source_id="cam_b"),
            },
        ),
        SyncViewGroup(
            anchor_timestamp_ms=100.0,
            frames_by_source={
                "cam_a": HandoffFrame(frame_id="cam_a_1", uri=cam_a_1.as_uri(), index=1, source_id="cam_a"),
                "cam_b": HandoffFrame(frame_id="cam_b_1", uri=cam_b_1.as_uri(), index=1, source_id="cam_b"),
            },
        ),
    ]
    return tmp_path, image_paths, sync_groups


@pytest.fixture
def sync_test_images(tmp_path: Path):
    """Create valid tiny JPEG images for tests that need PIL-readable files."""
    from PIL import Image
    import numpy as np
    cam_a_0 = tmp_path / "cam_a_0.jpg"
    cam_b_0 = tmp_path / "cam_b_0.jpg"
    cam_a_1 = tmp_path / "cam_a_1.jpg"
    cam_b_1 = tmp_path / "cam_b_1.jpg"
    for p in [cam_a_0, cam_b_0, cam_a_1, cam_b_1]:
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        Image.fromarray(arr).save(p)
    image_paths = [cam_a_0, cam_b_0, cam_a_1, cam_b_1]
    sync_groups = [
        SyncViewGroup(
            anchor_timestamp_ms=0.0,
            frames_by_source={
                "cam_a": HandoffFrame(frame_id="cam_a_0", uri=cam_a_0.as_uri(), index=0, source_id="cam_a"),
                "cam_b": HandoffFrame(frame_id="cam_b_0", uri=cam_b_0.as_uri(), index=0, source_id="cam_b"),
            },
        ),
        SyncViewGroup(
            anchor_timestamp_ms=100.0,
            frames_by_source={
                "cam_a": HandoffFrame(frame_id="cam_a_1", uri=cam_a_1.as_uri(), index=1, source_id="cam_a"),
                "cam_b": HandoffFrame(frame_id="cam_b_1", uri=cam_b_1.as_uri(), index=1, source_id="cam_b"),
            },
        ),
    ]
    return tmp_path, image_paths, sync_groups


def test_build_run_manifest_captures_runtime_configuration(tmp_path: Path) -> None:
    image = tmp_path / "view.jpg"
    image.write_bytes(b"test")
    output_dir = tmp_path / "out"

    manifest = build_run_manifest(
        image_paths=[image.resolve()],
        output_dir=output_dir,
        output_path=output_dir / "mesh.obj",
        model_name="naver/model",
        device="cpu",
        image_size=512,
        pairing_strategy="complete",
        dry_run=True,
    )

    assert manifest["backend"] == "mast3r"
    assert manifest["device"] == "cpu"
    assert manifest["output_path"].endswith("mesh.obj")


def test_runner_dry_run_writes_manifest(tmp_path: Path) -> None:
    from spatial_ingestion.reconstruction.cli import main as cli_main

    image_a = tmp_path / "a.jpg"
    image_b = tmp_path / "b.jpg"
    image_a.write_bytes(b"a")
    image_b.write_bytes(b"b")

    exit_code = cli_main(
        [
            str(tmp_path),
            "--dry-run",
            "-o",
            str(tmp_path / "out.obj"),
        ]
    )

    output_dirs = [d for d in tmp_path.iterdir() if d.is_dir() and d.name.startswith("out_")]
    assert len(output_dirs) == 1
    manifest = json.loads((output_dirs[0] / "run_manifest.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert manifest["dry_run"] is True


def test_resolve_device_auto_returns_supported_value() -> None:
    assert resolve_device("auto") in {"cpu", "cuda", "mps"}


def test_sync_pairs_are_image_dict_pairs_in_alignment(sync_test_data) -> None:
    from unittest.mock import MagicMock, patch

    from spatial_ingestion.reconstruction.alignment import run_sparse_alignment

    tmp_path, image_paths, sync_groups = sync_test_data

    captured_pairs: list[object] = []
    fake_output_dir = tmp_path / "cache"
    fake_output_dir.mkdir()

    fake_model = MagicMock()
    fake_model.to.return_value = fake_model

    def fake_load_images(paths, **kw):
        return [{"idx": i, "instance": p} for i, p in enumerate(paths)]

    def fake_sparse_ga(imgs, pairs_in, cache_path, model, **kw):
        captured_pairs.extend(pairs_in)
        return MagicMock()

    with (
        patch("spatial_ingestion.reconstruction.alignment.load_model") as mock_load,
        patch("spatial_ingestion.reconstruction.alignment.load_images", side_effect=fake_load_images),
        patch("spatial_ingestion.reconstruction.alignment.sparse_global_alignment", side_effect=fake_sparse_ga),
    ):
        mock_load.return_value = fake_model
        run_sparse_alignment(
            image_paths=image_paths,
            output_dir=tmp_path,
            model_name="fake/model",
            device="cpu",
            image_size=512,
            pairing_strategy="complete",
            sync_view_groups=sync_groups,
        )

    assert len(captured_pairs) == 4, f"Expected 4 sync pairs, got {len(captured_pairs)}"
    for pair in captured_pairs:
        assert isinstance(pair, tuple) and len(pair) == 2
        left, right = pair
        assert isinstance(left, dict), f"Expected dict, got {type(left)}: {left}"
        assert isinstance(right, dict), f"Expected dict, got {type(right)}: {right}"
        assert "idx" in left and "instance" in left
        assert "idx" in right and "instance" in right


def test_build_sync_pairs_creates_cross_camera_pairs(sync_test_data) -> None:
    tmp_path, image_paths, sync_groups = sync_test_data

    pairs = build_sync_pairs(image_paths, sync_groups)

    assert isinstance(pairs, list)
    assert len(pairs) == 4
    assert (0, 1) in pairs or (1, 0) in pairs
    assert (2, 3) in pairs or (3, 2) in pairs


def test_intrinsic_priors_reach_sparse_global_alignment(sync_test_images) -> None:
    from unittest.mock import MagicMock, patch

    from spatial_ingestion.reconstruction.alignment import run_sparse_alignment

    tmp_path, image_paths, sync_groups = sync_test_images

    frames = [
        HandoffFrame(
            frame_id="cam_a_0", uri=image_paths[0].as_uri(), index=0, source_id="cam_a",
            camera_intrinsics=CameraIntrinsics(focal_length_35mm=50.0),
        ),
        HandoffFrame(
            frame_id="cam_b_0", uri=image_paths[1].as_uri(), index=0, source_id="cam_b",
            camera_intrinsics=CameraIntrinsics(focal_length_35mm=50.0),
        ),
        HandoffFrame(
            frame_id="cam_a_1", uri=image_paths[2].as_uri(), index=1, source_id="cam_a",
            camera_intrinsics=CameraIntrinsics(focal_length_35mm=24.0),
        ),
        HandoffFrame(
            frame_id="cam_b_1", uri=image_paths[3].as_uri(), index=1, source_id="cam_b",
            camera_intrinsics=None,
        ),
    ]

    captured_init: list[Any] = [None]

    def fake_sparse_ga(imgs, pairs_in, cache_path, model, **kw):
        captured_init[0] = kw.get("init")
        return MagicMock()

    fake_model = MagicMock()
    fake_model.to.return_value = fake_model

    def fake_load_images(paths, **kw):
        return [{"idx": i, "instance": p} for i, p in enumerate(paths)]

    with (
        patch("spatial_ingestion.reconstruction.alignment.load_model") as mock_load,
        patch("spatial_ingestion.reconstruction.alignment.load_images", side_effect=fake_load_images),
        patch("spatial_ingestion.reconstruction.alignment.sparse_global_alignment", side_effect=fake_sparse_ga),
    ):
        mock_load.return_value = fake_model
        run_sparse_alignment(
            image_paths=image_paths,
            output_dir=tmp_path,
            model_name="fake/model",
            device="cpu",
            image_size=512,
            pairing_strategy="complete",
            sync_view_groups=sync_groups,
            frames=frames,
        )

    init = captured_init[0]
    assert init is not None, "init dict was not passed to sparse_global_alignment"
    assert len(init) == 3
    for str_path, K_dict in init.items():
        assert "intrinsics" in K_dict, f"Missing intrinsics for {str_path}"
        K = K_dict["intrinsics"]
        assert hasattr(K, "shape") and K.shape == (3, 3), f"Expected 3x3 K matrix, got {K}"
    cam_a_1_key = str(image_paths[2])
    K_24 = init[cam_a_1_key]["intrinsics"]
    focal_24 = K_24[0, 0].item()
    for str_path, K_dict in init.items():
        if str_path != cam_a_1_key:
            assert K_dict["intrinsics"][0, 0].item() != focal_24, "Expected different focal for 24mm vs 50mm priors"
