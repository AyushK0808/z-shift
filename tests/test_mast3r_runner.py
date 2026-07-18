import json
from pathlib import Path

from spatial_ingestion.reconstruction.models import HandoffFrame, SyncViewGroup
from spatial_ingestion.reconstruction.runners.mast3r import (
    _build_sync_pairs,
    build_run_manifest,
    main,
    resolve_device,
    resolve_image_input,
)


def test_resolve_image_input_supports_file_uri(tmp_path: Path) -> None:
    image = tmp_path / "view.jpg"
    image.write_bytes(b"test")

    resolved = resolve_image_input(image.as_uri())

    assert resolved == image.resolve()


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
    image_a = tmp_path / "a.jpg"
    image_b = tmp_path / "b.jpg"
    image_a.write_bytes(b"a")
    image_b.write_bytes(b"b")
    output_dir = tmp_path / "artifacts"

    exit_code = main(
        [
            "--output-dir",
            str(output_dir),
            "--dry-run",
            str(image_a),
            str(image_b),
        ]
    )

    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert manifest["dry_run"] is True
    assert manifest["image_paths"] == [str(image_a.resolve()), str(image_b.resolve())]


def test_resolve_device_auto_returns_supported_value() -> None:
    assert resolve_device("auto") in {"cpu", "cuda"}


def test_build_sync_pairs_creates_cross_camera_pairs(tmp_path: Path) -> None:
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

    pairs = _build_sync_pairs(image_paths, sync_groups)

    assert isinstance(pairs, list)
    assert len(pairs) == 4
    assert (0, 1) in pairs or (1, 0) in pairs
    assert (2, 3) in pairs or (3, 2) in pairs
