import json
from pathlib import Path

from spatial_ingestion.reconstruction.runners.mast3r import (
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
    assert manifest["artifacts"]["point_cloud"].endswith("point_cloud.ply")


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
