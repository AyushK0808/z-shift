from pathlib import Path

from spatial_ingestion.reconstruction.cli import collect_input_images, main, resolve_output_path


def test_collect_input_images_rejects_single_file(tmp_path: Path) -> None:
    image = tmp_path / "chair.png"
    image.write_bytes(b"data")

    try:
        collect_input_images(image)
    except ValueError as exc:
        assert "at least two views" in str(exc)
    else:
        raise AssertionError("expected single-image input to be rejected")


def test_collect_input_images_sorts_directory_entries(tmp_path: Path) -> None:
    image_b = tmp_path / "b.png"
    image_a = tmp_path / "a.png"
    image_b.write_bytes(b"b")
    image_a.write_bytes(b"a")

    assert collect_input_images(tmp_path) == [image_a, image_b]


def test_cli_dry_run_accepts_a_folder_with_multiple_views(tmp_path: Path) -> None:
    (tmp_path / "front.png").write_bytes(b"front")
    (tmp_path / "side.png").write_bytes(b"side")
    output = tmp_path / "output.obj"

    assert main([str(tmp_path), "--dry-run", "-o", str(output)]) == 0
    job_dirs = [d for d in tmp_path.iterdir() if d.is_dir() and d.name.startswith("output_")]
    assert len(job_dirs) == 1
    assert (job_dirs[0] / "run_manifest.json").exists()


def test_resolve_output_path_defaults_to_obj_in_reconstruction_dir(tmp_path: Path) -> None:
    target = resolve_output_path(tmp_path / "images", None)

    assert target.name == "images.obj"
    assert target.parent.name.startswith("images_")


def test_resolve_output_path_preserves_explicit_obj(tmp_path: Path) -> None:
    target = resolve_output_path(tmp_path / "images", str(tmp_path / "mesh.obj"))

    assert target.name == "mesh.obj"
    assert target.parent.parent == (tmp_path / "mesh.obj").resolve().parent
    assert target.parent.name.startswith("mesh_")


def test_resolve_output_path_preserves_explicit_glb(tmp_path: Path) -> None:
    target = resolve_output_path(tmp_path / "images", str(tmp_path / "mesh.glb"))

    assert target.name == "mesh.glb"
    assert target.parent.parent == (tmp_path / "mesh.glb").resolve().parent
    assert target.parent.name.startswith("mesh_")


def test_resolve_output_path_appends_mesh_obj_for_directory(tmp_path: Path) -> None:
    target = resolve_output_path(tmp_path / "images", str(tmp_path / "out"))

    assert target.suffix == ".obj"
    assert target.parent == (tmp_path / "out").resolve()
    assert target.stem.startswith("mesh_")
