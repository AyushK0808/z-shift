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
    assert (tmp_path / "run_manifest.json").is_file()


def test_resolve_output_path_defaults_to_obj_in_reconstruction_dir(tmp_path: Path) -> None:
    target = resolve_output_path(tmp_path / "images", None)

    assert target.name == "images.obj"


def test_resolve_output_path_preserves_explicit_obj(tmp_path: Path) -> None:
    target = resolve_output_path(tmp_path / "images", str(tmp_path / "mesh.obj"))

    assert target == (tmp_path / "mesh.obj").resolve()


def test_resolve_output_path_preserves_explicit_glb(tmp_path: Path) -> None:
    target = resolve_output_path(tmp_path / "images", str(tmp_path / "mesh.glb"))

    assert target == (tmp_path / "mesh.glb").resolve()


def test_resolve_output_path_appends_mesh_obj_for_directory(tmp_path: Path) -> None:
    target = resolve_output_path(tmp_path / "images", str(tmp_path / "out"))

    assert target == (tmp_path / "out" / "mesh.obj").resolve()
