from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import pyvista as pv

from spatial_ingestion.refinement import (
    MeshCleaningConfig,
    MeshValidationError,
    clean_ai_mesh,
    clean_mesh,
    load_mesh_file,
    write_mesh_file,
)


def _make_colored_sphere_with_hole() -> pv.PolyData:
    sphere = pv.Sphere(theta_resolution=32, phi_resolution=32)
    holey = sphere.clip(normal=(0.0, 0.0, 1.0), origin=(0.0, 0.0, 0.45)).extract_surface(
        algorithm=None
    )
    colors = np.zeros((holey.n_points, 3), dtype=np.uint8)
    colors[:, 0] = np.linspace(40, 220, holey.n_points, dtype=np.uint8)
    colors[:, 1] = 80
    colors[:, 2] = 200
    holey.point_data["rgb"] = colors
    holey.point_data.active_scalars_name = "rgb"
    return holey


def _make_disjoint_sheets() -> pv.PolyData:
    first = pv.Plane(i_resolution=4, j_resolution=4, direction=(0, 0, 1), center=(0.0, 0.0, 0.0))
    second = pv.Plane(i_resolution=4, j_resolution=4, direction=(0, 0, 1), center=(3.0, 0.0, 0.0))
    return first.merge(second, merge_points=False).extract_surface(algorithm=None)


def _make_room_like_mesh() -> pv.PolyData:
    wall = pv.Plane(i_resolution=8, j_resolution=8, direction=(0, 0, 1), center=(0.0, 0.0, 0.0))
    debris = pv.Cube(
        center=(4.0, 0.0, 0.0), x_length=0.05, y_length=0.05, z_length=0.05
    ).extract_surface(algorithm=None)
    return wall.merge(debris, merge_points=False).extract_surface(algorithm=None)


def test_object_mode_closes_holes_and_preserves_colors() -> None:
    mesh = _make_colored_sphere_with_hole()

    result = clean_mesh(
        mesh, MeshCleaningConfig(mode="object", smoothing_iters=0, verify_watertight=True)
    )

    output = result["mesh"]
    assert result["is_watertight"] is True
    assert result["open_edge_count"] == 0
    assert "rgb" in output.point_data
    assert output.point_data["rgb"].shape[1] == 3


def test_clean_mesh_defaults_and_alias_remain_compatible() -> None:
    mesh = _make_colored_sphere_with_hole()

    result = clean_mesh(mesh)

    assert clean_ai_mesh is clean_mesh
    assert result["mode"] == "object"


def test_room_mode_keeps_major_sheet_and_drops_small_debris() -> None:
    mesh = _make_room_like_mesh()

    result = clean_mesh(mesh, MeshCleaningConfig(mode="room", smoothing_iters=0, min_cell_count=20))

    output = result["mesh"]
    assert output.n_cells >= 64
    assert result["boundary_edge_count"] and result["boundary_edge_count"] > 0
    assert result["non_manifold_edge_count"] == 0


def test_multi_sheet_object_mode_keeps_all_components() -> None:
    mesh = _make_disjoint_sheets()

    result = clean_mesh(
        mesh, MeshCleaningConfig(mode="object", smoothing_iters=0, verify_watertight=False)
    )

    assert len(result["mesh"].split_bodies()) == 2


def test_nan_rejection() -> None:
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [np.nan, 1.0, 0.0]])
    mesh = pv.PolyData(points, faces=np.array([3, 0, 1, 2]))

    with pytest.raises(MeshValidationError):
        clean_mesh(mesh, MeshCleaningConfig(mode="object", smoothing_iters=0))


def test_decimation_reduces_triangle_count() -> None:
    mesh = pv.Sphere(theta_resolution=64, phi_resolution=64)

    result = clean_mesh(
        mesh,
        MeshCleaningConfig(
            mode="object", smoothing_iters=0, decimate_target_reduction=0.5, verify_watertight=False
        ),
    )

    assert result["output_cell_count"] < mesh.n_cells


def test_write_mesh_file_round_trips_glb(tmp_path: Path) -> None:
    mesh = _make_colored_sphere_with_hole()
    target = tmp_path / "cleaned.glb"

    write_mesh_file(mesh, target)

    assert target.exists() and target.stat().st_size > 0
    reloaded = load_mesh_file(target)
    assert isinstance(reloaded, pv.PolyData)
    assert reloaded.n_cells == mesh.n_cells
    assert reloaded.point_data["COLOR_0"].shape[1] == 4


def test_load_mesh_file_unwraps_glb_multiblock(tmp_path: Path) -> None:
    import trimesh

    sphere = trimesh.creation.icosphere(subdivisions=2)
    raw = tmp_path / "raw.glb"
    sphere.export(str(raw))

    loaded = load_mesh_file(raw)

    assert isinstance(loaded, pv.PolyData)
    assert loaded.n_cells == len(sphere.faces)


def test_load_mesh_file_warns_when_glb_has_multiple_primitives(tmp_path: Path, caplog) -> None:
    import logging

    import trimesh
    import trimesh.visual

    sphere = trimesh.creation.icosphere(subdivisions=2)
    box = trimesh.creation.box()
    scene = trimesh.Scene(geometry=[sphere, box])
    raw = tmp_path / "multi.glb"
    scene.export(str(raw))

    with caplog.at_level(logging.WARNING, logger="spatial_ingestion.refinement.core"):
        loaded = load_mesh_file(raw)

    assert isinstance(loaded, pv.PolyData)
    assert loaded.n_cells == len(sphere.faces)
    assert any("usable mesh blocks" in record.message for record in caplog.records)


def test_clean_mesh_accepts_glb_multiblock_directly(tmp_path: Path) -> None:
    import trimesh

    sphere = trimesh.creation.icosphere(subdivisions=2)
    raw = tmp_path / "raw.glb"
    sphere.export(str(raw))
    mesh = load_mesh_file(raw)

    result = clean_mesh(mesh, MeshCleaningConfig(mode="object", smoothing_iters=0))

    assert result["output_cell_count"] == len(sphere.faces)


def test_clean_mesh_preserves_vertex_colors_through_smoothing(tmp_path: Path) -> None:
    import trimesh
    import trimesh.visual

    from spatial_ingestion.refinement import to_trimesh

    sphere = trimesh.creation.icosphere(subdivisions=2)
    colors = np.zeros((len(sphere.vertices), 4), dtype=np.uint8)
    colors[:, 0] = 255
    colors[:, 2] = np.linspace(0, 255, len(sphere.vertices), dtype=np.uint8)
    colors[:, 3] = 255
    sphere.visual = trimesh.visual.ColorVisuals(vertex_colors=colors)
    raw = tmp_path / "raw.glb"
    sphere.export(str(raw))
    mesh = load_mesh_file(raw)
    assert "COLOR_0" in mesh.point_data

    result = clean_mesh(
        mesh,
        MeshCleaningConfig(mode="object", smoothing_iters=3, verify_watertight=False),
    )

    output = result["mesh"]
    assert "COLOR_0" in output.point_data
    transferred = np.asarray(output.point_data["COLOR_0"])
    assert transferred.shape == (output.n_points, 4)
    assert np.count_nonzero(transferred.sum(axis=1)) > output.n_points / 2
    from scipy.spatial import KDTree

    _, nearest = KDTree(np.asarray(mesh.points)).query(np.asarray(output.points))
    expected = colors[nearest].astype(np.float32) / 255.0
    assert np.allclose(transferred, expected, atol=0.05)

    tri = to_trimesh(output)
    tri_colors = (
        tri.visual.vertex_colors if isinstance(tri.visual, trimesh.visual.ColorVisuals) else None
    )
    assert tri_colors is not None
    assert np.count_nonzero(tri_colors.sum(axis=1)) > len(tri.vertices) / 2


def test_write_mesh_file_keeps_vertex_colors_for_obj_and_ply(tmp_path: Path) -> None:
    import trimesh
    import trimesh.visual

    mesh = _make_colored_sphere_with_hole()

    for suffix in (".obj", ".ply"):
        target = tmp_path / f"cleaned{suffix}"
        write_mesh_file(mesh, target)
        assert target.exists() and target.stat().st_size > 0
        loaded = trimesh.load(str(target))
        assert isinstance(loaded, trimesh.Trimesh), f"{suffix} did not load as a mesh"
        tri_colors = None
        if isinstance(loaded.visual, trimesh.visual.ColorVisuals):
            tri_colors = loaded.visual.vertex_colors
        assert tri_colors is not None, f"no vertex colors survived {suffix} export"
        assert np.count_nonzero(tri_colors.sum(axis=1)) > len(loaded.vertices) / 2

    ply_reloaded = load_mesh_file(tmp_path / "cleaned.ply")
    assert any(name in ply_reloaded.point_data for name in ("RGB", "RGBA"))


def test_preserve_data_arrays_only_transfers_recognized_color_names() -> None:
    from spatial_ingestion.refinement.core import preserve_data_arrays

    rng = np.random.default_rng(0)
    source = pv.Sphere(theta_resolution=16, phi_resolution=16)
    source.point_data["Confidence"] = rng.random((source.n_points, 3))
    source.point_data["COLOR_0"] = np.full((source.n_points, 4), [255, 0, 0, 255], dtype=np.uint8)
    target = pv.Sphere(theta_resolution=16, phi_resolution=16)
    target.points += 0.01

    result = preserve_data_arrays(source, target)

    assert "COLOR_0" in result.point_data
    assert "Confidence" not in result.point_data
