from __future__ import annotations

import numpy as np
import pyvista as pv
import pytest

from spatial_ingestion.refinement import MeshCleaningConfig, MeshValidationError, clean_ai_mesh, clean_mesh


def _make_colored_sphere_with_hole() -> pv.PolyData:
    sphere = pv.Sphere(theta_resolution=32, phi_resolution=32)
    holey = sphere.clip(normal=(0.0, 0.0, 1.0), origin=(0.0, 0.0, 0.45)).extract_surface(algorithm=None)
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
    debris = pv.Cube(center=(4.0, 0.0, 0.0), x_length=0.05, y_length=0.05, z_length=0.05).extract_surface(algorithm=None)
    return wall.merge(debris, merge_points=False).extract_surface(algorithm=None)


def test_object_mode_closes_holes_and_preserves_colors() -> None:
    mesh = _make_colored_sphere_with_hole()

    result = clean_mesh(mesh, MeshCleaningConfig(mode="object", smoothing_iters=0, verify_watertight=True))

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

    result = clean_mesh(mesh, MeshCleaningConfig(mode="object", smoothing_iters=0, verify_watertight=False))

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
        MeshCleaningConfig(mode="object", smoothing_iters=0, decimate_target_reduction=0.5, verify_watertight=False),
    )

    assert result["output_cell_count"] < mesh.n_cells