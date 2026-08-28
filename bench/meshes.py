"""Mesh sources and the size ladder shared by A1-A4.

Two sources, deliberately: an icosphere (closed, watertight, isotropic) and a
displaced grid standing in for a fused MASt3R pointmap (open, sheet-like).
The second one matters because `refinement.core.is_sheet_like` gates hole
filling on exactly that geometry class, so a ladder built only from spheres
would never exercise the branch real reconstruction output takes.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import pyvista as pv
import trimesh

__all__ = [
    "SOURCES",
    "analytic_colors",
    "base_mesh",
    "colored_pyvista",
    "fragmented_mesh",
    "ladder_mesh",
    "mesh_at_triangle_count",
    "to_pyvista",
]

SOURCES: tuple[str, ...] = ("icosphere", "pointmap_sheet")


def _pointmap_sheet(resolution: int = 128, seed: int = 0) -> trimesh.Trimesh:
    """A displaced height-field grid: the topology MASt3R pointmap fusion emits.

    Thin in one axis (so `is_sheet_like` fires), open at the boundary, and
    smoothly displaced by summed sinusoids so it has curvature to smooth
    rather than being a flat plane.
    """
    rng = np.random.default_rng(seed)
    axis = np.linspace(-0.5, 0.5, resolution)
    grid_x, grid_y = np.meshgrid(axis, axis, indexing="ij")

    height = np.zeros_like(grid_x)
    for octave in range(1, 5):
        frequency = 2.0 * octave
        phase_x, phase_y = rng.uniform(0, 2 * np.pi, 2)
        height += (
            np.sin(frequency * np.pi * grid_x + phase_x)
            * np.cos(frequency * np.pi * grid_y + phase_y)
            / (2.0**octave)
        )
    height *= 0.06

    vertices = np.column_stack([grid_x.ravel(), grid_y.ravel(), height.ravel()])
    faces = []
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            a = i * resolution + j
            b = a + 1
            c = a + resolution
            d = c + 1
            faces.append((a, c, b))
            faces.append((b, c, d))
    return trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces), process=False)


def base_mesh(source: str, seed: int = 0) -> trimesh.Trimesh:
    """A named base shape, before any resizing."""
    if source == "icosphere":
        return trimesh.creation.icosphere(subdivisions=4, radius=1.0)
    if source == "pointmap_sheet":
        return _pointmap_sheet(seed=seed)
    if source == "torus":
        return trimesh.creation.torus(major_radius=1.0, minor_radius=0.35)
    if source == "capsule":
        return trimesh.creation.capsule(height=1.5, radius=0.4, count=[48, 48])
    if source == "box":
        return trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    raise ValueError(f"unknown mesh source '{source}'")


def mesh_at_triangle_count(
    mesh: trimesh.Trimesh, n_tri: int, preserve_topology: bool = False
) -> trimesh.Trimesh:
    """Resize a mesh to approximately `n_tri` triangles.

    Subdivision (x4 per pass) overshoots, then `decimate_pro` lands on the
    target, so every rung of the ladder is the same shape at a different
    resolution rather than a different shape.

    `preserve_topology=False` hits the triangle target more exactly, which is
    what the size ladder wants. It can also tear a closed shell into several,
    so callers that care about component count -- `fragmented_mesh` -- must
    pass True and accept a looser triangle count.
    """
    if n_tri < 4:
        raise ValueError("n_tri must be at least 4")
    working = mesh.copy()
    while len(working.faces) < n_tri:
        working = working.subdivide()

    current = len(working.faces)
    if current <= n_tri:
        return working

    reduction = 1.0 - (n_tri / current)
    poly = to_pyvista(working).triangulate()
    decimated = poly.decimate_pro(
        min(reduction, 0.999), preserve_topology=preserve_topology
    )
    faces = np.asarray(decimated.faces).reshape(-1, 4)[:, 1:]
    return trimesh.Trimesh(
        vertices=np.asarray(decimated.points, dtype=float), faces=faces, process=False
    )


@lru_cache(maxsize=64)
def _ladder_cached(source: str, n_tri: int, seed: int) -> trimesh.Trimesh:
    return mesh_at_triangle_count(base_mesh(source, seed=seed), n_tri)


def ladder_mesh(source: str, n_tri: int, seed: int = 0) -> trimesh.Trimesh:
    """One rung of the size ladder, cached so repeats time cleaning, not building."""
    return _ladder_cached(source, n_tri, seed).copy()


def _uv_sphere_near(target_faces: int) -> trimesh.Trimesh:
    """A closed UV sphere with about `target_faces` triangles.

    Decimation is deliberately not used here. `decimate_pro` tears a closed
    icosphere into several shells at some reductions even with
    `preserve_topology=True`, which would silently multiply the component count
    this function exists to control. A UV sphere's resolution is tunable in
    fine steps and is always a single watertight shell.
    """
    best: trimesh.Trimesh | None = None
    best_error = float("inf")
    resolution = max(4, int(round(math.sqrt(max(target_faces, 1) / 4.0))))
    for candidate in range(max(4, resolution - 2), resolution + 4):
        sphere = trimesh.creation.uv_sphere(radius=1.0, count=[candidate, candidate])
        error = abs(len(sphere.faces) - target_faces)
        if error < best_error:
            best, best_error = sphere, error
    assert best is not None
    return best


def fragmented_mesh(n_tri: int, n_components: int, seed: int = 0) -> trimesh.Trimesh:
    """`n_components` disjoint pieces totalling about `n_tri` triangles.

    A fused MASt3R pointmap is not one connected surface: each view contributes
    its own sheet and confidence masking punches them into many disconnected
    islands. Component count is therefore an independent axis from triangle
    count, and `keep_object_components` walks every piece.
    """
    if n_components < 1:
        raise ValueError("n_components must be >= 1")
    per_piece = max(4, n_tri // n_components)
    rng = np.random.default_rng(seed)
    piece = _uv_sphere_near(per_piece)

    spread = float(np.cbrt(n_components)) * 4.0
    parts = []
    for _ in range(n_components):
        clone = piece.copy()
        clone.apply_translation(rng.uniform(-spread, spread, 3))
        parts.append(clone)
    return trimesh.util.concatenate(parts)


def to_pyvista(mesh: trimesh.Trimesh, colors: np.ndarray | None = None) -> pv.PolyData:
    """Convert a trimesh to PolyData, optionally attaching an RGB point array."""
    faces = np.asarray(mesh.faces, dtype=np.int64)
    padded = np.column_stack([np.full(len(faces), 3, dtype=np.int64), faces]).ravel()
    poly = pv.PolyData(np.asarray(mesh.vertices, dtype=float), padded)
    if colors is not None:
        poly.point_data["RGB"] = np.asarray(colors, dtype=np.uint8)
    return poly


def analytic_colors(vertices: np.ndarray) -> np.ndarray:
    """RGB as a closed-form function of position, for A4's colour-transfer error.

    Mapping the unit-normalised bounding box onto 0-255 per channel means the
    ground-truth colour of any output vertex is computable from its position
    alone, so transfer error is measurable without vertex correspondence.
    """
    points = np.asarray(vertices, dtype=float)
    low = points.min(axis=0)
    span = np.maximum(points.max(axis=0) - low, 1e-12)
    return np.clip(((points - low) / span) * 255.0, 0, 255).astype(np.uint8)


def colored_pyvista(mesh: trimesh.Trimesh) -> pv.PolyData:
    """A PolyData carrying the analytic colour field of `analytic_colors`."""
    return to_pyvista(mesh, colors=analytic_colors(mesh.vertices))
