import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import pyvista as pv

logger = logging.getLogger(__name__)

Mode = Literal["object", "room"]


class MeshValidationError(ValueError):
    """Raised when input data isn't a usable mesh for cleaning."""


class MeshProcessingError(RuntimeError):
    """Raised when a cleaning step fails unrecoverably (e.g. pyvista/VTK error)."""


@dataclass
class MeshCleaningConfig:
    mode: Mode = "object"
    smoothing_iters: int = 15
    pass_band: float = 0.1
    hole_size: Optional[float] = None          # None => auto-sized to close all detected holes
    min_cell_count: int = 500                  # 'room' mode: drop components at/below this size
    feature_angle: float = 45.0                # 'room' mode: sharp-edge preservation threshold
    merge_tolerance: float = 1e-5               # relative tolerance for duplicate-point merging
    decimate_target_reduction: Optional[float] = None  # e.g. 0.5 => drop ~50% of triangles
    verify_watertight: bool = True

    def __post_init__(self):
        if self.mode not in ("object", "room"):
            raise ValueError(f"Unknown mode '{self.mode}'; use 'object' or 'room'")
        if self.smoothing_iters < 0:
            raise ValueError("smoothing_iters must be >= 0")
        if self.min_cell_count < 0:
            raise ValueError("min_cell_count must be >= 0")
        if self.hole_size is not None and self.hole_size <= 0:
            raise ValueError("hole_size must be positive when specified")
        if self.decimate_target_reduction is not None and not (0.0 < self.decimate_target_reduction < 1.0):
            raise ValueError("decimate_target_reduction must be between 0 and 1")


def _validate_mesh(mesh: pv.DataSet) -> pv.DataSet:
    if not isinstance(mesh, pv.DataSet):
        raise MeshValidationError(f"Expected a pyvista DataSet, got {type(mesh).__name__}")
    if mesh.n_points == 0 or mesh.n_cells == 0:
        raise MeshValidationError("Mesh must contain at least one point and one cell")
    if not np.all(np.isfinite(mesh.points)):
        raise MeshValidationError("Mesh contains NaN/Inf point coordinates")
    return mesh


def _run_step(step_name: str, fn, *args, **kwargs):
    # Centralized error wrapping so any VTK/pyvista failure gets a clear, attributable message
    try:
        return fn(*args, **kwargs)
    except MeshValidationError:
        raise
    except Exception as exc:
        raise MeshProcessingError(f"Mesh cleaning step '{step_name}' failed: {exc}") from exc


def _bounding_diagonal(mesh: pv.DataSet) -> float:
    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
    return float(np.linalg.norm([xmax - xmin, ymax - ymin, zmax - zmin]))


def _auto_hole_size(mesh: pv.DataSet) -> float:
    # Default cap large enough to close virtually any hole relative to the model's own scale
    return _bounding_diagonal(mesh) * 0.5


def _count_open_edges(mesh: pv.PolyData) -> int:
    boundary = mesh.extract_feature_edges(
        boundary_edges=True,
        feature_edges=False,
        manifold_edges=False,
        non_manifold_edges=False,
    )
    return boundary.n_cells


def _filter_object(mesh: pv.DataSet) -> pv.PolyData:
    largest = mesh.connectivity(largest=True)
    surface = largest.extract_surface()
    if surface.n_cells == 0:
        raise MeshProcessingError("Largest-component extraction produced an empty surface")
    return surface


def _filter_room(mesh: pv.DataSet, min_cell_count: int) -> pv.PolyData:
    bodies = mesh.split_bodies()
    if len(bodies) == 0:
        raise MeshProcessingError("split_bodies() returned no components")

    valid_pieces: List[pv.DataSet] = [body for body in bodies if body.n_cells > min_cell_count]

    if not valid_pieces:
        largest_found = max((body.n_cells for body in bodies), default=0)
        raise MeshValidationError(
            f"No component exceeded min_cell_count={min_cell_count} "
            f"(largest component had {largest_found} cells); lower the threshold or check the input mesh"
        )

    merged = valid_pieces[0]
    for piece in valid_pieces[1:]:
        merged = merged.merge(piece)

    # merge() yields an UnstructuredGrid; downstream steps (hole fill, smoothing) need PolyData
    surface = merged.extract_surface()
    if surface.n_cells == 0:
        raise MeshProcessingError("Room-mode merge/extract_surface produced an empty surface")
    return surface


def _fill_holes(mesh: pv.PolyData, hole_size: Optional[float]) -> pv.PolyData:
    size = hole_size if hole_size is not None else _auto_hole_size(mesh)
    filled = mesh.fill_holes(hole_size=size)
    if filled.n_points == 0 or filled.n_cells == 0:
        raise MeshProcessingError("fill_holes produced an empty mesh")
    return filled


def _smooth(mesh: pv.PolyData, mode: Mode, iterations: int, pass_band: float, feature_angle: float) -> pv.PolyData:
    if iterations == 0:
        return mesh
    if mode == "object":
        return mesh.smooth_taubin(n_iter=iterations, pass_band=pass_band)
    return mesh.smooth_taubin(
        n_iter=iterations,
        pass_band=pass_band,
        feature_smoothing=True,
        feature_angle=feature_angle,
        boundary_smoothing=True,
    )


def _finalize(mesh: pv.PolyData, merge_tolerance: float, decimate_target_reduction: Optional[float]) -> pv.PolyData:
    # Merge coincident points and drop zero-area/duplicate cells left over from prior steps
    mesh = mesh.clean(tolerance=merge_tolerance)
    mesh = mesh.triangulate()

    if decimate_target_reduction:
        mesh = mesh.decimate_pro(decimate_target_reduction, preserve_topology=True)

    # Consistent, outward-facing normals matter for downstream rendering/shading
    mesh = mesh.compute_normals(auto_orient_normals=True, consistent_normals=True, splitting=False)
    return mesh


def clean_ai_mesh(mesh: pv.DataSet, config: Optional[MeshCleaningConfig] = None, **overrides: Any) -> Dict[str, Any]:
    """Pipeline entry point: cleans an AI-generated mesh (component filtering,
    hole filling, smoothing, decimation) and returns the polished result plus
    diagnostics, ready for the next pipeline stage. `mesh` is a pyvista
    DataSet already in memory; no file I/O happens here."""
    cfg = config or MeshCleaningConfig(**overrides)
    if config is not None and overrides:
        raise ValueError("Pass either config or keyword overrides, not both")

    mesh = _validate_mesh(mesh)
    warnings: List[str] = []

    logger.info("Cleaning mesh: mode=%s, in_points=%d, in_cells=%d", cfg.mode, mesh.n_points, mesh.n_cells)

    if cfg.mode == "object":
        filtered = _run_step("component_filter", _filter_object, mesh)
    else:
        filtered = _run_step("component_filter", _filter_room, mesh, cfg.min_cell_count)

    filled = _run_step("fill_holes", _fill_holes, filtered, cfg.hole_size)

    smoothed = _run_step(
        "smooth", _smooth, filled, cfg.mode, cfg.smoothing_iters, cfg.pass_band, cfg.feature_angle
    )

    final_mesh = _run_step(
        "finalize", _finalize, smoothed, cfg.merge_tolerance, cfg.decimate_target_reduction
    )

    open_edges = None
    is_watertight = None
    if cfg.verify_watertight:
        open_edges = _run_step("watertight_check", _count_open_edges, final_mesh)
        is_watertight = open_edges == 0
        if not is_watertight:
            warnings.append(f"Output mesh is not fully watertight ({open_edges} open boundary edges remain)")

    if final_mesh.n_points == 0 or final_mesh.n_cells == 0:
        raise MeshProcessingError("Final mesh is empty after cleaning")

    logger.info("Cleaning complete: out_points=%d, out_cells=%d, watertight=%s",
                final_mesh.n_points, final_mesh.n_cells, is_watertight)

    return {
        "mesh": final_mesh,
        "mode": cfg.mode,
        "input_point_count": mesh.n_points,
        "input_cell_count": mesh.n_cells,
        "output_point_count": final_mesh.n_points,
        "output_cell_count": final_mesh.n_cells,
        "is_watertight": is_watertight,
        "open_edge_count": open_edges,
        "warnings": warnings,
    }