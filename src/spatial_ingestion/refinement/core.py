import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence

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
    hole_size: Optional[float] = None
    min_cell_count: int = 500
    feature_angle: float = 45.0
    merge_tolerance: float = 1e-5
    decimate_target_reduction: Optional[float] = None
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


def validate_mesh_input(mesh: pv.DataSet) -> pv.DataSet:
    if not isinstance(mesh, pv.DataSet):
        raise MeshValidationError(f"Expected a pyvista DataSet, got {type(mesh).__name__}")
    if mesh.n_points == 0 or mesh.n_cells == 0:
        raise MeshValidationError("Mesh must contain at least one point and one cell")
    if not np.all(np.isfinite(mesh.points)):
        raise MeshValidationError("Mesh contains NaN/Inf point coordinates")
    return mesh


def run_step(step_name: str, fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except MeshValidationError:
        raise
    except Exception as exc:
        raise MeshProcessingError(f"Mesh cleaning step '{step_name}' failed: {exc}") from exc


def get_bounds_spans(mesh: pv.DataSet) -> np.ndarray:
    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
    return np.asarray([xmax - xmin, ymax - ymin, zmax - zmin], dtype=float)


def is_sheet_like(mesh: pv.DataSet) -> bool:
    spans = get_bounds_spans(mesh)
    spans = spans[spans > 1e-12]
    if spans.size < 3:
        return True
    return float(np.min(spans) / np.max(spans)) < 0.15


def get_default_hole_size(mesh: pv.DataSet) -> float:
    spans = get_bounds_spans(mesh)
    diagonal = float(np.linalg.norm(spans))
    return diagonal * 0.5


def get_component_pieces(mesh: pv.DataSet) -> List[pv.PolyData]:
    bodies = mesh.split_bodies()
    if len(bodies) == 0:
        raise MeshProcessingError("split_bodies() returned no components")
    pieces: List[pv.PolyData] = []
    for body in bodies:
        surface = body.extract_surface()
        if surface.n_cells > 0:
            pieces.append(surface)
    return pieces


def merge_components(pieces: Sequence[pv.PolyData]) -> pv.PolyData:
    merged = pieces[0].copy(deep=True)
    for piece in pieces[1:]:
        merged = merged.merge(piece, merge_points=False)
    surface = merged.extract_surface()
    if surface.n_cells == 0:
        raise MeshProcessingError("Component fusion produced an empty surface")
    return surface


def keep_object_components(mesh: pv.DataSet) -> pv.PolyData:
    pieces = get_component_pieces(mesh)
    return merge_components(pieces)


def filter_room_components(mesh: pv.DataSet, min_cell_count: int) -> pv.PolyData:
    pieces = get_component_pieces(mesh)
    valid_pieces = [body for body in pieces if body.n_cells > min_cell_count]

    if not valid_pieces:
        largest_found = max((body.n_cells for body in pieces), default=0)
        raise MeshValidationError(
            f"No component exceeded min_cell_count={min_cell_count} "
            f"(largest component had {largest_found} cells); lower the threshold or check the input mesh"
        )

    return merge_components(valid_pieces)


def fill_mesh_holes(mesh: pv.PolyData, hole_size: Optional[float]) -> pv.PolyData:
    if hole_size is None and is_sheet_like(mesh):
        return mesh

    size = hole_size if hole_size is not None else get_default_hole_size(mesh)
    filled = mesh.fill_holes(hole_size=size)
    if filled.n_points == 0 or filled.n_cells == 0:
        raise MeshProcessingError("fill_holes produced an empty mesh")
    return filled


def smooth_mesh(mesh: pv.PolyData, mode: Mode, iterations: int, pass_band: float, feature_angle: float) -> pv.PolyData:
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


def preserve_data_arrays(source: pv.DataSet, target: pv.DataSet) -> pv.DataSet:
    if target.n_points == 0:
        return target
    if not source.point_data and not source.cell_data:
        return target

    try:
        transferred = target.sample(source)
    except Exception:
        transferred = target.copy(deep=True)
        for name, array in source.point_data.items():
            if array.ndim != 2 or array.shape[1] not in (3, 4):
                continue
            nearest_values = np.empty((target.n_points, array.shape[1]), dtype=array.dtype)
            for index, point in enumerate(np.asarray(target.points)):
                closest_id = source.find_closest_point(point)
                nearest_values[index] = np.asarray(array[closest_id])
            transferred.point_data[name] = nearest_values
    return transferred


def finalize_mesh(mesh: pv.PolyData, merge_tolerance: float, decimate_target_reduction: Optional[float]) -> pv.PolyData:
    mesh = mesh.clean(tolerance=merge_tolerance)
    mesh = mesh.triangulate()

    if decimate_target_reduction:
        mesh = mesh.decimate_pro(decimate_target_reduction, preserve_topology=True)

    mesh = mesh.compute_normals(auto_orient_normals=True, consistent_normals=True)
    return mesh


def count_topology_issues(mesh: pv.PolyData) -> Dict[str, int]:
    boundary = mesh.extract_feature_edges(
        boundary_edges=True,
        feature_edges=False,
        manifold_edges=False,
        non_manifold_edges=False,
    )
    non_manifold = mesh.extract_feature_edges(
        boundary_edges=False,
        feature_edges=False,
        manifold_edges=False,
        non_manifold_edges=True,
    )
    return {
        "boundary_edge_count": int(boundary.n_cells),
        "non_manifold_edge_count": int(non_manifold.n_cells),
    }


def clean_mesh(mesh, config: Optional[MeshCleaningConfig] = None, **overrides: Any):
    """Clean an AI-generated mesh and return diagnostics plus the processed result."""
    if config is not None and overrides:
        raise ValueError("Pass either config or keyword overrides, not both")

    cfg = config or MeshCleaningConfig(**overrides)
    mesh = validate_mesh_input(mesh)
    warnings: List[str] = []

    logger.info("Cleaning mesh: mode=%s, in_points=%d, in_cells=%d", cfg.mode, mesh.n_points, mesh.n_cells)

    if cfg.mode == "object":
        filtered = run_step("component_filter", keep_object_components, mesh)
    else:
        filtered = run_step("component_filter", filter_room_components, mesh, cfg.min_cell_count)

    data_source = filtered if (filtered.point_data or filtered.cell_data) else mesh

    filled = run_step("fill_holes", fill_mesh_holes, filtered, cfg.hole_size)

    smoothed = run_step(
        "smooth",
        smooth_mesh,
        filled,
        cfg.mode,
        cfg.smoothing_iters,
        cfg.pass_band,
        cfg.feature_angle,
    )

    final_mesh = run_step("finalize", finalize_mesh, smoothed, cfg.merge_tolerance, cfg.decimate_target_reduction)
    if data_source.point_data or data_source.cell_data:
        final_mesh = run_step("transfer_data", preserve_data_arrays, data_source, final_mesh)

    topology = {"boundary_edge_count": None, "non_manifold_edge_count": None}
    is_watertight = None
    if cfg.verify_watertight:
        topology = run_step("watertight_check", count_topology_issues, final_mesh)
        is_watertight = topology["boundary_edge_count"] == 0 and topology["non_manifold_edge_count"] == 0
        issue_count = topology["boundary_edge_count"] + topology["non_manifold_edge_count"]
        if not is_watertight:
            warnings.append(
                "Output mesh has topology issues ("
                f"{topology['boundary_edge_count']} boundary edges, "
                f"{topology['non_manifold_edge_count']} non-manifold edges; {issue_count} total)"
            )

    if final_mesh.n_points == 0 or final_mesh.n_cells == 0:
        raise MeshProcessingError("Final mesh is empty after cleaning")

    logger.info(
        "Cleaning complete: out_points=%d, out_cells=%d, watertight=%s",
        final_mesh.n_points,
        final_mesh.n_cells,
        is_watertight,
    )

    open_edge_count = topology["boundary_edge_count"] if topology["boundary_edge_count"] is not None else None
    if cfg.verify_watertight:
        open_edge_count = topology["boundary_edge_count"] + topology["non_manifold_edge_count"]

    return {
        "mesh": final_mesh,
        "mode": cfg.mode,
        "input_point_count": mesh.n_points,
        "input_cell_count": mesh.n_cells,
        "output_point_count": final_mesh.n_points,
        "output_cell_count": final_mesh.n_cells,
        "is_watertight": is_watertight,
        "open_edge_count": open_edge_count,
        "boundary_edge_count": topology["boundary_edge_count"],
        "non_manifold_edge_count": topology["non_manifold_edge_count"],
        "warnings": warnings,
    }


clean_ai_mesh = clean_mesh