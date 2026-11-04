import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyvista as pv
import trimesh

from spatial_ingestion.instrumentation import StageLog
from spatial_ingestion.refinement.gpu import (
    Device,
    cell_arrays,
    edge_use_counts,
    region_ids,
    resolve_device,
)

logger = logging.getLogger(__name__)

Mode = Literal["object", "room"]
ComponentFilter = Literal["labels", "split_bodies"]

_COLOR_ARRAY_NAMES = ("RGBA", "RGB", "rgba", "rgb", "COLOR_0", "color_0")


def _has_recognized_color_data(data: pv.DataSet) -> bool:
    return any(name in data.point_data for name in _COLOR_ARRAY_NAMES)


class MeshValidationError(ValueError):
    """Raised when input data isn't a usable mesh for cleaning."""


class MeshProcessingError(RuntimeError):
    """Raised when a cleaning step fails unrecoverably (e.g. pyvista/VTK error)."""


@dataclass
class MeshCleaningConfig:
    mode: Mode = "object"
    smoothing_iters: int = 15
    pass_band: float = 0.1
    hole_size: float | None = None
    min_cell_count: int = 500
    feature_angle: float = 45.0
    merge_tolerance: float = 1e-5
    decimate_target_reduction: float | None = None
    verify_watertight: bool = True
    # Where the mesh-graph steps (component labelling, edge counting) run;
    # "auto" is CUDA when torch sees a GPU. See refinement/gpu.py.
    device: Device = "auto"
    # "split_bodies" is the original per-component VTK path, kept so the
    # bottleneck it caused (bench A1/A2) can still be measured.
    component_filter: ComponentFilter = "labels"
    # Object mode only. Off, the object is the largest connected component.
    # On, a dominant plane (floor, table) is cut away first and the fragments
    # lying inside the largest remaining piece's box are kept with it. A real
    # capture's support surface is usually reconstructed as one sheet the
    # subject sits on, and the more views the more of it there is: on the
    # CO3D teddybear at 24 views the floor outgrew the bear and "largest
    # component" kept the floor alone. Off by default so the synthetic
    # single-body benchmarks (A1-A3) measure what they always measured.
    isolate_object: bool = False

    def __post_init__(self):
        if self.mode not in ("object", "room"):
            raise ValueError(f"Unknown mode '{self.mode}'; use 'object' or 'room'")
        if self.device not in ("auto", "cuda", "cpu"):
            raise ValueError(f"Unknown device '{self.device}'; use 'auto', 'cuda' or 'cpu'")
        if self.component_filter not in ("labels", "split_bodies"):
            raise ValueError(
                f"Unknown component_filter '{self.component_filter}'; "
                "use 'labels' or 'split_bodies'"
            )
        if self.smoothing_iters < 0:
            raise ValueError("smoothing_iters must be >= 0")
        if self.min_cell_count < 0:
            raise ValueError("min_cell_count must be >= 0")
        if self.hole_size is not None and self.hole_size <= 0:
            raise ValueError("hole_size must be positive when specified")
        if self.decimate_target_reduction is not None and not (
            0.0 < self.decimate_target_reduction < 1.0
        ):
            raise ValueError("decimate_target_reduction must be between 0 and 1")


def _extract_mesh_from_multiblock(mesh: Any) -> pv.DataSet:
    if isinstance(mesh, pv.MultiBlock):
        blocks = _collect_usable_blocks(mesh)
        if not blocks:
            raise MeshValidationError("MultiBlock input did not contain any usable mesh blocks")
        if len(blocks) > 1:
            logger.warning(
                "MultiBlock input contains %d usable mesh blocks; using the first "
                "(%d points, %d cells) and dropping the rest",
                len(blocks),
                blocks[0].n_points,
                blocks[0].n_cells,
            )
        return blocks[0]
    return mesh


def _collect_usable_blocks(mesh: Any) -> list[pv.DataSet]:
    if isinstance(mesh, pv.MultiBlock):
        blocks: list[pv.DataSet] = []
        for block in mesh:
            blocks.extend(_collect_usable_blocks(block))
        return blocks
    if isinstance(mesh, pv.DataSet) and mesh.n_points > 0 and mesh.n_cells > 0:
        return [mesh]
    return []


def validate_mesh_input(mesh: Any) -> pv.DataSet:
    mesh = _extract_mesh_from_multiblock(mesh)
    if not isinstance(mesh, pv.DataSet):
        raise MeshValidationError(f"Expected a pyvista DataSet, got {type(mesh).__name__}")
    if mesh.n_points == 0 or mesh.n_cells == 0:
        raise MeshValidationError("Mesh must contain at least one point and one cell")
    if not np.all(np.isfinite(mesh.points)):
        raise MeshValidationError("Mesh contains NaN/Inf point coordinates")
    return mesh


def load_mesh_file(path: Path | str) -> pv.DataSet:
    """Read a mesh file, unwrapping MultiBlock containers (e.g. GLB scenes)."""
    path = Path(path)
    mesh = _extract_mesh_from_multiblock(pv.read(str(path)))
    if not isinstance(mesh, pv.DataSet):
        raise MeshValidationError(f"Unsupported mesh content in {path}: {type(mesh).__name__}")
    return mesh


def to_trimesh(mesh: pv.DataSet) -> trimesh.Trimesh:
    """Adapt a PyVista DataSet to a trimesh.Trimesh, preserving vertex colors."""
    tri = mesh.extract_surface(algorithm=None).triangulate()
    faces = tri.faces.reshape(-1, 4)[:, 1:]
    colors = None
    for name in _COLOR_ARRAY_NAMES:
        colors = tri.point_data.get(name)
        if colors is not None:
            break
    if colors is None and tri.active_scalars_name not in ("Normals", "TCoords", None):
        active = tri.active_scalars
        if active.ndim == 2 and active.shape[1] in (3, 4):
            colors = active
    vertex_colors = np.asarray(colors) if colors is not None else None
    return trimesh.Trimesh(vertices=tri.points, faces=faces, vertex_colors=vertex_colors)


def write_mesh_file(mesh: pv.DataSet, path: Path | str) -> None:
    """Save a mesh with vertex colors intact.

    .glb/.gltf/.obj/.ply/.stl go through trimesh (pyvista cannot write GLB and
    its save() drops vertex colors for .obj/.ply); remaining formats via pyvista.
    """
    path = Path(path)
    if path.suffix.lower() in {".glb", ".gltf", ".obj", ".ply", ".stl"}:
        to_trimesh(mesh).export(str(path))
    else:
        mesh.save(str(path))


def run_step(step_name: str, fn, *args, stage_log: StageLog | None = None, **kwargs):
    """Run one cleaning step, normalising failures and optionally timing it.

    `stage_log` is threaded through rather than wrapped around the call site so
    the timing boundary is exactly the step boundary, and so a step that raises
    still records the seconds it burned before failing.
    """
    if stage_log is None:
        return _invoke_step(step_name, fn, *args, **kwargs)
    with stage_log.stage(step_name):
        return _invoke_step(step_name, fn, *args, **kwargs)


def _invoke_step(step_name: str, fn, *args, **kwargs):
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


def get_component_pieces(mesh: pv.DataSet) -> list[pv.PolyData]:
    bodies = mesh.split_bodies()
    if len(bodies) == 0:
        raise MeshProcessingError("split_bodies() returned no components")
    pieces: list[pv.PolyData] = []
    for body in bodies:
        surface = body.extract_surface(algorithm=None)
        if surface.n_cells > 0:
            pieces.append(surface)
    return pieces


def merge_components(pieces: Sequence[pv.PolyData]) -> pv.PolyData:
    # A per-piece `merged.merge(piece)` loop re-copies the whole accumulated
    # mesh on every iteration -- O(k^2) in piece count. Passing the rest of
    # the list to a single `merge()` call appends them all in one pass
    # instead (bench/FINDINGS.md #1: ~64x faster at 1000 components, same
    # output, since merge_points=False makes this a pure concatenation).
    merged = pieces[0].merge(list(pieces[1:]), merge_points=False)
    surface = merged.extract_surface(algorithm=None)
    if surface.n_cells == 0:
        raise MeshProcessingError("Component fusion produced an empty surface")
    return surface


def label_components(
    mesh: pv.DataSet, device: Device = "auto", method: ComponentFilter = "labels"
) -> np.ndarray | None:
    """Per-cell component ids in `split_bodies()` order, or None to use split_bodies."""
    if method == "split_bodies":
        return None
    arrays = cell_arrays(mesh)
    if arrays is None:
        return None
    return region_ids(mesh.n_points, *arrays, device=resolve_device(device))


def extract_regions(mesh: pv.DataSet, regions: np.ndarray, keep: np.ndarray) -> pv.PolyData:
    # One extraction for every kept component, in input cell order. Pieces
    # share no points, so this equals merging them with merge_points=False.
    surface = mesh.extract_cells(np.flatnonzero(np.isin(regions, keep))).extract_surface(
        algorithm=None
    )
    if surface.n_cells == 0:
        raise MeshProcessingError("Component extraction produced an empty surface")
    return surface


# Gathering (isolate_object): a fragment joins the object when this share of
# its cells lies inside the largest piece's bounding box, padded by
# _GATHER_BOX_PAD of its size. Fragments below _GATHER_MIN_CELLS are debris.
_GATHER_BOX_PAD = 0.15
_GATHER_INSIDE_FRAC = 0.9
_GATHER_MIN_CELLS = 200


def gather_object_regions(mesh: pv.DataSet, regions: np.ndarray) -> np.ndarray:
    """The largest region plus every sizeable fragment inside its padded box.

    A TSDF mesh of a real capture breaks its subject into many pieces (the
    CO3D teddybear's largest held 8-18% of its cells once the floor was
    cut), so the largest alone is a partial subject.
    """
    counts = np.bincount(regions)
    largest = int(np.argmax(counts))
    centres = np.asarray(mesh.cell_centers().points)
    main = centres[regions == largest]
    lo, hi = main.min(axis=0), main.max(axis=0)
    pad = _GATHER_BOX_PAD * (hi - lo)
    inside = np.all((centres >= lo - pad) & (centres <= hi + pad), axis=1)
    inside_frac = np.bincount(regions, weights=inside, minlength=len(counts)) / counts
    keep = (inside_frac >= _GATHER_INSIDE_FRAC) & (counts >= _GATHER_MIN_CELLS)
    keep[largest] = True
    return np.flatnonzero(keep)


def keep_object_components(
    mesh: pv.DataSet,
    *,
    device: Device = "auto",
    method: ComponentFilter = "labels",
    gather: bool = False,
) -> pv.PolyData:
    # Keeps the single largest piece by cell count and discards the rest --
    # object-mode captures are one subject, and stray fragments below the
    # main body are debris, not additional geometry to preserve (bench/
    # FINDINGS.md #2; the architecture text already described this behaviour
    # before the code implemented it). `gather` also keeps the fragments
    # inside the largest piece's box (gather_object_regions).
    regions = label_components(mesh, device, method)
    if regions is None:
        if gather:
            logger.warning(
                "isolate_object gathers fragments only on the labels path; "
                "keeping the largest component alone"
            )
        pieces = get_component_pieces(mesh)
        return max(pieces, key=lambda piece: piece.n_cells)
    if gather:
        return extract_regions(mesh, regions, gather_object_regions(mesh, regions))
    # argmax takes the first of equal counts, as max() over split_bodies does.
    return extract_regions(mesh, regions, np.array([np.argmax(np.bincount(regions))]))


# A plane is a support surface when its faces are at least this share of the
# mesh; a subject's own flat side is smaller than the floor it was shot on.
_SUPPORT_PLANE_MIN_FRAC = 0.3
# If cutting the plane would leave less than this share, the mesh *was* the
# plane (a sheet, a wall scan) and is left alone.
_SUPPORT_PLANE_MIN_REMAINDER = 0.01
_RANSAC_ITERATIONS = 300
_RANSAC_SAMPLE = 200_000


def _fit_plane(points: np.ndarray) -> tuple[np.ndarray, float]:
    centroid = points.mean(axis=0)
    normal = np.linalg.svd(points - centroid, full_matrices=False)[2][-1]
    return normal, float(-normal @ centroid)


def find_support_plane(
    points: np.ndarray, *, seed: int = 0
) -> tuple[np.ndarray, float, float] | None:
    """The dominant plane through `points` as (unit normal, offset, inlier tolerance).

    RANSAC at 1% of the bounding-box diagonal finds the plane; the tolerance
    is then narrowed to the plane's own thickness (3 robust SDs of its
    inliers' distances), so the subject's base is not cut along with a noisy
    floor's full search band.
    """
    rng = np.random.default_rng(seed)
    if len(points) < 3:
        return None
    sample = points[rng.choice(len(points), min(len(points), _RANSAC_SAMPLE), replace=False)]
    search = 0.01 * float(np.linalg.norm(np.ptp(points, axis=0)))
    if search <= 0:
        return None

    best_count, best = 0, None
    for _ in range(_RANSAC_ITERATIONS):
        a, b, c = sample[rng.choice(len(sample), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        length = np.linalg.norm(normal)
        if length < 1e-12:
            continue
        normal /= length
        count = int(np.count_nonzero(np.abs((sample - a) @ normal) < search))
        if count > best_count:
            best_count, best = count, (normal, float(-normal @ a))
    if best is None:
        return None

    normal, offset = best
    inliers = sample[np.abs(sample @ normal + offset) < search]
    normal, offset = _fit_plane(inliers)
    distances = np.abs(inliers @ normal + offset)
    spread = 1.4826 * float(np.median(np.abs(distances - np.median(distances))))
    tolerance = min(search, max(3.0 * spread, 1e-3 * search))
    return normal, offset, tolerance


def cut_support_plane(mesh: pv.DataSet, *, seed: int = 0) -> tuple[pv.DataSet, dict[str, Any]]:
    """Drop the faces lying in a dominant plane, if one covers enough of the mesh."""
    info: dict[str, Any] = {"support_plane_removed": False}
    points = np.asarray(mesh.points, dtype=float)
    plane = find_support_plane(points, seed=seed)
    if plane is None:
        return mesh, info
    normal, offset, tolerance = plane

    near = np.abs(points @ normal + offset) < tolerance
    arrays = cell_arrays(mesh)
    if arrays is not None:
        offsets, connectivity = arrays
        on_plane = np.logical_and.reduceat(near[connectivity], offsets[:-1])
    else:
        on_plane = np.array(
            [bool(np.all(near[mesh.get_cell(i).point_ids])) for i in range(mesh.n_cells)]
        )
    plane_frac = float(on_plane.mean()) if len(on_plane) else 0.0
    info.update(
        support_plane_normal=[round(float(v), 6) for v in normal],
        support_plane_tolerance=round(tolerance, 6),
        support_plane_face_frac=round(plane_frac, 4),
    )
    if plane_frac < _SUPPORT_PLANE_MIN_FRAC or 1.0 - plane_frac < _SUPPORT_PLANE_MIN_REMAINDER:
        return mesh, info

    kept = mesh.extract_cells(np.flatnonzero(~on_plane)).extract_surface(algorithm=None)
    info["support_plane_removed"] = True
    logger.info(
        "Removed a support plane: %.1f%% of faces within %.4g of it", 100 * plane_frac, tolerance
    )
    return kept, info


def filter_room_components(
    mesh: pv.DataSet,
    min_cell_count: int,
    *,
    device: Device = "auto",
    method: ComponentFilter = "labels",
) -> pv.PolyData:
    regions = label_components(mesh, device, method)
    if regions is None:
        pieces = get_component_pieces(mesh)
        counts = np.array([body.n_cells for body in pieces])
    else:
        counts = np.bincount(regions)
    keep = np.flatnonzero(counts > min_cell_count)

    if keep.size == 0:
        largest_found = int(counts.max(initial=0))
        raise MeshValidationError(
            f"No component exceeded min_cell_count={min_cell_count} "
            f"(largest component had {largest_found} cells); "
            "lower the threshold or check the input mesh"
        )

    if regions is None:
        return merge_components([pieces[i] for i in keep])
    return extract_regions(mesh, regions, keep)


def fill_mesh_holes(
    mesh: pv.PolyData, hole_size: float | None, *, guard_mesh: pv.DataSet | None = None
) -> pv.PolyData:
    # The sheet-like guard is meant to ask "is this capture inherently thin
    # and open", not "did component filtering happen to leave a thin
    # result" -- so it is evaluated against the pre-filter input
    # (bench/FINDINGS.md #3). A raw MASt3R pointmap is typically fragmented
    # into many islands by confidence masking, so this reliably reads as
    # non-sheet-like on real reconstruction output even though the
    # single-piece synthetic case still trips it.
    check_mesh = guard_mesh if guard_mesh is not None else mesh
    if hole_size is None and is_sheet_like(check_mesh):
        return mesh

    size = hole_size if hole_size is not None else get_default_hole_size(mesh)
    filled = mesh.fill_holes(hole_size=size)
    if filled.n_points == 0 or filled.n_cells == 0:
        raise MeshProcessingError("fill_holes produced an empty mesh")
    return filled


def smooth_mesh(
    mesh: pv.PolyData, mode: Mode, iterations: int, pass_band: float, feature_angle: float
) -> pv.PolyData:
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
    """Transfer vertex color arrays (recognized color names) from source to target.

    Nearest-point transfer is used so colors survive geometry-changing steps
    (smoothing, decimation) without probe-interpolation zeros. Other point data
    (e.g. normals, confidence) and cell data are intentionally not transferred.
    """
    if target.n_points == 0:
        return target
    color_arrays = {
        name: np.asarray(array)
        for name, array in source.point_data.items()
        if name in _COLOR_ARRAY_NAMES
    }
    if not color_arrays:
        return target

    from scipy.spatial import KDTree

    tree = KDTree(np.asarray(source.points))
    _, nearest = tree.query(np.asarray(target.points), k=1, workers=-1)
    transferred = target.copy(deep=True)
    for name, array in color_arrays.items():
        transferred.point_data[name] = array[nearest]
    return transferred


def finalize_mesh(
    mesh: pv.PolyData, merge_tolerance: float, decimate_target_reduction: float | None
) -> pv.PolyData:
    mesh = mesh.clean(tolerance=merge_tolerance)
    mesh = mesh.triangulate()

    if decimate_target_reduction:
        mesh = mesh.decimate_pro(decimate_target_reduction, preserve_topology=True)

    mesh = mesh.compute_normals(auto_orient_normals=True, consistent_normals=True)
    return mesh


def count_topology_issues(mesh: pv.PolyData, *, device: Device = "auto") -> dict[str, int]:
    arrays = cell_arrays(mesh)
    if arrays is not None and np.all(np.diff(arrays[0]) == 3):
        return edge_use_counts(mesh.n_points, *arrays, device=resolve_device(device))

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


def clean_mesh(mesh, config: MeshCleaningConfig | None = None, **overrides: Any):
    """Clean an AI-generated mesh and return diagnostics plus the processed result."""
    if config is not None and overrides:
        raise ValueError("Pass either config or keyword overrides, not both")

    cfg = config or MeshCleaningConfig(**overrides)
    stage_log = StageLog()
    mesh = validate_mesh_input(mesh)
    warnings: list[str] = []
    device = resolve_device(cfg.device)

    logger.info(
        "Cleaning mesh: mode=%s, in_points=%d, in_cells=%d, device=%s",
        cfg.mode,
        mesh.n_points,
        mesh.n_cells,
        device,
    )

    plane_info: dict[str, Any] = {}
    if cfg.mode == "object":
        candidates = mesh
        if cfg.isolate_object:
            candidates, plane_info = run_step(
                "support_plane", cut_support_plane, mesh, stage_log=stage_log
            )
        filtered = run_step(
            "component_filter",
            keep_object_components,
            candidates,
            device=device,
            method=cfg.component_filter,
            gather=cfg.isolate_object,
            stage_log=stage_log,
        )
    else:
        filtered = run_step(
            "component_filter",
            filter_room_components,
            mesh,
            cfg.min_cell_count,
            device=device,
            method=cfg.component_filter,
            stage_log=stage_log,
        )

    data_source = filtered if _has_recognized_color_data(filtered) else mesh

    filled = run_step(
        "fill_holes", fill_mesh_holes, filtered, cfg.hole_size, guard_mesh=mesh, stage_log=stage_log
    )

    smoothed = run_step(
        "smooth",
        smooth_mesh,
        filled,
        cfg.mode,
        cfg.smoothing_iters,
        cfg.pass_band,
        cfg.feature_angle,
        stage_log=stage_log,
    )

    final_mesh = run_step(
        "finalize",
        finalize_mesh,
        smoothed,
        cfg.merge_tolerance,
        cfg.decimate_target_reduction,
        stage_log=stage_log,
    )
    if _has_recognized_color_data(data_source):
        final_mesh = run_step(
            "transfer_data",
            preserve_data_arrays,
            data_source,
            final_mesh,
            stage_log=stage_log,
        )

    topology = {"boundary_edge_count": None, "non_manifold_edge_count": None}
    is_watertight = None
    if cfg.verify_watertight:
        topology = run_step(
            "watertight_check",
            count_topology_issues,
            final_mesh,
            device=device,
            stage_log=stage_log,
        )
        is_watertight = (
            topology["boundary_edge_count"] == 0 and topology["non_manifold_edge_count"] == 0
        )
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

    open_edge_count: int | None = topology["boundary_edge_count"]
    if cfg.verify_watertight:
        be = topology["boundary_edge_count"] or 0
        nme = topology["non_manifold_edge_count"] or 0
        open_edge_count = be + nme

    return {
        "mesh": final_mesh,
        "mode": cfg.mode,
        "compute_device": device,
        "input_point_count": mesh.n_points,
        "input_cell_count": mesh.n_cells,
        "output_point_count": final_mesh.n_points,
        "output_cell_count": final_mesh.n_cells,
        "is_watertight": is_watertight,
        "open_edge_count": open_edge_count,
        "boundary_edge_count": topology["boundary_edge_count"],
        "non_manifold_edge_count": topology["non_manifold_edge_count"],
        **plane_info,
        "warnings": warnings,
        "stage_timings": stage_log.as_list(),
        "total_stage_seconds": stage_log.total_seconds,
    }


clean_ai_mesh = clean_mesh
