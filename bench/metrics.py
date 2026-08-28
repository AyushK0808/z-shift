"""P2 - geometric metrics used by A3, A4 and all of Tier B.

Every distance returned here is in *scene units*. Callers that compare across
shapes must normalise first (`unit_normalize`), and every reported F-score has
to state its tau. Nothing in this module is stochastic beyond the surface
sampling, which is seeded.
"""

from __future__ import annotations

import numpy as np
import trimesh
from scipy.spatial import KDTree

__all__ = [
    "bbox_diagonal",
    "chamfer_l1",
    "hausdorff_95",
    "normal_consistency",
    "precision_recall_f",
    "sample_points",
    "sample_points_with_normals",
    "unit_normalize",
    "volume_ratio",
]


def bbox_diagonal(mesh: trimesh.Trimesh) -> float:
    bounds = np.asarray(mesh.bounds, dtype=float)
    return float(np.linalg.norm(bounds[1] - bounds[0]))


def unit_normalize(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Centre a mesh at the origin and scale it to a unit bounding diagonal.

    Distances between differently-sized shapes are otherwise incomparable, and
    a Chamfer number without a stated scale is meaningless.
    """
    out = mesh.copy()
    diagonal = bbox_diagonal(out)
    out.apply_translation(-out.bounds.mean(axis=0))
    if diagonal > 0:
        out.apply_scale(1.0 / diagonal)
    return out


def sample_points(mesh: trimesh.Trimesh, n: int = 100_000, seed: int = 0) -> np.ndarray:
    points, _ = _sample(mesh, n, seed)
    return points


def sample_points_with_normals(
    mesh: trimesh.Trimesh, n: int = 100_000, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    points, face_index = _sample(mesh, n, seed)
    return points, np.asarray(mesh.face_normals)[face_index]


def _sample(mesh: trimesh.Trimesh, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if len(mesh.faces) == 0:
        raise ValueError("cannot sample a surface from a mesh with no faces")
    rng = np.random.default_rng(seed)
    points, face_index = trimesh.sample.sample_surface(mesh, n, seed=int(rng.integers(1 << 31)))
    return np.asarray(points, dtype=float), np.asarray(face_index, dtype=int)


def _dists(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Nearest-neighbour distance from every point of `a` to the set `b`."""
    return np.asarray(KDTree(b).query(a, k=1)[0], dtype=float)


def chamfer_l1(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric mean nearest-neighbour distance. Scene units."""
    return float(0.5 * (_dists(a, b).mean() + _dists(b, a).mean()))


def hausdorff_95(a: np.ndarray, b: np.ndarray) -> float:
    """95th-percentile symmetric distance, so one outlier vertex cannot dominate."""
    return float(max(np.percentile(_dists(a, b), 95), np.percentile(_dists(b, a), 95)))


def precision_recall_f(pred: np.ndarray, gt: np.ndarray, tau: float) -> tuple[float, float, float]:
    """MVS convention: accuracy = pred->gt, completeness = gt->pred."""
    precision = float((_dists(pred, gt) < tau).mean())
    recall = float((_dists(gt, pred) < tau).mean())
    f_score = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f_score


def normal_consistency(
    pred: trimesh.Trimesh,
    gt: trimesh.Trimesh,
    n: int = 50_000,
    seed: int = 0,
    exact: bool = False,
    gt_oversample: int = 4,
) -> float:
    """Mean |cos| between a sampled predicted normal and the nearest GT normal.

    `exact=True` uses `trimesh.proximity.closest_point`, the textbook
    definition, but that needs the optional `rtree` dependency and runs a
    Python-level query per point -- far too slow for a grid of hundreds of
    meshes. The default instead samples the GT surface `gt_oversample`x more
    densely than the query set and takes the normal of the nearest GT
    *sample*, which converges to the same value as sampling density rises.
    `tests/test_bench.py` pins it against an analytic sphere reference.
    """
    pred_points, pred_normals = sample_points_with_normals(pred, n, seed)

    if exact:
        try:
            import rtree  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "normal_consistency(exact=True) needs the optional 'rtree' package; "
                "install it or use the default KD-tree estimator"
            ) from exc
        _, _, gt_face_index = trimesh.proximity.closest_point(gt, pred_points)
        gt_normals = np.asarray(gt.face_normals)[np.asarray(gt_face_index, dtype=int)]
    else:
        gt_points, gt_all_normals = sample_points_with_normals(
            gt, max(n * gt_oversample, 1000), seed + 1
        )
        nearest = KDTree(gt_points).query(pred_points, k=1)[1]
        gt_normals = gt_all_normals[np.asarray(nearest, dtype=int)]

    return float(np.abs((pred_normals * gt_normals).sum(axis=1)).mean())


def volume_ratio(mesh: trimesh.Trimesh, reference: trimesh.Trimesh) -> float | None:
    """`mesh` volume as a fraction of `reference` volume.

    Only defined for watertight inputs; returns None otherwise rather than
    reporting trimesh's signed-volume estimate for an open surface, which is
    not a volume.
    """
    if not (mesh.is_watertight and reference.is_watertight):
        return None
    ref_volume = float(reference.volume)
    if ref_volume == 0:
        return None
    return float(mesh.volume) / ref_volume
