"""DTU's evaluation protocol, for a reconstruction that estimated its own cameras.

DTU's published numbers come from its MATLAB evaluation, or the Python port
most MVS papers use (github.com/jzhangbs/DTUeval-python). There the method was
given DTU's cameras, so its output is already in DTU's world frame. Accuracy
is scored only on reconstruction points inside the scan's observation mask,
completeness only on ground-truth points above the table plane, and both means
ignore distances over 20 mm.

MASt3R estimates its own cameras, so its output has an arbitrary frame and a
predicted scale. ICP against the ground truth cannot recover that from
scratch. The reconstruction also contains the table and backdrop, which the
ground truth leaves out, and the two clouds can differ in size several times
over. Every ICP seed then settles hundreds of millimetres off, and slowly.

The estimated cameras fix this instead. Each image's camera centre has a known
counterpart in DTU's calibration (rect_NNN_* <-> pos_NNN.txt). A similarity
transform fitted to those pairs (Umeyama, with scale) places the
reconstruction in DTU's frame from known correspondences, and the fitted scale
directly measures how metric MASt3R's output is. A rigid ICP on the masked
points then removes the residual left by camera-pose error. Both stages are
reported, so the ICP's contribution is visible rather than hidden.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import KDTree

from bench.gt_align import AlignmentResult, align_to_reference, apply_alignment, umeyama

__all__ = [
    "CAMERAS_FILE",
    "DTU_MAX_DIST_MM",
    "DtuReference",
    "camera_alignment",
    "estimate_normals",
    "load_reconstruction_cameras",
    "mesh_normal_consistency",
    "score_dtu",
    "view_number",
]

logger = logging.getLogger(__name__)

CAMERAS_FILE = "cameras.json"

# Distances beyond this are outliers to DTU's evaluation and excluded from the
# accuracy and completeness means (DTUeval-python's --max_dist).
DTU_MAX_DIST_MM = 20.0
# DTUeval-python's "patch" margin around the observation mask's bounding box,
# which decides which points count as in bounds for completeness.
_BOUNDS_MARGIN_MM = 60.0
# Below this, the camera centres are close to collinear and the rotation about
# their common line is poorly determined; the row says so.
_MIN_CAMERA_SPREAD = 0.05
# Camera-fit residual above which MASt3R's estimated poses are taken to have
# failed. A run whose cameras are tens of millimetres off DTU's cannot be
# scored meaningfully by those cameras; its F-score measures the pose failure,
# not the surface. Working runs on DTU sit at 3-6 mm, failures at 10-130 mm.
POSE_FAILURE_RMSE_MM = 10.0
# The camera fit is trimmed: a camera whose residual is past both this floor
# and _OUTLIER_MEDIAN_FACTOR times the median is left out and the fit redone.
# A plain least-squares fit lets one lost camera drag the whole placement: on
# B4's 40-view runs single views sat 100-280 mm off while the rest had a 4-8 mm
# median, and the lost view alone pushed the RMSE past the failure line.
_OUTLIER_FLOOR_MM = POSE_FAILURE_RMSE_MM
_OUTLIER_MEDIAN_FACTOR = 3.0
# More outliers than this and the rig as a whole has failed, however well the
# remaining cameras fit one another.
MAX_CAMERA_OUTLIER_FRAC = 0.2
# The ICP refinement is a diagnostic, not the score, so it runs on subsamples:
# 30 iterations against all ~5M structured-light points took 150 s a scoring.
_ICP_GT_POINTS = 500_000
_ICP_SOURCE_POINTS = 100_000
# DTU's ground truth carries no normals, so they are estimated from the
# structured-light points: the smallest principal axis of each point's
# neighbourhood. 16 neighbours of the 500k subsample span a few millimetres.
_NORMAL_NEIGHBOURS = 16
_NORMAL_CHUNK = 100_000

_VIEW_RE = re.compile(r"rect_(\d{3})_", re.IGNORECASE)


def view_number(image: str | Path) -> int | None:
    """DTU view number of a rectified image (rect_017_3_r5000.png -> 17)."""
    match = _VIEW_RE.search(Path(image).name)
    return int(match.group(1)) if match else None


@dataclass(frozen=True)
class DtuReference:
    """One scan's ground truth and the calibration files its evaluation needs."""

    stl_path: Path
    calibration_dir: Path
    obs_mask_path: Path
    plane_path: Path

    @cached_property
    def camera_centres(self) -> dict[int, np.ndarray]:
        """Camera centre in mm for every view, from its 3x4 projection matrix."""
        centres = {}
        for path in sorted(self.calibration_dir.glob("pos_*.txt")):
            projection = np.loadtxt(path)
            centres[int(path.stem.split("_")[1])] = -np.linalg.solve(
                projection[:, :3], projection[:, 3]
            )
        if not centres:
            raise FileNotFoundError(f"no pos_*.txt camera files in {self.calibration_dir}")
        return centres

    @cached_property
    def _obs_mask(self) -> tuple[np.ndarray, np.ndarray, float]:
        import scipy.io

        data = scipy.io.loadmat(self.obs_mask_path)
        return (
            data["ObsMask"].astype(bool),
            data["BB"].astype(float),
            float(np.asarray(data["Res"]).ravel()[0]),
        )

    @cached_property
    def plane(self) -> np.ndarray:
        import scipy.io

        return scipy.io.loadmat(self.plane_path)["P"].astype(float).ravel()

    @cached_property
    def stl_points(self) -> np.ndarray:
        """The full structured-light cloud: DTU scores against every point."""
        from bench.tier_b_common import load_gt_points

        return load_gt_points(self.stl_path, max_points=1 << 31)

    @cached_property
    def stl_tree(self) -> KDTree:
        return KDTree(self.stl_points)

    @cached_property
    def stl_sample(self) -> np.ndarray:
        """A 500k subsample for the ICP diagnostic, which needs no full density."""
        from bench.tier_b_common import load_gt_points

        return load_gt_points(self.stl_path, max_points=_ICP_GT_POINTS)

    @cached_property
    def stl_sample_tree(self) -> KDTree:
        return KDTree(self.stl_sample)

    @cached_property
    def stl_sample_normals(self) -> np.ndarray:
        """Unoriented PCA normals of `stl_sample`; consistency compares |cos|, so sign is moot."""
        return estimate_normals(self.stl_sample, self.stl_sample_tree)

    @cached_property
    def stl_above_plane(self) -> np.ndarray:
        return self.stl_points[self.above_plane(self.stl_points)]

    def above_plane(self, points: np.ndarray) -> np.ndarray:
        return points @ self.plane[:3] + self.plane[3] > 0

    def in_bounds(self, points: np.ndarray) -> np.ndarray:
        _, bounds, _ = self._obs_mask
        return np.all(
            (points >= bounds[0] - _BOUNDS_MARGIN_MM)
            & (points < bounds[1] + 2 * _BOUNDS_MARGIN_MM),
            axis=1,
        )

    def in_mask(self, points: np.ndarray) -> np.ndarray:
        """Points inside the scan's observed volume, as DTU's evaluation tests it."""
        mask, bounds, resolution = self._obs_mask
        grid = np.round((points - bounds[0]) / resolution).astype(np.int64)
        candidates = np.flatnonzero(
            self.in_bounds(points) & np.all((grid >= 0) & (grid < mask.shape), axis=1)
        )
        inside = np.zeros(len(points), dtype=bool)
        cells = grid[candidates]
        inside[candidates] = mask[cells[:, 0], cells[:, 1], cells[:, 2]]
        return inside


def estimate_normals(
    points: np.ndarray, tree: KDTree | None = None, k: int = _NORMAL_NEIGHBOURS
) -> np.ndarray:
    """Unit normal of every point: the least-variance axis of its k nearest neighbours.

    The sign is arbitrary. A point whose neighbourhood is degenerate (all
    coincident) gets an arbitrary axis, which the callers' distance cap and the
    sheer number of points make immaterial.
    """
    points = np.asarray(points, dtype=float)
    tree = tree if tree is not None else KDTree(points)
    normals = np.empty_like(points)
    for start in range(0, len(points), _NORMAL_CHUNK):
        chunk = points[start : start + _NORMAL_CHUNK]
        neighbours = points[tree.query(chunk, k=min(k, len(points)), workers=-1)[1]]
        centred = neighbours - neighbours.mean(axis=1, keepdims=True)
        covariance = np.einsum("nki,nkj->nij", centred, centred)
        normals[start : start + len(chunk)] = np.linalg.eigh(covariance)[1][:, :, 0]
    return normals


def load_reconstruction_cameras(output_dir: Path) -> dict[int, np.ndarray]:
    """Estimated camera centres by DTU view number, from Phase 2's cameras.json."""
    path = Path(output_dir) / CAMERAS_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing; it is written by Phase 2 export, so this "
            "reconstruction predates it and has to be re-run"
        )
    cameras = json.loads(path.read_text(encoding="utf-8"))["cameras"]
    centres = {}
    for camera in cameras:
        view = view_number(camera["image"])
        if view is not None:
            centres[view] = np.asarray(camera["cam2world"], dtype=float)[:3, 3]
    return centres


@dataclass(frozen=True)
class CameraFit:
    """The trimmed camera fit, and what it left out."""

    alignment: AlignmentResult  # fitted to the inlier cameras only
    spread: float
    views: list[int]
    residuals_mm: np.ndarray  # per view, under `alignment`
    inliers: np.ndarray
    rmse_all_mm: float  # untrimmed least-squares fit over every camera

    @property
    def outlier_views(self) -> list[int]:
        return [view for view, keep in zip(self.views, self.inliers, strict=True) if not keep]

    @property
    def outlier_frac(self) -> float:
        return float(np.mean(~self.inliers))

    @property
    def pose_failure(self) -> bool:
        return bool(
            self.alignment.rmse > POSE_FAILURE_RMSE_MM
            or self.outlier_frac > MAX_CAMERA_OUTLIER_FRAC
        )

    def as_row(self) -> dict[str, Any]:
        return {
            "camera_align_scale": round(self.alignment.scale, 6),
            "camera_align_rmse_mm": round(self.alignment.rmse, 4),
            "camera_align_rmse_all_mm": round(self.rmse_all_mm, 4),
            "camera_align_median_mm": round(float(np.median(self.residuals_mm)), 4),
            "camera_align_n_cameras": len(self.views),
            "camera_outliers": len(self.outlier_views),
            "camera_outlier_views": " ".join(str(view) for view in self.outlier_views),
            "camera_spread": round(self.spread, 4),
            "pose_failure": self.pose_failure,
        }


def camera_alignment(
    estimated: dict[int, np.ndarray], reference: dict[int, np.ndarray]
) -> CameraFit:
    """Similarity transform taking estimated camera centres onto DTU's, trimmed of lost views.

    Fits every camera, leaves out those far past the median residual
    (_OUTLIER_FLOOR_MM, _OUTLIER_MEDIAN_FACTOR), and refits until the inlier
    set stops changing. At least half the cameras, and three, always stay in.
    The spread is that of the reference centres, the ratio of their second to
    first principal extent: near 0 means near-collinear.
    """
    views = sorted(set(estimated) & set(reference))
    if len(views) < 3:
        raise ValueError(
            f"only {len(views)} camera(s) match DTU views; a similarity fit needs 3. "
            "Are the images DTU's rect_NNN_* files?"
        )
    source = np.stack([estimated[view] for view in views])
    target = np.stack([reference[view] for view in views])
    min_inliers = max(3, (len(views) + 1) // 2)

    def residuals(fit: AlignmentResult) -> np.ndarray:
        return np.linalg.norm(apply_alignment(source, fit) - target, axis=1)

    untrimmed = umeyama(source, target, with_scale=True)
    inliers = np.ones(len(views), dtype=bool)
    alignment = untrimmed
    for _ in range(len(views)):
        residual = residuals(alignment)
        median = float(np.median(residual[inliers]))
        threshold = max(_OUTLIER_FLOOR_MM, _OUTLIER_MEDIAN_FACTOR * median)
        trimmed = residual <= threshold
        if trimmed.sum() < min_inliers:
            # Keep the best-fitting half rather than drop below it.
            trimmed = np.zeros(len(views), dtype=bool)
            trimmed[np.argsort(residual)[:min_inliers]] = True
        if np.array_equal(trimmed, inliers):
            break
        inliers = trimmed
        alignment = umeyama(source[inliers], target[inliers], with_scale=True)

    singular = np.linalg.svd(target - target.mean(axis=0), compute_uv=False)
    spread = float(singular[1] / singular[0]) if singular[0] > 0 else 0.0
    if spread < _MIN_CAMERA_SPREAD:
        logger.warning(
            "camera centres are nearly collinear (spread %.3f); the rotation about "
            "their line is poorly constrained",
            spread,
        )
    fit = CameraFit(
        alignment=alignment,
        spread=spread,
        views=views,
        residuals_mm=residuals(alignment),
        inliers=inliers,
        rmse_all_mm=untrimmed.rmse,
    )
    if fit.outlier_views:
        logger.info(
            "camera fit left out view(s) %s; RMSE %.1f mm untrimmed, %.1f mm on the rest",
            fit.outlier_views,
            fit.rmse_all_mm,
            alignment.rmse,
        )
    return fit


def _mean(values: np.ndarray) -> float:
    return float(values.mean()) if len(values) else float("nan")


def _protocol_metrics(points: np.ndarray, reference: DtuReference, tau: float) -> dict[str, Any]:
    """DTU's scores. Distances past DTU_MAX_DIST_MM are excluded, as DTU excludes them.

    So the searches stop there: a KD-tree query has to prove there is nothing
    nearer, and for ground truth far from a partial reconstruction that proof
    is what made unbounded queries take minutes. `chamfer_l1`,
    `accuracy_mean` and `completeness_mean` are therefore DTU's capped means,
    the same values as the dtu_* columns, kept under the names the other
    experiments' code reads.
    """
    observed = points[reference.in_mask(points)]
    in_bounds = points[reference.in_bounds(points)]
    stl_above = reference.stl_above_plane
    bound = {"k": 1, "workers": -1, "distance_upper_bound": DTU_MAX_DIST_MM}

    d2s = reference.stl_tree.query(observed, **bound)[0] if len(observed) else np.empty(0)
    s2d = (
        KDTree(in_bounds).query(stl_above, **bound)[0]
        if len(in_bounds)
        else np.full(len(stl_above), np.inf)
    )
    precision = _mean(d2s < tau) if len(d2s) else 0.0
    recall = _mean(s2d < tau)
    f_score = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    accuracy = round(_mean(d2s[np.isfinite(d2s)]), 6)
    completeness = round(_mean(s2d[np.isfinite(s2d)]), 6)
    overall = round(0.5 * (accuracy + completeness), 6)
    return {
        "dtu_accuracy": accuracy,
        "dtu_completeness": completeness,
        "dtu_overall": overall,
        "accuracy_mean": accuracy,
        "completeness_mean": completeness,
        "chamfer_l1": overall,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f_score": round(f_score, 4),
        # Fractions beyond the cap, which the capped means cannot show.
        "accuracy_outlier_frac": round(_mean(~np.isfinite(d2s)), 4) if len(d2s) else 0.0,
        "completeness_outlier_frac": round(_mean(~np.isfinite(s2d)), 4),
        "n_points_in_mask": int(len(observed)),
    }


def score_dtu(
    points: np.ndarray,
    reference: DtuReference,
    output_dir: Path,
    *,
    tau: float,
    seed: int = 0,
    icp_iterations: int = 30,
) -> dict[str, Any]:
    """Place `points` in DTU's frame by the cameras and score them there.

    `points` are in the reconstruction's own frame and units, as Phase 2
    wrote them; the camera fit supplies both the frame and the scale.

    The headline scores use the camera fit alone. A rigid ICP refinement is
    also run and reported in `*_icp` columns, but it is not trusted for the
    headline: ICP minimises the very distances being scored, so on a sparse,
    partial reconstruction it can slide into a pose that scores better and is
    wrong. The first DTU smoke run did exactly that, rotating 14 degrees and
    shifting 50 mm away from a camera fit with a 2 mm residual. A large
    `icp_rotation_deg` or `icp_shift_mm` flags that.
    """
    points = np.asarray(points, dtype=float)
    camera_fit = camera_alignment(load_reconstruction_cameras(output_dir), reference.camera_centres)
    cameras = camera_fit.alignment
    placed = apply_alignment(points, cameras)
    scores = _protocol_metrics(placed, reference, tau)

    observed = placed[reference.in_mask(placed)]
    icp_row: dict[str, Any] = {}
    if len(observed) >= 3:
        # Rigid only: the cameras already fixed the scale, and an ICP that may
        # also rescale can shrink a partial reconstruction onto the surface.
        rng = np.random.default_rng(seed)
        source = (
            observed[rng.choice(len(observed), _ICP_SOURCE_POINTS, replace=False)]
            if len(observed) > _ICP_SOURCE_POINTS
            else observed
        )
        _, icp = align_to_reference(
            source,
            reference.stl_sample,
            with_scale=False,
            iterations=icp_iterations,
            pca_seeds=False,
            max_correspondence_distance=DTU_MAX_DIST_MM,
            tree=reference.stl_sample_tree,
        )
        refined = _protocol_metrics(apply_alignment(placed, icp), reference, tau)
        centroid = observed.mean(axis=0, keepdims=True)
        cos_angle = np.clip((np.trace(icp.rotation) - 1) / 2, -1, 1)
        icp_row = {
            "dtu_overall_icp": refined["dtu_overall"],
            "f_score_icp": refined["f_score"],
            "precision_icp": refined["precision"],
            "recall_icp": refined["recall"],
            "icp_rotation_deg": round(float(np.degrees(np.arccos(cos_angle))), 4),
            # How far ICP moved the masked points' centroid; the transform's
            # own translation also absorbs the rotation about DTU's origin.
            "icp_shift_mm": round(
                float(np.linalg.norm(apply_alignment(centroid, icp) - centroid)), 4
            ),
            "icp_rmse_mm": round(icp.rmse, 4),
        }
    else:
        logger.warning("no reconstruction points fall inside the observation mask")

    return {
        "protocol": "dtu_obsmask",
        "align_method": "camera_sim3",
        "tau": tau,
        "seed": seed,
        **scores,
        "n_reconstruction_points": int(len(points)),
        "n_gt_points": int(len(reference.stl_points)),
        "mask_retention": round(scores["n_points_in_mask"] / max(len(points), 1), 4),
        **camera_fit.as_row(),
        # The trimmed camera fit is the alignment the scores come from.
        **cameras.as_row(),
        **icp_row,
    }


def mesh_normal_consistency(
    mesh: Any,
    reference: DtuReference,
    output_dir: Path,
    *,
    n_samples: int = 100_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Mean |cos| between sampled mesh normals and the nearest ground-truth normal.

    The mesh is placed in DTU's frame by the same trimmed camera fit as the
    other scores (normals take the rotation only), kept to the observation
    mask, and matched to the nearest ground-truth point within DTU's 20 mm cap.
    The ground-truth normals are estimated from the points, so this measures
    agreement with the structured-light surface as DTU samples it; it is not a
    standard DTU metric.
    """
    from bench.metrics import sample_points_with_normals

    fit = camera_alignment(load_reconstruction_cameras(output_dir), reference.camera_centres)
    points, normals = sample_points_with_normals(mesh, n_samples, seed)
    placed = apply_alignment(points, fit.alignment)
    rotated = normals @ fit.alignment.rotation.T
    inside = reference.in_mask(placed)
    placed, rotated = placed[inside], rotated[inside]
    empty = {"normal_consistency": float("nan"), "normal_consistency_n": 0}
    if len(placed) == 0:
        return empty
    distance, index = reference.stl_sample_tree.query(
        placed, k=1, workers=-1, distance_upper_bound=DTU_MAX_DIST_MM
    )
    matched = np.isfinite(distance)
    if not matched.any():
        return empty
    cosine = np.abs(
        np.einsum("ij,ij->i", rotated[matched], reference.stl_sample_normals[index[matched]])
    )
    return {
        "normal_consistency": round(float(cosine.mean()), 5),
        "normal_consistency_p10": round(float(np.percentile(cosine, 10)), 5),
        "normal_consistency_n": int(matched.sum()),
    }
