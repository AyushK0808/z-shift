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
    "load_reconstruction_cameras",
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
# The ICP refinement is a diagnostic, not the score, so it runs on subsamples:
# 30 iterations against all ~5M structured-light points took 150 s a scoring.
_ICP_GT_POINTS = 500_000
_ICP_SOURCE_POINTS = 100_000

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


def camera_alignment(
    estimated: dict[int, np.ndarray], reference: dict[int, np.ndarray]
) -> tuple[AlignmentResult, float]:
    """Similarity transform taking estimated camera centres onto DTU's.

    Returns the fit and the spread of the reference centres, the ratio of
    their second to first principal extent: near 0 means near-collinear.
    """
    views = sorted(set(estimated) & set(reference))
    if len(views) < 3:
        raise ValueError(
            f"only {len(views)} camera(s) match DTU views; a similarity fit needs 3. "
            "Are the images DTU's rect_NNN_* files?"
        )
    target = np.stack([reference[view] for view in views])
    alignment = umeyama(np.stack([estimated[view] for view in views]), target, with_scale=True)
    singular = np.linalg.svd(target - target.mean(axis=0), compute_uv=False)
    spread = float(singular[1] / singular[0]) if singular[0] > 0 else 0.0
    if spread < _MIN_CAMERA_SPREAD:
        logger.warning(
            "camera centres are nearly collinear (spread %.3f); the rotation about "
            "their line is poorly constrained",
            spread,
        )
    return alignment, spread


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
    cameras, spread = camera_alignment(
        load_reconstruction_cameras(output_dir), reference.camera_centres
    )
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
        "camera_align_scale": round(cameras.scale, 6),
        "camera_align_rmse_mm": round(cameras.rmse, 4),
        "camera_align_n_cameras": cameras.n_correspondences,
        "camera_spread": round(spread, 4),
        "pose_failure": bool(cameras.rmse > POSE_FAILURE_RMSE_MM),
        # The camera fit is the alignment the scores come from.
        **cameras.as_row(),
        **icp_row,
    }
