"""Geometry analysis helpers for template skeleton fitting.

The old Phase 5 fitter placed joints at fixed fractions of the bounding box,
which is correct only for perfectly axis-aligned, Y-up, ideally-shaped meshes.
This module reads the actual mesh shape instead:

- ``orient_up``   -- conservative support-plane detection so templates are fit
                     in a Y-up frame even when the reconstruction is tilted.
- ``cross_sections`` -- the volumetric centerline of the mesh (per-slice
                     centroids of occupied voxels along an axis).
- ``extremal_cluster`` / ``split_cluster`` -- limb-endpoint estimation from
                     where the mesh's own surface protrudes.

Everything here is deterministic (no RNG) and vectorised enough for the
mesh sizes the pipeline sees.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

UP = 1  # Y axis index in the template frame


@dataclass(frozen=True)
class SliceProfile:
    """Per-slice summary along a chosen axis (default: the Y/up axis)."""

    mid: np.ndarray  # (n,) axis coordinates at slice midpoints
    center: np.ndarray  # (n, 3) centroids (other two axes), smoothed
    radius: np.ndarray  # (n,) mean horizontal distance from centroid
    count: np.ndarray  # (n,) occupied voxels per slice

    def center_at(self, fraction: float) -> np.ndarray:
        """Interpolated centroid of the cross-section profile at a height fraction."""
        if len(self.mid) == 0:
            return np.zeros(3)
        lo, hi = float(self.mid[0]), float(self.mid[-1])
        target = lo + fraction * (hi - lo)
        idx = int(np.searchsorted(self.mid, target))
        idx = int(min(max(idx, 0), len(self.mid) - 1))
        return np.asarray(self.center[idx], dtype=float)


def orient_up(mesh: trimesh.Trimesh, bottom_frac: float = 0.06) -> trimesh.Trimesh:
    """Return a copy of ``mesh`` rotated so its support plane is +Y.

    Only reorients when a stable, flat "bottom" can be found (typical for
    ground/turntable captures where the object sat on a platen). Rounded or
    noisy bottoms are left alone -- an incorrect guess is worse than none.
    """
    verts = np.asarray(mesh.vertices, dtype=float)
    if len(verts) < 32:
        return mesh.copy()

    y = verts[:, UP]
    k = max(int(len(verts) * bottom_frac), 16)
    lowest = verts[np.argsort(y)[:k]]

    centroid = lowest.mean(axis=0)
    _, _, vt = np.linalg.svd(
        (lowest - centroid) / (float(mesh.extents.max()) or 1.0), full_matrices=False
    )
    normal = vt[-1]
    # Plane must point "up"; fold to the hemisphere containing +Y.
    if normal[UP] < 0:
        normal = -normal

    # Flatness check: the bottom must actually be a plane.
    residual = np.abs((lowest - centroid) @ normal).std()
    scale = float(mesh.extents.max()) or 1.0
    if residual / scale > 0.02:
        return mesh.copy()

    deviation = float(np.degrees(np.arccos(np.clip(normal[UP], -1.0, 1.0))))
    if deviation < 1.0:
        return mesh.copy()
    if deviation > 40.0:
        return mesh.copy()

    rot = trimesh.transformations.rotation_matrix(
        np.radians(deviation), _perpendicular_axis(normal)
    )
    out = mesh.copy()
    out.apply_transform(rot)
    # Normalise again so the canonical frame stays unit cube.
    ext = out.extents.max()
    if ext > 0:
        out.apply_scale(1.0 / ext)
    return out


def _perpendicular_axis(v: np.ndarray) -> np.ndarray:
    """A unit vector perpendicular to ``v`` (used as the rotation axis)."""
    ref = np.zeros(3)
    ref[0 if abs(v[1]) < 0.9 else 1] = 1.0
    axis = np.cross(np.asarray(v, dtype=float), ref)
    norm = np.linalg.norm(axis)
    if norm < 1e-9:
        return np.array([1.0, 0.0, 0.0])
    return axis / norm


def _voxel_points(mesh: trimesh.Trimesh, resolution: int = 96) -> np.ndarray:
    ext = float(mesh.extents.max()) or 1.0
    pitch = ext / resolution
    vox = trimesh.voxel.creation.voxelize(mesh, pitch=pitch)
    if vox is None:
        return np.asarray(mesh.vertices, dtype=float)
    try:
        filled = vox.fill()
    except Exception:
        filled = vox
    # Drop voxels touching the padding plane so pure-surface noise doesn't dominate.
    pts = np.asarray(filled.points, dtype=float)
    if len(pts) < 32:
        pts = np.asarray(mesh.vertices, dtype=float)
    return pts


def cross_sections(
    mesh: trimesh.Trimesh,
    axis: int = UP,
    resolution: int = 96,
    min_count: int = 2,
    smooth: int = 3,
) -> SliceProfile:
    """Volumetric cross-section profile of ``mesh`` along ``axis``."""
    pts = _voxel_points(mesh, resolution)
    lo, hi = float(pts[:, axis].min()), float(pts[:, axis].max())
    if hi - lo <= 0:
        return SliceProfile(
            mid=np.array([hi]),
            center=np.atleast_2d(pts.mean(axis=0)),
            radius=np.ones(1),
            count=np.ones(1),
        )

    extent = float(mesh.extents.max()) or 1.0
    n = max(16, int(round(resolution * (hi - lo) / extent)))
    edges = np.linspace(lo, hi, n + 1)

    centers = np.zeros((n, 3))
    radii = np.zeros(n)
    counts = np.zeros(n, dtype=int)
    for i in range(n):
        mask = (pts[:, axis] >= edges[i]) & (pts[:, axis] < edges[i + 1])
        if mask.sum() < min_count:
            continue
        slice_pts = pts[mask]
        centers[i] = slice_pts.mean(axis=0)
        centers[i, axis] = 0.5 * (edges[i] + edges[i + 1])
        radii[i] = float(
            np.linalg.norm(
                slice_pts[:, [a for a in range(3) if a != axis]]
                - centers[i, [a for a in range(3) if a != axis]],
                axis=1,
            ).mean()
        )
        counts[i] = int(mask.sum())

    filled_rows = counts > 0
    mid = 0.5 * (edges[:-1] + edges[1:])

    for col in range(3):
        if col == axis:
            continue
        val = centers[filled_rows, col]
        centers[filled_rows, col] = _smooth(val, smooth)

    return SliceProfile(
        mid=mid[filled_rows],
        center=centers[filled_rows],
        radius=_smooth(radii[filled_rows], smooth),
        count=counts[filled_rows],
    )


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) <= 1:
        return values
    window = max(1, int(window))
    kernel = np.ones(window, dtype=float) / window
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(values)]


def extremal_cluster(verts: np.ndarray, axis: int, sign: int, k: int = 24) -> np.ndarray:
    """Centroid of the ``k`` most extreme vertices along ``axis`` in one direction."""
    if len(verts) == 0:
        return np.zeros(3)
    vals = verts[:, axis]
    order = np.argsort(vals if sign < 0 else -vals)
    return verts[order[: min(k, len(verts))]].mean(axis=0)


def split_cluster(verts: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
    """Two centroids split by the sign of ``axis`` (left then right)."""
    if len(verts) == 0:
        return np.zeros(3), np.zeros(3)
    side = verts[verts[:, axis] <= 0]
    other = verts[verts[:, axis] > 0]
    if len(side) == 0:
        side = verts[np.argsort(verts[:, axis])[: max(1, len(verts) // 4)]]
    if len(other) == 0:
        other = verts[np.argsort(-verts[:, axis])[: max(1, len(verts) // 4)]]
    return side.mean(axis=0), other.mean(axis=0)


def widest_horizontal_axis(mesh: trimesh.Trimesh) -> int:
    """Return 0 (X) or 2 (Z): the horizontal bounding axis with the wider extent."""
    ext = mesh.extents
    return 0 if ext[0] >= ext[2] else 2


def limb_extrema(
    mesh: trimesh.Trimesh,
    lo_frac: float = 0.45,
    hi_frac: float = 0.78,
    k: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    """Left/right limb end points from the band's most extreme protrusions.

    The band (a horizontal slab of the mesh) is split along whichever
    horizontal axis has the widest spread, so arms that hang sideways and arms
    that point forward both resolve into a (left, right) pair. Falls back to the
    bounding-box extreme when the band is too small to be meaningful.
    """
    pts = np.asarray(mesh.vertices, dtype=float)
    band = horizontal_band(pts, lo_frac, hi_frac)
    if len(band) < max(8, min(len(pts), 8)):
        axis = widest_horizontal_axis(mesh)
        centroid = np.asarray(mesh.centroid, dtype=float)
        other = 2 if axis == 0 else 0
        return (
            np.array([mesh.bounds[0][axis], centroid[UP], centroid[other]]),
            np.array([mesh.bounds[1][axis], centroid[UP], centroid[other]]),
        )
    spread = band[:, [0, 2]].std(axis=0)
    axis = 0 if spread[0] >= spread[1] else 2
    left = extremal_cluster(band, axis, -1, k)
    right = extremal_cluster(band, axis, +1, k)
    return left, right


def foot_points(
    mesh: trimesh.Trimesh,
    lo_frac: float = 0.0,
    hi_frac: float = 0.14,
) -> tuple[np.ndarray, np.ndarray]:
    """Left/right foot anchors: bottom-band clusters snapped to the floor."""
    pts = np.asarray(mesh.vertices, dtype=float)
    band = horizontal_band(pts, lo_frac, hi_frac)
    if len(band) < max(8, min(len(pts), 8)):
        centroid = np.asarray(mesh.centroid, dtype=float)
        left = np.array([mesh.bounds[0][0], float(pts[:, UP].min()), centroid[2]])
        right = np.array([mesh.bounds[1][0], float(pts[:, UP].min()), centroid[2]])
        return left, right
    spread = band[:, [0, 2]].std(axis=0)
    axis = 0 if spread[0] >= spread[1] else 2
    left, right = split_cluster(band, axis)
    y = band[:, UP]
    left[UP] = float(y[band[:, axis] <= 0].min())
    right[UP] = float(y[band[:, axis] > 0].min())
    return left, right


def horizontal_band(pts: np.ndarray, lo_frac: float, hi_frac: float) -> np.ndarray:
    """Vertices whose height (Y) fraction of the mesh falls inside [lo, hi]."""
    y = pts[:, UP]
    lo, hi = float(y.min()), float(y.max())
    span = hi - lo
    if span <= 0:
        return pts
    mask = (y >= lo + lo_frac * span) & (y <= lo + hi_frac * span)
    return pts[mask]
