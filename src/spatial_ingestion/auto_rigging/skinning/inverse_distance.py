from __future__ import annotations

import logging

import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph
import trimesh
from scipy.spatial import cKDTree  # type: ignore

from spatial_ingestion.auto_rigging.models import Skeleton, SkinningWeights

logger = logging.getLogger(__name__)

_EPS = 1e-10


def _voxel_geodesic_distances(
    mesh: trimesh.Trimesh,
    joint_positions: np.ndarray,
    resolution: int = 64,
) -> np.ndarray:
    """(V, J) shortest-path distances through the *interior* of the mesh.

    The mesh is voxelised and flood-filled, then for each joint a multi-source
    Dijkstra runs over the 6-connectivity voxel graph, so distances follow the
    volume instead of cutting across empty space. This is what removes the
    left-arm/right-arm bleed that plain Euclidean distance-to-joint produces
    when two limbs happen to sit close together in space.

    Falls back to surface-voxel distances (a surface geodesic approximation)
    when the mesh is open and cannot be flood-filled.
    """
    vertices = np.asarray(mesh.vertices, dtype=float)
    extent = float(mesh.extents.max()) or 1.0
    pitch = extent / resolution

    try:
        vox = trimesh.voxel.creation.voxelize(mesh, pitch=pitch)
        if vox is None:
            return _euclidean_distances(vertices, joint_positions)
        try:
            filled = vox.fill()
        except Exception:
            filled = vox
        world = np.asarray(filled.points, dtype=float)
    except Exception:
        return _euclidean_distances(vertices, joint_positions)

    n = len(world)
    if n < 2:
        return _euclidean_distances(vertices, joint_positions)
    if n > 120_000:
        # Downgrade the grid so memory and Dijkstra cost stay bounded.
        new_resolution = max(16, resolution // 2)
        return _voxel_geodesic_distances(mesh, joint_positions, resolution=new_resolution)

    # 6-connectivity adjacency of occupied voxels (axis-aligned neighbours only).
    tree = cKDTree(world)
    pairs = tree.query_pairs(1.5 * pitch, output_type="ndarray")
    if len(pairs) == 0:
        return _euclidean_distances(vertices, joint_positions)
    dx = np.linalg.norm(world[pairs[:, 0]] - world[pairs[:, 1]], axis=1)
    keep = np.abs(dx - pitch) < 0.05 * pitch
    pairs = pairs[keep]
    dx = dx[keep]
    rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
    cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
    data = np.concatenate([dx, dx])
    graph = sp.csr_matrix((data, (rows, cols)), shape=(n, n))

    _, joint_vox = tree.query(np.asarray(joint_positions, dtype=float))
    joint_vox = np.atleast_1d(np.asarray(joint_vox, dtype=int))
    _, vert_vox = tree.query(vertices)

    eucl = _euclidean_distances(vertices, joint_positions)
    dists = np.empty((len(vertices), len(joint_positions)), dtype=float)
    for j, src in enumerate(joint_vox):
        path = scipy.sparse.csgraph.dijkstra(graph, directed=False, indices=int(src))
        path = np.asarray(path, dtype=float)[vert_vox]
        path[~np.isfinite(path)] = eucl[~np.isfinite(path), j]
        dists[:, j] = path
    return dists


def _euclidean_distances(vertices: np.ndarray, joint_positions: np.ndarray) -> np.ndarray:
    """(V, J) plain Euclidean distance matrix; the fallback when the mesh
    cannot be voxelised meaningfully."""
    diff = vertices[:, None, :] - joint_positions[None, :, :]
    return np.linalg.norm(diff, axis=2)


class InverseDistanceSkinner:
    """Volumetric-geodesic inverse-distance skinning.

    Weights are an RBF of the shortest distance through the mesh *volume* from
    each joint to each vertex, so the map respects where the shape actually
    is. The result is smoothed along mesh edges and sparsified to the top
    ``max_influences`` per vertex.
    """

    def __init__(
        self,
        power: float = 2.0,
        laplacian_smoothing_iters: int = 2,
        voxel_resolution: int = 64,
    ) -> None:
        self._power = power
        self._smoothing_iters = laplacian_smoothing_iters
        self._voxel_resolution = voxel_resolution

    def compute(
        self,
        mesh: trimesh.Trimesh,
        skeleton: Skeleton,
        max_influences: int = 4,
    ) -> SkinningWeights:
        vertices = np.asarray(mesh.vertices, dtype=float)
        joints = skeleton.joints
        n_vertices = len(vertices)
        n_joints = len(joints)
        effective_max_influences = min(max_influences, max(1, n_joints))

        if n_vertices == 0 or n_joints == 0:
            return SkinningWeights(
                joint_names=[j.name for j in joints],
                weights=[],
                max_influences=effective_max_influences,
            )

        joint_positions = np.asarray([joint.position for joint in joints], dtype=float)

        # 1. Volumetric-geodesic distances from every joint (Euclidean fallback).
        distances = _voxel_geodesic_distances(mesh, joint_positions, self._voxel_resolution)

        if np.isnan(distances).any():
            distances = _euclidean_distances(vertices, joint_positions)

        # 2. RBF decay with a characteristic scale tied to the shape.
        per_vertex_closest = np.nanmin(distances, axis=1)
        sigma = float(np.median(per_vertex_closest)) * 2.0
        sigma = max(sigma, 0.05)
        raw_weights = np.exp(-0.5 * (distances / sigma) ** self._power)

        # 3. Guarantee no all-zero rows.
        zero_row_mask = raw_weights.sum(axis=1) < 1e-12
        if np.any(zero_row_mask):
            closest_joint = np.argmin(distances[zero_row_mask], axis=1)
            for r_idx, j_idx in zip(np.where(zero_row_mask)[0], closest_joint, strict=True):
                raw_weights[r_idx, j_idx] = 1.0

        row_sums = raw_weights.sum(axis=1, keepdims=True)
        weights = raw_weights / np.maximum(row_sums, 1e-12)

        # 4. Laplacian smoothing along the actual mesh edges.
        if self._smoothing_iters > 0 and len(mesh.edges_unique) > 0 and n_joints > 1:
            edges = mesh.edges_unique
            row = np.concatenate([edges[:, 0], edges[:, 1]])
            col = np.concatenate([edges[:, 1], edges[:, 0]])
            data = np.ones(len(row), dtype=float)
            adj = sp.coo_matrix((data, (row, col)), shape=(n_vertices, n_vertices)).tocsr()

            degrees = np.asarray(adj.sum(axis=1)).flatten()
            degrees[degrees == 0] = 1.0
            laplacian_op = sp.diags(1.0 / degrees) @ adj

            alpha = 0.3
            for _ in range(self._smoothing_iters):
                weights = (1.0 - alpha) * weights + alpha * (laplacian_op @ weights)

        # 5. Sparsify to the top max_influences per vertex.
        if effective_max_influences < n_joints:
            top_k_indices = np.argpartition(-weights, effective_max_influences, axis=1)[
                :, :effective_max_influences
            ]
            sparsified = np.zeros_like(weights)
            row_idx = np.arange(n_vertices)[:, np.newaxis]
            sparsified[row_idx, top_k_indices] = weights[row_idx, top_k_indices]
            weights = sparsified

        # 6. Renormalise rows to sum exactly to 1.
        final_sums = weights.sum(axis=1, keepdims=True)
        final_sums[final_sums == 0] = 1.0
        normalized_weights = weights / final_sums

        return SkinningWeights(
            joint_names=[joint.name for joint in joints],
            weights=normalized_weights.tolist(),
            max_influences=effective_max_influences,
        )
