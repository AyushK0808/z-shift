from __future__ import annotations

import numpy as np
import trimesh

from spatial_ingestion.auto_rigging.models import Skeleton, SkinningWeights


class InverseDistanceSkinner:
    """Baseline skinning using inverse distance from vertices to joints."""

    def compute(
        self,
        mesh: trimesh.Trimesh,
        skeleton: Skeleton,
        max_influences: int = 4,
    ) -> SkinningWeights:
        vertices = np.asarray(mesh.vertices, dtype=float)
        joint_positions = np.asarray([joint.position for joint in skeleton.joints], dtype=float)
        if len(vertices) == 0:
            raise ValueError("mesh has no vertices")
        if len(joint_positions) == 0:
            raise ValueError("skeleton has no joints")

        distances = np.linalg.norm(vertices[:, None, :] - joint_positions[None, :, :], axis=2)
        raw_weights = 1.0 / np.maximum(distances, 1e-6)

        influence_count = min(max_influences, raw_weights.shape[1])
        if influence_count < raw_weights.shape[1]:
            keep = np.argpartition(raw_weights, -influence_count, axis=1)[:, -influence_count:]
            masked = np.zeros_like(raw_weights)
            rows = np.arange(raw_weights.shape[0])[:, None]
            masked[rows, keep] = raw_weights[rows, keep]
            raw_weights = masked

        row_sums = raw_weights.sum(axis=1, keepdims=True)
        weights = raw_weights / np.maximum(row_sums, 1e-12)
        return SkinningWeights(
            joint_names=[joint.name for joint in skeleton.joints],
            weights=weights.tolist(),
            max_influences=influence_count,
        )
