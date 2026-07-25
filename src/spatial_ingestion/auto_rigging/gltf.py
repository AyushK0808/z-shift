from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spatial_ingestion.auto_rigging.models import RiggedMesh


@dataclass(frozen=True)
class GltfSkinPayload:
    joint_names: list[str]
    joints_0: list[tuple[int, int, int, int]]
    weights_0: list[tuple[float, float, float, float]]


class GltfSkinPayloadBuilder:
    """Builds glTF-compatible JOINTS_0 / WEIGHTS_0 arrays.

    Full skinned GLB writing is a larger step; this class nails down the
    packing contract used by glTF 2.0 so exporters can stay boring later.
    """

    def build(self, rigged_mesh: RiggedMesh) -> GltfSkinPayload:
        joint_names = rigged_mesh.skinning.joint_names
        raw_weights = np.asarray(rigged_mesh.skinning.weights, dtype=float)
        if raw_weights.ndim != 2:
            raise ValueError("skinning weights must be a 2D matrix")
        if raw_weights.shape[0] != rigged_mesh.vertex_count:
            raise ValueError("skinning weight row count must match vertex count")
        if raw_weights.shape[1] != len(joint_names):
            raise ValueError("skinning weight column count must match joint count")

        joints_0: list[tuple[int, int, int, int]] = []
        weights_0: list[tuple[float, float, float, float]] = []
        for row in raw_weights:
            selected = self._top_four(row)
            joint_indices = [index for index, _ in selected]
            packed_weights = self._normalize([weight for _, weight in selected])
            joints_0.append(self._pad_ints(joint_indices))
            weights_0.append(self._pad_floats(packed_weights))

        return GltfSkinPayload(
            joint_names=joint_names,
            joints_0=joints_0,
            weights_0=weights_0,
        )

    @staticmethod
    def _top_four(row: np.ndarray) -> list[tuple[int, float]]:
        if row.size == 0:
            return []
        count = min(4, row.size)
        indices = np.argpartition(row, -count)[-count:]
        sorted_indices = sorted(indices.tolist(), key=lambda index: row[index], reverse=True)
        return [(index, float(row[index])) for index in sorted_indices if row[index] > 0]

    @staticmethod
    def _normalize(values: list[float]) -> list[float]:
        total = sum(values)
        if total <= 0:
            return [1.0]
        return [value / total for value in values]

    @staticmethod
    def _pad_ints(values: list[int]) -> tuple[int, int, int, int]:
        padded = (values + [0, 0, 0, 0])[:4]
        return (padded[0], padded[1], padded[2], padded[3])

    @staticmethod
    def _pad_floats(values: list[float]) -> tuple[float, float, float, float]:
        padded = (values + [0.0, 0.0, 0.0, 0.0])[:4]
        return (padded[0], padded[1], padded[2], padded[3])

