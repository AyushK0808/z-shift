from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

from spatial_ingestion.auto_rigging.models import Joint, RiggedMesh


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
            if not selected:
                raise ValueError(
                    "skinning weight rows must contain at least one positive influence; "
                    "refusing to silently bind an empty row to joint 0"
                )
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
            raise ValueError("cannot normalize an empty or zero-sum influence row")
        return [value / total for value in values]

    @staticmethod
    def _pad_ints(values: list[int]) -> tuple[int, int, int, int]:
        padded = (values + [0, 0, 0, 0])[:4]
        return (padded[0], padded[1], padded[2], padded[3])

    @staticmethod
    def _pad_floats(values: list[float]) -> tuple[float, float, float, float]:
        padded = (values + [0.0, 0.0, 0.0, 0.0])[:4]
        return (padded[0], padded[1], padded[2], padded[3])


class SkinnedGlbExporter:
    """Writes a minimal glTF 2.0 binary with skinning data embedded."""

    _ARRAY_BUFFER = 34962
    _ELEMENT_ARRAY_BUFFER = 34963
    _FLOAT = 5126
    _UNSIGNED_BYTE = 5121
    _UNSIGNED_SHORT = 5123
    _UNSIGNED_INT = 5125

    def __init__(self, payload_builder: GltfSkinPayloadBuilder | None = None) -> None:
        self._payload_builder = payload_builder or GltfSkinPayloadBuilder()

    def export(self, mesh: trimesh.Trimesh, rigged_mesh: RiggedMesh, path: Path) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        gltf, binary = self._build_gltf(mesh, rigged_mesh)
        _write_glb(path, gltf, binary)
        return path.as_uri()

    def _build_gltf(
        self, mesh: trimesh.Trimesh, rigged_mesh: RiggedMesh
    ) -> tuple[dict[str, object], bytes]:
        if len(mesh.vertices) != rigged_mesh.vertex_count:
            raise ValueError("mesh vertex count must match rigged mesh vertex count")
        if len(mesh.faces) != rigged_mesh.face_count:
            raise ValueError("mesh face count must match rigged mesh face count")

        skin_payload = self._payload_builder.build(rigged_mesh)
        buffer = _BinaryBuffer()
        buffer_views: list[dict[str, object]] = []
        accessors: list[dict[str, object]] = []

        positions = np.asarray(mesh.vertices, dtype="<f4")
        position_view = buffer.append(positions)
        buffer_views.append(
            {
                "buffer": 0,
                "byteOffset": position_view.byte_offset,
                "byteLength": position_view.byte_length,
                "target": self._ARRAY_BUFFER,
            }
        )
        accessors.append(
            {
                "bufferView": len(buffer_views) - 1,
                "componentType": self._FLOAT,
                "count": int(len(positions)),
                "type": "VEC3",
                "min": positions.min(axis=0).astype(float).tolist(),
                "max": positions.max(axis=0).astype(float).tolist(),
            }
        )
        position_accessor = len(accessors) - 1

        indices = np.asarray(mesh.faces.reshape(-1), dtype="<u4")
        index_view = buffer.append(indices)
        buffer_views.append(
            {
                "buffer": 0,
                "byteOffset": index_view.byte_offset,
                "byteLength": index_view.byte_length,
                "target": self._ELEMENT_ARRAY_BUFFER,
            }
        )
        accessors.append(
            {
                "bufferView": len(buffer_views) - 1,
                "componentType": self._UNSIGNED_INT,
                "count": int(len(indices)),
                "type": "SCALAR",
            }
        )
        index_accessor = len(accessors) - 1

        joint_values = np.asarray(skin_payload.joints_0, dtype="<u2")
        joints_view = buffer.append(joint_values)
        buffer_views.append(
            {
                "buffer": 0,
                "byteOffset": joints_view.byte_offset,
                "byteLength": joints_view.byte_length,
                "target": self._ARRAY_BUFFER,
            }
        )
        accessors.append(
            {
                "bufferView": len(buffer_views) - 1,
                "componentType": self._UNSIGNED_SHORT,
                "count": int(len(joint_values)),
                "type": "VEC4",
            }
        )
        joints_accessor = len(accessors) - 1

        weight_values = np.asarray(skin_payload.weights_0, dtype="<f4")
        weights_view = buffer.append(weight_values)
        buffer_views.append(
            {
                "buffer": 0,
                "byteOffset": weights_view.byte_offset,
                "byteLength": weights_view.byte_length,
                "target": self._ARRAY_BUFFER,
            }
        )
        accessors.append(
            {
                "bufferView": len(buffer_views) - 1,
                "componentType": self._FLOAT,
                "count": int(len(weight_values)),
                "type": "VEC4",
            }
        )
        weights_accessor = len(accessors) - 1

        color_accessor = self._append_vertex_colors(mesh, buffer, buffer_views, accessors)
        inverse_bind_accessor = self._append_inverse_bind_matrices(
            rigged_mesh, buffer, buffer_views, accessors
        )

        nodes, root_node = _build_joint_nodes(rigged_mesh.skeleton.joints)
        mesh_node = len(nodes)
        nodes.append({"name": "rigged_mesh", "mesh": 0, "skin": 0})

        attributes: dict[str, int] = {
            "POSITION": position_accessor,
            "JOINTS_0": joints_accessor,
            "WEIGHTS_0": weights_accessor,
        }
        if color_accessor is not None:
            attributes["COLOR_0"] = color_accessor

        primitive: dict[str, object] = {
            "attributes": attributes,
            "indices": index_accessor,
            "material": 0,
        }
        gltf: dict[str, object] = {
            "asset": {
                "version": "2.0",
                "generator": "z-shift Phase 5 skinned GLB exporter",
            },
            "scene": 0,
            "scenes": [{"nodes": [root_node, mesh_node]}],
            "nodes": nodes,
            "skins": [
                {
                    "name": f"{rigged_mesh.skeleton.articulation_type.value}_skin",
                    "skeleton": root_node,
                    "joints": list(range(len(rigged_mesh.skeleton.joints))),
                    "inverseBindMatrices": inverse_bind_accessor,
                }
            ],
            "meshes": [{"name": "rigged_mesh", "primitives": [primitive]}],
            "materials": [
                {
                    "name": "vertex_color_material",
                    "doubleSided": True,
                    "pbrMetallicRoughness": {
                        "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                        "metallicFactor": 0.0,
                        "roughnessFactor": 1.0,
                    },
                }
            ],
            "accessors": accessors,
            "bufferViews": buffer_views,
            "buffers": [{"byteLength": buffer.padded_length}],
        }
        return gltf, buffer.bytes_padded()

    def _append_vertex_colors(
        self,
        mesh: trimesh.Trimesh,
        buffer: _BinaryBuffer,
        buffer_views: list[dict[str, object]],
        accessors: list[dict[str, object]],
    ) -> int | None:
        colors = getattr(mesh.visual, "vertex_colors", None)
        if colors is None or len(colors) != len(mesh.vertices):
            return None
        color_values = np.asarray(colors, dtype=np.uint8)
        if color_values.ndim != 2 or color_values.shape[1] not in (3, 4):
            return None
        if color_values.shape[1] == 3:
            alpha = np.full((color_values.shape[0], 1), 255, dtype=np.uint8)
            color_values = np.concatenate([color_values, alpha], axis=1)

        color_view = buffer.append(color_values)
        buffer_views.append(
            {
                "buffer": 0,
                "byteOffset": color_view.byte_offset,
                "byteLength": color_view.byte_length,
                "target": self._ARRAY_BUFFER,
            }
        )
        accessors.append(
            {
                "bufferView": len(buffer_views) - 1,
                "componentType": self._UNSIGNED_BYTE,
                "count": int(len(color_values)),
                "type": "VEC4",
                "normalized": True,
            }
        )
        return len(accessors) - 1

    def _append_inverse_bind_matrices(
        self,
        rigged_mesh: RiggedMesh,
        buffer: _BinaryBuffer,
        buffer_views: list[dict[str, object]],
        accessors: list[dict[str, object]],
    ) -> int:
        matrices = []
        for joint in rigged_mesh.skeleton.joints:
            matrix = np.eye(4, dtype=np.float32)
            matrix[:3, 3] = -np.asarray(joint.position, dtype=np.float32)
            matrices.append(matrix.flatten(order="F"))
        values = np.asarray(matrices, dtype="<f4")
        view = buffer.append(values)
        buffer_views.append(
            {
                "buffer": 0,
                "byteOffset": view.byte_offset,
                "byteLength": view.byte_length,
            }
        )
        accessors.append(
            {
                "bufferView": len(buffer_views) - 1,
                "componentType": self._FLOAT,
                "count": int(len(values)),
                "type": "MAT4",
            }
        )
        return len(accessors) - 1


@dataclass(frozen=True)
class _BufferSlice:
    byte_offset: int
    byte_length: int


class _BinaryBuffer:
    def __init__(self) -> None:
        self._data = bytearray()

    @property
    def padded_length(self) -> int:
        return _aligned_length(len(self._data))

    def append(self, values: np.ndarray) -> _BufferSlice:
        self._align()
        raw = values.tobytes(order="C")
        view = _BufferSlice(byte_offset=len(self._data), byte_length=len(raw))
        self._data.extend(raw)
        return view

    def bytes_padded(self) -> bytes:
        data = bytearray(self._data)
        data.extend(b"\x00" * (_aligned_length(len(data)) - len(data)))
        return bytes(data)

    def _align(self) -> None:
        self._data.extend(b"\x00" * (_aligned_length(len(self._data)) - len(self._data)))


def _build_joint_nodes(joints: list[Joint]) -> tuple[list[dict[str, object]], int]:
    joint_by_name = {joint.name: joint for joint in joints}
    joint_index = {joint.name: index for index, joint in enumerate(joints)}
    children: dict[str, list[int]] = {joint.name: [] for joint in joints}
    root_name = joints[0].name
    for joint in joints:
        if joint.parent and joint.parent in children:
            children[joint.parent].append(joint_index[joint.name])
        elif joint.parent is None:
            root_name = joint.name

    nodes: list[dict[str, object]] = []
    for joint in joints:
        position = np.asarray(joint.position, dtype=float)
        if joint.parent and joint.parent in joint_by_name:
            parent_position = np.asarray(joint_by_name[joint.parent].position, dtype=float)
            translation = position - parent_position
        else:
            translation = position
        node: dict[str, object] = {
            "name": joint.name,
            "translation": translation.astype(float).tolist(),
        }
        if children[joint.name]:
            node["children"] = children[joint.name]
        nodes.append(node)
    return nodes, joint_index[root_name]


def _write_glb(path: Path, gltf: dict[str, object], binary: bytes) -> None:
    json_payload = json.dumps(gltf, separators=(",", ":"), sort_keys=True).encode("utf-8")
    json_payload += b" " * (_aligned_length(len(json_payload)) - len(json_payload))
    binary += b"\x00" * (_aligned_length(len(binary)) - len(binary))

    total_length = 12 + 8 + len(json_payload) + 8 + len(binary)
    with path.open("wb") as file:
        file.write(struct.pack("<III", 0x46546C67, 2, total_length))
        file.write(struct.pack("<I4s", len(json_payload), b"JSON"))
        file.write(json_payload)
        file.write(struct.pack("<I4s", len(binary), b"BIN\x00"))
        file.write(binary)


def _aligned_length(length: int) -> int:
    return (length + 3) & ~3
