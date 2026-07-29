from __future__ import annotations

import numpy as np
import trimesh

from spatial_ingestion.auto_rigging.models import ArticulationType, Bone, Joint, Skeleton


class TemplateSkeletonFitter:
    """Fits simple skeleton templates to a mesh bounding box.

    This is the Phase 5 MVP baseline: predictable, deterministic, and good
    enough to validate downstream skinning/export contracts before adding
    learned rigging backends.

    Known limitation: templates assume a world-axis-aligned mesh with Y as the
    vertical axis. Inputs with a different up-axis or arbitrary rotation should
    be reoriented before this baseline fitter is used.
    """

    def fit(self, mesh: trimesh.Trimesh, articulation_type: ArticulationType) -> Skeleton:
        bounds = np.asarray(mesh.bounds, dtype=float)
        if bounds.shape != (2, 3):
            raise ValueError("mesh bounds must be shaped as (2, 3)")

        mins, maxs = bounds
        center = (mins + maxs) / 2.0
        size = np.maximum(maxs - mins, 1e-6)

        if articulation_type == ArticulationType.STATIC:
            return self._static(center)
        if articulation_type == ArticulationType.BIPED:
            return self._biped(mins, maxs, center, size)
        if articulation_type == ArticulationType.QUADRUPED:
            return self._quadruped(mins, maxs, center, size)
        if articulation_type == ArticulationType.WINGED:
            return self._winged(mins, maxs, center, size)
        raise ValueError(f"unsupported articulation type: {articulation_type}")

    @staticmethod
    def _joint(name: str, position: np.ndarray, parent: str | None = None) -> Joint:
        return Joint(
            name=name,
            position=(float(position[0]), float(position[1]), float(position[2])),
            parent=parent,
        )

    @classmethod
    def _bone(cls, parent: str, child: str) -> Bone:
        return Bone(name=f"{parent}_to_{child}", parent_joint=parent, child_joint=child)

    def _static(self, center: np.ndarray) -> Skeleton:
        root = self._joint("root", center)
        return Skeleton(
            articulation_type=ArticulationType.STATIC,
            joints=[root],
            bones=[],
            root_joint="root",
        )

    def _biped(
        self,
        mins: np.ndarray,
        maxs: np.ndarray,
        center: np.ndarray,
        size: np.ndarray,
    ) -> Skeleton:
        y_low, y_high = mins[1], maxs[1]
        x_left = center[0] - size[0] * 0.28
        x_right = center[0] + size[0] * 0.28
        z_mid = center[2]
        joints = [
            self._joint("hips", np.array([center[0], y_low + size[1] * 0.45, z_mid])),
            self._joint("spine", np.array([center[0], y_low + size[1] * 0.65, z_mid]), "hips"),
            self._joint("head", np.array([center[0], y_low + size[1] * 0.92, z_mid]), "spine"),
            self._joint("left_hand", np.array([x_left, y_low + size[1] * 0.62, z_mid]), "spine"),
            self._joint("right_hand", np.array([x_right, y_low + size[1] * 0.62, z_mid]), "spine"),
            self._joint("left_foot", np.array([x_left, y_low, z_mid]), "hips"),
            self._joint("right_foot", np.array([x_right, y_low, z_mid]), "hips"),
        ]
        bones = [
            self._bone("hips", "spine"),
            self._bone("spine", "head"),
            self._bone("spine", "left_hand"),
            self._bone("spine", "right_hand"),
            self._bone("hips", "left_foot"),
            self._bone("hips", "right_foot"),
        ]
        return Skeleton(articulation_type=ArticulationType.BIPED, joints=joints, bones=bones, root_joint="hips")

    def _quadruped(
        self,
        mins: np.ndarray,
        maxs: np.ndarray,
        center: np.ndarray,
        size: np.ndarray,
    ) -> Skeleton:
        x_front = maxs[0]
        x_back = mins[0]
        z_left = center[2] - size[2] * 0.30
        z_right = center[2] + size[2] * 0.30
        y_body = center[1] + size[1] * 0.10
        y_ground = mins[1]
        joints = [
            self._joint("body", np.array([center[0], y_body, center[2]])),
            self._joint("neck", np.array([x_front - size[0] * 0.20, y_body + size[1] * 0.15, center[2]]), "body"),
            self._joint("head", np.array([x_front, y_body + size[1] * 0.20, center[2]]), "neck"),
            self._joint("front_left_foot", np.array([x_front - size[0] * 0.15, y_ground, z_left]), "body"),
            self._joint("front_right_foot", np.array([x_front - size[0] * 0.15, y_ground, z_right]), "body"),
            self._joint("back_left_foot", np.array([x_back + size[0] * 0.15, y_ground, z_left]), "body"),
            self._joint("back_right_foot", np.array([x_back + size[0] * 0.15, y_ground, z_right]), "body"),
        ]
        bones = [
            self._bone("body", "neck"),
            self._bone("neck", "head"),
            self._bone("body", "front_left_foot"),
            self._bone("body", "front_right_foot"),
            self._bone("body", "back_left_foot"),
            self._bone("body", "back_right_foot"),
        ]
        return Skeleton(
            articulation_type=ArticulationType.QUADRUPED,
            joints=joints,
            bones=bones,
            root_joint="body",
        )

    def _winged(
        self,
        mins: np.ndarray,
        maxs: np.ndarray,
        center: np.ndarray,
        size: np.ndarray,
    ) -> Skeleton:
        y_body = center[1]
        joints = [
            self._joint("body", center),
            self._joint("head", np.array([center[0], maxs[1], center[2]]), "body"),
            self._joint("left_wing_tip", np.array([mins[0], y_body, center[2]]), "body"),
            self._joint("right_wing_tip", np.array([maxs[0], y_body, center[2]]), "body"),
            self._joint("tail", np.array([center[0], y_body - size[1] * 0.25, mins[2]]), "body"),
        ]
        bones = [
            self._bone("body", "head"),
            self._bone("body", "left_wing_tip"),
            self._bone("body", "right_wing_tip"),
            self._bone("body", "tail"),
        ]
        return Skeleton(articulation_type=ArticulationType.WINGED, joints=joints, bones=bones, root_joint="body")
