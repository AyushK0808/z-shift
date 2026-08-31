from __future__ import annotations

import logging

import numpy as np
import trimesh

from spatial_ingestion.auto_rigging.models import ArticulationType, Bone, Joint, Skeleton
from spatial_ingestion.auto_rigging.skeleton.mesh_analysis import (
    UP,
    cross_sections,
    extremal_cluster,
    foot_points,
    horizontal_band,
    limb_extrema,
    widest_horizontal_axis,
)

logger = logging.getLogger(__name__)


class TemplateSkeletonFitter:
    """Fits template skeletons using the mesh's own geometry.

    Anchors are taken from the volumetric centreline (per-slice centroids of
    the occupied voxels) and from limb extrema (where the surface actually
    protrudes), so the rig tracks the reconstruction instead of its bounding
    box. Every analysis has a bounding-box fallback, so no geometry-degenerate
    input crashes the fit: it degrades to the old deterministic fractions.
    """

    def fit(
        self,
        mesh: trimesh.Trimesh,
        articulation_type: ArticulationType,
        detailed: bool = False,
    ) -> Skeleton:
        bounds = mesh.bounds
        mins = bounds[0]
        maxs = bounds[1]
        center = mesh.centroid
        size = maxs - mins

        if articulation_type == ArticulationType.STATIC:
            return self._static(mesh, center)
        if articulation_type == ArticulationType.BIPED:
            return self._biped(mesh, mins, maxs, center, size, detailed=detailed)
        if articulation_type == ArticulationType.QUADRUPED:
            return self._quadruped(mesh, mins, maxs, center, size, detailed=detailed)
        if articulation_type == ArticulationType.WINGED:
            return self._winged(mesh, mins, maxs, center, size, detailed=detailed)
        raise ValueError(f"Unknown articulation type: {articulation_type}")

    def _static(self, mesh: trimesh.Trimesh, center: np.ndarray) -> Skeleton:
        return Skeleton(
            articulation_type=ArticulationType.STATIC,
            joints=[Joint(name="root", position=tuple(center.tolist()), parent=None)],
            bones=[],
            root_joint="root",
        )

    # ------------------------------------------------------------------
    # Biped
    # ------------------------------------------------------------------

    def _biped(
        self,
        mesh: trimesh.Trimesh,
        mins: np.ndarray,
        maxs: np.ndarray,
        center: np.ndarray,
        size: np.ndarray,
        detailed: bool = False,
    ) -> Skeleton:
        pts = np.asarray(mesh.vertices, dtype=float)
        profile = cross_sections(mesh)

        def c_at(fraction: float) -> np.ndarray:
            return profile.center_at(fraction)

        # Head: the centroid of the top 10% band (falls back to centreline).
        head = c_at(0.93)
        top_band = horizontal_band(pts, 0.90, 1.0)
        if len(top_band) >= min(8, len(pts)):
            head = top_band.mean(axis=0)

        # Hands: where the shoulder band protrudes most on each side.
        hand_l, hand_r = limb_extrema(mesh, lo_frac=0.45, hi_frac=0.78)
        # Feet: bottom-band clusters snapped to the floor.
        foot_l, foot_r = foot_points(mesh, lo_frac=0.0, hi_frac=0.14)

        if not detailed:
            joints = [
                self._joint("hips", c_at(0.45)),
                self._joint("spine", c_at(0.65), "hips"),
                self._joint("head", head, "spine"),
                self._joint("left_hand", hand_l, "spine"),
                self._joint("right_hand", hand_r, "spine"),
                self._joint("left_foot", foot_l, "hips"),
                self._joint("right_foot", foot_r, "hips"),
            ]
            bones = [
                self._bone("hips", "spine"),
                self._bone("spine", "head"),
                self._bone("spine", "left_hand"),
                self._bone("spine", "right_hand"),
                self._bone("hips", "left_foot"),
                self._bone("hips", "right_foot"),
            ]
            return Skeleton(
                articulation_type=ArticulationType.BIPED,
                joints=joints,
                bones=bones,
                root_joint="hips",
            )

        # --- Detailed 17-joint anatomical chain --------------------------
        pelvis = c_at(0.30)
        spine = c_at(0.46)
        chest = c_at(0.60)
        neck = c_at(0.78)

        # Shoulder sits partway from the chest centreline toward the hand.
        shoulder_l = chest + (hand_l - chest) * 0.45
        shoulder_r = chest + (hand_r - chest) * 0.45
        upper_arm_l = shoulder_l + (hand_l - shoulder_l) * 0.5
        upper_arm_r = shoulder_r + (hand_r - shoulder_r) * 0.5

        thigh_l = pelvis + (foot_l - pelvis) * 0.30
        thigh_r = pelvis + (foot_r - pelvis) * 0.30
        calf_l = thigh_l + (foot_l - thigh_l) * 0.5
        calf_r = thigh_r + (foot_r - thigh_r) * 0.5

        joints = [
            self._joint("pelvis", pelvis),
            self._joint("spine", spine, "pelvis"),
            self._joint("chest", chest, "spine"),
            self._joint("neck", neck, "chest"),
            self._joint("head", head, "neck"),
            self._joint("left_shoulder", shoulder_l, "chest"),
            self._joint("left_upper_arm", upper_arm_l, "left_shoulder"),
            self._joint("left_hand", hand_l, "left_upper_arm"),
            self._joint("right_shoulder", shoulder_r, "chest"),
            self._joint("right_upper_arm", upper_arm_r, "right_shoulder"),
            self._joint("right_hand", hand_r, "right_upper_arm"),
            self._joint("left_thigh", thigh_l, "pelvis"),
            self._joint("left_calf", calf_l, "left_thigh"),
            self._joint("left_foot", foot_l, "left_calf"),
            self._joint("right_thigh", thigh_r, "pelvis"),
            self._joint("right_calf", calf_r, "right_thigh"),
            self._joint("right_foot", foot_r, "right_calf"),
        ]
        bones = [
            self._bone("pelvis", "spine"),
            self._bone("spine", "chest"),
            self._bone("chest", "neck"),
            self._bone("neck", "head"),
            self._bone("chest", "left_shoulder"),
            self._bone("left_shoulder", "left_upper_arm"),
            self._bone("left_upper_arm", "left_hand"),
            self._bone("chest", "right_shoulder"),
            self._bone("right_shoulder", "right_upper_arm"),
            self._bone("right_upper_arm", "right_hand"),
            self._bone("pelvis", "left_thigh"),
            self._bone("left_thigh", "left_calf"),
            self._bone("left_calf", "left_foot"),
            self._bone("pelvis", "right_thigh"),
            self._bone("right_thigh", "right_calf"),
            self._bone("right_calf", "right_foot"),
        ]
        return Skeleton(
            articulation_type=ArticulationType.BIPED,
            joints=joints,
            bones=bones,
            root_joint="pelvis",
        )

    # ------------------------------------------------------------------
    # Quadruped
    # ------------------------------------------------------------------

    def _quadruped(
        self,
        mesh: trimesh.Trimesh,
        mins: np.ndarray,
        maxs: np.ndarray,
        center: np.ndarray,
        size: np.ndarray,
        detailed: bool = False,
    ) -> Skeleton:
        pts = np.asarray(mesh.vertices, dtype=float)
        long_axis = widest_horizontal_axis(mesh)  # front-back axis
        other_axis = 2 if long_axis == 0 else 0  # left-right axis
        profile = cross_sections(mesh, axis=long_axis)

        def q_at(fraction: float) -> np.ndarray:
            row = profile.center_at(fraction)
            return row

        y_low = mins[1]
        y_high = maxs[1]

        # Feet: bottom band split into four quadrant clusters.
        bottom = horizontal_band(pts, 0.0, 0.14)
        if len(bottom) >= 8:
            front_left, front_right, back_left, back_right = _quadruped_feet(
                bottom, long_axis, other_axis
            )
        else:
            front_left = np.array([maxs[long_axis], y_low, mins[other_axis]])
            front_right = np.array([maxs[long_axis], y_low, maxs[other_axis]])
            back_left = np.array([mins[long_axis], y_low, mins[other_axis]])
            back_right = np.array([mins[long_axis], y_low, maxs[other_axis]])

        # Head: the farthest protrusion toward the front (+long axis), raised
        # into the upper half of the body so it isn't the front paw.
        head = extremal_cluster(pts, long_axis, +1)
        head[UP] = max(float(head[UP]), y_low + 0.6 * (y_high - y_low))

        if not detailed:
            body = q_at(0.5)
            # Neck: front of the body, raised halfway up.
            neck = q_at(0.78)
            neck[UP] = min(neck[UP] + 0.18 * (y_high - y_low), y_high)
            joints = [
                self._joint("body", body),
                self._joint("neck", neck, "body"),
                self._joint("head", head, "neck"),
                self._joint("front_left_foot", front_left, "body"),
                self._joint("front_right_foot", front_right, "body"),
                self._joint("back_left_foot", back_left, "body"),
                self._joint("back_right_foot", back_right, "body"),
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

        # --- Detailed articulated quadruped ------------------------------
        body = q_at(0.5)
        chest = q_at(0.72)
        pelvis = q_at(0.28)
        front_p = chest + (front_left - chest) * 0.35
        front_r = chest + (front_right - chest) * 0.35
        back_p = pelvis + (back_left - pelvis) * 0.35
        back_r = pelvis + (back_right - pelvis) * 0.35

        neck = chest + (head - chest) * 0.4
        neck = neck.copy()
        neck[UP] = y_low + 0.7 * (y_high - y_low)

        joints = [
            self._joint("body", body),
            self._joint("chest", chest, "body"),
            self._joint("pelvis", pelvis, "body"),
            self._joint("neck", neck, "chest"),
            self._joint("head", head, "neck"),
            self._joint("front_shoulder_l", front_p, "chest"),
            self._joint("front_knee_l", front_p + (front_left - front_p) * 0.5, "front_shoulder_l"),
            self._joint("front_left_foot", front_left, "front_knee_l"),
            self._joint("front_shoulder_r", front_r, "chest"),
            self._joint(
                "front_knee_r", front_r + (front_right - front_r) * 0.5, "front_shoulder_r"
            ),
            self._joint("front_right_foot", front_right, "front_knee_r"),
            self._joint("back_hip_l", back_p, "pelvis"),
            self._joint("back_knee_l", back_p + (back_left - back_p) * 0.5, "back_hip_l"),
            self._joint("back_left_foot", back_left, "back_knee_l"),
            self._joint("back_hip_r", back_r, "pelvis"),
            self._joint("back_knee_r", back_r + (back_right - back_r) * 0.5, "back_hip_r"),
            self._joint("back_right_foot", back_right, "back_knee_r"),
        ]
        bones = [
            self._bone("body", "chest"),
            self._bone("body", "pelvis"),
            self._bone("chest", "neck"),
            self._bone("neck", "head"),
            self._bone("chest", "front_shoulder_l"),
            self._bone("front_shoulder_l", "front_knee_l"),
            self._bone("front_knee_l", "front_left_foot"),
            self._bone("chest", "front_shoulder_r"),
            self._bone("front_shoulder_r", "front_knee_r"),
            self._bone("front_knee_r", "front_right_foot"),
            self._bone("pelvis", "back_hip_l"),
            self._bone("back_hip_l", "back_knee_l"),
            self._bone("back_knee_l", "back_left_foot"),
            self._bone("pelvis", "back_hip_r"),
            self._bone("back_hip_r", "back_knee_r"),
            self._bone("back_knee_r", "back_right_foot"),
        ]
        return Skeleton(
            articulation_type=ArticulationType.QUADRUPED,
            joints=joints,
            bones=bones,
            root_joint="body",
        )

    # ------------------------------------------------------------------
    # Winged
    # ------------------------------------------------------------------

    def _winged(
        self,
        mesh: trimesh.Trimesh,
        mins: np.ndarray,
        maxs: np.ndarray,
        center: np.ndarray,
        size: np.ndarray,
        detailed: bool = False,
    ) -> Skeleton:
        pts = np.asarray(mesh.vertices, dtype=float)
        profile = cross_sections(mesh)
        body = profile.center_at(0.5)

        head = extremal_cluster(pts, UP, +1)
        wing_axis = widest_horizontal_axis(mesh)
        other_axis = 2 if wing_axis == 0 else 0
        band = horizontal_band(pts, 0.30, 0.70)
        left_wing_tip = (
            extremal_cluster(band, wing_axis, -1)
            if len(band) >= 8
            else np.array([mins[wing_axis], body[UP], center[other_axis]])
        )
        right_wing_tip = (
            extremal_cluster(band, wing_axis, +1)
            if len(band) >= 8
            else np.array([maxs[wing_axis], body[UP], center[other_axis]])
        )
        tail = extremal_cluster(pts, other_axis, -1)

        if not detailed:
            joints = [
                self._joint("body", body),
                self._joint("head", head, "body"),
                self._joint("left_wing_tip", left_wing_tip, "body"),
                self._joint("right_wing_tip", right_wing_tip, "body"),
                self._joint("tail", tail, "body"),
            ]
            bones = [
                self._bone("body", "head"),
                self._bone("body", "left_wing_tip"),
                self._bone("body", "right_wing_tip"),
                self._bone("body", "tail"),
            ]
            return Skeleton(
                articulation_type=ArticulationType.WINGED,
                joints=joints,
                bones=bones,
                root_joint="body",
            )

        left_base = body + (left_wing_tip - body) * 0.30
        left_mid = left_base + (left_wing_tip - left_base) * 0.5
        right_base = body + (right_wing_tip - body) * 0.30
        right_mid = right_base + (right_wing_tip - right_base) * 0.5

        joints = [
            self._joint("body", body),
            self._joint("head", head, "body"),
            self._joint("tail", tail, "body"),
            self._joint("left_wing_base", left_base, "body"),
            self._joint("left_wing_mid", left_mid, "left_wing_base"),
            self._joint("left_wing_tip", left_wing_tip, "left_wing_mid"),
            self._joint("right_wing_base", right_base, "body"),
            self._joint("right_wing_mid", right_mid, "right_wing_base"),
            self._joint("right_wing_tip", right_wing_tip, "right_wing_mid"),
        ]
        bones = [
            self._bone("body", "head"),
            self._bone("body", "tail"),
            self._bone("body", "left_wing_base"),
            self._bone("left_wing_base", "left_wing_mid"),
            self._bone("left_wing_mid", "left_wing_tip"),
            self._bone("body", "right_wing_base"),
            self._bone("right_wing_base", "right_wing_mid"),
            self._bone("right_wing_mid", "right_wing_tip"),
        ]
        return Skeleton(
            articulation_type=ArticulationType.WINGED,
            joints=joints,
            bones=bones,
            root_joint="body",
        )

    @staticmethod
    def _joint(name: str, position: np.ndarray, parent: str | None = None) -> Joint:
        pos = [float(v) for v in position]
        return Joint(name=name, position=(pos[0], pos[1], pos[2]), parent=parent)

    @staticmethod
    def _bone(parent: str, child: str) -> Bone:
        return Bone(name=f"{parent}_{child}", parent_joint=parent, child_joint=child)


def _quadruped_feet(
    bottom: np.ndarray, long_axis: int, other_axis: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Four floor-snapped foot clusters from the bottom band of a quadruped.

    Split by the sign of the long (front/back) axis and the other (left/right)
    horizontal axis, returning (front_left, front_right, back_left, back_right).
    """
    y = bottom[:, UP]

    def cluster(sign_long: int, sign_other: int) -> np.ndarray:
        mask = (sign_long * bottom[:, long_axis] >= 0) & (sign_other * bottom[:, other_axis] >= 0)
        if not mask.any():
            mask = sign_long * bottom[:, long_axis] >= 0
        foot = bottom[mask].mean(axis=0)
        foot[UP] = float(y[mask].min())
        return foot

    fl, fr = cluster(+1, +1), cluster(+1, -1)
    bl, br = cluster(-1, +1), cluster(-1, -1)
    # Convention: the foot with the smaller "other-axis" value is called *left*.
    front_left, front_right = (fl, fr) if fl[other_axis] < fr[other_axis] else (fr, fl)
    back_left, back_right = (bl, br) if bl[other_axis] < br[other_axis] else (br, bl)
    return front_left, front_right, back_left, back_right
