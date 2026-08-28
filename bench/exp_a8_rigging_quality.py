"""A8 - auto-rigging weight quality.

Existing tests check only that weights sum to 1 and that influences are capped.
That is a schema check, not a quality result. This measures four things the
schema cannot see:

  weight smoothness   mean L1 distance between the weight vectors of
                      edge-adjacent vertices. Inverse distance to a
                      bounding-box joint is not smooth across joint
                      boundaries, and quantifying that is an honest limitation.
  rejection rate      fraction of vertices with no positive influence.
  deformation sanity  linear blend skinning with one joint rotated. Vertices
                      whose weight for that joint is exactly 0 must not move at
                      all; face-normal flips count as deformation damage.
  template sensitivity the pipeline warns that template fitting assumes an
                      axis-aligned Y-up mesh. Rotating the input and re-fitting
                      turns that warning string into a displacement number.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from bench.csvio import ResultWriter, mean_std
from bench.harness import experiment_parser, finish
from bench.meshes import base_mesh
from spatial_ingestion.auto_rigging.models import ArticulationType, AutoRigConfig
from spatial_ingestion.auto_rigging.pipeline import AutoRiggingPipeline

EXP_ID = "a8_rigging_quality"

MAX_INFLUENCES: tuple[int, ...] = (1, 2, 4, 8)
ROTATIONS_DEG: tuple[float, ...] = (0.0, 15.0, 30.0)
JOINT_ROTATIONS_DEG: tuple[float, ...] = (15.0, 45.0, 90.0)

logger = logging.getLogger(__name__)


def _shapes() -> dict[str, trimesh.Trimesh]:
    """Shapes whose proportions match each template's assumed body plan."""
    biped = trimesh.creation.capsule(height=1.6, radius=0.35, count=[32, 32])
    # Capsule is built along +Z; the templates assume Y is up.
    biped.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))

    quadruped = trimesh.creation.box(extents=(1.6, 0.6, 0.7))
    quadruped = quadruped.subdivide().subdivide().subdivide()

    winged = trimesh.creation.box(extents=(2.0, 0.25, 0.9))
    winged = winged.subdivide().subdivide().subdivide()

    return {
        "biped_capsule": biped,
        "quadruped_box": quadruped,
        "winged_slab": winged,
        "pointmap_sheet": base_mesh("pointmap_sheet"),
    }


def _weight_smoothness(mesh: trimesh.Trimesh, weights: np.ndarray) -> dict[str, float]:
    edges = np.asarray(mesh.edges_unique, dtype=int)
    if edges.size == 0:
        return {"weight_smoothness_mean": float("nan"), "weight_smoothness_p95": float("nan")}
    delta = np.abs(weights[edges[:, 0]] - weights[edges[:, 1]]).sum(axis=1)
    return {
        "weight_smoothness_mean": round(float(delta.mean()), 6),
        "weight_smoothness_p95": round(float(np.percentile(delta, 95)), 6),
    }


def _linear_blend_skin(
    vertices: np.ndarray,
    weights: np.ndarray,
    joint_positions: np.ndarray,
    joint_index: int,
    angle_deg: float,
) -> np.ndarray:
    """Rotate one joint about the Y axis through its own position; blend linearly."""
    angle = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotation = np.array([[cos_a, 0.0, sin_a], [0.0, 1.0, 0.0], [-sin_a, 0.0, cos_a]])
    pivot = joint_positions[joint_index]

    rotated = (vertices - pivot) @ rotation.T + pivot
    influence = weights[:, joint_index][:, None]
    # Every other joint contributes the identity transform, so the blend is
    # exactly `w * rotated + (1 - w) * rest`.
    return influence * rotated + (1.0 - influence) * vertices


def _deformation_stats(
    mesh: trimesh.Trimesh, weights: np.ndarray, joints: np.ndarray, angle_deg: float
) -> dict[str, Any]:
    if weights.shape[1] < 2:
        # The `static` template has a single root joint, so there is nothing to
        # rotate relative to anything else. Emit the full key set anyway, so
        # every row has the same columns.
        return {
            "deform_joint": -1,
            "deform_angle_deg": angle_deg,
            "deform_zero_weight_vertices": 0,
            "deform_zero_weight_max_displacement": 0.0,
            "deform_max_displacement": 0.0,
            "deform_flipped_faces": 0,
            "deform_flipped_face_frac": 0.0,
            "deform_watertight_after": bool(mesh.is_watertight),
        }

    # Pick the joint with the most non-zero influence, so the test actually moves geometry.
    joint_index = int(np.argmax((weights > 0).sum(axis=0)))
    vertices = np.asarray(mesh.vertices, dtype=float)
    deformed = _linear_blend_skin(vertices, weights, joints, joint_index, angle_deg)

    zero_weight = weights[:, joint_index] == 0.0
    zero_displacement = (
        float(np.linalg.norm(deformed[zero_weight] - vertices[zero_weight], axis=1).max())
        if zero_weight.any()
        else 0.0
    )

    before = trimesh.Trimesh(vertices=vertices, faces=mesh.faces, process=False)
    after = trimesh.Trimesh(vertices=deformed, faces=mesh.faces, process=False)
    dots = (np.asarray(before.face_normals) * np.asarray(after.face_normals)).sum(axis=1)

    return {
        "deform_joint": joint_index,
        "deform_angle_deg": angle_deg,
        "deform_zero_weight_vertices": int(zero_weight.sum()),
        "deform_zero_weight_max_displacement": round(zero_displacement, 9),
        "deform_max_displacement": round(
            float(np.linalg.norm(deformed - vertices, axis=1).max()), 6
        ),
        "deform_flipped_faces": int((dots < 0).sum()),
        "deform_flipped_face_frac": round(float((dots < 0).mean()), 6),
        "deform_watertight_after": bool(after.is_watertight),
    }


def _fit_joints(
    mesh: trimesh.Trimesh, articulation: ArticulationType
) -> tuple[np.ndarray, list[str]]:
    result = AutoRiggingPipeline().rig_mesh(
        mesh, AutoRigConfig(articulation_type=articulation), export_metadata=False
    )
    skeleton = result.rigged_mesh.skeleton
    return (
        np.asarray([joint.position for joint in skeleton.joints], dtype=float),
        [joint.name for joint in skeleton.joints],
    )


def _template_sensitivity(
    mesh: trimesh.Trimesh, articulation: ArticulationType, angle_deg: float
) -> dict[str, Any]:
    """Joint displacement caused by violating the axis-aligned Y-up assumption.

    Both fits happen in the pipeline's own normalised frame. The rotated fit is
    mapped back through the inverse rotation before comparison, so a template
    that tracked the shape correctly would score 0.
    """
    baseline, _ = _fit_joints(mesh, articulation)
    if angle_deg == 0.0:
        return {"joint_displacement_mean": 0.0, "joint_displacement_max": 0.0}

    rotation = trimesh.transformations.rotation_matrix(np.deg2rad(angle_deg), [1, 0, 0])
    rotated_mesh = mesh.copy()
    rotated_mesh.apply_transform(rotation)
    rotated_joints, _ = _fit_joints(rotated_mesh, articulation)

    inverse = np.asarray(rotation[:3, :3]).T
    realigned = rotated_joints @ inverse.T
    displacement = np.linalg.norm(realigned - baseline, axis=1)
    return {
        "joint_displacement_mean": round(float(displacement.mean()), 6),
        "joint_displacement_max": round(float(displacement.max()), 6),
    }


def _trial(
    shape_name: str,
    mesh: trimesh.Trimesh,
    articulation: ArticulationType,
    max_influences: int,
    rotation_deg: float,
    joint_angle_deg: float,
) -> dict[str, Any]:
    result = AutoRiggingPipeline().rig_mesh(
        mesh,
        AutoRigConfig(
            articulation_type=articulation, max_skinning_influences=max_influences
        ),
        export_metadata=False,
    )
    rigged = result.rigged_mesh
    weights = np.asarray(rigged.skinning.weights, dtype=float)
    joints = np.asarray(
        [joint.position for joint in rigged.skeleton.joints], dtype=float
    )

    # rig_mesh normalises internally; rebuild the same normalised mesh so edge
    # topology and joint positions live in one frame.
    normalized = mesh.copy()
    extents = normalized.extents
    scale = float(max(extents)) if len(extents) else 1.0
    normalized.apply_translation(-normalized.bounding_box.centroid)
    if scale > 0:
        normalized.apply_scale(1.0 / scale)

    row: dict[str, Any] = {
        "shape": shape_name,
        "articulation": articulation.value,
        "max_influences_requested": max_influences,
        "max_influences_effective": rigged.skinning.max_influences,
        "n_joints": len(rigged.skeleton.joints),
        "n_vertices": rigged.vertex_count,
        "input_rotation_deg": rotation_deg,
        "weight_sum_max_error": round(float(np.abs(weights.sum(axis=1) - 1.0).max()), 9),
        "rejection_rate": round(float((weights.max(axis=1) <= 0.0).mean()), 6),
        "mean_active_influences": round(float((weights > 0).sum(axis=1).mean()), 4),
    }
    row.update(_weight_smoothness(normalized, weights))
    row.update(_deformation_stats(normalized, weights, joints, joint_angle_deg))
    row.update(_template_sensitivity(mesh, articulation, rotation_deg))
    return row


def run(results_dir: Path | None = None, *, quick: bool = False, seed: int = 0) -> ResultWriter:
    writer = ResultWriter(EXP_ID, results_dir)
    shapes = _shapes()
    if quick:
        shapes = dict(list(shapes.items())[:1])
    articulations = (
        (ArticulationType.BIPED,) if quick else tuple(ArticulationType)
    )
    influences = MAX_INFLUENCES[:2] if quick else MAX_INFLUENCES
    rotations = ROTATIONS_DEG[:1] if quick else ROTATIONS_DEG
    joint_angles = JOINT_ROTATIONS_DEG[:1] if quick else JOINT_ROTATIONS_DEG

    for shape_name, mesh in shapes.items():
        for articulation in articulations:
            for max_influences in influences:
                for rotation_deg in rotations:
                    for joint_angle in joint_angles:
                        writer.add(
                            seed=seed,
                            **_trial(
                                shape_name,
                                mesh,
                                articulation,
                                max_influences,
                                rotation_deg,
                                joint_angle,
                            ),
                        )
        logger.info("%s done (%d rows)", shape_name, len(writer))
    return writer


def _summarise(writer: ResultWriter) -> None:
    print(
        f"  {'shape':<16}{'articulation':<12}{'k':>3}{'smooth':>9}{'reject':>8}"
        f"{'zeroW_disp':>12}{'flipped':>9}"
    )
    keys: list[tuple[Any, ...]] = []
    for row in writer.rows:
        key = (row["shape"], row["articulation"], row["max_influences_requested"])
        if key in keys:
            continue
        keys.append(key)
        group = [
            r
            for r in writer.rows
            if (r["shape"], r["articulation"], r["max_influences_requested"]) == key
        ]
        smooth, _, _ = mean_std([r["weight_smoothness_mean"] for r in group])
        reject, _, _ = mean_std([r["rejection_rate"] for r in group])
        zero_disp = max(r["deform_zero_weight_max_displacement"] for r in group)
        flipped, _, _ = mean_std([r["deform_flipped_face_frac"] for r in group])
        print(
            f"  {key[0]:<16}{key[1]:<12}{key[2]:>3}{smooth:>9.4f}{reject:>8.3f}"
            f"{zero_disp:>12.2e}{flipped:>9.4f}"
        )


def main(argv: list[str] | None = None) -> int:
    args = experiment_parser(EXP_ID, __doc__ or "").parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    writer = run(results_dir=args.results_dir, quick=args.quick, seed=args.seed)
    finish(writer)
    _summarise(writer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
