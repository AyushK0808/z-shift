"""Generate CO3D teddybear reconstruction/refinement/rigging quality outputs.

These values come from the local outputs produced by
``notebooks/co3d_teddybear_rig.ipynb``:

* out/new_raw_4cdcf66f3342/run_manifest.json
* out/new_raw_4cdcf66f3342/refinement_manifest.json
* out/rigging_new/skeleton.json
* out/rigging_new/skinning_weights.json

They are diagnostic quality scores and validity checks for the produced 3D
asset. They are not ground-truth reconstruction-accuracy metrics; CO3D
teddybear ground-truth geometry is not present in this workspace.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
# A re-run by bench/exp_b9_case_study.py keeps everything under one run
# directory (run_manifest.json, refinement_manifest.json, rig/*.json); point
# ZSHIFT_CASE_STUDY_DIR at it. Unset, the original notebook layout is read.
_CASE_STUDY_DIR = os.environ.get("ZSHIFT_CASE_STUDY_DIR")
if _CASE_STUDY_DIR:
    _RUN_DIR = Path(_CASE_STUDY_DIR)
    _RIG_DIR = _RUN_DIR / "rig"
else:
    _RUN_DIR = ROOT / "out" / "new_raw_4cdcf66f3342"
    _RIG_DIR = ROOT / "out" / "rigging_new"
RUN_MANIFEST = _RUN_DIR / "run_manifest.json"
REFINE_MANIFEST = _RUN_DIR / "refinement_manifest.json"
SKELETON_JSON = _RIG_DIR / "skeleton.json"
SKINNING_JSON = _RIG_DIR / "skinning_weights.json"
RIGGING_MANIFEST = _RIG_DIR / "rigging_manifest.json"

CSV_PATH = ROOT / "paper" / "outputs" / "co3d-teddybear-quality-scores.csv"
FIGURE_PATH = ROOT / "paper" / "figures" / "fig13-co3d-teddybear-quality.png"


def load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def clipped_score(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def main() -> int:
    run = load_json(RUN_MANIFEST)
    refine = load_json(REFINE_MANIFEST)
    skeleton = load_json(SKELETON_JSON)
    skinning = load_json(SKINNING_JSON)
    rigging = load_json(RIGGING_MANIFEST)

    outputs = run["outputs"]
    n_images = int(outputs["n_images"])
    directed_complete_pairs = n_images * (n_images - 1)
    pairing_coverage = outputs["n_pairs"] / directed_complete_pairs

    cell_reduction = 1.0 - refine["output_cell_count"] / refine["input_cell_count"]
    point_reduction = 1.0 - refine["output_point_count"] / refine["input_point_count"]
    non_manifold_cleanliness = 1.0 - min(
        refine["non_manifold_edge_count"] / max(refine["output_cell_count"], 1), 1.0
    )

    joints = skeleton["joints"]
    bones = skeleton["bones"]
    joint_names = {joint["name"] for joint in joints}
    parents_valid = all(
        joint["parent"] is None or joint["parent"] in joint_names for joint in joints
    )
    bones_valid = all(
        bone["parent_joint"] in joint_names and bone["child_joint"] in joint_names for bone in bones
    )
    root_valid = skeleton["root_joint"] in joint_names
    skeleton_graph_score = 1.0 if parents_valid and bones_valid and root_valid else 0.0

    weights = np.asarray(skinning["weights"], dtype=float)
    row_sums = weights.sum(axis=1)
    nonzero_counts = (weights > 1e-8).sum(axis=1)
    skinning_coverage = len(weights) / max(refine["output_point_count"], 1)
    weight_sum_mae = float(np.abs(row_sums - 1.0).mean())
    weight_normalization_score = clipped_score(1.0 - weight_sum_mae)
    nonnegative_weight_score = float(((weights >= -1e-8) & (weights <= 1.0 + 1e-8)).mean())
    max_influence_score = float((nonzero_counts <= skinning["max_influences"]).mean())

    warnings = rigging.get("warnings", [])
    warning_free_score = 1.0 if not warnings else 0.0

    rows = [
        (
            "reconstruction",
            "confidence_retention",
            outputs["conf_retention"],
            "0-1",
            "run_manifest outputs.conf_retention",
            "MASt3R points retained after confidence masking",
        ),
        (
            "reconstruction",
            "pairing_coverage",
            pairing_coverage,
            "0-1",
            "run_manifest outputs.n_pairs",
            f"{outputs['n_pairs']} / {directed_complete_pairs} directed complete pairs",
        ),
        (
            "reconstruction",
            "tsdf_success",
            0.0 if outputs["tsdf_fallback"] else 1.0,
            "0-1",
            "run_manifest outputs.tsdf_fallback",
            "1 means TSDF fallback was not used",
        ),
        (
            "refinement",
            "cell_reduction",
            cell_reduction,
            "0-1",
            "refinement_manifest cell counts",
            f"{refine['input_cell_count']} -> {refine['output_cell_count']} cells",
        ),
        (
            "refinement",
            "point_reduction",
            point_reduction,
            "0-1",
            "refinement_manifest point counts",
            f"{refine['input_point_count']} -> {refine['output_point_count']} points",
        ),
        (
            "refinement",
            "boundary_edge_cleanliness",
            1.0 if refine["boundary_edge_count"] == 0 else 0.0,
            "0-1",
            "refinement_manifest boundary_edge_count",
            f"{refine['boundary_edge_count']} boundary edges",
        ),
        (
            "refinement",
            "non_manifold_cleanliness",
            non_manifold_cleanliness,
            "0-1",
            "refinement_manifest non_manifold_edge_count/output_cell_count",
            f"{refine['non_manifold_edge_count']} non-manifold edges",
        ),
        (
            "refinement",
            "watertight_score",
            1.0 if refine["is_watertight"] else 0.0,
            "0-1",
            "refinement_manifest is_watertight",
            "1 means watertight output",
        ),
        (
            "rigging",
            "skeleton_graph_score",
            skeleton_graph_score,
            "0-1",
            "skeleton.json",
            f"{len(joints)} joints, {len(bones)} bones, root={skeleton['root_joint']}",
        ),
        (
            "rigging",
            "skinning_coverage",
            skinning_coverage,
            "0-1",
            "skinning_weights.json/refinement_manifest",
            f"{len(weights)} skinned vertices / {refine['output_point_count']} refined points",
        ),
        (
            "rigging",
            "weight_normalization_score",
            weight_normalization_score,
            "0-1",
            "skinning_weights.json",
            f"mean |sum(weights)-1| = {weight_sum_mae:.3e}",
        ),
        (
            "rigging",
            "nonnegative_weight_score",
            nonnegative_weight_score,
            "0-1",
            "skinning_weights.json",
            "fraction of weights in [0,1]",
        ),
        (
            "rigging",
            "max_influence_score",
            max_influence_score,
            "0-1",
            "skinning_weights.json",
            f"max influences <= {skinning['max_influences']}",
        ),
        (
            "rigging",
            "warning_free_score",
            warning_free_score,
            "0-1",
            "rigging_manifest warnings",
            "; ".join(warnings) if warnings else "no warnings",
        ),
        (
            "runtime",
            "reconstruction_seconds",
            run["total_stage_seconds"],
            "seconds",
            "run_manifest total_stage_seconds",
            "",
        ),
        (
            "runtime",
            "refinement_seconds",
            refine["total_stage_seconds"],
            "seconds",
            "refinement_manifest total_stage_seconds",
            "",
        ),
    ]

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["group", "metric", "value", "unit", "source", "note"])
        writer.writerows(rows)

    score_rows = [row for row in rows if row[3] == "0-1"]
    plt.rcParams.update(
        {
            "font.size": 6,
            "axes.titlesize": 6.5,
            "axes.labelsize": 5.5,
            "xtick.labelsize": 5,
            "ytick.labelsize": 5,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(3.5, 2.1))
    left, right = axes

    selected = [
        "confidence_retention",
        "cell_reduction",
        "point_reduction",
        "non_manifold_cleanliness",
        "watertight_score",
    ]
    labels = {
        "confidence_retention": "conf. retained",
        "cell_reduction": "cells removed",
        "point_reduction": "points removed",
        "non_manifold_cleanliness": "NM clean",
        "watertight_score": "watertight",
        "skeleton_graph_score": "skeleton graph",
        "skinning_coverage": "skinned verts",
        "weight_normalization_score": "weight norm.",
        "max_influence_score": "max influence",
        "warning_free_score": "no warnings",
    }
    selected_rows = [row for row in score_rows if row[1] in selected]
    left.barh([labels[row[1]] for row in selected_rows], [row[2] for row in selected_rows])
    left.set_xlim(0, 1)
    left.set_xlabel("score / fraction")
    left.set_title("(a) reconstruction/refinement")
    left.grid(axis="x", alpha=0.25, linewidth=0.4)

    rig_selected = [
        "skeleton_graph_score",
        "skinning_coverage",
        "weight_normalization_score",
        "max_influence_score",
        "warning_free_score",
    ]
    rig_rows = [row for row in score_rows if row[1] in rig_selected]
    right.barh([labels[row[1]] for row in rig_rows], [row[2] for row in rig_rows])
    right.set_xlim(0, 1)
    right.set_xlabel("score / fraction")
    right.set_title("(b) rigging validity")
    right.grid(axis="x", alpha=0.25, linewidth=0.4)

    for axis in axes:
        axis.tick_params(length=2, width=0.5, pad=1.5)
    figure.tight_layout(pad=0.3, w_pad=0.9)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=600)
    plt.close(figure)

    print(f"wrote {os.path.relpath(CSV_PATH, ROOT)}")
    print(f"wrote {os.path.relpath(FIGURE_PATH, ROOT)}")
    for group, metric, value, unit, _, note in rows:
        if unit == "0-1":
            print(f"{group:14s} {metric:28s} {value:.4f}  {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
