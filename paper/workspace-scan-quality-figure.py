"""Score a workspace reconstruction against ground-truth 3D geometry.

This script produces the table CSV and paper figure for actual reconstruction
quality metrics:

* Chamfer-L1
* Hausdorff-95
* precision / recall / F-score at a stated tau
* optional normal consistency when both inputs are meshes

It deliberately requires ``--gt`` and ``--tau``. A reconstruction can be dense,
watertight, or visually plausible without being geometrically accurate; true
quality scores need reference geometry and a stated distance threshold.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import trimesh

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.gt_align import align_to_reference
from bench.metrics import chamfer_l1, hausdorff_95, normal_consistency, precision_recall_f

DEFAULT_RECONSTRUCTION = ROOT / "out" / "new_raw_4cdcf66f3342" / "point_cloud.ply"
DEFAULT_CSV = ROOT / "paper" / "outputs" / "workspace-reconstruction-quality.csv"
DEFAULT_FIGURE = ROOT / "paper" / "figures" / "fig13-workspace-reconstruction-quality.png"


def _load_geometry(path: Path) -> Any:
    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = [
            geom
            for geom in loaded.geometry.values()
            if isinstance(geom, trimesh.Trimesh) and len(geom.vertices)
        ]
        if not meshes:
            raise ValueError(f"{path} contains no mesh geometry")
        return trimesh.util.concatenate(meshes)
    return loaded


def _points_from_geometry(path: Path, max_points: int, seed: int) -> np.ndarray:
    geom = _load_geometry(path)
    if isinstance(geom, trimesh.points.PointCloud):
        points = np.asarray(geom.vertices, dtype=float)
    elif isinstance(geom, trimesh.Trimesh):
        if len(geom.faces):
            n = min(max_points, max(len(geom.faces), 1) * 3)
            points, _ = trimesh.sample.sample_surface(geom, n, seed=seed)
        else:
            points = np.asarray(geom.vertices, dtype=float)
    else:
        raise ValueError(f"unsupported geometry type in {path}: {type(geom)!r}")

    if len(points) > max_points:
        rng = np.random.default_rng(seed)
        points = points[rng.choice(len(points), size=max_points, replace=False)]
    if len(points) < 3:
        raise ValueError(f"{path} has fewer than 3 usable points")
    return np.asarray(points, dtype=float)


def _maybe_mesh(path: Path) -> trimesh.Trimesh | None:
    geom = _load_geometry(path)
    if isinstance(geom, trimesh.Trimesh) and len(geom.faces):
        return geom
    return None


def _fmt(value: float | int | str | bool | None) -> str:
    if value is None:
        return "--"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if math.isnan(value):
            return "--"
        if abs(value) >= 100:
            return f"{value:.2f}"
        if abs(value) >= 1:
            return f"{value:.4f}"
        return f"{value:.6f}"
    return str(value)


def score(args: argparse.Namespace) -> dict[str, Any]:
    reconstruction = _points_from_geometry(args.reconstruction, args.max_points, args.seed)
    gt = _points_from_geometry(args.gt, args.max_points, args.seed + 1)
    scaled = reconstruction * args.reconstruction_scale
    aligned, alignment = align_to_reference(scaled, gt, with_scale=args.with_scale)
    precision, recall, f_score = precision_recall_f(aligned, gt, args.tau)

    recon_mesh = _maybe_mesh(args.reconstruction)
    gt_mesh = _maybe_mesh(args.gt)
    normal_score = None
    if recon_mesh is not None and gt_mesh is not None:
        normal_score = normal_consistency(
            recon_mesh, gt_mesh, n=args.normal_samples, seed=args.seed
        )

    return {
        "reconstruction": str(args.reconstruction),
        "gt": str(args.gt),
        "units": args.units,
        "tau": args.tau,
        "with_scale": args.with_scale,
        "reconstruction_scale": args.reconstruction_scale,
        "chamfer_l1": chamfer_l1(aligned, gt),
        "hausdorff_95": hausdorff_95(aligned, gt),
        "precision": precision,
        "recall": recall,
        "f_score": f_score,
        "normal_consistency": normal_score,
        "n_reconstruction_points": len(aligned),
        "n_gt_points": len(gt),
        **alignment.as_row(),
    }


def write_csv(path: Path, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        ("F-score", metrics["f_score"], f"tau={metrics['tau']} {metrics['units']}"),
        ("Precision", metrics["precision"], f"tau={metrics['tau']} {metrics['units']}"),
        ("Recall", metrics["recall"], f"tau={metrics['tau']} {metrics['units']}"),
        ("Chamfer-L1", metrics["chamfer_l1"], metrics["units"]),
        ("Hausdorff-95", metrics["hausdorff_95"], metrics["units"]),
        ("Normal consistency", metrics["normal_consistency"], "mean |cos|"),
        ("Alignment RMSE", metrics["align_rmse"], metrics["units"]),
        ("Alignment scale", metrics["align_scale"], "factor"),
        ("Reconstruction points scored", metrics["n_reconstruction_points"], "points"),
        ("Ground-truth points scored", metrics["n_gt_points"], "points"),
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value", "unit_or_note"])
        for row in rows:
            writer.writerow(row)


def write_figure(path: Path, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 6,
            "axes.titlesize": 6.5,
            "axes.labelsize": 5.5,
            "xtick.labelsize": 5,
            "ytick.labelsize": 5,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(3.5, 1.85))
    left, right = axes

    score_labels = ["F-score", "precision", "recall"]
    score_values = [metrics["f_score"], metrics["precision"], metrics["recall"]]
    left.barh(score_labels, score_values, color=["#4C78A8", "#59A14F", "#F28E2B"])
    left.set_xlim(0, 1)
    left.set_xlabel(f"score at tau={metrics['tau']} {metrics['units']}")
    left.set_title("(a) threshold scores")
    left.grid(axis="x", alpha=0.25, linewidth=0.4)
    for idx, value in enumerate(score_values):
        left.text(min(value + 0.03, 0.94), idx, f"{value:.3f}", va="center", fontsize=5)

    distance_labels = ["Chamfer-L1", "Hausdorff-95", "align RMSE"]
    distance_values = [
        metrics["chamfer_l1"],
        metrics["hausdorff_95"],
        metrics["align_rmse"],
    ]
    right.bar(distance_labels, distance_values, color=["#9C755F", "#E15759", "#B07AA1"])
    right.set_ylabel(metrics["units"])
    right.set_title("(b) distance errors")
    right.grid(axis="y", alpha=0.25, linewidth=0.4)
    right.tick_params(axis="x", rotation=18)

    for axis in axes:
        axis.tick_params(length=2, width=0.5, pad=1.5)
    figure.tight_layout(pad=0.35, w_pad=0.8)
    figure.savefig(path, dpi=600)
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reconstruction", type=Path, default=DEFAULT_RECONSTRUCTION)
    parser.add_argument("--gt", type=Path, required=True, help="ground-truth mesh or point cloud")
    parser.add_argument("--tau", type=float, required=True, help="F-score threshold in --units")
    parser.add_argument("--units", default="scene units")
    parser.add_argument("--reconstruction-scale", type=float, default=1.0)
    parser.add_argument("--with-scale", action="store_true", help="allow ICP to fit scale")
    parser.add_argument("--max-points", type=int, default=500_000)
    parser.add_argument("--normal-samples", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    args = parser.parse_args(argv)

    if not args.reconstruction.exists():
        raise FileNotFoundError(args.reconstruction)
    if not args.gt.exists():
        raise FileNotFoundError(args.gt)

    metrics = score(args)
    write_csv(args.csv, metrics)
    write_figure(args.figure, metrics)

    print(f"wrote {args.csv}")
    print(f"wrote {args.figure}")
    print(
        "F={f_score:.4f} chamfer={chamfer_l1:.6f} hausdorff95={hausdorff_95:.6f} "
        "precision={precision:.4f} recall={recall:.4f}".format(**metrics)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
