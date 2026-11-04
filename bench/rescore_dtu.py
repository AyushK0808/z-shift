"""Re-score finished DTU runs (B2, B4, B6) from their saved outputs. No MASt3R, no GPU.

Scoring changed after these runs were made: the camera fit that places a
reconstruction in DTU's frame is now trimmed of lost views
(dtu_eval.camera_alignment). Before, one camera 150 mm off dragged the whole
placement, failed the run, and skewed its accuracy. A reconstruction is
already on disk for every row, as Phase 2 wrote it (point_cloud.ply and
cameras.json, and for B6 the raw mesh), so only the scoring has to be redone.

Every score column of a row is replaced; timing, memory and the env_* columns
of the run that made the reconstruction are kept, and rescored_git_commit /
rescored_utc say when and by which code the scores were recomputed. B6's
refined rows re-run refinement on the saved raw mesh, which is deterministic
and takes seconds.

    uv run python -m bench.rescore_dtu --exp b4 \\
        --manifest ~/zshift/dtu/manifest.json --outputs ~/zshift/tier_b/full/outputs/b4
"""

from __future__ import annotations

import argparse
import csv
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from bench import RESULTS_DIR
from bench.instrument import env_metadata
from bench.tier_b_common import (
    SceneSet,
    point_cloud_from_output,
    reconstruction_scale_for,
    score_against_gt,
)

EXP_IDS = {
    "b2": "b2_reconstruction_accuracy",
    "b4": "b4_frame_budget_ablation",
    "b6": "b6_refinement_effect",
}

logger = logging.getLogger(__name__)


def b2_output_dir(outputs: Path, row: dict[str, str]) -> Path:
    return outputs / row["scene"] / f"views{row['n_images']}"


def b4_output_dir(outputs: Path, row: dict[str, str]) -> Path:
    seed_root = outputs / f"seed{row['seed']}"
    if row["part"] == "variant_ablation" or row.get("reused_from") == "variant_ablation":
        return seed_root / row["scene"] / row["variant"]
    return seed_root / f"budget_{row['budget']}" / row["scene"] / row["variant"]


def _score(points: np.ndarray, scene: Any, output_dir: Path, seed: int) -> dict[str, Any]:
    return score_against_gt(
        points,
        np.empty((0, 3)),
        tau=scene.tau,
        seed=seed,
        reconstruction_scale=reconstruction_scale_for(scene.units),
        reference=scene.dtu_reference(),
        output_dir=output_dir,
    )


def rescore_rows(
    exp: str, rows: list[dict[str, str]], scenes: dict[str, Any], outputs: Path
) -> list[dict[str, Any]]:
    stamp = {
        "rescored_git_commit": env_metadata().get("git_commit", ""),
        "rescored_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    raw_scores: dict[str, dict[str, Any]] = {}
    raw_meshes: dict[str, Any] = {}
    out: list[dict[str, Any]] = []

    for row in rows:
        if row.get("status") not in (None, "", "ok") or row["scene"] not in scenes:
            out.append(row)
            continue
        scene = scenes[row["scene"]]
        seed = int(float(row.get("seed") or 0))

        if exp == "b6":
            from bench.exp_b6_refinement_effect import _mesh_points
            from spatial_ingestion.refinement import (
                MeshCleaningConfig,
                clean_mesh,
                load_mesh_file,
                to_trimesh,
            )

            output_dir = outputs / scene.name
            if scene.name not in raw_meshes:
                raw_meshes[scene.name] = load_mesh_file(output_dir / f"{scene.name}.glb")
            raw_poly = raw_meshes[scene.name]
            if row["stage"] == "raw":
                scores = _score(_mesh_points(to_trimesh(raw_poly), seed), scene, output_dir, seed)
                raw_scores[scene.name] = scores
            else:
                result = clean_mesh(
                    raw_poly,
                    MeshCleaningConfig(
                        mode=row["mode"],  # type: ignore[arg-type]
                        smoothing_iters=int(float(row["smoothing_iters"])),
                        verify_watertight=True,
                    ),
                )
                refined = to_trimesh(result["mesh"])
                if len(refined.faces) != int(float(row["n_faces"])):
                    raise RuntimeError(
                        f"{scene.name} {row['mode']}/{row['smoothing_iters']}: refinement gave "
                        f"{len(refined.faces)} faces, the run recorded {row['n_faces']}"
                    )
                scores = _score(_mesh_points(refined, seed), scene, output_dir, seed)
                raw = raw_scores[scene.name]
                scores["chamfer_delta_vs_raw"] = round(scores["chamfer_l1"] - raw["chamfer_l1"], 6)
                scores["f_score_delta_vs_raw"] = round(scores["f_score"] - raw["f_score"], 4)
        else:
            output_dir = (b2_output_dir if exp == "b2" else b4_output_dir)(outputs, row)
            points = point_cloud_from_output(output_dir, seed=seed)
            scores = _score(points, scene, output_dir, seed)

        before = row.get("pose_failure")
        out.append({**row, **scores, **stamp})
        logger.info(
            "%s %s: pose_failure %s -> %s, dtu_overall %s -> %s, outlier views [%s]",
            row["scene"],
            row.get("variant") or row.get("stage") or row.get("n_images"),
            before,
            scores["pose_failure"],
            row.get("dtu_overall"),
            scores["dtu_overall"],
            scores["camera_outlier_views"],
        )
    return out


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        fieldnames += [key for key in row if key not in fieldnames]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bench.rescore_dtu", description=__doc__)
    parser.add_argument("--exp", choices=sorted(EXP_IDS), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--outputs", type=Path, required=True, help="the experiment's --output-root"
    )
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    path = args.results_dir / f"{EXP_IDS[args.exp]}.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    scenes = {scene.name: scene for scene in SceneSet.from_manifest(args.manifest).scenes}
    write_rows(path, rescore_rows(args.exp, rows, scenes, args.outputs))
    print(f"re-scored {len(rows)} rows -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
