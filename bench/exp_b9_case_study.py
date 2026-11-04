"""B9 - the CO3D teddybear end-to-end trace (Table II). Requires MASt3R + GPU.

Re-runs the paper's illustrative case study, which was a single traced run of
notebooks/co3d_teddybear_rig.ipynb on Kaggle: eight evenly spaced frames of a
CO3D teddybear capture through Phase 1 ingest, Phase 2 MASt3R (complete
pairing, seed 42), Phase 3 object-mode refinement and Phase 5 biped rigging.
Its run directory was not kept, so this rebuilds it on the local machine and
records every number Table II prints.

Refinement runs twice on the same raw mesh, once per component filter
(`filter_method`, as in A1/A2). The first, `split_bodies` on the CPU, is the
pipeline as submitted and is what Table II's 3069 s component-filtering row
measured; its output is what the rig is fitted to. The second is the batched
`labels` path that replaced it, timed on the same mesh.

Data: CO3D v2's single-sequence teddybear subset (CC BY-NC 4.0, Reizenstein
et al., ICCV 2021). Only the RGB frames are pulled from the 606 MB chunk, by
HTTP byte range (bench.fetch_dtu.HttpRangeFile), about 70 MB for all three
sequences in it.

    uv run python -m bench.exp_b9_case_study --data-dir data/co3d

Writes one row per filter_method to results/b9_co3d_case_study.csv. The run
directory (<output-root>/run) keeps run_manifest.json,
refinement_manifest.json and rig/*.json, the layout
paper/co3d-teddybear-quality-figure.py reads via ZSHIFT_CASE_STUDY_DIR.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from bench.csvio import ResultWriter
from bench.harness import experiment_parser, finish
from bench.instrument import peak_rss_mb
from bench.tier_b_common import clear_alignment_cache, require_cuda_device

EXP_ID = "b9_co3d_case_study"

CO3D_TEDDYBEAR_URL = "https://dl.fbaipublicfiles.com/co3dv2_231130/teddybear_001_singlesequence.zip"
# The sequence holding paper/outputs/teddy.jpeg, the case study's published
# input photograph: it matches this sequence's frame000001, which is also the
# first of the eight evenly spaced frames select_frames takes.
DEFAULT_SEQUENCE = "187_20215_38541"
# Table II: 8 views, complete pairing (56 directed pairs).
N_FRAMES = 8
SEED = 42
# Component filters to refine with, each on its own device. The first is the
# pipeline's own refinement, and the rig is fitted to its output.
FILTER_DEVICES: dict[str, str] = {"split_bodies": "cpu", "labels": "auto"}
FILTERS: tuple[str, ...] = ("split_bodies", "labels")

logger = logging.getLogger(__name__)


def fetch_frames(dest: Path, url: str = CO3D_TEDDYBEAR_URL) -> dict[str, Path]:
    """Every sequence's RGB frames under <dest>/<sequence>/images; returns them by name."""
    from bench.fetch_dtu import HttpRangeFile, _extract

    existing = {path.parent.name: path for path in dest.glob("*/images") if path.is_dir()}
    marker = dest / ".complete"
    if marker.exists() and existing:
        return existing

    logger.info("reading the CO3D teddybear chunk's directory: %s", url)
    with zipfile.ZipFile(HttpRangeFile(url)) as archive:
        for info in archive.infolist():
            parts = info.filename.split("/")
            if len(parts) != 4 or parts[2] != "images" or not parts[3].endswith(".jpg"):
                continue
            target = dest / parts[1] / "images" / parts[3]
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            _extract(archive, info, target)
    marker.write_text(url, encoding="utf-8")
    return {path.parent.name: path for path in dest.glob("*/images") if path.is_dir()}


def select_frames(image_dir: Path, count: int = N_FRAMES) -> list[Path]:
    """The notebook's curation: `count` evenly spaced frames, floor-indexed."""
    frames = sorted(image_dir.glob("*.jpg"))
    if not frames:
        raise FileNotFoundError(f"no frames under {image_dir}")
    picks = sorted(set(np.linspace(0, len(frames) - 1, count).astype(int).tolist()))
    return [frames[index] for index in picks]


def _stage_seconds(records: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for record in records:
        row[f"{prefix}{record['stage']}_s"] = record["seconds"]
        if record.get("peak_rss_mb") is not None:
            row[f"{prefix}{record['stage']}_peak_rss_mb"] = record["peak_rss_mb"]
    return row


def run(
    results_dir: Path | None = None,
    *,
    data_dir: Path,
    sequence: str = DEFAULT_SEQUENCE,
    n_frames: int = N_FRAMES,
    seed: int = SEED,
    image_size: int = 512,
    output_root: Path | None = None,
    rig: bool = True,
    filters: tuple[str, ...] = FILTERS,
) -> ResultWriter:
    from spatial_ingestion.auto_rigging.models import ArticulationType, AutoRigConfig
    from spatial_ingestion.final_pipeline.handoff import run_ingested_pipeline
    from spatial_ingestion.reconstruction.cli import DEFAULT_MODEL
    from spatial_ingestion.reconstruction.models import Mast3rRunParams
    from spatial_ingestion.refinement import MeshCleaningConfig, clean_mesh, load_mesh_file

    writer = ResultWriter(EXP_ID, results_dir)
    sequences = fetch_frames(data_dir)
    if sequence not in sequences:
        raise SystemExit(f"sequence {sequence} not in the chunk; have {sorted(sequences)}")

    root = (output_root or Path("data") / "bench" / EXP_ID).resolve()
    subset_dir = root / "inputs"
    if subset_dir.exists():
        shutil.rmtree(subset_dir)
    subset_dir.mkdir(parents=True)
    picked = select_frames(sequences[sequence], n_frames)
    inputs = []
    for index, source in enumerate(picked):
        target = subset_dir / f"frame_{index:03d}{source.suffix}"
        shutil.copy2(source, target)
        inputs.append(target)

    run_dir = root / "run"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    # The notebook's settings, except the frame count, which Table II's run
    # (56 pairs) shows was eight.
    mast3r_params = Mast3rRunParams(
        model_name=DEFAULT_MODEL,
        device=require_cuda_device(),
        image_size=image_size,
        pairing_strategy="complete",
        tsdf_thresh=0.2,
        min_conf_thr=1.5,
        seed=seed,
        deterministic=True,
    )

    def refinement(filter_method: str) -> MeshCleaningConfig:
        return MeshCleaningConfig(
            mode="object",
            smoothing_iters=15,
            pass_band=0.1,
            decimate_target_reduction=None,
            verify_watertight=True,
            component_filter=filter_method,  # type: ignore[arg-type]
            device=FILTER_DEVICES[filter_method],  # type: ignore[arg-type]
        )

    start = time.perf_counter()
    result = run_ingested_pipeline(
        inputs,
        mast3r_params=mast3r_params,
        output_path=run_dir / "mesh.glb",
        use_case="editing",
        input_type="image_folder",
        refinement_config=refinement(filters[0]),
        refined_output_path=run_dir / "mesh_refined.glb",
        rig=rig,
        rigging_config=(
            AutoRigConfig(
                articulation_type=ArticulationType.BIPED,
                max_skinning_influences=4,
                normalize_mesh=True,
                output_dir=run_dir / "rig",
                rigged_output_path=run_dir / "rig" / "rigged_mesh.glb",
            )
            if rig
            else None
        ),
        rigged_output_path=run_dir / "rig" / "rigged_mesh.glb" if rig else None,
        rig_output_dir=run_dir / "rig" if rig else None,
        deliverables_root=root / "deliverables",
    )
    pipeline_seconds = round(time.perf_counter() - start, 3)
    pipeline = getattr(result, "pipeline_result", result)

    manifest = json.loads(Path(pipeline.reconstruction_manifest_path).read_text(encoding="utf-8"))
    outputs = manifest.get("outputs", {})
    reconstruction_records = manifest.get("stage_timings", [])
    reconstruction_seconds = round(sum(r["seconds"] for r in reconstruction_records), 3)
    rig_row: dict[str, Any] = {}
    if rig and pipeline.skeleton_path:
        skeleton = json.loads(Path(pipeline.skeleton_path).read_text(encoding="utf-8"))
        skinning = json.loads(Path(pipeline.skinning_weights_path).read_text(encoding="utf-8"))
        rig_row = {
            "rig_joints": len(skeleton["joints"]),
            "rig_bones": len(skeleton["bones"]),
            "rig_skinned_vertices": len(skinning["weights"]),
            "rig_max_influences": skinning["max_influences"],
        }

    base_row = {
        "sequence": sequence,
        "frames": " ".join(path.name for path in picked),
        "n_images": len(inputs),
        "image_size": image_size,
        "seed": seed,
        "pairing_strategy": mast3r_params.pairing_strategy,
        "tsdf_fallback": manifest.get("tsdf_fallback"),
        "reconstruction_seconds": reconstruction_seconds,
        **{f"out_{key}": value for key, value in outputs.items()},
        **_stage_seconds(reconstruction_records, "stage_"),
    }

    def refine_row(filter_method: str, diagnostics: dict[str, Any]) -> dict[str, Any]:
        seconds = float(diagnostics.get("total_stage_seconds") or 0.0)
        return {
            "filter_method": filter_method,
            "refine_device": diagnostics.get("compute_device"),
            "refine_seconds": round(seconds, 3),
            "total_seconds": round(reconstruction_seconds + seconds, 3),
            "refine_in_cells": diagnostics.get("input_cell_count"),
            "refine_out_cells": diagnostics.get("output_cell_count"),
            "refine_out_points": diagnostics.get("output_point_count"),
            "is_watertight": diagnostics.get("is_watertight"),
            **_stage_seconds(diagnostics.get("stage_timings", []), "refine_"),
        }

    # The pipeline's own refinement; the rig was fitted to its output.
    first = pipeline.refinement_diagnostics or {}
    writer.add(
        **base_row,
        **refine_row(filters[0], first),
        pipeline_wall_seconds=pipeline_seconds,
        peak_rss_mb=peak_rss_mb(),
        **rig_row,
    )

    # Every other filter on the same raw mesh, for the timing comparison.
    raw_mesh = load_mesh_file(pipeline.raw_mesh_path)
    for filter_method in filters[1:]:
        other = clean_mesh(raw_mesh, refinement(filter_method))
        writer.add(
            **base_row,
            **refine_row(filter_method, other),
            same_output_as_first=(
                other["output_cell_count"] == first.get("output_cell_count")
                and other["output_point_count"] == first.get("output_point_count")
            ),
        )
    clear_alignment_cache(run_dir)
    return writer


def main(argv: list[str] | None = None) -> int:
    parser = experiment_parser(EXP_ID, __doc__ or "")
    parser.add_argument("--data-dir", type=Path, default=Path("data") / "co3d")
    parser.add_argument("--sequence", default=DEFAULT_SEQUENCE)
    parser.add_argument("--n-images", type=int, default=N_FRAMES)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--no-rig", action="store_true")
    parser.add_argument(
        "--filters",
        nargs="+",
        choices=sorted(FILTER_DEVICES),
        default=list(FILTERS),
        help="component filters to refine with; the first is the pipeline's own",
    )
    parser.set_defaults(seed=SEED)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    writer = run(
        results_dir=args.results_dir,
        data_dir=args.data_dir,
        sequence=args.sequence,
        n_frames=args.n_images,
        seed=args.seed,
        image_size=args.image_size,
        output_root=args.output_root,
        rig=not args.no_rig,
        filters=tuple(args.filters),
    )
    finish(writer)
    for row in writer.rows:
        print(
            f"  {row['filter_method']:<13} device={row.get('refine_device')} "
            f"pairs={row.get('out_n_pairs')} align_s={row.get('stage_sparse_alignment_s')} "
            f"refine_s={row.get('refine_seconds')} cells {row.get('refine_in_cells')}"
            f"->{row.get('refine_out_cells')} total_s={row.get('total_seconds')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
