"""Run the Tier B GPU experiments on a local CUDA machine.

Local counterpart of notebooks/tier_b_gpu.ipynb, for a desktop or laptop GPU
instead of Kaggle. It runs in two lanes:

    smoke  every selected experiment on a small copy of the first scene --
           minutes, not publishable, exists to catch a crash before the long
           lane starts (and to download the MASt3R checkpoint once)
    full   every experiment once per scene, each (experiment, scene) in its
           own subprocess

One subprocess per scene is deliberate. An experiment writes its CSV only when
it finishes, so a crash or an out-of-memory error at scene five of a single
long process would throw away four scenes of finished work. Here each scene
leaves its own CSV under data/bench/tier_b/, a re-run skips scenes whose CSV
already exists, and the per-scene CSVs are merged into bench/results/<exp>.csv
at the end.

    uv run python -m bench.run_tier_b --manifest data/dtu/manifest.json
    uv run python -m bench.run_tier_b --manifest data/dtu/manifest.json --lane full
    uv run python -m bench.run_tier_b --manifest data/dtu/manifest.json --preflight-only
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bench import REPO_ROOT, RESULTS_DIR

logger = logging.getLogger("bench.tier_b")


@dataclass(frozen=True)
class Experiment:
    key: str
    exp_id: str
    module: str
    # Ceiling for one scene in the full lane. A run that exceeds it is a hang
    # or a thrash, not progress.
    timeout_min: int
    smoke_args: tuple[str, ...]
    # The full lane runs this many scenes at most (None = every scene).
    scene_limit: int | None = None


EXPERIMENTS: dict[str, Experiment] = {
    exp.key: exp
    for exp in (
        # Ground-truth accuracy of MASt3R-as-integrated (reviewer 2, item 1).
        Experiment(
            "b2",
            "b2_reconstruction_accuracy",
            "bench.exp_b2_reconstruction_accuracy",
            timeout_min=120,
            smoke_args=("--n-images", "4"),
        ),
        # Raw vs refined against GT: whether the cells refinement drops were
        # off-surface (reviewer 2, item 3). Four CPU refinements per scene, and
        # per-component filtering took 51 min on the 1.9M-triangle CO3D mesh
        # alone, so the ceiling is generous.
        Experiment(
            "b6",
            "b6_refinement_effect",
            "bench.exp_b6_refinement_effect",
            timeout_min=600,
            smoke_args=("--n-images", "4"),
        ),
        # Motion-rank vs capture order vs uniform at a 40-frame budget, plus a
        # budget sweep on the first scene (reviewer 2, item 1: retention order).
        Experiment(
            "b4",
            "b4_frame_budget_ablation",
            "bench.exp_b4_frame_budget_ablation",
            timeout_min=720,
            smoke_args=("--no-budget-sweep",),
            scene_limit=3,
        ),
        # complete vs swin pairing; optional, the 30-image complete cell alone
        # is 870 pairs.
        Experiment(
            "b3",
            "b3_pairing_ablation",
            "bench.exp_b3_pairing_ablation",
            timeout_min=600,
            smoke_args=("--frame-counts", "4"),
            scene_limit=3,
        ),
    )
}
# B6 last: its CPU refinement is the long, uncertain part, so the GPU results
# land first.
DEFAULT_EXPERIMENTS = ("b2", "b4", "b6")

# The smoke scene: the first few frames of the first scene, copied so that B4,
# which takes a scene's frames whole, also gets a small input.
SMOKE_FRAMES = 8

# MASt3R ViT-L weights alone are ~2.7 GB; below this much free VRAM at start
# the first 40-frame alignment is likely to run out of memory.
MIN_FREE_VRAM_GB = 5.0

_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001


def keep_awake(enabled: bool) -> None:
    """Stop Windows sleeping mid-run; a no-op elsewhere."""
    if sys.platform != "win32":
        return
    import ctypes

    flags = _ES_CONTINUOUS | (_ES_SYSTEM_REQUIRED if enabled else 0)
    ctypes.windll.kernel32.SetThreadExecutionState(flags)  # type: ignore[attr-defined]


# --- preflight ------------------------------------------------------------


def preflight(manifest_path: Path, work_dir: Path) -> dict[str, Any]:
    """Fail before hours of compute, not during them. Returns a machine report."""
    report: dict[str, Any] = {"python": sys.version.split()[0], "executable": sys.executable}
    problems: list[str] = []

    try:
        import torch
    except ImportError as exc:
        raise SystemExit(f"torch is not importable ({exc}); run `uv sync --dev` first") from exc

    report["torch"] = torch.__version__
    report["torch_cuda"] = torch.version.cuda
    if not torch.cuda.is_available():
        problems.append(
            "torch.cuda.is_available() is False. On Windows this usually means a CPU-only "
            "torch wheel; `uv sync --dev` installs the cu128 build pinned in pyproject.toml."
        )
    else:
        major, minor = torch.cuda.get_device_capability(0)
        arch = f"sm_{major}{minor}"
        free, total = torch.cuda.mem_get_info(0)
        report.update(
            gpu=torch.cuda.get_device_name(0),
            gpu_arch=arch,
            torch_arch_list=torch.cuda.get_arch_list(),
            vram_total_gb=round(total / 1024**3, 2),
            vram_free_gb=round(free / 1024**3, 2),
        )
        if arch not in torch.cuda.get_arch_list():
            # An RTX 50-series card is sm_120 and needs a CUDA 12.8+ build.
            problems.append(
                f"this torch build has no kernels for {arch} ({report['gpu']}); "
                f"it supports {torch.cuda.get_arch_list()}. Install a cu128 build."
            )
        else:
            probe = torch.randn(512, 512, device="cuda")
            float((probe @ probe).sum())
            torch.cuda.synchronize()
        if free / 1024**3 < MIN_FREE_VRAM_GB:
            logger.warning(
                "only %.1f GB of %.1f GB VRAM is free. Close browsers, games and other GPU "
                "apps before the full lane; MASt3R's weights alone take ~2.7 GB.",
                free / 1024**3,
                total / 1024**3,
            )

    try:
        import dust3r  # noqa: F401
        from mast3r.cloud_opt.sparse_ga import sparse_global_alignment  # noqa: F401
    except ImportError as exc:
        problems.append(
            f"MASt3R is not importable ({exc}). Run scripts/setup-mast3r.sh, and re-run it "
            "after every `uv sync`, which removes the editable mast3r/dust3r installs."
        )

    # Phase 2 export and Phase 3 refinement pull in pyvista/VTK, which need
    # system GL libraries on Linux; fail here rather than an hour into B6.
    try:
        import spatial_ingestion.reconstruction.pipeline  # noqa: F401
        import spatial_ingestion.refinement  # noqa: F401
    except ImportError as exc:
        problems.append(
            f"pipeline modules are not importable ({exc}). On Ubuntu/WSL a missing "
            "libGL.so.1 is fixed by `sudo apt install libgl1 libxrender1`."
        )

    from bench.tier_b_common import SceneSet

    scenes = SceneSet.from_manifest(manifest_path).scenes
    report["scenes"] = [scene.name for scene in scenes]
    if not scenes:
        problems.append(f"{manifest_path} lists no scenes")
    for scene in scenes:
        if scene.gt_path is None or not scene.gt_path.exists():
            problems.append(f"{scene.name}: ground truth missing ({scene.gt_path})")
        try:
            report.setdefault("n_images", {})[scene.name] = len(scene.image_paths())
        except FileNotFoundError as exc:
            problems.append(f"{scene.name}: {exc}")

    work_dir.mkdir(parents=True, exist_ok=True)
    report["disk_free_gb"] = round(shutil.disk_usage(work_dir).free / 1024**3, 1)
    if report["disk_free_gb"] < 30:
        logger.warning("only %.0f GB free under %s", report["disk_free_gb"], work_dir)

    for key, value in report.items():
        logger.info("preflight %-16s %s", key, value)
    if problems:
        for problem in problems:
            logger.error("preflight: %s", problem)
        raise SystemExit("preflight failed; nothing was run")
    return report


# --- scenes ---------------------------------------------------------------


def build_smoke_manifest(manifest_path: Path, work_dir: Path) -> Path:
    """A one-scene manifest holding the first SMOKE_FRAMES frames of scene one."""
    from bench.tier_b_common import SceneSet

    scene = SceneSet.from_manifest(manifest_path).scenes[0]
    smoke_dir = work_dir / "smoke_scene"
    image_dir = smoke_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    for path in scene.image_paths(SMOKE_FRAMES):
        target = image_dir / path.name
        if not target.exists():
            shutil.copy2(path, target)

    manifest = {
        "tau": scene.tau,
        "units": scene.units,
        "scenes": [
            {
                "name": f"smoke_{scene.name}",
                "image_dir": "images",
                "gt_path": str(scene.gt_path),
                "notes": f"first {SMOKE_FRAMES} frames of {scene.name}; smoke test only",
            }
        ],
    }
    path = smoke_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


# --- running --------------------------------------------------------------


def _stream(pipe: Any, sink: Any, echo: bool) -> None:
    for line in iter(pipe.readline, ""):
        sink.write(line)
        sink.flush()
        if echo:
            sys.stdout.write(line)
            sys.stdout.flush()
    pipe.close()


def run_one(
    argv: list[str], log_path: Path, timeout_min: float, env: dict[str, str]
) -> tuple[str, float]:
    """Run one experiment subprocess, tee its output, enforce a ceiling."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with log_path.open("a", encoding="utf-8") as sink:
        sink.write(f"\n$ {' '.join(argv)}\n")
        process = subprocess.Popen(  # noqa: S603 - argv is built here, not from input
            argv,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        reader = threading.Thread(target=_stream, args=(process.stdout, sink, True), daemon=True)
        reader.start()
        try:
            code = process.wait(timeout=timeout_min * 60)
            status = "ok" if code == 0 else f"exit {code}"
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            status = f"timeout after {timeout_min:.0f} min"
        reader.join(timeout=10)
    return status, round(time.perf_counter() - start, 1)


def child_env(work_dir: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    # Persist the subsampled-GT cache across runs and reboots instead of the
    # temp dir, so DTU's multi-million-point clouds are parsed once.
    env.setdefault("ZSHIFT_GT_CACHE_DIR", str(work_dir / "gt_cache"))
    return env


def merge_csvs(parts: list[Path], target: Path) -> int:
    """Concatenate per-scene CSVs, taking the union of their columns."""
    rows: list[dict[str, str]] = []
    fieldnames: list[str] = []
    for part in parts:
        with part.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for name in reader.fieldnames or []:
                if name not in fieldnames:
                    fieldnames.append(name)
            rows.extend(reader)
    for index, row in enumerate(rows):
        row["row_index"] = str(index)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def base_argv(
    exp: Experiment, manifest: Path, results_dir: Path, output_root: Path, args
) -> list[str]:
    argv = [
        sys.executable,
        "-u",
        "-m",
        exp.module,
        "--manifest",
        str(manifest),
        "--results-dir",
        str(results_dir),
        "--output-root",
        str(output_root),
        "--seed",
        str(args.seed),
        "--image-size",
        str(args.image_size),
        "--verbose",
    ]
    if args.with_scale:
        argv.append("--with-scale")
    return argv


def run_lane(lane: str, manifest: Path, args, work_dir: Path, status: dict[str, Any]) -> bool:
    from bench.tier_b_common import SceneSet

    env = child_env(work_dir)
    lane_dir = work_dir / lane
    scenes = [scene.name for scene in SceneSet.from_manifest(manifest).scenes]
    all_ok = True

    for key in args.experiments:
        exp = EXPERIMENTS[key]
        lane_scenes = scenes if lane == "smoke" else scenes[: exp.scene_limit or None]
        parts: list[Path] = []
        for scene_index, scene in enumerate(lane_scenes):
            scene_dir = lane_dir / exp.key / scene
            csv_path = scene_dir / f"{exp.exp_id}.csv"
            record_key = f"{lane}/{exp.key}/{scene}"
            if csv_path.exists() and not args.force:
                logger.info("skip %s: %s already exists", record_key, csv_path)
                parts.append(csv_path)
                status["runs"].setdefault(record_key, {"status": "ok (cached)"})
                continue

            argv = base_argv(exp, manifest, scene_dir, lane_dir / "outputs" / exp.key, args)
            argv += ["--scenes", scene]
            if lane == "smoke":
                argv += list(exp.smoke_args)
            elif exp.key == "b4" and scene_index > 0:
                # The quality-vs-budget curve needs one scene; B4 runs it on
                # whichever scene it sees first, which here is every scene.
                argv.append("--no-budget-sweep")

            timeout = exp.timeout_min * (0.5 if lane == "smoke" else 1)
            logger.info("=== %s starting (ceiling %d min) ===", record_key, timeout)
            run_status, seconds = run_one(
                argv, work_dir / "logs" / f"{lane}_{exp.key}.log", timeout, env
            )
            ok = run_status == "ok" and csv_path.exists()
            if run_status == "ok" and not csv_path.exists():
                run_status = "ok but wrote no CSV (scene skipped -- missing GT?)"
            status["runs"][record_key] = {"status": run_status, "seconds": seconds}
            logger.info("=== %s: %s in %.1f min ===", record_key, run_status, seconds / 60)
            write_status(work_dir, status)
            if ok:
                parts.append(csv_path)
            else:
                all_ok = False
                if lane == "smoke":
                    return False

        if lane == "full" and parts:
            target = (args.results_dir or RESULTS_DIR) / f"{exp.exp_id}.csv"
            n_rows = merge_csvs(parts, target)
            missing = len(lane_scenes) - len(parts)
            note = f" ({missing} scene(s) missing; re-run to fill them)" if missing else ""
            logger.info("merged %d rows from %d scene(s) -> %s%s", n_rows, len(parts), target, note)
            status["merged"][exp.exp_id] = {
                "path": str(target),
                "rows": n_rows,
                "scenes": len(parts),
            }
    return all_ok


def write_status(work_dir: Path, status: dict[str, Any]) -> None:
    (work_dir / "status.json").write_text(
        json.dumps(status, indent=2, default=str), encoding="utf-8"
    )


# --- headline summary -----------------------------------------------------


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value or "-")


def print_headlines(results_dir: Path) -> None:
    from bench.csvio import read_rows

    def rows_of(exp_id: str) -> list[dict[str, Any]]:
        path = results_dir / f"{exp_id}.csv"
        return read_rows(path) if path.exists() else []

    b2 = rows_of("b2_reconstruction_accuracy")
    if b2:
        print("\nB2 reconstruction accuracy (tau in GT units)")
        for row in b2:
            print(
                f"  {row['scene']:<10} F={_fmt(row.get('f_score'))} P={_fmt(row.get('precision'))} "
                f"R={_fmt(row.get('recall'))} chamfer={_fmt(row.get('chamfer_l1'), 3)} "
                f"align_rmse={_fmt(row.get('align_rmse'), 3)} tau={row.get('tau')}"
            )
    b6 = rows_of("b6_refinement_effect")
    if b6:
        print(
            "\nB6 raw vs refined (precision up with recall held = dropped cells were off-surface)"
        )
        for row in b6:
            if row.get("status") not in (None, "ok"):
                continue
            label = (
                row["stage"]
                if row["stage"] == "raw"
                else f"{row['mode']}/sm{row['smoothing_iters']}"
            )
            print(
                f"  {row['scene']:<10} {label:<12} F={_fmt(row.get('f_score'))} "
                f"P={_fmt(row.get('precision'))} R={_fmt(row.get('recall'))} "
                f"faces={row.get('n_faces')} nc={_fmt(row.get('normal_consistency'))}"
            )
    b4 = rows_of("b4_frame_budget_ablation")
    if b4:
        print("\nB4 frame budget (V1 motion order, V2 re-sorted, V3 uniform)")
        for row in b4:
            print(
                f"  {row['part']:<17} {row['scene']:<10} {row['variant']:<20} "
                f"budget={row['budget']:<3} F={_fmt(row.get('f_score'))} "
                f"chamfer={_fmt(row.get('chamfer_l1'))} "
                f"align_s={_fmt(row.get('stage_sparse_alignment_s'), 1)}"
            )
    if any(_fmt(row.get("f_score")) == "0.000" for row in b2 + b4):
        print(
            "\nAn F-score of 0 with a large align_rmse usually means a scale mismatch between "
            "MASt3R's metric output and the GT units; re-run with --with-scale and compare."
        )


# --- main -----------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bench.run_tier_b", description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--lane", choices=("all", "smoke", "full"), default="all")
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=sorted(EXPERIMENTS),
        default=list(DEFAULT_EXPERIMENTS),
    )
    parser.add_argument("--work-dir", type=Path, default=Path("data") / "bench" / "tier_b")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="where merged CSVs go (default bench/results)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--with-scale", action="store_true")
    parser.add_argument(
        "--force", action="store_true", help="re-run scenes that already have a CSV"
    )
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    work_dir = args.work_dir.resolve()
    manifest = args.manifest.resolve()

    report = preflight(manifest, work_dir)
    if args.preflight_only:
        return 0
    if args.image_size != 512:
        logger.warning(
            "image size %d deviates from the protocol's 512; say so in the paper", args.image_size
        )

    status: dict[str, Any] = {"preflight": report, "args": vars(args), "runs": {}, "merged": {}}
    keep_awake(True)
    start = time.perf_counter()
    try:
        if args.lane in ("all", "smoke"):
            smoke_manifest = build_smoke_manifest(manifest, work_dir)
            if not run_lane("smoke", smoke_manifest, args, work_dir, status):
                logger.error(
                    "smoke lane failed; see %s. The full lane was not started.", work_dir / "logs"
                )
                return 1
        ok = True
        if args.lane in ("all", "full"):
            ok = run_lane("full", manifest, args, work_dir, status)
    finally:
        keep_awake(False)
        status["elapsed_hours"] = round((time.perf_counter() - start) / 3600, 2)
        write_status(work_dir, status)

    print(
        f"\nTier B finished in {status['elapsed_hours']:.2f} h; "
        f"status -> {work_dir / 'status.json'}"
    )
    for key, record in status["runs"].items():
        print(f"  {key:<32} {record['status']}")
    if args.lane in ("all", "full"):
        print_headlines(args.results_dir or RESULTS_DIR)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
