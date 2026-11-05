"""Print the headline numbers of the reviewer-response experiments (B10-B13) from their CSVs."""

from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path

from bench import RESULTS_DIR


def _rows(name: str) -> list[dict[str, str]]:
    path = RESULTS_DIR / f"{name}.csv"
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _num(row: dict[str, str], key: str) -> float | None:
    try:
        value = float(row[key])
    except (KeyError, ValueError):
        return None
    return None if value != value else value


def _mean(values: list[float | None]) -> str:
    kept = [v for v in values if v is not None]
    return f"{statistics.fmean(kept):.3f}" if kept else "-"


def b10() -> None:
    rows = _rows("b10_real_mesh_refinement")
    if not rows:
        return
    print("\nB10  refinement on real DTU meshes (mean over scans)")
    accuracy = defaultdict(list)
    for row in rows:
        if row["part"] == "accuracy" and row["status"] == "ok":
            accuracy[row["variant"]].append(row)
    print(f"  {'variant':<17}{'faces kept':>11}{'F@2mm':>9}{'overall mm':>12}{'normal cons.':>14}")
    for variant, group in accuracy.items():
        print(
            f"  {variant:<17}{_mean([_num(r, 'face_fraction_kept') for r in group]):>11}"
            f"{_mean([_num(r, 'f_score') for r in group]):>9}"
            f"{_mean([_num(r, 'dtu_overall') for r in group]):>12}"
            f"{_mean([_num(r, 'normal_consistency') for r in group]):>14}"
        )
    print(
        f"  {'filter':<14}{'scene':<9}{'status':<9}{'n':>3}"
        f"{'mean s':>10}{'sd s':>8}{'out cells':>11}"
    )
    for row in rows:
        if row["part"] == "timing":
            print(
                f"  {row['filter_method']:<14}{row['scene']:<9}{row['status']:<9}"
                f"{row['n_repeats']:>3}{row['seconds_mean']:>10}{row['seconds_sd']:>8}"
                f"{row['out_cells']:>11}"
            )


def b11() -> None:
    rows = _rows("b11_co3d_accuracy")
    if not rows:
        return
    print(
        "\nB11  CO3D teddybear vs CO3D's reference cloud "
        "(pseudo-reference; fractions of bbox diagonal)"
    )
    print(
        f"  {'run':<8}{'variant':<20}{'registr.':>9}{'accuracy':>10}"
        f"{'compl.':>8}{'F@1%':>7}{'F@2%':>7}"
    )
    for row in rows:
        print(
            f"  {row['run']:<8}{row['variant']:<20}{row['registration_chamfer_frac']:>9}"
            f"{row['accuracy_frac']:>10}{row['completeness_frac']:>8}"
            f"{row['f_score_1pct']:>7}{row['f_score_2pct']:>7}"
        )


def b12() -> None:
    rows = _rows("b12_dtu_sync")
    if not rows:
        return
    print("\nB12  sync on the real DTU rig images, injected clocks")
    print(
        f"  {'scan':<8}{'trials':>7}{'median err ms':>15}"
        f"{'mean err ms':>13}{'sign ok':>9}{'groups':>8}"
    )
    by_scene = defaultdict(list)
    for row in rows:
        by_scene[row["scene"]].append(row)
    for scene, group in by_scene.items():
        errors = [e for e in (_num(r, "offset_abs_error_ms") for r in group) if e is not None]
        sign = sum(r["sign_correct"] == "True" for r in group) / len(group)
        print(
            f"  {scene:<8}{len(group):>7}{statistics.median(errors):>15.1f}"
            f"{statistics.fmean(errors):>13.1f}{sign:>9.3f}"
            f"{_mean([_num(r, 'group_formation_rate') for r in group]):>8}"
        )


def b13() -> None:
    rows = _rows("b13_backend_swap")
    if not rows:
        return
    print("\nB13  second backend through the same artifacts (10 views)")
    print(
        f"  {'backend':<8}{'scan':<8}{'contract':>9}{'refine':>8}{'pose fail':>10}"
        f"{'overall mm':>11}{'F@2mm':>7}{'s':>7}"
    )
    for row in rows:
        if row["status"] != "ok":
            print(f"  {row['backend']:<8}{row['scene']:<8}{row['status']}")
            continue
        print(
            f"  {row['backend']:<8}{row['scene']:<8}{row['contract_ok']:>9}{row['refine_ok']:>8}"
            f"{row['pose_failure']:>10}{row['dtu_overall']:>11}{row['f_score']:>7}"
            f"{float(row['wall_seconds']):>7.0f}"
        )


def main() -> int:
    print(f"results: {Path(RESULTS_DIR)}")
    for section in (b10, b11, b12, b13):
        section()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
