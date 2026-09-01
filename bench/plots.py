"""Regenerate every figure from the CSVs in bench/results/.

Nothing here reads a number that is not in a committed CSV, so a figure can
never drift from the data behind it. Missing CSVs are skipped with a note
rather than raising, so a partial results set still plots what it has.

    uv run python -m bench.plots                # all figures
    uv run python -m bench.plots --only fig4    # one figure
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from bench import FIGURES_DIR, RESULTS_DIR  # noqa: E402
from bench.csvio import aggregate, read_rows  # noqa: E402

logger = logging.getLogger("bench.plots")

FIGURE_DPI = 200


def _ok(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("status", "ok") == "ok"]


def fig4_refine_scaling(results_dir: Path, figures_dir: Path) -> Path:
    """Fig. 4 - refinement cost on both axes, and on both together."""
    rows = _ok(read_rows("a1_refine_scaling", results_dir))
    triangles = [r for r in rows if r["ladder"] == "triangles"]
    components = [r for r in rows if r["ladder"] == "components"]
    interaction = [r for r in rows if r["ladder"] == "interaction"]

    # Stacked in a single column (rather than a row of 3) so the figure fits
    # a single-column \columnwidth placement in the paper.
    figure, (left, middle, right) = plt.subplots(3, 1, figsize=(4.0, 8.8))

    for source in dict.fromkeys(r["source"] for r in triangles):
        for smoothing in sorted({r["smoothing_iters"] for r in triangles}):
            for watertight in sorted({r["verify_watertight"] for r in triangles}):
                subset = [
                    r
                    for r in triangles
                    if r["source"] == source
                    and r["smoothing_iters"] == smoothing
                    and r["verify_watertight"] == watertight
                ]
                stats = aggregate(subset, ["rung"], "seconds")
                if not stats:
                    continue
                left.errorbar(
                    [key[0] for key in stats],
                    [value[0] for value in stats.values()],
                    yerr=[value[1] for value in stats.values()],
                    marker="o",
                    capsize=3,
                    linewidth=1.2,
                    markersize=4,
                    label=f"{source}, sm={smoothing}, wt={watertight}",
                )
    left.set_xlabel("input triangles (1 component)")
    left.set_ylabel("clean_mesh wall clock (s)")
    left.set_title("(a) cost vs triangle count")
    left.legend(fontsize=5)

    for smoothing in sorted({r["smoothing_iters"] for r in components}):
        for watertight in sorted({r["verify_watertight"] for r in components}):
            subset = [
                r
                for r in components
                if r["smoothing_iters"] == smoothing and r["verify_watertight"] == watertight
            ]
            stats = aggregate(subset, ["rung"], "seconds")
            if not stats:
                continue
            middle.errorbar(
                [key[0] for key in stats],
                [value[0] for value in stats.values()],
                yerr=[value[1] for value in stats.values()],
                marker="s",
                capsize=3,
                linewidth=1.2,
                markersize=4,
                label=f"sm={smoothing}, wt={watertight}",
            )
    middle.set_xlabel("connected components (~100k triangles)")
    middle.set_ylabel("clean_mesh wall clock (s)")
    middle.set_title("(b) cost vs component count")
    middle.legend(fontsize=6)

    # (c) both axes raised together, against the single-axis costs at the same
    # triangle count. The gap between the two curves is the interaction.
    if interaction:
        stats = aggregate(interaction, ["n_tri_nominal"], "seconds")
        right.errorbar(
            [key[0] for key in stats],
            [value[0] for value in stats.values()],
            yerr=[value[1] for value in stats.values()],
            marker="D",
            capsize=3,
            linewidth=1.4,
            markersize=5,
            color="C3",
            label="500-1000 components",
        )
    connected = [r for r in triangles if r["smoothing_iters"] == 0 and not r["verify_watertight"]]
    stats = aggregate(connected, ["rung"], "seconds")
    if stats:
        right.errorbar(
            [key[0] for key in stats],
            [value[0] for value in stats.values()],
            yerr=[value[1] for value in stats.values()],
            marker="o",
            capsize=3,
            linewidth=1.2,
            markersize=4,
            color="C0",
            label="1 component",
        )
    # Table I's reported observation, for direct comparison.
    right.axhline(7 * 60, color="black", linestyle="--", linewidth=1)
    right.text(6e4, 7 * 60 * 1.1, "Table I: ~7 min at ~2.6M tri", fontsize=6, va="bottom")
    right.set_xlabel("input triangles")
    right.set_ylabel("clean_mesh wall clock (s)")
    right.set_title("(c) both axes together (Table I config)")
    right.legend(fontsize=6, loc="lower right")

    for axis in (left, middle, right):
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.grid(True, which="both", alpha=0.3)

    # No figure.suptitle: at the narrow single-column figsize used here, a
    # one-line suptitle this long overflows the canvas and gets clipped: the
    # paper's LaTeX caption already carries this description.
    return _save(figure, figures_dir / "fig4_refine_scaling.png")


def fig5_refine_quality(results_dir: Path, figures_dir: Path) -> Path:
    """Fig. 5 - what refinement does to accuracy, to surfaces, and to volume.

    Plots the *deltas* rather than before/after pairs. Before and after
    overlap heavily once averaged over holes, fragments and seeds, and the
    finding is the sign of the change, not the absolute level.
    """
    rows = read_rows("a3_refine_quality", results_dir)
    grid = _ok([r for r in rows if r["part"] == "refinement"])
    baseline = [r for r in rows if r["part"] == "smoothing_baseline"]

    figure, (left, middle, right) = plt.subplots(1, 3, figsize=(14, 4.4))
    bases = list(dict.fromkeys(r["base"] for r in grid))
    colors = {base: f"C{index}" for index, base in enumerate(bases)}
    styles = {"object": "-", "room": "--"}

    for axis, key, title, ylabel in (
        (
            left,
            "chamfer_delta",
            "(a) accuracy: Chamfer change",
            "after - before  (negative is better)",
        ),
        (
            middle,
            "normal_consistency_delta",
            "(b) surfaces: normal-consistency change",
            "after - before  (positive is better)",
        ),
    ):
        for base in bases:
            for mode in ("object", "room"):
                subset = [r for r in grid if r["base"] == base and r["mode"] == mode]
                stats = aggregate(subset, ["noise_sigma"], key)
                if not stats:
                    continue
                axis.errorbar(
                    [k[0] for k in stats],
                    [v[0] for v in stats.values()],
                    yerr=[v[1] / max(v[2] ** 0.5, 1) for v in stats.values()],
                    marker="o" if mode == "object" else "s",
                    linestyle=styles[mode],
                    color=colors[base],
                    capsize=3,
                    markersize=4,
                    linewidth=1.3,
                    label=f"{base} / {mode}",
                )
        axis.axhline(0.0, color="black", linestyle=":", linewidth=1)
        axis.set_xlabel("corruption sigma (fraction of bbox diagonal)")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(True, alpha=0.3)
    left.legend(fontsize=6, ncol=2)

    for smoother in dict.fromkeys(r["smoother"] for r in baseline):
        subset = [
            r
            for r in baseline
            if r["smoother"] == smoother and r["volume_ratio_vs_clean"] is not None
        ]
        stats = aggregate(subset, ["smoothing_iters"], "volume_ratio_vs_clean")
        if not stats:
            continue
        right.errorbar(
            [k[0] for k in stats],
            [v[0] for v in stats.values()],
            yerr=[v[1] for v in stats.values()],
            marker="o",
            capsize=3,
            markersize=4,
            label=smoother,
        )
    right.axhline(1.0, color="black", linestyle=":", linewidth=1)
    right.set_xlabel("smoothing iterations")
    right.set_ylabel("volume / clean volume")
    right.set_title("(c) shrinkage: Taubin vs Laplacian")
    right.grid(True, alpha=0.3)
    right.legend(fontsize=7)

    # Error bars are standard errors of the mean, not standard deviations:
    # each point averages over hole counts, fragment counts and seeds, which
    # are deliberately different conditions rather than repeats of one.
    figure.suptitle(
        "Refinement effect vs corruption level (error bars: SEM over holes/fragments/seeds)",
        fontsize=9,
    )
    return _save(figure, figures_dir / "fig5_refine_quality.png")


def fig6_pair_gaps(results_dir: Path, figures_dir: Path) -> Path:
    """Fig. 6 - pair temporal gap by frame-selection variant."""
    rows = read_rows("a5_frame_budget", results_dir)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for axis, strategy in zip(axes, ("swin", "complete"), strict=False):
        subset = [r for r in rows if r["pairing_strategy"] == strategy]
        variants = list(dict.fromkeys(r["variant"] for r in subset))
        positions = range(len(variants))
        medians = [
            float(
                sum(r["pair_index_gap_median"] for r in subset if r["variant"] == v)
                / max(sum(1 for r in subset if r["variant"] == v), 1)
            )
            for v in variants
        ]
        p95s = [
            float(
                sum(r["pair_index_gap_p95"] for r in subset if r["variant"] == v)
                / max(sum(1 for r in subset if r["variant"] == v), 1)
            )
            for v in variants
        ]
        width = 0.38
        axis.bar([p - width / 2 for p in positions], medians, width, label="median gap")
        axis.bar([p + width / 2 for p in positions], p95s, width, label="95th pct gap")
        axis.set_xticks(list(positions))
        axis.set_xticklabels(variants, rotation=12, fontsize=7)
        axis.set_ylabel("pair gap (source frame indices)")
        axis.set_title(f"{strategy} pairing")
        axis.grid(True, axis="y", alpha=0.3)
        axis.legend(fontsize=7)

    figure.suptitle("Temporal adjacency of pairs, by frame-selection variant", fontsize=10)
    return _save(figure, figures_dir / "fig6_pair_gaps.png")


def fig7_sync_recovery(results_dir: Path, figures_dir: Path) -> Path:
    """Fig. 7 - offset recovery error and group formation rate."""
    rows = read_rows("a6_sync_offset", results_dir)
    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 4.5))

    for jitter in sorted({r["jitter_ms"] for r in rows}):
        subset = [r for r in rows if r["jitter_ms"] == jitter]
        stats = aggregate(subset, ["true_offset_ms"], "offset_abs_error_ms")
        left.errorbar(
            [k[0] for k in stats],
            [v[0] for v in stats.values()],
            yerr=[v[1] for v in stats.values()],
            marker="o",
            capsize=3,
            markersize=4,
            label=f"jitter sigma = {jitter} ms",
        )
    left.set_xlabel("injected offset (ms)")
    left.set_ylabel("mean |recovered - true| (ms)")
    left.set_title("(a) offset recovery error")
    left.grid(True, alpha=0.3)
    left.legend(fontsize=7)

    for noise in sorted({r["motion_noise"] for r in rows}):
        subset = [r for r in rows if r["motion_noise"] == noise]
        stats = aggregate(subset, ["tolerance_ms"], "group_formation_rate")
        right.errorbar(
            [k[0] for k in stats],
            [v[0] for v in stats.values()],
            yerr=[v[1] for v in stats.values()],
            marker="s",
            capsize=3,
            markersize=4,
            label=f"motion noise = {noise}",
        )
    right.set_xlabel("tolerance_ms")
    right.set_ylabel("groups formed / anchor frames")
    right.set_title("(b) group formation rate")
    right.grid(True, alpha=0.3)
    right.legend(fontsize=7)

    return _save(figure, figures_dir / "fig7_sync_recovery.png")


def fig_a7_sampling(results_dir: Path, figures_dir: Path) -> Path:
    """Budget concentration, adaptive vs matched uniform."""
    rows = read_rows("a7_motion_sampling", results_dir)
    shipped = [r for r in rows if r["threshold_variant"] == "shipped"]

    figure, axis = plt.subplots(figsize=(7, 4.5))
    for magnitude in dict.fromkeys(r["magnitude"] for r in shipped):
        for strategy in dict.fromkeys(r["strategy"] for r in shipped):
            subset = [
                r for r in shipped if r["magnitude"] == magnitude and r["strategy"] == strategy
            ]
            stats = aggregate(subset, ["motion_fraction"], "budget_concentration")
            axis.errorbar(
                [k[0] for k in stats],
                [v[0] for v in stats.values()],
                yerr=[v[1] for v in stats.values()],
                marker="o" if strategy == "adaptive" else "x",
                linestyle="-" if strategy == "adaptive" else "--",
                capsize=3,
                markersize=5,
                label=f"{strategy}, {magnitude}",
            )
    limits = [0.0, 1.0]
    axis.plot(
        limits,
        limits,
        color="black",
        linestyle=":",
        linewidth=1,
        label="chance (= motion fraction)",
    )
    axis.set_xlabel("fraction of clip in motion")
    axis.set_ylabel("fraction of selected frames inside motion windows")
    axis.set_title("Budget concentration at matched frame counts")
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=7)
    return _save(figure, figures_dir / "fig_a7_sampling.png")


def fig_a8_rigging(results_dir: Path, figures_dir: Path) -> Path:
    """Weight smoothness vs influence cap, and template rotation sensitivity."""
    rows = [
        row
        for row in read_rows("a8_rigging_quality", results_dir)
        if row["articulation"] != "static"
    ]
    # Stacked in a single column (rather than a row of 2) so the figure fits
    # a single-column \columnwidth placement in the paper.
    figure, (left, right) = plt.subplots(2, 1, figsize=(4.0, 7.5))

    for shape in dict.fromkeys(r["shape"] for r in rows):
        subset = [r for r in rows if r["shape"] == shape]
        stats = aggregate(subset, ["max_influences_requested"], "weight_smoothness_mean")
        left.errorbar(
            [k[0] for k in stats],
            [v[0] for v in stats.values()],
            yerr=[v[1] for v in stats.values()],
            marker="o",
            capsize=3,
            markersize=4,
            label=shape,
        )
    left.set_xscale("log", base=2)
    left.set_xlabel("max skinning influences")
    left.set_ylabel("mean |w_i - w_j| over mesh edges")
    left.set_title("(a) weight smoothness")
    left.grid(True, alpha=0.3)
    left.legend(fontsize=7)

    for articulation in dict.fromkeys(r["articulation"] for r in rows):
        subset = [r for r in rows if r["articulation"] == articulation]
        stats = aggregate(subset, ["input_rotation_deg"], "joint_displacement_mean")
        right.errorbar(
            [k[0] for k in stats],
            [v[0] for v in stats.values()],
            yerr=[v[1] for v in stats.values()],
            marker="s",
            capsize=3,
            markersize=4,
            label=articulation,
        )
    right.set_xlabel("input rotation about X (degrees)")
    right.set_ylabel("mean joint displacement (normalised units)")
    right.set_title("(b) template sensitivity to non-axis-aligned input")
    right.grid(True, alpha=0.3)
    right.legend(fontsize=7)

    return _save(figure, figures_dir / "fig_a8_rigging.png")


def fig_a2_stage_profile(results_dir: Path, figures_dir: Path) -> Path:
    """Stacked per-stage share of refinement time."""
    rows = read_rows("a2_stage_profile", results_dir)
    configurations = list(dict.fromkeys(r["configuration"] for r in rows))
    stages = list(dict.fromkeys(r["stage"] for r in rows))

    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    bottom = [0.0] * len(configurations)
    for stage in stages:
        values = []
        for configuration in configurations:
            subset = [
                r for r in rows if r["configuration"] == configuration and r["stage"] == stage
            ]
            values.append(sum(r["seconds"] for r in subset) / len(subset) if subset else 0.0)
        axis.bar(configurations, values, bottom=bottom, label=stage)
        bottom = [b + v for b, v in zip(bottom, values, strict=True)]

    axis.set_ylabel("mean seconds per clean_mesh call")
    axis.set_title("Per-stage refinement cost")
    axis.grid(True, axis="y", alpha=0.3)
    axis.legend(fontsize=7)
    return _save(figure, figures_dir / "fig_a2_stage_profile.png")


def fig8_fragment_removal(results_dir: Path, figures_dir: Path) -> Path:
    """Fig. 8 - what refinement does to stray fragments, before and after.

    Drawn as before -> after arrows rather than paired bars. The two modes see
    the *same* corrupted input, so a shared open marker carries the "before"
    and the arrow length is the whole finding; bars would have repeated the
    before column twice and forced a truncated axis on panels (b) and (d).
    """
    rows = _ok(
        [r for r in read_rows("a3_refine_quality", results_dir) if r["part"] == "refinement"]
    )
    if not rows:
        raise ValueError("no ok refinement rows in a3_refine_quality")

    injected = sorted({int(r["n_fragments_injected"]) for r in rows})
    positions = {count: index for index, count in enumerate(injected)}
    colors = {"object": "C1", "room": "C0"}
    offsets = {"object": -0.14, "room": 0.14}

    # The fragment-free cell count is what "all debris removed" looks like;
    # panel (b) is only readable against it.
    clean_cells = aggregate([r for r in rows if r["n_fragments_injected"] == 0], [], "out_cells")[
        ()
    ][0]

    # Stacked in a single column (rather than a 2x2 grid) so the figure fits
    # a single-column \columnwidth placement in the paper.
    figure, axes = plt.subplots(4, 1, figsize=(4.0, 9.1))
    panels = (
        (
            axes[0],
            "components_in",
            "components_out",
            "(a) stray fragments: connected components",
            "connected components",
            None,
        ),
        (
            axes[1],
            "in_cells",
            "out_cells",
            "(b) stray surface: cells retained",
            "mesh cells",
            clean_cells,
        ),
        (
            axes[2],
            "before_chamfer_l1",
            "after_chamfer_l1",
            "(c) cost of keeping them: Chamfer-L1",
            "Chamfer-L1 (bbox-diagonal units)",
            None,
        ),
        (
            axes[3],
            "boundary_edges_in",
            "boundary_edges_out",
            "(d) side effect: boundary edges",
            "boundary edges",
            None,
        ),
    )

    for axis, before_key, after_key, title, ylabel, reference in panels:
        for mode in ("object", "room"):
            subset = [r for r in rows if r["mode"] == mode]
            before = aggregate(subset, ["n_fragments_injected"], before_key)
            after = aggregate(subset, ["n_fragments_injected"], after_key)
            for key, (before_mean, _, _) in before.items():
                count = int(key[0])
                after_mean, after_std, after_n = after[key]
                x = positions[count] + offsets[mode]
                axis.annotate(
                    "",
                    xy=(x, after_mean),
                    xytext=(x, before_mean),
                    arrowprops={
                        "arrowstyle": "-|>",
                        "color": colors[mode],
                        "linewidth": 1.6,
                        "shrinkA": 0,
                        "shrinkB": 0,
                    },
                    zorder=3,
                )
                axis.errorbar(
                    x,
                    after_mean,
                    yerr=after_std / max(after_n**0.5, 1),
                    marker="o",
                    markersize=7,
                    color=colors[mode],
                    capsize=3,
                    linestyle="none",
                    zorder=4,
                )
                axis.plot(
                    x,
                    before_mean,
                    marker="o",
                    markersize=11,
                    markerfacecolor="none",
                    markeredgecolor="0.35",
                    markeredgewidth=1.5,
                    linestyle="none",
                    zorder=5,
                )

        if reference is not None:
            axis.axhline(
                reference,
                color="black",
                linestyle=":",
                linewidth=1,
                label="fragment-free mesh",
            )
            axis.legend(fontsize=7, loc="upper left")
        axis.set_xticks(list(positions.values()))
        axis.set_xticklabels([str(count) for count in injected])
        axis.set_xlabel("fragments injected")
        axis.set_ylabel(ylabel)
        axis.set_title(title, fontsize=10)
        axis.grid(True, alpha=0.3, axis="y")
        axis.set_xlim(-0.5, len(injected) - 0.5)

    handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor="none",
            markeredgecolor="0.35",
            markeredgewidth=1.5,
            markersize=11,
            label="before refinement (shared input)",
        ),
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            color="C1",
            markersize=7,
            label="after: object mode",
        ),
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            color="C0",
            markersize=7,
            label="after: room mode",
        ),
    ]
    axes[0].legend(handles=handles, fontsize=7, loc="upper left", framealpha=0.9)
    # No figure.suptitle: at the narrow single-column figsize used here, a
    # one-line suptitle this long overflows the canvas and gets clipped: the
    # paper's LaTeX caption already carries this description.
    return _save(figure, figures_dir / "fig8_fragment_removal.png")


FIGURES: dict[str, Callable[[Path, Path], Path]] = {
    "fig4": fig4_refine_scaling,
    "fig5": fig5_refine_quality,
    "fig6": fig6_pair_gaps,
    "fig7": fig7_sync_recovery,
    "fig_a7": fig_a7_sampling,
    "fig_a8": fig_a8_rigging,
    "fig_a2": fig_a2_stage_profile,
    "fig8": fig8_fragment_removal,
}


def _save(figure: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=FIGURE_DPI)
    plt.close(figure)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bench.plots", description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--only", nargs="*", default=None, choices=sorted(FIGURES))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    selected = args.only or sorted(FIGURES)
    written, skipped = [], []
    for name in selected:
        try:
            written.append(FIGURES[name](args.results_dir, args.figures_dir))
        except FileNotFoundError as exc:
            skipped.append((name, f"missing CSV: {exc}"))
        except (KeyError, IndexError, ValueError) as exc:
            skipped.append((name, f"{type(exc).__name__}: {exc}"))

    for path in written:
        print(f"wrote {path}")
    for name, reason in skipped:
        print(f"skipped {name}: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
