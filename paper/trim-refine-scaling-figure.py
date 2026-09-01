"""Regenerates fig9-refine-scaling.png as a 2-panel figure (component count;
both axes together), dropping the triangle-count-alone panel that used to be
panel (a) in bench/plots.py's fig4_refine_scaling().

Rationale: paper.tex Table I already isolates triangle count as the weaker
driver in tabular form, so the triangle-count panel was pure duplication in
the paper context, even though bench/plots.py keeps the 3-panel version for
its own purposes (fig4 there stands on its own, without an equivalent
table). This script only changes the paper's copy of the figure -- it reads
the same committed CSV (bench/results/a1_refine_scaling.csv) and does not
touch bench/plots.py or bench/figures/fig4_refine_scaling.png.

Panels are renumbered (a) component count, (b) both axes together, matching
the new 2-panel caption in paper.tex.

Requires matplotlib. Run from anywhere; writes into paper/figures/ next to
this script.
"""

import csv
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))

CSV_PATH = os.path.join("..", "bench", "results", "a1_refine_scaling.csv")


def _coerce(value):
    if value is None or value == "":
        return None
    lowered = value.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    try:
        number = float(value)
    except ValueError:
        return value
    looks_integral = number.is_integer() and "." not in value and "e" not in lowered
    return int(number) if looks_integral else number


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return [{k: _coerce(v) for k, v in row.items()} for row in csv.DictReader(handle)]


def mean_std(values):
    clean = [
        float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))
    ]
    n = len(clean)
    if n == 0:
        return (float("nan"), float("nan"), 0)
    mean = sum(clean) / n
    if n == 1:
        return (mean, 0.0, 1)
    variance = sum((v - mean) ** 2 for v in clean) / (n - 1)
    return (mean, math.sqrt(variance), n)


def aggregate(rows, group_by, value_key):
    buckets = {}
    for row in rows:
        value = row.get(value_key)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        key = tuple(row.get(col) for col in group_by)
        buckets.setdefault(key, []).append(float(value))
    return {key: mean_std(vals) for key, vals in sorted(buckets.items(), key=repr)}


rows = [r for r in read_rows(CSV_PATH) if r.get("status", "ok") == "ok"]
triangles = [r for r in rows if r["ladder"] == "triangles"]
components = [r for r in rows if r["ladder"] == "components"]
interaction = [r for r in rows if r["ladder"] == "interaction"]

# Side by side, and rendered at the size it is actually printed at.
# Two things were wrong with the previous stacked version: it printed ~4.6 in
# tall at \columnwidth (about half a column for two small log-log plots), and
# the figure was rendered 7 in wide then scaled to a 3.5 in column, which
# halves every font -- a 6.5 pt label reaches the page at ~3.2 pt. Rendering
# at figsize == the placed width keeps the point sizes below honest.
plt.rcParams.update(
    {
        "font.size": 6,
        "axes.titlesize": 6.5,
        "axes.labelsize": 6,
        "xtick.labelsize": 5,
        "ytick.labelsize": 5,
        "lines.linewidth": 0.9,
        "lines.markersize": 2.5,
    }
)
figure, (left, right) = plt.subplots(1, 2, figsize=(3.5, 1.62))

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
        left.errorbar(
            [key[0] for key in stats],
            [value[0] for value in stats.values()],
            yerr=[value[1] for value in stats.values()],
            marker="s",
            capsize=1.5,
            linewidth=0.9,
            markersize=2.2,
            label=f"sm={smoothing}, wt={watertight}",
        )
left.set_xlabel("connected components")
left.set_ylabel("wall clock (s)")
left.set_title("(a) cost vs component count")
left.legend(fontsize=4.2, handlelength=1.2, borderpad=0.25, labelspacing=0.2, handletextpad=0.4)

if interaction:
    stats = aggregate(interaction, ["n_tri_nominal"], "seconds")
    right.errorbar(
        [key[0] for key in stats],
        [value[0] for value in stats.values()],
        yerr=[value[1] for value in stats.values()],
        marker="D",
        capsize=1.5,
        linewidth=1.0,
        markersize=2.8,
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
        capsize=1.5,
        linewidth=0.9,
        markersize=2.2,
        color="C0",
        label="1 component",
    )
right.axhline(7 * 60, color="black", linestyle="--", linewidth=0.8)
right.text(6e4, 7 * 60 * 0.85, "~7 min", fontsize=4.5, va="top")
right.set_xlabel("input triangles")
right.set_ylabel("wall clock (s)")
right.set_title("(b) both axes together")
right.legend(
    fontsize=4.2,
    loc="lower right",
    handlelength=1.2,
    borderpad=0.25,
    labelspacing=0.2,
    handletextpad=0.4,
)

for axis in (left, right):
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.grid(True, which="major", alpha=0.25, linewidth=0.4)
    axis.tick_params(length=2, width=0.5, pad=1.5)

figure.tight_layout(pad=0.25)
figure.savefig("figures/fig9-refine-scaling.png", dpi=600)
plt.close(figure)
print("wrote figures/fig9-refine-scaling.png (2-panel, side by side)")
