"""Regenerates fig12-refine-quality.png as a compact 2x2 panel figure, from
the same committed CSV that bench/plots.py's fig5_refine_quality() reads.

Rationale: this is the paper's copy of that figure, laid out for a
single-column \\columnwidth placement rather than for bench/'s own wide
1x3 layout. The version this replaced was three panels stacked vertically,
which printed ~5.7 in tall -- well over half a column for three small line
plots -- and was rendered several times wider than the column it lands in,
so every font was scaled down on the way onto the page.

Two changes fix both problems. The panels go into a 2x2 grid (a, b, c plus
the shared legend in the fourth cell, since (a) and (b) plot the same six
series), which is squarer and therefore shorter at a fixed width. And the
figure is rendered at figsize == the width it is actually placed at, so the
point sizes set below are the point sizes that reach the page.

This does not touch bench/plots.py or bench/figures/fig5_refine_quality.png,
which keep the 1x3 layout for their own, non-paper purposes. Do *not* copy
bench/figures/fig5_refine_quality.png over figures/fig12-refine-quality.png
-- that reintroduces the tall version and silently undoes this.

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

CSV_PATH = os.path.join("..", "bench", "results", "a3_refine_quality.csv")


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


rows = read_rows(CSV_PATH)
grid = [r for r in rows if r["part"] == "refinement" and r.get("status", "ok") == "ok"]
baseline = [r for r in rows if r["part"] == "smoothing_baseline"]

plt.rcParams.update(
    {
        "font.size": 6,
        "axes.titlesize": 6.5,
        "axes.labelsize": 5.5,
        "xtick.labelsize": 5,
        "ytick.labelsize": 5,
    }
)

figure, axes = plt.subplots(2, 2, figsize=(3.5, 2.75))
(left, middle), (right, legend_cell) = axes

bases = list(dict.fromkeys(r["base"] for r in grid))
colors = {base: f"C{index}" for index, base in enumerate(bases)}
styles = {"object": "-", "room": "--"}

handles, labels = [], []
for axis, key, title, ylabel in (
    (left, "chamfer_delta", "(a) Chamfer change", "after - before (neg. better)"),
    (middle, "normal_consistency_delta", "(b) normal-consistency", "after - before (pos. better)"),
):
    for base in bases:
        for mode in ("object", "room"):
            subset = [r for r in grid if r["base"] == base and r["mode"] == mode]
            stats = aggregate(subset, ["noise_sigma"], key)
            if not stats:
                continue
            bar = axis.errorbar(
                [k[0] for k in stats],
                [v[0] for v in stats.values()],
                yerr=[v[1] / max(v[2] ** 0.5, 1) for v in stats.values()],
                marker="o" if mode == "object" else "s",
                linestyle=styles[mode],
                color=colors[base],
                capsize=1.5,
                markersize=2.2,
                linewidth=0.9,
                label=f"{base} / {mode}",
            )
            if axis is left:
                handles.append(bar)
                labels.append(f"{base} / {mode}")
    axis.axhline(0.0, color="black", linestyle=":", linewidth=0.7)
    axis.set_xlabel("corruption sigma (frac. bbox diag.)")
    axis.set_ylabel(ylabel)
    axis.set_title(title)

for smoother in dict.fromkeys(r["smoother"] for r in baseline):
    subset = [
        r for r in baseline if r["smoother"] == smoother and r["volume_ratio_vs_clean"] is not None
    ]
    stats = aggregate(subset, ["smoothing_iters"], "volume_ratio_vs_clean")
    if not stats:
        continue
    right.errorbar(
        [k[0] for k in stats],
        [v[0] for v in stats.values()],
        yerr=[v[1] for v in stats.values()],
        marker="o",
        capsize=1.5,
        markersize=2.2,
        linewidth=0.9,
        label=smoother,
    )
right.axhline(1.0, color="black", linestyle=":", linewidth=0.7)
right.set_xlabel("smoothing iterations")
right.set_ylabel("volume / clean volume")
right.set_title("(c) Taubin vs Laplacian")
right.legend(fontsize=4.5, handlelength=1.2, borderpad=0.25, labelspacing=0.2, handletextpad=0.4)

# Fourth cell carries the legend (a) and (b) share, rather than a fourth plot:
# six series repeated inside two ~1.5 in panels would cover the data.
legend_cell.axis("off")
legend_cell.legend(
    handles,
    labels,
    loc="center",
    fontsize=5,
    frameon=False,
    handlelength=1.6,
    labelspacing=0.45,
    handletextpad=0.5,
    title="(a), (b) series",
    title_fontsize=5.5,
)

for axis in (left, middle, right):
    axis.grid(True, alpha=0.25, linewidth=0.4)
    axis.tick_params(length=2, width=0.5, pad=1.5)

figure.tight_layout(pad=0.25)
figure.savefig("figures/fig12-refine-quality.png", dpi=600)
plt.close(figure)
print("wrote figures/fig12-refine-quality.png (2x2, column width)")
