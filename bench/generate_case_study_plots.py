"""Regenerates fig7-stage-timings.png and fig8-mesh-reduction.png in results/
from the case-study numbers reported in Table II of paper.tex. These are not
part of the paper (Table II already prints the same numbers in tabular form)
-- they're kept here as an optional visual cross-check. Update the literal
values below if the case study in paper.tex is ever re-run.

Requires matplotlib. Run from anywhere; writes into results/ next to this
script.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

os.chdir(os.path.dirname(os.path.abspath(__file__)))

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Georgia", "Times New Roman", "DejaVu Serif"]
plt.rcParams["axes.edgecolor"] = "#475569"
plt.rcParams["axes.labelcolor"] = "#0f172a"
plt.rcParams["text.color"] = "#0f172a"
plt.rcParams["xtick.color"] = "#334155"
plt.rcParams["ytick.color"] = "#334155"
plt.rcParams["axes.linewidth"] = 0.8

VIOLET_FILL, VIOLET_EDGE = "#ede9fe", "#7c3aed"  # reconstruction (matches fig3)
TEAL_FILL, TEAL_EDGE = "#ccfbf1", "#0d9488"  # refinement (matches fig4)
GREY = "#475569"

# ---------------------------------------------------------------------------
# Figure 7: per-stage wall time for the traced CO3D teddybear run (Table II)
# ---------------------------------------------------------------------------
stages = [
    ("Sparse alignment\n(MASt3R)", 1533.2, VIOLET_FILL, VIOLET_EDGE),
    ("Mesh export", 71.8, VIOLET_FILL, VIOLET_EDGE),
    ("Component\nfiltering", 3069.3, TEAL_FILL, TEAL_EDGE),
    ("Hole filling", 0.33, TEAL_FILL, TEAL_EDGE),
    ("Taubin\nsmoothing", 0.12, TEAL_FILL, TEAL_EDGE),
    ("Finalization", 0.85, TEAL_FILL, TEAL_EDGE),
    ("Color transfer", 0.18, TEAL_FILL, TEAL_EDGE),
    ("Watertightness\ncheck", 0.53, TEAL_FILL, TEAL_EDGE),
]
labels = [s[0] for s in stages][::-1]
times = [s[1] for s in stages][::-1]
fills = [s[2] for s in stages][::-1]
edges = [s[3] for s in stages][::-1]

fig, ax = plt.subplots(figsize=(3.4, 3.0), dpi=600)
y = range(len(labels))
bars = ax.barh(y, times, color=fills, edgecolor=edges, linewidth=1.1, height=0.62)
ax.set_xscale("log")
ax.set_xlim(0.05, 6000)
ax.set_yticks(list(y))
ax.set_yticklabels(labels, fontsize=7.3)
ax.set_xlabel("Wall time (s, log scale)", fontsize=8)
ax.tick_params(axis="x", labelsize=7)
ax.tick_params(axis="y", length=0)
for spine in ("top", "right", "left"):
    ax.spines[spine].set_visible(False)
ax.xaxis.grid(True, which="major", color="#e2e8f0", linewidth=0.7, zorder=0)
ax.set_axisbelow(True)

for rect, t in zip(bars, times, strict=True):
    label = f"{t:g} s" if t < 10 else f"{t:,.1f} s"
    ax.text(
        rect.get_width() * 1.15,
        rect.get_y() + rect.get_height() / 2,
        label,
        va="center",
        ha="left",
        fontsize=6.6,
        color=GREY,
    )

legend_handles = [
    Patch(facecolor=VIOLET_FILL, edgecolor=VIOLET_EDGE, label="Reconstruction"),
    Patch(facecolor=TEAL_FILL, edgecolor=TEAL_EDGE, label="Refinement"),
]
ax.legend(handles=legend_handles, loc="lower right", fontsize=6.8, frameon=False, borderaxespad=0.2)

fig.tight_layout(pad=0.4)
fig.savefig("results/fig7-stage-timings.png", dpi=600)
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 8: mesh size before vs. after refinement (cells and points)
# ---------------------------------------------------------------------------
groups = ["Cells\n(faces)", "Points\n(vertices)"]
before = [1.901916e6, 0.669182e6]
after = [0.411714e6, 0.111892e6]

fig, ax = plt.subplots(figsize=(3.4, 2.5), dpi=600)
x = range(len(groups))
w = 0.32
b1 = ax.bar(
    [i - w / 2 for i in x],
    before,
    width=w,
    color=VIOLET_FILL,
    edgecolor=VIOLET_EDGE,
    linewidth=1.1,
    label="Raw reconstruction",
)
b2 = ax.bar(
    [i + w / 2 for i in x],
    after,
    width=w,
    color=TEAL_FILL,
    edgecolor=TEAL_EDGE,
    linewidth=1.1,
    label="Refined (object mode)",
)

ax.set_xticks(list(x))
ax.set_xticklabels(groups, fontsize=8)
ax.set_ylabel("Count (millions)", fontsize=8)
ax.tick_params(axis="y", labelsize=7)
ax.yaxis.set_major_formatter(lambda v, pos: f"{v / 1e6:g}M" if v >= 1e6 else f"{v / 1e3:.0f}K")
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
ax.yaxis.grid(True, which="major", color="#e2e8f0", linewidth=0.7, zorder=0)
ax.set_axisbelow(True)

for rects, vals in ((b1, before), (b2, after)):
    for rect, v in zip(rects, vals, strict=True):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() * 1.02,
            f"{v / 1e6:.2f}M" if v >= 1e6 else f"{v / 1e3:.0f}K",
            ha="center",
            va="bottom",
            fontsize=6.6,
            color=GREY,
        )

ax.legend(fontsize=6.8, frameon=False, loc="upper right")
fig.tight_layout(pad=0.4)
fig.savefig("results/fig8-mesh-reduction.png", dpi=600)
plt.close(fig)

print("wrote results/fig7-stage-timings.png and results/fig8-mesh-reduction.png")
