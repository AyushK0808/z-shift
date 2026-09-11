#!/usr/bin/env python3
"""Render the paper's full-width component architecture figure."""

from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle


OUT = Path(__file__).parent / "figures" / "fig1-overview.png"

INK = "#334155"
MUTED = "#64748b"
GROUP_EDGE = "#cbd5e1"
GROUP_FILL = "#ffffff"
STORE_FILL = "#fff4cc"
STORE_EDGE = "#b96800"
REJECT_FILL = "#fee2e2"

STAGES = {
    "ingest": ("#dbeafe", "#2563eb"),
    "recon": ("#ede9fe", "#7c3aed"),
    "refine": ("#ccfbf1", "#0d9488"),
    "deliver": ("#ffedd5", "#ea580c"),
}


def group(ax, x, y, width, height, number, title, stage):
    fill, accent = STAGES[stage]
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=GROUP_FILL,
            edgecolor=GROUP_EDGE,
            linewidth=0.9,
            zorder=0,
        )
    )
    header_height = 0.62
    ax.add_patch(
        FancyBboxPatch(
            (x, y + height - header_height),
            width,
            header_height,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=fill,
            edgecolor=accent,
            linewidth=1.0,
            zorder=1,
        )
    )
    # Covers the lower rounded corners of the header, leaving a straight divider.
    ax.add_patch(
        Rectangle(
            (x, y + height - header_height),
            width,
            0.10,
            facecolor=fill,
            edgecolor="none",
            zorder=2,
        )
    )
    ax.text(
        x + 0.15,
        y + height - header_height / 2,
        f"{number}  {title}",
        ha="left",
        va="center",
        fontsize=6.7,
        fontweight="bold",
        color=INK,
        zorder=4,
    )


def service(ax, x, y, width, height, text, stage, fontsize=5.8):
    fill, accent = STAGES[stage]
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.02,rounding_size=0.07",
            facecolor=fill,
            edgecolor=accent,
            linewidth=1.0,
            zorder=4,
        )
    )
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#172033",
        linespacing=1.12,
        zorder=5,
    )


def external(ax, x, y, width, height, text, fill="#e0f2fe", fontsize=5.6):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.02,rounding_size=0.07",
            facecolor=fill,
            edgecolor=INK,
            linewidth=1.0,
            zorder=4,
        )
    )
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#172033",
        linespacing=1.12,
        zorder=5,
    )


def cylinder(ax, x, y, width, height, text, fontsize=5.4):
    cap_height = min(0.17, height * 0.23)
    ax.add_patch(
        Rectangle(
            (x, y + cap_height / 2),
            width,
            height - cap_height,
            facecolor=STORE_FILL,
            edgecolor=STORE_EDGE,
            linewidth=1.0,
            zorder=4,
        )
    )
    for center_y in (y + cap_height / 2, y + height - cap_height / 2):
        ax.add_patch(
            Ellipse(
                (x + width / 2, center_y),
                width,
                cap_height,
                facecolor=STORE_FILL,
                edgecolor=STORE_EDGE,
                linewidth=1.0,
                zorder=4,
            )
        )
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#172033",
        linespacing=1.08,
        zorder=5,
    )


def arrow(ax, start, end, dashed=False):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=0.9,
            linestyle="--" if dashed else "-",
            color=MUTED,
            shrinkA=1,
            shrinkB=1,
            zorder=3,
        )
    )


def main():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "savefig.dpi": 300,
        }
    )
    fig, ax = plt.subplots(figsize=(7.1, 2.9), dpi=300)
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 7.25)
    ax.axis("off")

    backbone_y = 4.12

    external(
        ax,
        0.10,
        3.20,
        1.28,
        1.84,
        "Inputs\n\nImages\nVideo\nSynced views",
        fontsize=5.25,
    )

    group(ax, 1.68, 1.10, 2.62, 5.35, "1", "Media ingestion", "ingest")
    service(
        ax,
        2.00,
        3.48,
        1.98,
        1.28,
        "Ingestion service\n\nClassify · sample\nNormalize · sync",
        "ingest",
        fontsize=5.45,
    )
    cylinder(ax, 2.25, 1.72, 1.48, 0.86, "Live frame\nbuffer")
    arrow(ax, (2.99, 3.48), (2.99, 2.58))

    cylinder(
        ax,
        4.62,
        3.24,
        1.42,
        1.76,
        "Spatial\nschema\n\nFrames · timing\nIntrinsics",
        fontsize=5.15,
    )

    group(ax, 6.34, 1.10, 2.92, 5.35, "2", "Reconstruction", "recon")
    service(
        ax,
        6.67,
        3.48,
        2.26,
        1.28,
        "Reconstruction\n\nBudget · pairing\nMASt3R · TSDF\nPoint-map fallback",
        "recon",
        fontsize=5.1,
    )
    cylinder(ax, 6.75, 1.72, 1.28, 0.86, "Alignment\ncache")
    cylinder(ax, 8.18, 1.72, 0.66, 0.86, "Run\nmanifest", fontsize=4.7)
    arrow(ax, (7.39, 3.48), (7.39, 2.58))
    arrow(ax, (8.51, 3.48), (8.51, 2.58), dashed=True)

    cylinder(
        ax,
        9.57,
        3.24,
        1.48,
        1.76,
        "Geometry\nartifacts\n\nMesh\nPoint cloud",
        fontsize=5.05,
    )

    group(ax, 11.35, 1.10, 2.62, 5.35, "3", "Refinement", "refine")
    service(
        ax,
        11.67,
        3.48,
        1.98,
        1.28,
        "Refinement\n\nFilter · fill · smooth\nColor · validate",
        "refine",
        fontsize=5.25,
    )
    cylinder(ax, 11.92, 1.72, 1.48, 0.86, "Refinement\nmanifest")
    arrow(ax, (12.66, 3.48), (12.66, 2.58), dashed=True)

    group(ax, 14.27, 1.10, 3.63, 5.35, "4", "Delivery", "deliver")
    service(
        ax,
        14.60,
        3.48,
        1.20,
        1.28,
        "Compatibility\nrouter",
        "deliver",
        fontsize=5.0,
    )
    service(ax, 16.18, 4.64, 1.36, 0.72, "Editing · GLB", "deliver", fontsize=5.2)
    service(ax, 16.18, 3.76, 1.36, 0.72, "Viewing · PLY", "deliver", fontsize=5.2)
    external(
        ax,
        16.18,
        2.88,
        1.36,
        0.72,
        "Reject\nunsupported",
        fill=REJECT_FILL,
        fontsize=4.8,
    )
    arrow(ax, (15.80, backbone_y), (16.18, 5.00))
    arrow(ax, (15.80, backbone_y), (16.18, backbone_y))
    arrow(ax, (15.80, backbone_y), (16.18, 3.24))

    # Main data path: a single horizontal backbone with no crossed labels.
    arrow(ax, (1.38, backbone_y), (2.00, backbone_y))
    arrow(ax, (3.98, backbone_y), (4.62, backbone_y))
    arrow(ax, (6.04, backbone_y), (6.67, backbone_y))
    arrow(ax, (8.93, backbone_y), (9.57, backbone_y))
    arrow(ax, (11.05, backbone_y), (11.67, backbone_y))
    arrow(ax, (13.65, backbone_y), (14.60, backbone_y))

    ax.text(
        9.0,
        0.42,
        "Cylinders: buffered, cached, or persistent artifacts    "
        "Solid arrows: media / geometry    Dashed arrows: manifest records",
        ha="center",
        va="center",
        fontsize=5.6,
        color="#475569",
    )

    fig.subplots_adjust(left=0.006, right=0.994, top=0.985, bottom=0.035)
    fig.savefig(OUT, facecolor="white", bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
