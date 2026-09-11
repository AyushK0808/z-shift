#!/usr/bin/env python3
"""Render the paper's Mermaid diagrams to print-resolution PNG files."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parent
DIAGRAM_DIR = PAPER_DIR / "diagrams"
FIGURE_DIR = PAPER_DIR / "figures"
MERMAID_CONFIG = PAPER_DIR / "mermaid-config.json"
PUPPETEER_CONFIG = PAPER_DIR / "puppeteer-config.json"


def main() -> None:
    npx = shutil.which("npx")
    if npx is None:
        raise SystemExit(
            "npx was not found. Install Node.js, then rerun this script."
        )

    sources = sorted(DIAGRAM_DIR.glob("*.mmd"))
    if not sources:
        raise SystemExit(f"No Mermaid sources found in {DIAGRAM_DIR}")

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for source in sources:
        output = FIGURE_DIR / f"{source.stem}.png"
        print(f"rendering {source.stem}", flush=True)
        subprocess.run(  # noqa: S603 -- executable and arguments are controlled here
            [
                npx,
                "-y",
                "@mermaid-js/mermaid-cli@11",
                "-i",
                str(source),
                "-o",
                str(output),
                "-c",
                str(MERMAID_CONFIG),
                "-p",
                str(PUPPETEER_CONFIG),
                "-b",
                "white",
                "-w",
                "1800",
                "-s",
                "4",
            ],
            cwd=PAPER_DIR,
            check=True,
        )

    print(f"done: rendered {len(sources)} diagram(s)")


if __name__ == "__main__":
    main()
