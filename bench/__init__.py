"""Benchmark harness for the z-shift experiment protocol.

Every experiment module writes exactly one CSV into ``bench/results/`` and
nothing else, so every table and figure in the paper traces back to a file in
this tree rather than to a number typed by hand.

Layout mirrors the protocol:

``instrument``  stage timing + peak RSS (P1)
``metrics``     chamfer / hausdorff / F-score / normal consistency (P2)
``fixtures``    synthetic mesh + video generators (P3)
``meshes``      the mesh-size ladder shared by A1-A4
``csvio``       result-row accumulation and CSV emission
``exp_a*``      CPU-only experiments (no MASt3R required)
``exp_b*``      experiments that need MASt3R and a GPU
``plots``       regenerates every figure from the CSVs
"""

from __future__ import annotations

from pathlib import Path

BENCH_ROOT = Path(__file__).resolve().parent
REPO_ROOT = BENCH_ROOT.parent
RESULTS_DIR = BENCH_ROOT / "results"
FIGURES_DIR = BENCH_ROOT / "figures"

__all__ = ["BENCH_ROOT", "FIGURES_DIR", "REPO_ROOT", "RESULTS_DIR"]
