"""Shared CLI scaffolding so every experiment module runs the same way."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from bench.csvio import ResultWriter

__all__ = ["experiment_parser", "finish"]


def experiment_parser(exp_id: str, description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=f"bench.{exp_id}", description=description)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="override the directory the CSV is written to (default bench/results)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="run a reduced grid; for smoke-testing the harness, not for publication",
    )
    parser.add_argument("--seed", type=int, default=0, help="base seed for the run")
    parser.add_argument("-v", "--verbose", action="store_true", help="log progress per cell")
    return parser


def finish(writer: ResultWriter) -> Path:
    path = writer.write()
    logging.getLogger("bench").info("%s: %d rows -> %s", writer.exp_id, len(writer), path)
    print(f"{writer.exp_id}: {len(writer)} rows -> {path}")
    return path
