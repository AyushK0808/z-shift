from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spatial_ingestion.test_harness.runner import run_harness


if __name__ == "__main__":
    run_harness()

