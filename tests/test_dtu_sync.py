"""B12: the rig-sync experiment runs on real image files and scores recovery against a clock."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from bench.exp_b12_dtu_sync import _another_camera, _scores, run


def _scene(root: Path, n_frames: int = 30) -> Path:
    """A folder of frames in which a bright block moves at an uneven speed."""
    images = root / "scan1"
    images.mkdir(parents=True)
    rng = np.random.default_rng(0)
    background = rng.integers(30, 60, (90, 160), dtype=np.uint8)
    position = 0
    for index in range(n_frames):
        frame = background.copy()
        position += 2 if index % 10 < 5 else 12
        frame[20:60, position % 120 : position % 120 + 30] = 220
        cv2.imwrite(str(images / f"rect_{index + 1:03d}_3_r5000.png"), frame)
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps({"tau": 2.0, "scenes": [{"name": "scan1", "image_dir": "scan1"}]}),
        encoding="utf-8",
    )
    return manifest


def test_motion_scores_follow_the_real_frames(tmp_path: Path) -> None:
    _scene(tmp_path)
    frames = [
        cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        for path in sorted((tmp_path / "scan1").iterdir())
    ]
    scores = _scores(frames)
    assert scores[0] == 1.0
    # Fast phases of the block's motion score higher than slow ones.
    assert max(scores[6:10]) > max(scores[1:5])


def test_another_camera_scores_differently_but_not_arbitrarily(tmp_path: Path) -> None:
    manifest = _scene(tmp_path)
    frames = [
        cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        for path in sorted((tmp_path / "scan1").iterdir())
    ]
    anchor = np.asarray(_scores(frames))
    other = np.asarray(_scores(_another_camera(frames, np.random.default_rng(1))))
    assert not np.array_equal(anchor, other)
    assert np.corrcoef(anchor[1:], other[1:])[0, 1] > 0.5
    assert manifest.exists()


def test_quick_run_writes_scored_rows(tmp_path: Path) -> None:
    manifest = _scene(tmp_path)
    writer = run(tmp_path / "out", manifest=manifest, quick=True)
    # 1 seed x 2 source counts x 3 offsets x 1 jitter
    assert len(writer) == 6
    row = writer.rows[0]
    assert row["data_source"] == "dtu_real_images_injected_clock"
    assert row["n_frames"] == 30
    assert {"true_offset_ms", "recovered_offset_ms", "offset_abs_error_ms", "sign_correct"} <= set(
        row
    )
