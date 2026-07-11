from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


class FFmpegTools:
    """Thin boundary around FFmpeg/ffprobe so callers do not depend on CLI details."""

    def __init__(self, ffprobe_binary: str = "ffprobe") -> None:
        self._ffprobe = ffprobe_binary

    def probe(self, video_path: Path) -> dict[str, Any]:
        if shutil.which(self._ffprobe) is None:
            return {"available": False}

        command = [
            self._ffprobe,
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(video_path),
        ]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=20,
        )
        if result.returncode != 0:
            return {"available": True, "error": result.stderr.strip()}
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError:
            payload = {"raw": result.stdout}
        payload["available"] = True
        return payload

