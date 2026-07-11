from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import ExifTags, Image

from spatial_ingestion.metadata.schema import CameraIntrinsics


class ExifExtractor:
    def extract(self, image_path: Path) -> CameraIntrinsics:
        raw: dict[str, Any] = {}
        try:
            with Image.open(image_path) as image:
                exif = image.getexif()
                for tag_id, value in exif.items():
                    name = ExifTags.TAGS.get(tag_id, str(tag_id))
                    raw[name] = self._json_safe(value)
        except Exception:
            return CameraIntrinsics()

        return CameraIntrinsics(
            focal_length_mm=self._ratio_to_float(raw.get("FocalLength")),
            focal_length_35mm=self._ratio_to_float(raw.get("FocalLengthIn35mmFilm")),
            make=self._as_str(raw.get("Make")),
            model=self._as_str(raw.get("Model")),
            lens_model=self._as_str(raw.get("LensModel")),
            raw_exif=raw,
        )

    @staticmethod
    def _ratio_to_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            if hasattr(value, "numerator") and hasattr(value, "denominator"):
                return float(value.numerator) / float(value.denominator)
            return float(value)
        except (TypeError, ValueError, ZeroDivisionError):
            return None

    @staticmethod
    def _as_str(value: Any) -> str | None:
        return str(value).strip() if value is not None else None

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if isinstance(value, bytes):
            return value.decode(errors="ignore")
        if isinstance(value, tuple):
            return [ExifExtractor._json_safe(item) for item in value]
        if hasattr(value, "numerator") and hasattr(value, "denominator"):
            return ExifExtractor._ratio_to_float(value)
        try:
            if isinstance(value, (str, int, float, bool)) or value is None:
                return value
            return str(value)
        except Exception:
            return None

