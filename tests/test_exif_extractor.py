from pathlib import Path

from PIL import Image
from PIL.TiffImagePlugin import IFDRational

from spatial_ingestion.batch_normalization.exif import ExifExtractor

# FocalLength, FocalLengthIn35mmFilm and LensModel are standard EXIF tags that
# live in the Exif SubIFD (IFD0 tag 0x8769 points to it), not in IFD0 itself.
# Every real camera/phone JPEG stores them there, so a fixture that skips the
# SubIFD does not exercise the path that matters.
_EXIF_SUBIFD = 0x8769
_FOCAL_LENGTH = 0x920A
_FOCAL_LENGTH_35MM = 0xA405
_LENS_MODEL = 0xA434


def _write_camera_jpeg(path: Path) -> None:
    image = Image.new("RGB", (16, 16), (200, 100, 50))
    exif = Image.Exif()
    exif[0x010F] = "Apple"  # Make
    exif[0x0110] = "iPhone 14"  # Model
    exif[_EXIF_SUBIFD] = {
        _FOCAL_LENGTH: IFDRational(24, 1),
        _FOCAL_LENGTH_35MM: IFDRational(35, 1),
        _LENS_MODEL: "iPhone 14 back camera",
    }
    image.save(path, exif=exif)


def test_extract_reads_focal_length_from_exif_subifd(tmp_path: Path) -> None:
    path = tmp_path / "photo.jpg"
    _write_camera_jpeg(path)

    intrinsics = ExifExtractor().extract(path)

    assert intrinsics.focal_length_mm == 24.0
    assert intrinsics.focal_length_35mm == 35.0
    assert intrinsics.make == "Apple"
    assert intrinsics.model == "iPhone 14"
    assert intrinsics.lens_model == "iPhone 14 back camera"


def test_extract_returns_empty_intrinsics_without_exif(tmp_path: Path) -> None:
    path = tmp_path / "no_exif.jpg"
    Image.new("RGB", (16, 16), (0, 0, 0)).save(path)

    intrinsics = ExifExtractor().extract(path)

    assert intrinsics.focal_length_35mm is None
    assert intrinsics.make is None
