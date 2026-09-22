"""Stage 01 — NORMALIZE. Get every accepted format into one RGBA canvas.

Two things here are load-bearing and easy to get wrong:

1. The file type is sniffed from magic bytes, never from the extension. A .png
   that is actually a PDF is a routine upload and breaks everything downstream.
2. EXIF orientation is applied before anything is measured. Skip it and a phone
   photo is analyzed rotated, so every edge, bleed and safe-zone result is wrong
   while looking perfectly plausible.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .types import Canvas, UndecodableFile

MAX_BYTES = 25 * 1024 * 1024
MAX_EDGE_PX = 12_000
MAX_PIXELS = 80_000_000  # guards against a decompression bomb

# Pillow's own bomb guard, set below our own limit so it trips first.
Image.MAX_IMAGE_PIXELS = MAX_PIXELS

_RASTER_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"\xff\xd8\xff", "jpeg"),
    (b"GIF87a", "gif"),
    (b"GIF89a", "gif"),
    (b"BM", "bmp"),
    (b"II*\x00", "tiff"),
    (b"MM\x00*", "tiff"),
)


def sniff_format(head: bytes) -> str:
    """Identify the format from its leading bytes. Returns "" if unknown."""
    for magic, name in _RASTER_MAGIC:
        if head.startswith(magic):
            return name
    if head.startswith(b"RIFF") and head[8:12] == b"WEBP":
        return "webp"
    if head.startswith(b"%PDF"):
        return "pdf"
    stripped = head.lstrip()
    if stripped.startswith(b"<svg") or (stripped.startswith(b"<?xml") and b"<svg" in head):
        return "svg"
    return ""


def normalize(path: str | Path) -> Canvas:
    """Decode an upload into a Canvas, or raise UndecodableFile."""
    path = Path(path)
    if not path.is_file():
        raise UndecodableFile(f"no such file: {path}")

    size = path.stat().st_size
    if size == 0:
        raise UndecodableFile("file is empty")
    if size > MAX_BYTES:
        raise UndecodableFile(
            f"file is {size / 1024 / 1024:.1f} MB, over the {MAX_BYTES // 1024 // 1024} MB limit"
        )

    with path.open("rb") as fh:
        head = fh.read(64)
    fmt = sniff_format(head)
    if not fmt:
        raise UndecodableFile("unrecognized file type")
    if fmt in ("pdf", "svg"):
        # Stage 01 for vector sources rasterizes at 600 DPI and also reads the
        # page box, fonts and colour space from the source. Not built yet, and
        # claiming otherwise would be worse than refusing.
        raise UndecodableFile(f"{fmt} input is not supported yet")

    try:
        with Image.open(path) as im:
            im.load()
            metadata_dpi = _read_dpi(im)
            im = ImageOps.exif_transpose(im)  # must happen before any measurement
            if im.width > MAX_EDGE_PX or im.height > MAX_EDGE_PX:
                raise UndecodableFile(
                    f"image is {im.width}x{im.height} px, over the {MAX_EDGE_PX} px edge limit"
                )
            rgba = np.asarray(im.convert("RGBA"), dtype=np.uint8)
    except UndecodableFile:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise UndecodableFile(f"could not decode {fmt}: {exc}") from exc

    return Canvas(
        rgba=np.ascontiguousarray(rgba),
        source_kind="raster",
        source_format=fmt,
        metadata_dpi=metadata_dpi,
    )


def _read_dpi(im: Image.Image) -> tuple[float, float] | None:
    """Read the embedded DPI tag.

    Recorded for the report only. It is routinely wrong or absent, and no
    verdict is ever allowed to depend on it.
    """
    dpi = im.info.get("dpi")
    if not dpi:
        return None
    try:
        x, y = float(dpi[0]), float(dpi[1])
    except (TypeError, ValueError, IndexError):
        return None
    if x <= 0 or y <= 0:
        return None
    return (x, y)
