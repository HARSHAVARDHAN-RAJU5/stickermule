"""Stage 01: the two things that silently corrupt everything downstream."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from preflight.normalize import UndecodableFile, normalize, sniff_format

from .factories import INK, WHITE, blank, rect, save


class TestSniffing:
    @pytest.mark.parametrize(
        "head, expected",
        [
            (b"\x89PNG\r\n\x1a\n\x00\x00", "png"),
            (b"\xff\xd8\xff\xe0\x00\x10JFIF", "jpeg"),
            (b"%PDF-1.7\n", "pdf"),
            (b"RIFF\x00\x00\x00\x00WEBPVP8 ", "webp"),
            (b"<svg xmlns='http://www.w3.org/2000/svg'>", "svg"),
            (b"\x00\x01\x02\x03garbage", ""),
        ],
    )
    def test_format_comes_from_the_bytes(self, head, expected):
        assert sniff_format(head) == expected

    def test_a_pdf_named_png_is_caught(self, tmp_path):
        """A routine upload, and it would decode as nothing at all."""
        path = tmp_path / "artwork.png"
        path.write_bytes(b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n")

        with pytest.raises(UndecodableFile, match="pdf"):
            normalize(path)


class TestGuards:
    def test_empty_file(self, tmp_path):
        path = tmp_path / "art.png"
        path.write_bytes(b"")

        with pytest.raises(UndecodableFile, match="empty"):
            normalize(path)

    def test_missing_file(self, tmp_path):
        with pytest.raises(UndecodableFile, match="no such file"):
            normalize(tmp_path / "nope.png")

    def test_truncated_png(self, tmp_path):
        path = save(blank(64, fill=WHITE), tmp_path / "art.png")
        data = path.read_bytes()
        path.write_bytes(data[: len(data) // 2])

        with pytest.raises(UndecodableFile):
            normalize(path)


def test_exif_orientation_is_applied_before_measurement(tmp_path):
    """A rotated phone photo measured unrotated puts every edge in the wrong place.

    The mark sits at the top-left of the upright image. Tagged orientation 6,
    the stored pixels are rotated 90 degrees clockwise, so a decoder that
    ignores EXIF finds it at the bottom-left instead.
    """
    art = blank(200, fill=WHITE)
    rect(art, 0, 0, 39, 19, INK)  # a wide mark, top-left of the upright image
    upright = Image.fromarray(art, mode="RGBA")

    # Orientation 6 means "rotate 90 CW to display", so the stored pixels are
    # the upright image turned 90 CCW.
    stored = upright.rotate(90, expand=True)
    path = tmp_path / "photo.jpg"
    exif = Image.Exif()
    exif[274] = 6  # Orientation: rotate 90 CW to display
    stored.convert("RGB").save(path, exif=exif, quality=95)

    canvas = normalize(path)
    dark = np.asarray(canvas.rgb).mean(axis=2) < 128
    ys, xs = np.nonzero(dark)

    assert ys.mean() < 100, "mark should be in the top half after transposition"
    assert xs.mean() < 100, "mark should be in the left half after transposition"
    assert np.ptp(xs) > np.ptp(ys), "the mark is wider than it is tall when upright"


def test_canvas_is_always_rgba_uint8(tmp_path):
    path = save(blank(32, fill=WHITE), tmp_path / "art.png")

    canvas = normalize(path)

    assert canvas.rgba.shape == (32, 32, 4)
    assert canvas.rgba.dtype == np.uint8
    assert canvas.source_kind == "raster"
