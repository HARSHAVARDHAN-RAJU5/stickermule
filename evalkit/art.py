"""Synthetic artwork primitives.

Every evaluation file is generated rather than checked in, so the ground truth
is exact and the whole set is reproducible from the repo with one command. The
unit tests build on the same primitives.

These are deliberately clean: hard edges, flat colour, no photographic noise
unless asked for. Real customer uploads are messier, and that gap is the single
biggest reason to read the results table as an optimistic bound.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

TRANSPARENT = (0, 0, 0, 0)
WHITE = (255, 255, 255, 255)
OFF_WHITE = (242, 242, 240, 255)
INK = (24, 28, 32, 255)
RED = (198, 40, 40, 255)

RNG = np.random.default_rng(20260923)  # fixed: the set must be reproducible


def save(rgba: np.ndarray, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(path)
    return path


def blank(width_px: int, height_px: int | None = None, fill=TRANSPARENT) -> np.ndarray:
    height_px = height_px if height_px is not None else width_px
    art = np.zeros((height_px, width_px, 4), dtype=np.uint8)
    art[:, :] = fill
    return art


def rect(art: np.ndarray, x0: int, y0: int, x1: int, y1: int, colour=INK) -> np.ndarray:
    """Filled rectangle, inclusive bounds."""
    art[y0 : y1 + 1, x0 : x1 + 1] = colour
    return art


def disc(art: np.ndarray, radius_frac: float = 0.4, colour=INK) -> np.ndarray:
    h, w = art.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    r = min(h, w) * radius_frac
    art[(xx - w / 2) ** 2 + (yy - h / 2) ** 2 <= r**2] = colour
    return art


def ring(art: np.ndarray, outer: float = 0.45, inner: float = 0.30, colour=INK):
    """An annulus — a real design shape that leaves most of its box empty."""
    h, w = art.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    d2 = (xx - w / 2) ** 2 + (yy - h / 2) ** 2
    span = min(h, w)
    art[(d2 <= (span * outer) ** 2) & (d2 >= (span * inner) ** 2)] = colour
    return art


def punch(art: np.ndarray, radius_frac: float = 0.04) -> np.ndarray:
    """Knock a small transparent hole in an otherwise opaque file."""
    return disc(art, radius_frac, TRANSPARENT)


def speckle(art: np.ndarray, sigma: float = 6.0) -> np.ndarray:
    """Photographic noise, the kind a JPEG'd upload carries."""
    noise = RNG.normal(0.0, sigma, size=art.shape[:2] + (3,))
    art[:, :, :3] = np.clip(art[:, :, :3].astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return art


def horizontal_gradient(art: np.ndarray, left: int, right: int, box: tuple[int, int, int, int]):
    """Fill a box with a left-to-right luminance ramp.

    A background that is *almost* uniform. This is the case the corner-patch
    test is expected to miss, and it is in the set precisely so the miss shows
    up in the results table.
    """
    x0, y0, x1, y1 = box
    ramp = np.linspace(left, right, x1 - x0 + 1, dtype=np.float32)
    art[y0 : y1 + 1, x0 : x1 + 1, :3] = ramp[None, :, None].astype(np.uint8)
    art[y0 : y1 + 1, x0 : x1 + 1, 3] = 255
    return art


def text_block(
    art: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    line_h: int = 14,
    line_gap: int = 12,
    colour=INK,
) -> np.ndarray:
    """Rows of short bars standing in for lines of small text.

    Real lettering is the thing that looks wrong when it is printed blurry or
    clipped, and it is what "fine detail" is meant to catch. Bars give the same
    dense edge pattern without depending on a font being installed.
    """
    y = y0
    while y + line_h <= y1:
        x = x0
        while x < x1:
            word = int(RNG.integers(10, 34))
            art[y : y + line_h, x : min(x + word, x1)] = colour
            x += word + 7
        y += line_h + line_gap
    return art


def two_tone(width_px: int, left=INK, right=RED) -> np.ndarray:
    """Full bleed, two flat colours meeting in a hard line.

    An edge, but not fine detail. The check must not mistake one for the other.
    """
    art = blank(width_px, fill=left)
    art[:, width_px // 2 :] = right
    return art


def sticker_mockup(width_px: int = 900, backdrop=(214, 214, 214)) -> np.ndarray:
    """A picture of a finished sticker, not the artwork for one.

    White die-cut rim already drawn in, soft shadow underneath, sitting on a
    grey backdrop. This is what a product-page screenshot looks like.
    """
    art = blank(width_px, fill=(*backdrop, 255))
    h = w = width_px
    yy, xx = np.ogrid[:h, :w]
    cx = cy = w / 2

    shadow = (xx - cx - w * 0.02) ** 2 + (yy - cy - h * 0.02) ** 2 <= (w * 0.35) ** 2
    art[shadow, :3] = np.array(backdrop, np.uint8) - 26

    blurred = cv2.GaussianBlur(art[:, :, :3], (0, 0), w * 0.012)
    art[:, :, :3] = blurred

    rim = (xx - cx) ** 2 + (yy - cy) ** 2 <= (w * 0.34) ** 2
    art[rim] = WHITE
    face = (xx - cx) ** 2 + (yy - cy) ** 2 <= (w * 0.29) ** 2
    art[face] = RED
    return art


def photo_texture(width_px: int = 900, seed: int = 11) -> np.ndarray:
    """Continuous-tone imagery running to all four edges: a full-bleed photo.

    Texture everywhere means flood filling inward from the corners cannot find
    a background, so the geometric checks have nothing trustworthy to measure.
    """
    rng = np.random.default_rng(seed)
    base = rng.integers(60, 200, (width_px, width_px, 3)).astype(np.uint8)
    base = cv2.GaussianBlur(base, (0, 0), 1.2)
    for y in range(0, width_px, max(8, width_px // 15)):
        cv2.line(base, (0, y), (width_px, y + width_px // 22), (232, 220, 180), 3)
    art = blank(width_px)
    art[:, :, :3] = base
    art[:, :, 3] = 255
    return art
