"""Synthetic artwork builders for the unit tests.

Every fixture is generated rather than checked in, so the expected findings are
exact and the whole suite is reproducible from the repo. The labelled 30-file
evaluation set is built on the same primitives.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

TRANSPARENT = (0, 0, 0, 0)
WHITE = (255, 255, 255, 255)
INK = (24, 28, 32, 255)


def save(rgba: np.ndarray, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(path)
    return path


def blank(size_px: int, fill: tuple[int, int, int, int] = TRANSPARENT) -> np.ndarray:
    art = np.zeros((size_px, size_px, 4), dtype=np.uint8)
    art[:, :] = fill
    return art


def disc(art: np.ndarray, radius_frac: float = 0.4, colour=INK) -> np.ndarray:
    """A filled circle in the middle — a stand-in for a real die-cut design."""
    h, w = art.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    r = min(h, w) * radius_frac
    inside = (xx - w / 2) ** 2 + (yy - h / 2) ** 2 <= r**2
    art[inside] = colour
    return art


def rect(art: np.ndarray, x0: int, y0: int, x1: int, y1: int, colour=INK) -> np.ndarray:
    """Filled rectangle, inclusive bounds."""
    art[y0 : y1 + 1, x0 : x1 + 1] = colour
    return art


def clean_die_cut(size_px: int = 900) -> np.ndarray:
    """Transparent background, a disc of ink. What a good upload looks like."""
    return disc(blank(size_px))


def opaque_design(size_px: int = 900) -> np.ndarray:
    """No alpha at all: a disc of ink on a solid white field."""
    return disc(blank(size_px, WHITE))


def white_box_with_alpha(size_px: int = 900, margin_px: int = 40) -> np.ndarray:
    """The classic mistake.

    A transparent margin makes the file *look* like it has a cut-out, but what
    sits behind the design is still a solid white rectangle.
    """
    art = blank(size_px)
    art[margin_px:-margin_px, margin_px:-margin_px] = WHITE
    return disc(art, radius_frac=0.25)


def inset_content(size_px: int = 600, inset_px: int = 60) -> np.ndarray:
    """Opaque white field with an ink block held `inset_px` from every edge."""
    art = blank(size_px, WHITE)
    return rect(art, inset_px, inset_px, size_px - 1 - inset_px, size_px - 1 - inset_px)


def content_near_bottom(size_px: int = 600, gap_px: int = 5) -> np.ndarray:
    """Ink running toward the bottom edge, corners left clean.

    The corners matter: they are the flood-fill seeds that establish what the
    background is.
    """
    art = blank(size_px, WHITE)
    return rect(art, size_px // 4, size_px // 2, 3 * size_px // 4, size_px - 1 - gap_px)
