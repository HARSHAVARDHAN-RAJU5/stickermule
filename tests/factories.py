"""Artwork builders for the unit tests.

The primitives live in `evalkit.art` so the tests and the evaluation set are
built from exactly the same code. What lives here are the named scenarios the
unit tests reach for.
"""

from __future__ import annotations

from evalkit.art import (  # noqa: F401 - re-exported for the tests
    INK,
    OFF_WHITE,
    RED,
    TRANSPARENT,
    WHITE,
    blank,
    disc,
    rect,
    ring,
    save,
)


def clean_die_cut(size_px: int = 900):
    """Transparent background, a disc of ink. What a good upload looks like."""
    return disc(blank(size_px))


def opaque_design(size_px: int = 900):
    """No alpha at all: a disc of ink on a solid white field."""
    return disc(blank(size_px, fill=WHITE))


def white_box_with_alpha(size_px: int = 900, margin_px: int = 40):
    """The classic mistake.

    A transparent margin makes the file *look* like it has a cut-out, but what
    sits behind the design is still a solid white rectangle.
    """
    art = blank(size_px)
    art[margin_px:-margin_px, margin_px:-margin_px] = WHITE
    return disc(art, radius_frac=0.25)


def inset_content(size_px: int = 600, inset_px: int = 60):
    """Opaque white field with an ink block held `inset_px` from every edge."""
    art = blank(size_px, fill=WHITE)
    return rect(art, inset_px, inset_px, size_px - 1 - inset_px, size_px - 1 - inset_px)


def content_near_bottom(size_px: int = 600, gap_px: int = 5):
    """Ink running toward the bottom edge, corners left clean.

    The corners matter: they are the flood-fill seeds that establish what the
    background is.
    """
    art = blank(size_px, fill=WHITE)
    return rect(art, size_px // 4, size_px // 2, 3 * size_px // 4, size_px - 1 - gap_px)
