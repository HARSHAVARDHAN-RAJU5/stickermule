"""The labelled evaluation set.

Each case carries the severity every check *should* report and the route the
job *should* take, so the harness can score the agent without a human in the
loop. The labels say what a careful person would say about the file, written
down before the agent runs — not a copy of whatever the agent produced.

Some cases are here because the agent once got them wrong. They stay in the
set. docs/results.md names them and says plainly that their scores are
therefore not out-of-sample.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import yaml

from preflight.types import Material, ProductSpec, Severity, Shape, State

from .art import (
    INK,
    OFF_WHITE,
    RED,
    WHITE,
    blank,
    disc,
    horizontal_gradient,
    punch,
    rect,
    ring,
    photo_texture,
    save,
    speckle,
    sticker_mockup,
    text_block,
    two_tone,
)

PASS, WARN, FAIL = Severity.PASS, Severity.WARN, Severity.FAIL
APPROVED = State.APPROVED
FIX = State.NEEDS_CUSTOMER_FIX
REVIEW = State.HUMAN_REVIEW
REJECTED = State.REJECTED


@dataclass(frozen=True)
class Case:
    name: str
    product: ProductSpec
    expect_route: State
    expect: dict[str, Severity]
    note: str
    art: Callable[[], np.ndarray] | None = None
    raw_bytes: bytes | None = None
    # True when the agent is known to get this wrong for a documented reason.
    # It still counts against the score; the flag only lets the report say why.
    known_limitation: bool = False

    @property
    def filename(self) -> str:
        return f"{self.name}.png"


def die_cut(w: float = 3.0, h: float = 3.0) -> ProductSpec:
    return ProductSpec(w, h, Shape.DIE_CUT, Material.WHITE_VINYL)


def square(w: float = 2.0, h: float = 2.0) -> ProductSpec:
    return ProductSpec(w, h, Shape.SQUARE, Material.WHITE_VINYL)


def _inset(size: int, inset: int) -> np.ndarray:
    art = blank(size, fill=WHITE)
    return rect(art, inset, inset, size - 1 - inset, size - 1 - inset)


def _white_box(fill=WHITE, margin: int = 40) -> np.ndarray:
    art = blank(900)
    art[margin:-margin, margin:-margin] = fill
    return disc(art, radius_frac=0.25)


CASES: list[Case] = [
    # --- clean files: the agent must not invent problems -----------------
    Case(
        "clean_die_cut_300",
        die_cut(),
        APPROVED,
        {"RES_DPI": PASS, "ALPHA_MISSING": PASS, "ALPHA_FAKE": PASS},
        "900 px at 3 in, transparent background, a real silhouette",
        art=lambda: disc(blank(900)),
    ),
    Case(
        "clean_die_cut_600",
        die_cut(),
        APPROVED,
        {"RES_DPI": PASS, "ALPHA_MISSING": PASS, "ALPHA_FAKE": PASS},
        "twice the resolution needed",
        art=lambda: disc(blank(1800)),
    ),
    Case(
        "clean_ring",
        die_cut(),
        APPROVED,
        {"RES_DPI": PASS, "ALPHA_MISSING": PASS, "ALPHA_FAKE": PASS},
        "an annulus leaves most of its bounding box empty",
        art=lambda: ring(blank(900)),
    ),
    Case(
        "clean_square",
        square(),
        APPROVED,
        {"RES_DPI": PASS, "DETAIL_AT_CUT": PASS, "SAFE_ZONE": PASS, "STICKER_MOCKUP": PASS},
        "opaque background is correct for a fixed shape",
        art=lambda: _inset(600, 60),
    ),
    Case(
        "clean_rect",
        ProductSpec(3.0, 2.0, Shape.RECT),
        APPROVED,
        {"RES_DPI": PASS, "DETAIL_AT_CUT": PASS, "SAFE_ZONE": PASS, "STICKER_MOCKUP": PASS},
        "non-square product at exactly 300 dpi on both axes",
        art=lambda: rect(blank(900, 600, WHITE), 60, 60, 839, 539),
    ),
    # --- resolution: the artwork decides the middle band ------------------
    Case(
        "res_soft_200_smooth",
        die_cut(),
        APPROVED,
        {"RES_DPI": PASS, "ALPHA_MISSING": PASS, "ALPHA_FAKE": PASS},
        "200 dpi, but one large smooth shape; prints fine, no human needed",
        art=lambda: disc(blank(600)),
    ),
    Case(
        "res_soft_299_smooth",
        die_cut(),
        APPROVED,
        {"RES_DPI": PASS, "ALPHA_MISSING": PASS, "ALPHA_FAKE": PASS},
        "one pixel under target, nothing fine in the artwork",
        art=lambda: disc(blank(899)),
    ),
    Case(
        "res_soft_200_with_text",
        die_cut(),
        FIX,
        {"RES_DPI": FAIL, "ALPHA_MISSING": PASS, "ALPHA_FAKE": PASS},
        "the same 200 dpi, but full of small lettering, which will print soft",
        art=lambda: text_block(blank(600), 60, 60, 540, 540),
    ),
    Case(
        "res_fail_96",
        die_cut(),
        FIX,
        {"RES_DPI": FAIL, "ALPHA_MISSING": PASS, "ALPHA_FAKE": PASS},
        "the classic web-logo upload",
        art=lambda: disc(blank(288)),
    ),
    Case(
        "res_fail_72",
        die_cut(),
        FIX,
        {"RES_DPI": FAIL, "ALPHA_MISSING": PASS, "ALPHA_FAKE": PASS},
        "screenshot resolution",
        art=lambda: disc(blank(216)),
    ),
    Case(
        "res_fail_50",
        die_cut(),
        FIX,
        {"RES_DPI": FAIL, "ALPHA_MISSING": PASS, "ALPHA_FAKE": PASS},
        "a thumbnail ordered at 3 inches",
        art=lambda: disc(blank(150)),
    ),
    Case(
        "res_axis_ok",
        die_cut(4.0, 2.0),
        APPROVED,
        {"RES_DPI": PASS, "ALPHA_MISSING": PASS, "ALPHA_FAKE": PASS},
        "square file on a wide product; the narrow axis still clears 300",
        art=lambda: disc(blank(1200)),
    ),
    Case(
        "res_axis_fail",
        die_cut(10.0, 2.0),
        FIX,
        {"RES_DPI": FAIL, "ALPHA_MISSING": PASS, "ALPHA_FAKE": PASS},
        "same file on a bumper-sticker width; the wide axis decides",
        art=lambda: disc(blank(1200)),
    ),
    # --- transparency ----------------------------------------------------
    Case(
        "alpha_missing_flat",
        die_cut(),
        FIX,
        {"RES_DPI": PASS, "ALPHA_MISSING": FAIL, "STICKER_MOCKUP": PASS},
        "die-cut ordered, file fully opaque",
        art=lambda: disc(blank(900, fill=WHITE)),
    ),
    Case(
        "alpha_missing_noisy",
        die_cut(),
        FIX,
        {"RES_DPI": PASS, "ALPHA_MISSING": FAIL, "STICKER_MOCKUP": PASS},
        "same mistake, but photographic noise on the background",
        art=lambda: speckle(disc(blank(900, fill=WHITE))),
    ),
    Case(
        "alpha_below_threshold",
        die_cut(),
        FIX,
        {"RES_DPI": PASS, "ALPHA_MISSING": FAIL, "STICKER_MOCKUP": PASS},
        "0.5% transparent pixels is not a cut-out",
        art=lambda: punch(blank(900, fill=WHITE), 0.04),
    ),
    Case(
        "alpha_hole_in_a_box",
        die_cut(),
        FIX,
        {"RES_DPI": PASS, "ALPHA_MISSING": PASS, "ALPHA_FAKE": FAIL},
        "enough alpha to pass the first check, but still a white box with a hole",
        art=lambda: punch(blank(900, fill=WHITE), 0.06),
    ),
    Case(
        "fake_white_box",
        die_cut(),
        FIX,
        {"RES_DPI": PASS, "ALPHA_MISSING": PASS, "ALPHA_FAKE": FAIL},
        "transparent margin, solid white rectangle behind the design",
        art=lambda: _white_box(WHITE),
    ),
    Case(
        "fake_off_white_box",
        die_cut(),
        FIX,
        {"RES_DPI": PASS, "ALPHA_MISSING": PASS, "ALPHA_FAKE": FAIL},
        "the same mistake at 242 grey, which still prints as a box",
        art=lambda: _white_box(OFF_WHITE),
    ),
    Case(
        "fake_gradient_box",
        die_cut(),
        FIX,
        {"RES_DPI": PASS, "ALPHA_MISSING": PASS, "ALPHA_FAKE": FAIL},
        "a background that is almost uniform; four corner samples missed it, "
        "which is why the check now measures the whole perimeter band",
        art=lambda: disc(
            horizontal_gradient(blank(900), 255, 225, (40, 40, 859, 859)), 0.25
        ),
    ),
    Case(
        "real_dark_block",
        die_cut(),
        APPROVED,
        {"RES_DPI": PASS, "ALPHA_MISSING": PASS, "ALPHA_FAKE": PASS},
        "a solid dark design that genuinely fills its box; flagging it is a false alarm",
        art=lambda: rect(blank(900), 100, 100, 799, 799, INK),
    ),
    Case(
        "real_red_block",
        die_cut(),
        APPROVED,
        {"RES_DPI": PASS, "ALPHA_MISSING": PASS, "ALPHA_FAKE": PASS},
        "same, in a saturated colour",
        art=lambda: rect(blank(900), 100, 100, 799, 799, RED),
    ),
    Case(
        "combo_opaque_and_lowres",
        die_cut(),
        FIX,
        {"RES_DPI": FAIL, "ALPHA_MISSING": FAIL, "STICKER_MOCKUP": PASS},
        "low resolution and no transparency together",
        art=lambda: disc(blank(288, fill=WHITE)),
    ),
    # --- safe zone -------------------------------------------------------
    Case(
        "safe_warn_30px",
        square(),
        REVIEW,
        {"RES_DPI": PASS, "DETAIL_AT_CUT": WARN, "SAFE_ZONE": WARN, "STICKER_MOCKUP": PASS},
        "0.10 in of clearance: inside 1/8 in, outside 1/16 in",
        art=lambda: _inset(600, 30),
    ),
    Case(
        "safe_warn_20px",
        square(),
        REVIEW,
        {"RES_DPI": PASS, "DETAIL_AT_CUT": WARN, "SAFE_ZONE": WARN, "STICKER_MOCKUP": PASS},
        "0.065 in, just above the fail line",
        art=lambda: _inset(600, 20),
    ),
    Case(
        "safe_fail_10px",
        square(),
        FIX,
        {"RES_DPI": PASS, "DETAIL_AT_CUT": FAIL, "SAFE_ZONE": FAIL, "STICKER_MOCKUP": PASS},
        "0.03 in; the blade will cut into it",
        art=lambda: _inset(600, 10),
    ),
    Case(
        "safe_fail_touching",
        square(),
        FIX,
        {"RES_DPI": PASS, "DETAIL_AT_CUT": PASS, "SAFE_ZONE": FAIL},
        "a block running off the bottom edge. Flat colour is not fine detail, "
        "so only the geometric check should catch this one",
        art=lambda: rect(blank(600, fill=WHITE), 150, 300, 449, 599),
    ),
    Case(
        "safe_circle_corners",
        ProductSpec(2.0, 2.0, Shape.CIRCLE),
        FIX,
        {"RES_DPI": PASS, "DETAIL_AT_CUT": FAIL, "SAFE_ZONE": FAIL, "STICKER_MOCKUP": PASS},
        "square artwork on a circular product; the corners fall outside the cut",
        art=lambda: rect(blank(600, fill=WHITE), 60, 60, 539, 539),
    ),
    Case(
        "safe_rounded_corner",
        ProductSpec(2.0, 2.0, Shape.ROUNDED_RECT),
        FIX,
        {"RES_DPI": PASS, "DETAIL_AT_CUT": WARN, "SAFE_ZONE": FAIL},
        "content parked in a corner the rounded die never reaches. Its own "
        "corner lands 0.117 in inside the die, which is the warn band",
        art=lambda: disc(rect(blank(600), 0, 0, 40, 40, INK), 0.3),
    ),
    Case(
        "combo_lowres_and_edge",
        square(),
        FIX,
        {"RES_DPI": FAIL, "DETAIL_AT_CUT": FAIL, "SAFE_ZONE": FAIL, "STICKER_MOCKUP": PASS},
        "both problems reported in one message, not one round trip each",
        art=lambda: _inset(200, 3),
    ),
    # --- lettering near the cut ------------------------------------------
    Case(
        "text_safely_inside",
        square(),
        APPROVED,
        {"RES_DPI": PASS, "DETAIL_AT_CUT": PASS, "SAFE_ZONE": PASS, "STICKER_MOCKUP": PASS},
        "dense lettering, but well clear of the blade",
        art=lambda: text_block(blank(600, fill=WHITE), 120, 120, 480, 480),
    ),
    Case(
        "text_at_edge",
        square(),
        FIX,
        {"RES_DPI": PASS, "DETAIL_AT_CUT": FAIL, "SAFE_ZONE": FAIL, "STICKER_MOCKUP": PASS},
        "lettering running to the edge; words will be sliced",
        art=lambda: text_block(blank(600, fill=WHITE), 10, 10, 590, 590),
    ),
    # --- full bleed: legitimate, and no longer escalated ------------------
    Case(
        "bleed_flat",
        square(),
        APPROVED,
        {"RES_DPI": PASS, "DETAIL_AT_CUT": PASS},
        "solid colour to all four edges: a normal full-bleed sticker, and the "
        "case that used to be escalated because no background could be found",
        art=lambda: rect(blank(600, fill=WHITE), 0, 0, 599, 599, INK),
    ),
    Case(
        "bleed_two_tone",
        square(),
        APPROVED,
        {"RES_DPI": PASS, "DETAIL_AT_CUT": PASS},
        "two flat colours meeting in a hard line: an edge, but not fine detail",
        art=lambda: two_tone(600),
    ),
    Case(
        "bleed_text_at_edge",
        square(),
        FIX,
        {"RES_DPI": PASS, "DETAIL_AT_CUT": FAIL, "SAFE_ZONE": FAIL, "STICKER_MOCKUP": PASS},
        "lettering running off the edge of a coloured field; both checks "
        "should catch it, from different directions",
        art=lambda: text_block(blank(600, fill=RED), 5, 5, 595, 595, colour=WHITE),
    ),
    # --- a picture of a sticker, instead of the artwork -------------------
    Case(
        "mockup_rendered_sticker",
        square(3.0, 3.0),
        REVIEW,
        {"RES_DPI": PASS, "DETAIL_AT_CUT": PASS, "SAFE_ZONE": PASS,
         "STICKER_MOCKUP": WARN},
        "a product-page screenshot: white die-cut rim and drop shadow already "
        "drawn in. Every other check passes it, and it would print the fake "
        "rim and a photograph of the shadow",
        art=lambda: sticker_mockup(900),
    ),
    Case(
        "real_white_outline",
        square(3.0, 3.0),
        APPROVED,
        {"RES_DPI": PASS, "DETAIL_AT_CUT": PASS, "SAFE_ZONE": PASS,
         "STICKER_MOCKUP": PASS},
        "a design with a genuine white outline but no shadow behind it; "
        "flagging this would be a false alarm on real artwork",
        art=lambda: disc(disc(blank(900, fill=WHITE), 0.34, WHITE), 0.29, RED),
    ),
    # --- a full-bleed photograph ------------------------------------------
    Case(
        "bleed_photo_texture",
        square(3.0, 3.0),
        APPROVED,
        {"RES_DPI": PASS, "DETAIL_AT_CUT": PASS},
        "continuous-tone imagery to all four edges. The background cannot be "
        "separated, so the geometric checks decline rather than measuring "
        "against noise",
        art=lambda: photo_texture(900),
    ),
    # --- intake -----------------------------------------------------------
    Case(
        "intake_pdf_named_png",
        square(),
        REJECTED,
        {},
        "a PDF with a .png extension must be refused, not half-decoded",
        raw_bytes=b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n1 0 obj\n<<>>\nendobj\n",
    ),
]


def build(out_dir: str | Path = "evalset") -> Path:
    """Generate every file and write labels.yaml beside them."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    records = []
    for case in CASES:
        path = out / case.filename
        if case.raw_bytes is not None:
            path.write_bytes(case.raw_bytes)
        else:
            assert case.art is not None
            save(case.art(), path)
        records.append(
            {
                "name": case.name,
                "file": case.filename,
                "product": {
                    "width_in": case.product.width_in,
                    "height_in": case.product.height_in,
                    "shape": case.product.shape.value,
                    "material": case.product.material.value,
                },
                "expect": {
                    "route": case.expect_route.value,
                    "findings": {k: v.value for k, v in case.expect.items()},
                },
                "known_limitation": case.known_limitation,
                "note": case.note,
            }
        )

    labels = out / "labels.yaml"
    labels.write_text(
        "# Ground truth for the evaluation set. Generated by `python -m evalkit build`.\n"
        + yaml.safe_dump({"cases": records}, sort_keys=False, width=100),
        encoding="utf-8",
    )
    return labels
