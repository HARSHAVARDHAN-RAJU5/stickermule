"""STICKER_MOCKUP — the file is a picture of a finished sticker.

People upload what they see on a product page: a rendering of the sticker,
already carrying the white die-cut rim the machine is supposed to add, sitting
on a grey backdrop with a drop shadow underneath it.

Every other check passes such a file. It is sharp, it is well inside the cut
line, and it has no transparency problem for a square order. Printed, the
customer gets a sticker with a fake white border and a photograph of a shadow
on it.

Two signs together:

  a light rim    the design's own outer edge is nearly white, because the rim
                 of the finished sticker has been drawn into the artwork
  a drop shadow  the backdrop immediately around the design is darker than the
                 backdrop further out

Artwork that is ready to print has neither. It also has no shadow, because a
shadow is something a finished sticker casts, not something artwork contains.

**WARN, not FAIL.** A designer may legitimately draw a white outline and a soft
shadow as part of the art. The agent cannot tell that apart from a mockup, and
it should not reject a customer's genuine design on a guess. It says what it
sees and a person spends two seconds on it.
"""

from __future__ import annotations

from ..analyze import Features
from ..config import Config
from ..types import Finding, Severity
from . import check


@check("STICKER_MOCKUP")
def sticker_mockup(features: Features, config: Config) -> Finding | None:
    if features.has_real_alpha:
        # A file with a real cut-out is artwork, not a photograph of a sticker.
        return None
    if not features.has_content:
        return None

    cfg = config.for_check("STICKER_MOCKUP")
    min_rim = float(cfg["min_rim_lightness"])
    min_halo = float(cfg["min_shadow_halo"])
    min_coherence = float(cfg["min_background_coherence"])
    max_border = float(cfg["max_content_on_border"])

    if features.background_coherence < min_coherence:
        return None  # no trustworthy backdrop to measure a shadow against
    if features.content_touches_border > max_border:
        return None  # full bleed: nothing is floating on a backdrop

    rim = features.content_rim_lightness
    halo = features.shadow_halo
    looks_rendered = rim >= min_rim and halo >= min_halo

    detail = {
        "rim_lightness": round(rim, 1),
        "shadow_halo": round(halo, 2),
    }

    if looks_rendered:
        return Finding(
            code="STICKER_MOCKUP",
            severity=Severity.WARN,
            summary=(
                f"the design already has a white rim (lightness {rim:.0f}) and a "
                f"shadow behind it; this looks like a picture of a finished "
                f"sticker rather than the artwork to print"
            ),
            measured=round(halo, 2),
            threshold=min_halo,
            overlay={"type": "bbox", "box": features.content_bbox},
            detail=detail,
        )

    return Finding(
        code="STICKER_MOCKUP",
        severity=Severity.PASS,
        summary="no rendered rim or drop shadow found",
        measured=round(halo, 2),
        threshold=min_halo,
        detail=detail,
    )
