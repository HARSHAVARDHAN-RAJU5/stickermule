"""RES_DPI — effective print resolution at the ordered size."""

from __future__ import annotations

import math

from ..analyze import Features
from ..config import Config
from ..types import Finding, Severity
from . import check


@check("RES_DPI")
def res_dpi(features: Features, config: Config) -> Finding:
    cfg = config.for_check("RES_DPI")
    fail_below = float(cfg["fail_below"])
    warn_below = float(cfg["warn_below"])
    target = float(cfg["target"])

    detail_max = float(cfg["detail_score_max"])

    dpi = features.dpi
    product = features.product

    # What the customer would have to upload to clear the target.
    need_w = math.ceil(target * product.width_in)
    need_h = math.ceil(target * product.height_in)

    if dpi < fail_below:
        severity, threshold = Severity.FAIL, fail_below
        summary = f"{dpi:.0f} DPI at the ordered size; will print visibly soft"
    elif dpi < warn_below:
        # The middle band used to be a blanket warning, which sent every
        # borderline file to a person. What actually decides it is what the
        # picture contains: 200 DPI on a photograph prints fine, 200 DPI on
        # small lettering does not. So the artwork decides, not a coin flip.
        if features.detail_score > detail_max:
            severity, threshold = Severity.FAIL, warn_below
            summary = (
                f"{dpi:.0f} DPI at the ordered size, and the artwork contains fine "
                f"detail such as lettering, which will print soft"
            )
        else:
            severity, threshold = Severity.PASS, warn_below
            summary = (
                f"{dpi:.0f} DPI at the ordered size; below target, but the artwork "
                f"has no fine detail that would show it"
            )
    else:
        severity, threshold = Severity.PASS, warn_below
        summary = f"{dpi:.0f} DPI at the ordered size"

    return Finding(
        code="RES_DPI",
        severity=severity,
        summary=summary,
        measured=round(dpi, 1),
        threshold=threshold,
        unit="dpi",
        suggested=(target, float(need_w), float(need_h)),
        detail={
            "px_width": features.canvas.px_width,
            "px_height": features.canvas.px_height,
            "ordered_in": [product.width_in, product.height_in],
            # Recorded for the report, never used for the verdict.
            "metadata_dpi": features.canvas.metadata_dpi,
            "required_px": [need_w, need_h],
            "detail_score": round(features.detail_score, 3),
        },
    )
