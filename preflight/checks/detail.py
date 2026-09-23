"""DETAIL_AT_CUT — is anything that matters about to be cut off?

This replaces the old BACKGROUND_UNKNOWN escalation.

That check existed because the agent could not separate a design from its
background when ink ran to all four corners, so it gave up and asked a human.
But separating the background was never the goal. The goal is to know whether
something important is about to be sliced, and that can be measured directly.

Flat colour running off the edge is deliberate — that is what full bleed is.
Lettering running off the edge is a mistake. The difference is the density of
detail near the cut, not the presence of an edge, so a single hard boundary
between two flat colours does not trip it.

Because this needs no background separation, it works on files the geometric
safe-zone check cannot touch.
"""

from __future__ import annotations

import numpy as np

from ..analyze import Features
from ..config import Config
from ..types import Finding, Severity
from . import check


@check("DETAIL_AT_CUT")
def detail_at_cut(features: Features, config: Config) -> Finding | None:
    if features.product.is_die_cut:
        # The cut path is derived from the art itself, with a fixed outward
        # offset, so nothing can be closer to it than that offset.
        return None

    cfg = config.for_check("DETAIL_AT_CUT")
    fail_below = float(cfg["fail_below_in"])
    warn_below = float(cfg["warn_below_in"])
    min_density = float(cfg["min_density"])

    if features.background_coherence < float(cfg["min_background_coherence"]):
        # Continuous-tone imagery covering the whole sticker: a photograph,
        # not a design on a background. Texture reaching the cut is the
        # picture itself, and a full-bleed photograph is an ordinary order.
        # Failing it would reject a large class of perfectly good files.
        return Finding(
            code="DETAIL_AT_CUT",
            severity=Severity.PASS,
            summary=(
                "continuous-tone artwork covering the whole sticker; detail at "
                "the edge is the image itself, which is how full bleed works"
            ),
            measured=round(features.background_coherence, 2),
            threshold=float(cfg["min_background_coherence"]),
            detail={"continuous_tone": True},
        )

    detail = features.detail_density >= min_density
    if not detail.any():
        return Finding(
            code="DETAIL_AT_CUT",
            severity=Severity.PASS,
            summary="no fine detail anywhere in the artwork",
            unit="in",
            detail={"detail_area_pct": 0.0},
        )

    distances = features.distance_inside_cut[detail]
    nearest_px = float(distances.min())
    nearest_in = features.inches(nearest_px)

    masked = np.where(detail, features.distance_inside_cut, np.inf)
    y, x = np.unravel_index(int(np.argmin(masked)), masked.shape)
    edge = features.nearest_edge((int(x), int(y)))

    if nearest_in < fail_below:
        severity, threshold = Severity.FAIL, fail_below
        summary = (
            f"fine detail such as lettering comes within {nearest_in:.3f} in of the "
            f"cut line near the {edge}; it will be cut off"
        )
    elif nearest_in < warn_below:
        severity, threshold = Severity.WARN, warn_below
        summary = f"fine detail comes within {nearest_in:.3f} in of the cut line near the {edge}"
    else:
        severity, threshold = Severity.PASS, warn_below
        summary = f"fine detail clears the cut line by {nearest_in:.3f} in"

    return Finding(
        code="DETAIL_AT_CUT",
        severity=severity,
        summary=summary,
        measured=round(nearest_in, 4),
        threshold=threshold,
        unit="in",
        suggested=(warn_below,),
        overlay={"type": "detail", "worst_point_px": [int(x), int(y)]},
        detail={
            "nearest_edge": edge,
            "detail_area_pct": round(float(detail.mean()) * 100, 1),
        },
    )
