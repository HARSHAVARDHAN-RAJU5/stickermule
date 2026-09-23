"""SAFE_ZONE — content crowding the cut line.

Skipped for die-cut products. There the cut path is derived from the artwork
itself with a fixed outward offset, so the clearance is guaranteed by
construction and measuring it would only be measuring our own arithmetic. A
customer-supplied die line would change that, and is the obvious next step.
"""

from __future__ import annotations

from ..analyze import Features
from ..config import Config
from ..types import Finding, Severity
from . import check


@check("SAFE_ZONE")
def safe_zone(features: Features, config: Config) -> Finding | None:
    if features.product.is_die_cut:
        return None
    if not features.has_content:
        return None

    cfg = config.for_check("SAFE_ZONE")
    if features.background_coherence < float(cfg["min_background_coherence"]):
        # The background came back as scattered speckle rather than one
        # region, which is what a photograph running to the edges produces.
        # The clearance would be measured against noise, so there is nothing
        # honest to report. DETAIL_AT_CUT covers these files instead.
        return None

    fail_below = float(cfg["fail_below_in"])
    warn_below = float(cfg["warn_below_in"])

    clearance_px = features.min_content_clearance_px
    assert clearance_px is not None  # has_content guarantees it
    clearance_in = features.inches(clearance_px)

    point = features.min_clearance_point
    edge = features.nearest_edge(point) if point else "edge"

    if clearance_in < fail_below:
        severity, threshold = Severity.FAIL, fail_below
        summary = (
            f"content comes within {clearance_in:.3f} in of the cut line "
            f"near the {edge}; it will be cut into"
        )
    elif clearance_in < warn_below:
        severity, threshold = Severity.WARN, warn_below
        summary = (
            f"content comes within {clearance_in:.3f} in of the cut line near the {edge}"
        )
    else:
        severity, threshold = Severity.PASS, warn_below
        summary = f"content clears the cut line by {clearance_in:.3f} in"

    return Finding(
        code="SAFE_ZONE",
        severity=severity,
        summary=summary,
        measured=round(clearance_in, 4),
        threshold=threshold,
        unit="in",
        suggested=(warn_below,),
        overlay={
            "type": "safe_zone",
            "inset_in": warn_below,
            "worst_point_px": list(point) if point else None,
        },
        detail={"nearest_edge": edge, "clearance_px": round(clearance_px, 1)},
    )
