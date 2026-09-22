"""Transparency checks for die-cut orders.

Two different mistakes, two different codes, because they need two different
sentences to the customer:

  ALPHA_MISSING  the file has no transparency at all
  ALPHA_FAKE     the file has an alpha channel, but what is behind the design
                 is still a solid light rectangle drawn into the art

Both are skipped entirely for products with a fixed shape, where an opaque
background is exactly what the customer should be sending.
"""

from __future__ import annotations

import numpy as np

from ..analyze import Features
from ..config import Config
from ..types import Finding, Severity
from . import check


@check("ALPHA_MISSING")
def alpha_missing(features: Features, config: Config) -> Finding | None:
    if not features.product.is_die_cut:
        return None

    cfg = config.for_check("ALPHA_MISSING")
    minimum = float(cfg["min_alpha_coverage_pct"])
    coverage = features.alpha_coverage_pct

    if coverage >= minimum:
        return Finding(
            code="ALPHA_MISSING",
            severity=Severity.PASS,
            summary=f"transparent background present ({coverage:.1f}% of pixels)",
            measured=round(coverage, 1),
            threshold=minimum,
            unit="%",
        )

    return Finding(
        code="ALPHA_MISSING",
        severity=Severity.FAIL,
        summary=(
            f"die-cut ordered but the file is fully opaque "
            f"({coverage:.1f}% transparent pixels)"
        ),
        measured=round(coverage, 1),
        threshold=minimum,
        unit="%",
        detail={"shape": features.product.shape.value},
    )


@check("ALPHA_FAKE")
def alpha_fake(features: Features, config: Config) -> Finding | None:
    if not features.product.is_die_cut:
        return None
    if not features.has_real_alpha:
        # ALPHA_MISSING already owns this case; reporting it twice would put
        # two sentences about the same mistake in front of the customer.
        return None
    if not features.has_content:
        return None

    cfg = config.for_check("ALPHA_FAKE")
    fill_max = float(cfg["box_fill_ratio_max"])
    delta_e_max = float(cfg["corner_delta_e_max"])
    near_white_min_l = float(cfg["near_white_min_l"])
    patch_px = int(cfg["corner_patch_px"])

    fill_ratio = features.content_fill_ratio or 0.0
    corners = _bbox_corner_patches(features, patch_px)
    delta_e = _max_pairwise_delta_e(corners)
    mean_l = float(corners[:, 0].mean())

    looks_like_a_box = (
        fill_ratio >= fill_max and delta_e <= delta_e_max and mean_l >= near_white_min_l
    )

    detail = {
        "corner_delta_e": round(delta_e, 2),
        "corner_mean_l": round(mean_l, 1),
        "content_source": features.content_source,
    }

    if looks_like_a_box:
        return Finding(
            code="ALPHA_FAKE",
            severity=Severity.FAIL,
            summary=(
                f"opaque region fills {fill_ratio * 100:.0f}% of its bounding box and is "
                f"a uniform light colour; the background looks like a white box, not a cut-out"
            ),
            measured=round(fill_ratio * 100, 1),
            threshold=round(fill_max * 100, 1),
            unit="%",
            overlay={"type": "bbox", "box": features.content_bbox},
            detail=detail,
        )

    return Finding(
        code="ALPHA_FAKE",
        severity=Severity.PASS,
        summary=f"opaque region fills {fill_ratio * 100:.0f}% of its bounding box",
        measured=round(fill_ratio * 100, 1),
        threshold=round(fill_max * 100, 1),
        unit="%",
        detail=detail,
    )


def _bbox_corner_patches(features: Features, patch_px: int) -> np.ndarray:
    """Mean L*a*b* of the four corners of the *content* bounding box.

    Sampling the canvas corners would only measure the transparent margin,
    which tells us nothing about what sits behind the design.
    """
    bbox = features.content_bbox
    assert bbox is not None  # guarded by the caller
    x0, y0, x1, y1 = bbox
    region = features.lab[y0 : y1 + 1, x0 : x1 + 1]
    h, w = region.shape[:2]
    p = max(1, min(patch_px, h // 2, w // 2))
    quadrants = (
        region[:p, :p],
        region[:p, w - p :],
        region[h - p :, :p],
        region[h - p :, w - p :],
    )
    return np.stack([q.reshape(-1, 3).mean(axis=0) for q in quadrants])


def _max_pairwise_delta_e(patches: np.ndarray) -> float:
    """Largest CIE76 colour difference between any two patches."""
    worst = 0.0
    for i in range(len(patches)):
        for j in range(i + 1, len(patches)):
            worst = max(worst, float(np.linalg.norm(patches[i] - patches[j])))
    return worst
