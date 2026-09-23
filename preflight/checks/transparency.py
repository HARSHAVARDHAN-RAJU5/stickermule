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
    near_white_min_l = float(cfg["near_white_min_l"])
    max_chroma = float(cfg["max_chroma"])
    max_l_spread = float(cfg["max_l_spread"])
    band_px = int(cfg["band_px"])

    fill_ratio = features.content_fill_ratio or 0.0
    band = _perimeter_band(features, band_px)
    median_l = float(np.median(band[:, 0]))
    l_spread = float(np.subtract(*np.percentile(band[:, 0], [75, 25])))
    median_chroma = float(np.median(np.linalg.norm(band[:, 1:], axis=1)))

    # A background is large, light, grey and smooth. All four, or it is a design.
    looks_like_a_box = (
        fill_ratio >= fill_max
        and median_l >= near_white_min_l
        and median_chroma <= max_chroma
        and l_spread <= max_l_spread
    )

    detail = {
        "band_median_l": round(median_l, 1),
        "band_median_chroma": round(median_chroma, 1),
        "band_l_spread": round(l_spread, 1),
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


def _perimeter_band(features: Features, band_px: int) -> np.ndarray:
    """L*a*b* of the opaque pixels in a band around the content bounding box.

    Sampling the canvas corners would only measure the transparent margin.
    Sampling four corner patches of the bounding box cannot tell a flat white
    box from a soft gradient, because the two corners differ. The band is the
    whole edge of whatever sits behind the design, which is the thing being
    judged.
    """
    bbox = features.content_bbox
    assert bbox is not None  # guarded by the caller
    x0, y0, x1, y1 = bbox
    region = features.lab[y0 : y1 + 1, x0 : x1 + 1]
    content = features.content_mask[y0 : y1 + 1, x0 : x1 + 1]
    h, w = region.shape[:2]

    b = max(1, min(band_px, h // 2, w // 2))
    edge = np.zeros((h, w), dtype=bool)
    edge[:b, :] = edge[-b:, :] = True
    edge[:, :b] = edge[:, -b:] = True

    band = region[edge & content]
    # A silhouette can leave the band empty; fall back to the whole region
    # rather than reporting on nothing.
    return band if band.size else region.reshape(-1, 3)
