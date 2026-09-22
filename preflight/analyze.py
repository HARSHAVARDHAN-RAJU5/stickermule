"""Stage 02 — ANALYZE. Derive the features once; share them with every check.

Nothing here decides anything. It turns a Canvas into measurements, so that
each check downstream is a few lines of comparison against a threshold rather
than its own private pile of image processing.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import cv2
import numpy as np

from .types import Canvas, ProductSpec, Shape

ALPHA_CONTENT_THRESHOLD = 8  # alpha above this counts as content
ALPHA_OPAQUE_THRESHOLD = 250  # alpha below this counts as "not fully opaque"
FLOOD_TOLERANCE = 12  # per-channel, for background flood fill
DIE_CUT_OFFSET_IN = 0.08  # the border a die-cut contour is offset outward by


@dataclass
class Features:
    """Everything the checks are allowed to look at."""

    canvas: Canvas
    product: ProductSpec

    # --- scale ---------------------------------------------------------

    @cached_property
    def dpi(self) -> float:
        """Effective print resolution at the ordered size."""
        return self.canvas.effective_dpi(self.product)

    def inches(self, px: float) -> float:
        return px / self.dpi

    def px(self, inches: float) -> float:
        return inches * self.dpi

    # --- transparency --------------------------------------------------

    @cached_property
    def alpha_coverage_pct(self) -> float:
        """Percent of pixels that are not fully opaque."""
        alpha = self.canvas.alpha
        return float(np.count_nonzero(alpha < ALPHA_OPAQUE_THRESHOLD)) / alpha.size * 100.0

    @cached_property
    def has_real_alpha(self) -> bool:
        """Whether the file carries a genuinely used transparent region.

        An alpha channel that is present but entirely opaque is not
        transparency; it is a white box with an extra channel.
        """
        return self.alpha_coverage_pct >= 1.0

    # --- content -------------------------------------------------------

    @cached_property
    def content_mask(self) -> np.ndarray:
        """Boolean mask of the pixels that carry the design.

        From the alpha channel when there is one. Otherwise from a flood fill
        inward from all four corners — which is what catches a design saved
        onto an opaque background.
        """
        if self.has_real_alpha:
            return self.canvas.alpha > ALPHA_CONTENT_THRESHOLD
        return ~self._flood_background()

    @cached_property
    def content_source(self) -> str:
        return "alpha" if self.has_real_alpha else "flood_fill"

    @cached_property
    def has_content(self) -> bool:
        return bool(self.content_mask.any())

    def _flood_background(self) -> np.ndarray:
        """Flood fill inward from each corner; the union is the background."""
        h, w = self.canvas.px_height, self.canvas.px_width
        bgr = cv2.cvtColor(self.canvas.rgb, cv2.COLOR_RGB2BGR)
        background = np.zeros((h, w), dtype=bool)
        lo = up = (FLOOD_TOLERANCE,) * 3
        for seed in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)):
            mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
            cv2.floodFill(
                bgr.copy(),
                mask,
                seedPoint=seed,
                newVal=(0, 0, 0),
                loDiff=lo,
                upDiff=up,
                flags=4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8),
            )
            background |= mask[1:-1, 1:-1].astype(bool)
        return background

    @cached_property
    def content_bbox(self) -> tuple[int, int, int, int] | None:
        """(x0, y0, x1, y1) inclusive bounds of the content, or None."""
        if not self.has_content:
            return None
        rows = np.flatnonzero(self.content_mask.any(axis=1))
        cols = np.flatnonzero(self.content_mask.any(axis=0))
        return int(cols[0]), int(rows[0]), int(cols[-1]), int(rows[-1])

    # --- colour --------------------------------------------------------

    @cached_property
    def lab(self) -> np.ndarray:
        """CIE L*a*b*, properly scaled (L* in 0-100, a*/b* signed)."""
        lab = cv2.cvtColor(self.canvas.rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        lab[:, :, 0] *= 100.0 / 255.0
        lab[:, :, 1:] -= 128.0
        return lab

    def corner_patches(self, patch_px: int) -> np.ndarray:
        """Mean L*a*b* of each of the four corner patches, shape (4, 3)."""
        h, w = self.canvas.px_height, self.canvas.px_width
        p = max(1, min(int(patch_px), h // 2, w // 2))
        regions = (
            self.lab[:p, :p],
            self.lab[:p, w - p :],
            self.lab[h - p :, :p],
            self.lab[h - p :, w - p :],
        )
        return np.stack([r.reshape(-1, 3).mean(axis=0) for r in regions])

    # --- geometry ------------------------------------------------------

    @cached_property
    def cut_mask(self) -> np.ndarray:
        """Boolean mask of everything inside the cut line.

        For a die-cut product the cut line is derived from the art itself:
        the content silhouette, closed up and offset outward. For a product
        with a fixed shape the cut line is that shape, and art that runs into
        it is the customer's problem to fix.
        """
        h, w = self.canvas.px_height, self.canvas.px_width
        shape = self.product.shape

        if shape is Shape.DIE_CUT:
            if not self.has_content:
                return np.zeros((h, w), dtype=bool)
            offset_px = max(1, int(round(self.px(DIE_CUT_OFFSET_IN))))
            src = self.content_mask.astype(np.uint8) * 255
            k = 2 * offset_px + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            dilated = cv2.dilate(src, kernel, iterations=1)
            closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
            return closed.astype(bool)

        mask = np.zeros((h, w), dtype=np.uint8)
        if shape is Shape.CIRCLE:
            cv2.ellipse(
                mask,
                center=(w // 2, h // 2),
                axes=(max(1, w // 2 - 1), max(1, h // 2 - 1)),
                angle=0,
                startAngle=0,
                endAngle=360,
                color=255,
                thickness=-1,
            )
        elif shape is Shape.ROUNDED_RECT:
            r = max(1, int(round(self.px(0.125))))
            r = min(r, w // 2 - 1, h // 2 - 1) if min(w, h) > 4 else 1
            cv2.rectangle(mask, (r, 0), (w - 1 - r, h - 1), 255, -1)
            cv2.rectangle(mask, (0, r), (w - 1, h - 1 - r), 255, -1)
            for cx, cy in ((r, r), (w - 1 - r, r), (r, h - 1 - r), (w - 1 - r, h - 1 - r)):
                cv2.circle(mask, (cx, cy), r, 255, -1)
        else:  # SQUARE, RECT — the trim is the canvas edge
            mask[:, :] = 255
        return mask.astype(bool)

    @cached_property
    def distance_inside_cut(self) -> np.ndarray:
        """Per-pixel distance, in pixels, to the nearest point outside the cut.

        Padded before the transform so that a cut mask filling the whole canvas
        still measures against the canvas edge instead of reporting infinity.
        """
        pad = 2
        m = np.pad(self.cut_mask.astype(np.uint8), pad, mode="constant", constant_values=0)
        dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)[pad:-pad, pad:-pad]
        # distanceTransform counts whole pixels to the nearest outside pixel, so
        # a pixel sitting on the cut line reports 1.0 rather than 0. The cut
        # falls between the two, half a pixel away.
        return np.maximum(dist - 0.5, 0.0)

    @cached_property
    def _closest_approach(self) -> tuple[float, tuple[int, int]] | None:
        if not self.has_content:
            return None
        masked = np.where(self.content_mask, self.distance_inside_cut, np.inf)
        y, x = np.unravel_index(int(np.argmin(masked)), masked.shape)
        return float(masked[y, x]), (int(x), int(y))

    @cached_property
    def min_content_clearance_px(self) -> float | None:
        """Closest approach of the content to the cut line, in pixels."""
        approach = self._closest_approach
        return approach[0] if approach else None

    @cached_property
    def min_clearance_point(self) -> tuple[int, int] | None:
        """(x, y) of the content pixel that comes closest to the cut line."""
        approach = self._closest_approach
        return approach[1] if approach else None

    def nearest_edge(self, point: tuple[int, int]) -> str:
        """Which edge a point sits closest to — for naming it in the message."""
        x, y = point
        h, w = self.canvas.px_height, self.canvas.px_width
        distances = {"left": x, "right": w - 1 - x, "top": y, "bottom": h - 1 - y}
        return min(distances, key=distances.__getitem__)

    @cached_property
    def content_fill_ratio(self) -> float | None:
        """How much of its own bounding box the content fills.

        A logo leaves gaps; a rectangle of background fills its box entirely.
        This is what separates a real cut-out from a white box with an alpha
        channel bolted on.
        """
        bbox = self.content_bbox
        if bbox is None:
            return None
        x0, y0, x1, y1 = bbox
        area = (x1 - x0 + 1) * (y1 - y0 + 1)
        return float(np.count_nonzero(self.content_mask[y0 : y1 + 1, x0 : x1 + 1])) / area


def analyze(canvas: Canvas, product: ProductSpec) -> Features:
    return Features(canvas=canvas, product=product)
