"""Core types.

Everything downstream is written against these. A check never sees a file path
and never sees an order; it sees Features and a ProductSpec, and it returns a
Finding. That is what keeps checks pure and testable in isolation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class Severity(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


class Shape(str, Enum):
    DIE_CUT = "die_cut"
    CIRCLE = "circle"
    SQUARE = "square"
    RECT = "rect"
    ROUNDED_RECT = "rounded_rect"


class Material(str, Enum):
    WHITE_VINYL = "white_vinyl"
    CLEAR = "clear"
    HOLOGRAPHIC = "holographic"


class State(str, Enum):
    RECEIVED = "RECEIVED"
    NORMALIZED = "NORMALIZED"
    ANALYZED = "ANALYZED"
    CHECKED = "CHECKED"
    AUTO_FIXED = "AUTO_FIXED"
    DRAFTED = "DRAFTED"
    # terminal
    APPROVED = "APPROVED"
    NEEDS_CUSTOMER_FIX = "NEEDS_CUSTOMER_FIX"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    REJECTED = "REJECTED"


TERMINAL_STATES = frozenset(
    {State.APPROVED, State.NEEDS_CUSTOMER_FIX, State.HUMAN_REVIEW, State.REJECTED}
)


class UndecodableFile(Exception):
    """Raised by stage 01 when the upload cannot become a Canvas.

    This is the only early exit in the pipeline.
    """


@dataclass(frozen=True)
class ProductSpec:
    """What the customer actually ordered.

    Nearly every check is meaningless without the ordered physical size, so
    this travels with the artwork from intake onward.
    """

    width_in: float
    height_in: float
    shape: Shape
    material: Material = Material.WHITE_VINYL

    def __post_init__(self) -> None:
        if self.width_in <= 0 or self.height_in <= 0:
            raise ValueError("product dimensions must be positive")

    @property
    def aspect_ratio(self) -> float:
        return self.width_in / self.height_in

    @property
    def is_die_cut(self) -> bool:
        return self.shape is Shape.DIE_CUT


@dataclass
class Canvas:
    """One decoded upload, normalized to RGBA.

    `metadata_dpi` is recorded but never used for a verdict: the only
    resolution that matters is pixels divided by the ordered inches.
    """

    rgba: np.ndarray  # (H, W, 4) uint8
    source_kind: str  # "raster" | "vector"
    source_format: str  # "png", "jpeg", "pdf", ...
    metadata_dpi: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if self.rgba.ndim != 3 or self.rgba.shape[2] != 4:
            raise ValueError(f"canvas must be (H, W, 4), got {self.rgba.shape}")
        if self.rgba.dtype != np.uint8:
            raise ValueError(f"canvas must be uint8, got {self.rgba.dtype}")

    @property
    def px_height(self) -> int:
        return int(self.rgba.shape[0])

    @property
    def px_width(self) -> int:
        return int(self.rgba.shape[1])

    @property
    def alpha(self) -> np.ndarray:
        return self.rgba[:, :, 3]

    @property
    def rgb(self) -> np.ndarray:
        return self.rgba[:, :, :3]

    def effective_dpi(self, product: ProductSpec) -> float:
        """Resolution the art will actually print at, at the ordered size.

        The lower of the two axes decides, because that axis is the one that
        goes soft on press.
        """
        return min(
            self.px_width / product.width_in,
            self.px_height / product.height_in,
        )


@dataclass(frozen=True)
class Finding:
    """One check's verdict.

    Carrying `measured` and `threshold` separately is not bookkeeping: it is
    what lets the customer message be specific ("96 DPI, needs 300") and what
    lets the grounding guard in stage 06 verify every number the model writes.
    """

    code: str
    severity: Severity
    summary: str  # factual, internal; not customer-facing copy
    measured: float | None = None
    threshold: float | None = None
    unit: str = ""
    suggested: tuple[float, ...] = ()  # other numbers the writer may use
    overlay: dict[str, Any] | None = None  # geometry for the annotated preview
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def is_fail(self) -> bool:
        return self.severity is Severity.FAIL

    @property
    def is_warn(self) -> bool:
        return self.severity is Severity.WARN

    def numbers(self) -> set[float]:
        """Every number this finding entitles a message to mention."""
        vals = [v for v in (self.measured, self.threshold, *self.suggested) if v is not None]
        return {float(v) for v in vals if not math.isnan(float(v))}

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity.value,
            "summary": self.summary,
            "measured": self.measured,
            "threshold": self.threshold,
            "unit": self.unit,
            "suggested": list(self.suggested),
            "detail": self.detail,
        }


@dataclass(frozen=True)
class Verdict:
    """The rollup of all findings into one of the three routes."""

    route: State
    findings: tuple[Finding, ...]
    reason: str

    @property
    def fails(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.is_fail)

    @property
    def warns(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.is_warn)
