"""Stage 06, part one: the facts a message is allowed to contain.

A writer never sees a Finding. It sees these sentences, written here, by code,
with every number already rounded the way a customer should read it. The guard
checks the model's message against the same facts, so the model cannot say
anything the checks did not measure.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..types import Finding, ProductSpec, Severity

SHAPE_WORDS = {
    "die_cut": "die-cut",
    "circle": "circle",
    "square": "square",
    "rect": "rectangle",
    "rounded_rect": "rounded rectangle",
}

# The codes a customer can act on. CHECK_ERROR is our problem, not theirs, and
# never reaches a customer message.
CUSTOMER_FACING = (
    "RES_DPI",
    "ALPHA_MISSING",
    "ALPHA_FAKE",
    "SAFE_ZONE",
    "DETAIL_AT_CUT",
    "STICKER_MOCKUP",
)


@dataclass(frozen=True)
class Fact:
    code: str
    severity: Severity
    text: str  # the problem and what to upload instead, in customer words
    numbers: frozenset[float]  # every number this fact entitles a message to use


def _inches(value: float) -> str:
    if value < 0.005:
        return "right up to"
    return f"within {value:.2f} in of"


def _text(finding: Finding, product: ProductSpec) -> str:
    code, m = finding.code, finding.measured
    side = finding.detail.get("nearest_edge", "")
    near = f" near the {side} edge" if side else ""

    if code == "RES_DPI":
        target, need_w, need_h = finding.suggested[:3]
        return (
            f"At the ordered size of {product.width_in:g} x {product.height_in:g} in, "
            f"the file works out to {m:g} DPI (dots per inch), so it will print blurry. "
            f"Stickers print sharply at {target:g} DPI, which at this size means an "
            f"image at least {need_w:g} x {need_h:g} pixels. Please upload a larger "
            f"original, not an enlarged copy of this one."
        )
    if code == "ALPHA_MISSING":
        return (
            "A die-cut sticker is cut around the outline of the design, so the file "
            "needs a transparent background. This file has none, so it would be cut "
            "as a plain rectangle. Please upload a PNG with a transparent background."
        )
    if code == "ALPHA_FAKE":
        return (
            "The design sits on a solid light-coloured box. That box is part of the "
            "picture, so it would print as a white rectangle behind the design. "
            "Please remove the box so only the design itself is left on a "
            "transparent background."
        )
    if code in ("SAFE_ZONE", "DETAIL_AT_CUT"):
        keep = finding.suggested[0] if finding.suggested else 0.125
        what = "Part of the design" if code == "SAFE_ZONE" else "Small detail such as lettering"
        risk = "will be cut into" if code == "SAFE_ZONE" else "will be cut off"
        if finding.severity is Severity.WARN:
            risk = "is closer than we recommend"
        return (
            f"{what} comes {_inches(m or 0.0)} the cut line{near}, so it {risk}. "
            f"Please keep everything important at least {keep:g} in (1/8 inch) "
            f"inside the edge."
        )
    if code == "STICKER_MOCKUP":
        return (
            "The file looks like a picture of a finished sticker: it already has a "
            "white border and a drop shadow drawn in, and both would be printed. "
            "Please upload the original artwork instead."
        )
    raise ValueError(f"no customer wording for {code}")


def facts_for(findings: tuple[Finding, ...], product: ProductSpec) -> list[Fact]:
    """The customer-facing problems in one run, worst first."""
    order = {Severity.FAIL: 0, Severity.WARN: 1}
    chosen = sorted(
        (f for f in findings if f.severity is not Severity.PASS and f.code in CUSTOMER_FACING),
        key=lambda f: (order[f.severity], CUSTOMER_FACING.index(f.code)),
    )
    return [
        Fact(
            code=f.code,
            severity=f.severity,
            text=_text(f, product),
            numbers=frozenset(f.numbers()),
        )
        for f in chosen
    ]


def allowed_numbers(facts: list[Fact], product: ProductSpec) -> set[float]:
    """Every number a message about these facts may contain."""
    allowed = {float(product.width_in), float(product.height_in), float(len(facts))}
    for fact in facts:
        allowed |= fact.numbers
    return allowed
