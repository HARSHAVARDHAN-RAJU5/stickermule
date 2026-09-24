"""Stage 06, part two: the grounding guard.

A model writes the message; this decides whether the message may be sent. It
is plain code, so it cannot be talked round. It rejects a message that:

  - writes a number the findings do not contain,
  - leaves out a problem that was found,
  - brings up a kind of problem that was not found,
  - promises something (a refund, an approval, a fix we will make),
  - is empty or too long.

Every reason comes back as a plain sentence. The same sentences are shown to
the model when it is asked to try again, and to a person reading the log.
"""

from __future__ import annotations

import re

from ..types import ProductSpec, Severity
from .facts import Fact, allowed_numbers

MAX_WORDS = 170

# A number, optionally a fraction. The look-behind stops "3.5" being read as
# "5", but lets "3x3" read as two threes.
_NUMBER = re.compile(r"(?<![\d.])(\d[\d,]*(?:\.\d+)?|\.\d+)(?:\s*/\s*(\d+))?")
# "1. " or "2) " at the start of a line is list numbering, not a claim.
_LIST_MARK = re.compile(r"^\s*\d+[.)]\s", re.MULTILINE)

# Words that mean the message is talking about this problem.
TOPIC = {
    "RES_DPI": ("dpi", "resolution", "blurry", "blur", "pixel", "sharp"),
    "ALPHA_MISSING": ("transparen",),
    "ALPHA_FAKE": ("box", "background"),
    "SAFE_ZONE": ("edge", "cut line", "cut into", "cut off", "trim"),
    "DETAIL_AT_CUT": ("edge", "cut line", "cut into", "cut off", "trim"),
    "STICKER_MOCKUP": ("shadow", "mockup", "mock-up", "border"),
}

# Words that only belong in a message when that problem was found. Narrower
# than TOPIC on purpose: a transparency message may say "background", and a
# die-cut message may say "cut", without inventing anything.
INVENTED = {
    "resolution": (("RES_DPI",), ("dpi", "resolution", "blurry", "pixelat", "low-res")),
    "transparency": (("ALPHA_MISSING", "ALPHA_FAKE"), ("transparen",)),
    "edges": (("SAFE_ZONE", "DETAIL_AT_CUT"), ("cut off", "cut into", "safe zone", "too close")),
    "a mockup": (("STICKER_MOCKUP",), ("shadow", "mockup", "mock-up")),
}

PROMISES = (
    "approved",
    "refund",
    "discount",
    "guarantee",
    "free of charge",
    "for free",
    "no charge",
    "we'll fix",
    "we will fix",
    "we fixed",
    "we've fixed",
    "we can fix",
)


def _numbers(text: str) -> list[tuple[str, float, int]]:
    """(as written, value, decimal places) for every number in the text."""
    found = []
    for match in _NUMBER.finditer(_LIST_MARK.sub("", text)):
        whole, denominator = match.group(1).replace(",", ""), match.group(2)
        if denominator:
            if float(denominator) == 0:
                continue
            found.append((match.group(0), float(whole) / float(denominator), 6))
        else:
            places = len(whole.split(".")[1]) if "." in whole else 0
            found.append((match.group(0), float(whole), places))
    return found


def _is_allowed(value: float, places: int, allowed: set[float]) -> bool:
    """True if some allowed number, rounded as written, gives this value.

    "0.04 in" is a fair way to write 0.0383; "99%" is a fair way to write 98.9.
    Nothing else is.
    """
    return any(abs(round(a, places) - value) < 1e-6 for a in allowed)


def check(message: str, facts: list[Fact], product: ProductSpec) -> list[str]:
    """Reasons the message may not be sent. An empty list means it may."""
    problems: list[str] = []
    text = message.strip()
    lower = text.lower()

    if not text:
        return ["the message is empty"]

    words = len(text.split())
    if words > MAX_WORDS:
        problems.append(f"it is {words} words long; the limit is {MAX_WORDS}")

    allowed = allowed_numbers(facts, product)
    for written, value, places in _numbers(text):
        if not _is_allowed(value, places, allowed):
            problems.append(f'it says "{written}", which is not a number the checks measured')

    for fact in facts:
        if not any(word in lower for word in TOPIC[fact.code]):
            kind = "warning" if fact.severity is Severity.WARN else "problem"
            problems.append(f"it does not mention the {fact.code} {kind}: {fact.text}")

    found = {fact.code for fact in facts}
    for name, (codes, words_) in INVENTED.items():
        if found.isdisjoint(codes):
            hit = next((w for w in words_ if w in lower), None)
            if hit:
                problems.append(
                    f'it says "{hit}", but no problem with {name} was found'
                )

    for promise in PROMISES:
        if promise in lower:
            problems.append(f'it says "{promise}"; the message must not promise anything')

    return problems
