"""Stage 06 EXPLAIN: word the message to the customer.

The route is already decided when this runs, and nothing here can change it.
A model is asked for a message; the guard checks it; a rejected draft gets one
more try with the guard's reasons attached; if that fails too, or the model
cannot be reached, the fixed template is sent instead. The customer always
gets a message that says only what the checks found.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from ..types import Finding, ProductSpec
from .facts import SHAPE_WORDS, Fact, facts_for
from .guard import check
from .writers import Writer, WriterUnavailable, get_writer

__all__ = ["Attempt", "Draft", "explain", "get_writer", "template"]

SYSTEM = """\
You write a short message to a customer whose sticker artwork cannot be printed as uploaded.

Rules:
- Use only the facts you are given. Do not add problems, causes or advice they do not support.
- Every number you write must appear in the facts, written the same way. Do not convert units. Do not number your points.
- Cover every problem listed: say what is wrong in plain words and what to upload instead.
- Do not promise anything: no refunds, discounts, dates, approvals, and do not offer to fix the file for them.
- Friendly and direct. Under 120 words. Plain text, no markdown, no subject line, do not sign it."""


@dataclass(frozen=True)
class Attempt:
    text: str
    problems: tuple[str, ...]
    elapsed_ms: float
    tokens_in: int | None = None
    tokens_out: int | None = None
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.error is None and not self.problems

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "problems": list(self.problems),
            "passed": self.passed,
            "elapsed_ms": self.elapsed_ms,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "error": self.error,
        }


@dataclass(frozen=True)
class Draft:
    text: str
    writer: str  # who was asked: "template", "gemini:gemini-3.1-flash-lite", ...
    source: str  # who wrote what is sent: "model", "template" or "fallback"
    attempts: tuple[Attempt, ...]
    elapsed_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "writer": self.writer,
            "source": self.source,
            "attempts": [a.to_dict() for a in self.attempts],
            "elapsed_ms": self.elapsed_ms,
        }


def _order(product: ProductSpec) -> str:
    return (
        f"{product.width_in:g} x {product.height_in:g} in, "
        f"{SHAPE_WORDS[product.shape.value]}, {product.material.value.replace('_', ' ')}"
    )


def template(facts: list[Fact], product: ProductSpec) -> str:
    """The fixed wording. Used with no model, and whenever a model fails."""
    lines = [
        f"Hi, thanks for your order. Before we print your {_order(product)} sticker, "
        f"the artwork needs a change:",
        "",
        *[f"- {fact.text}" for fact in facts],
        "",
        "Upload a new file and we will check it again straight away.",
    ]
    return "\n".join(lines)


def _prompt(facts: list[Fact], product: ProductSpec) -> str:
    listed = "\n".join(f"- {fact.text}" for fact in facts)
    return (
        f"The customer ordered: {_order(product)} sticker.\n\n"
        f"Problems found with their artwork:\n{listed}\n\n"
        f"Write the message."
    )


def _retry(prompt: str, problems: tuple[str, ...]) -> str:
    reasons = "\n".join(f"- {p}" for p in problems)
    return (
        f"{prompt}\n\nYour previous draft was rejected by an automatic checker:\n"
        f"{reasons}\n\nWrite it again and fix every one of these."
    )


def explain(
    findings: tuple[Finding, ...],
    product: ProductSpec,
    writer: Writer | None,
    max_attempts: int = 2,
) -> Draft | None:
    """A guarded customer message, or None if nothing needs saying."""
    started = time.perf_counter()
    facts = facts_for(findings, product)
    if not facts:
        return None

    def done(text: str, name: str, source: str, attempts: list[Attempt]) -> Draft:
        elapsed = round((time.perf_counter() - started) * 1000, 2)
        return Draft(text, name, source, tuple(attempts), elapsed)

    if writer is None:
        return done(template(facts, product), "template", "template", [])

    attempts: list[Attempt] = []
    prompt = _prompt(facts, product)
    for _ in range(max_attempts):
        t0 = time.perf_counter()
        try:
            reply = writer.complete(SYSTEM, prompt)
        except WriterUnavailable as exc:
            ms = round((time.perf_counter() - t0) * 1000, 2)
            attempts.append(Attempt("", (), ms, error=str(exc)))
            break  # an unreachable model will not be reachable a second later
        ms = round((time.perf_counter() - t0) * 1000, 2)
        text = reply.text.strip()
        problems = tuple(check(text, facts, product))
        attempts.append(Attempt(text, problems, ms, reply.tokens_in, reply.tokens_out))
        if not problems:
            return done(text, writer.name, "model", attempts)
        prompt = _retry(_prompt(facts, product), problems)

    return done(template(facts, product), writer.name, "fallback", attempts)
