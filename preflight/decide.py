"""Stage 04 — DECIDE. Roll findings up into one of three routes.

The rules are four lines. The judgement is in the auto-approvable warn set,
which lives in config: it is the single lever an operations lead owns, and
tightening it after a bad week must not need a deploy.
"""

from __future__ import annotations

from collections.abc import Sequence

from .config import Config
from .types import Finding, State, Verdict


def decide(findings: Sequence[Finding], config: Config) -> Verdict:
    findings = tuple(findings)
    fails = [f for f in findings if f.is_fail]
    warns = [f for f in findings if f.is_warn]

    if fails:
        codes = ", ".join(sorted(f.code for f in fails))
        return Verdict(
            route=State.NEEDS_CUSTOMER_FIX,
            findings=findings,
            reason=f"failing checks: {codes}",
        )

    blocking = sorted({w.code for w in warns} - config.auto_approvable_warns)
    if blocking:
        return Verdict(
            route=State.HUMAN_REVIEW,
            findings=findings,
            reason=f"warnings outside the auto-approvable set: {', '.join(blocking)}",
        )

    if warns:
        codes = ", ".join(sorted({w.code for w in warns}))
        return Verdict(
            route=State.APPROVED,
            findings=findings,
            reason=f"approved with notes: {codes}",
        )

    return Verdict(route=State.APPROVED, findings=findings, reason="all checks passed")
