"""Stage 03 — CHECK. The rule engine.

Every check is a pure function of (Features, Config) returning a Finding, or
None when the check does not apply to this product. No check touches a file,
a model, or another check.

A check that raises is not allowed to take the run down with it: the engine
converts the exception into a CHECK_ERROR finding, which is deliberately not
auto-approvable, so a broken check sends the job to a human rather than
quietly approving it.
"""

from __future__ import annotations

from typing import Callable, Iterator

from ..analyze import Features
from ..config import Config
from ..types import Finding, Severity

CheckFn = Callable[[Features, Config], "Finding | None"]

REGISTRY: dict[str, CheckFn] = {}


def check(code: str) -> Callable[[CheckFn], CheckFn]:
    """Register a check under a stable code."""

    def decorate(fn: CheckFn) -> CheckFn:
        if code in REGISTRY:
            raise RuntimeError(f"duplicate check code: {code}")
        REGISTRY[code] = fn
        fn.code = code  # type: ignore[attr-defined]
        return fn

    return decorate


def run_checks(features: Features, config: Config) -> list[Finding]:
    """Run every registered check. Order of findings follows registry order."""
    return list(_iter_findings(features, config))


def _iter_findings(features: Features, config: Config) -> Iterator[Finding]:
    for code, fn in REGISTRY.items():
        try:
            finding = fn(features, config)
        except Exception as exc:  # noqa: BLE001 - a broken check must not pass the job
            yield Finding(
                code="CHECK_ERROR",
                severity=Severity.WARN,
                summary=f"{code} raised {type(exc).__name__}: {exc}",
                detail={"check": code, "error": type(exc).__name__},
            )
            continue
        if finding is not None:
            yield finding


# Importing the modules is what populates the registry. Kept at the bottom so
# the decorator exists before they load.
from . import detail, mockup, resolution, safezone, transparency  # noqa: E402,F401
