"""DECIDE: the rollup, and the config lever that governs it."""

from __future__ import annotations

from preflight.config import Config
from preflight.decide import decide
from preflight.types import Finding, Severity, State


def finding(code: str, severity: Severity) -> Finding:
    return Finding(code=code, severity=severity, summary=code)


CONFIG = Config(checks={}, auto_approvable_warns=frozenset({"CUT_RADIUS", "INK_LIMIT"}))


def test_all_pass_approves():
    verdict = decide([finding("RES_DPI", Severity.PASS)], CONFIG)

    assert verdict.route is State.APPROVED
    assert verdict.reason == "all checks passed"


def test_any_fail_goes_to_the_customer():
    verdict = decide(
        [finding("RES_DPI", Severity.PASS), finding("SAFE_ZONE", Severity.FAIL)],
        CONFIG,
    )

    assert verdict.route is State.NEEDS_CUSTOMER_FIX
    assert "SAFE_ZONE" in verdict.reason


def test_a_fail_outranks_an_unapprovable_warn():
    """One route out, and the customer-fixable problem is the one to report."""
    verdict = decide(
        [finding("SAFE_ZONE", Severity.FAIL), finding("RES_DPI", Severity.WARN)],
        CONFIG,
    )

    assert verdict.route is State.NEEDS_CUSTOMER_FIX


def test_approvable_warns_still_approve():
    verdict = decide([finding("CUT_RADIUS", Severity.WARN)], CONFIG)

    assert verdict.route is State.APPROVED
    assert "CUT_RADIUS" in verdict.reason


def test_a_warn_outside_the_set_escalates():
    verdict = decide([finding("RES_DPI", Severity.WARN)], CONFIG)

    assert verdict.route is State.HUMAN_REVIEW
    assert "RES_DPI" in verdict.reason


def test_a_broken_check_escalates_rather_than_approving():
    """CHECK_ERROR is deliberately absent from every auto-approvable set."""
    verdict = decide(
        [finding("RES_DPI", Severity.PASS), finding("CHECK_ERROR", Severity.WARN)],
        CONFIG,
    )

    assert verdict.route is State.HUMAN_REVIEW


def test_the_approvable_set_is_the_only_lever():
    """Tightening the agent is a config edit, not a deploy."""
    warns = [finding("CUT_RADIUS", Severity.WARN)]
    loose = Config(checks={}, auto_approvable_warns=frozenset({"CUT_RADIUS"}))
    tight = Config(checks={}, auto_approvable_warns=frozenset())

    assert decide(warns, loose).route is State.APPROVED
    assert decide(warns, tight).route is State.HUMAN_REVIEW
