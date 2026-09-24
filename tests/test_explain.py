"""Stage 06: the message writer and the guard that stands in front of it.

The writers here are fakes that say exactly what each test needs, including
the lies a real model might tell. No test calls a real model.
"""

from __future__ import annotations

import pytest

from evalkit.dataset import CASES, build
from preflight.eventlog import EventLog
from preflight.explain import explain, template
from preflight.explain.facts import facts_for
from preflight.explain.guard import check
from preflight.explain.writers import Reply, WriterUnavailable
from preflight.pipeline import run
from preflight.types import Finding, ProductSpec, Severity, Shape, State

THREE_IN = ProductSpec(3.0, 3.0, Shape.DIE_CUT)

LOW_RES = Finding(
    code="RES_DPI",
    severity=Severity.FAIL,
    summary="96 DPI at the ordered size; will print visibly soft",
    measured=96.0,
    threshold=150.0,
    unit="dpi",
    suggested=(300.0, 900.0, 900.0),
)

GOOD = (
    "Your file is 96 DPI at 3 x 3 in, so it will print blurry. Please upload an "
    "image of at least 900 x 900 pixels, which gives 300 DPI."
)


class Scripted:
    """A writer that returns the given replies in order."""

    name = "fake:scripted"

    def __init__(self, *replies: str):
        self.replies = list(replies)
        self.calls = 0

    def complete(self, system: str, prompt: str) -> Reply:
        self.calls += 1
        return Reply(self.replies.pop(0))


class Down:
    name = "fake:down"

    def complete(self, system: str, prompt: str) -> Reply:
        raise WriterUnavailable("connection refused")


def problems(message: str, findings=(LOW_RES,), product=THREE_IN) -> list[str]:
    return check(message, facts_for(tuple(findings), product), product)


# --- the guard ----------------------------------------------------------


def test_a_faithful_message_passes():
    assert problems(GOOD) == []


def test_an_invented_number_is_caught():
    found = problems(GOOD.replace("96 DPI", "72 DPI"))
    assert any('"72"' in p for p in found)


def test_a_number_may_be_rounded_but_not_changed():
    edge = Finding(
        code="SAFE_ZONE",
        severity=Severity.FAIL,
        summary="content comes within 0.038 in of the cut line near the left",
        measured=0.0383,
        threshold=0.0625,
        unit="in",
        suggested=(0.125,),
    )
    square = ProductSpec(2.0, 2.0, Shape.SQUARE)
    ok = "Part of the design is within 0.04 in of the cut line. Keep it 1/8 inch from the edge."
    assert problems(ok, [edge], square) == []
    assert problems(ok.replace("0.04", "0.05"), [edge], square)
    assert problems(ok.replace("1/8", "1/32"), [edge], square)


def test_list_numbering_and_thousands_separators_are_read_correctly():
    wide = ProductSpec(10.0, 2.0, Shape.DIE_CUT)
    finding = Finding(
        code="RES_DPI",
        severity=Severity.FAIL,
        summary="",
        measured=120.0,
        threshold=150.0,
        unit="dpi",
        suggested=(300.0, 3000.0, 600.0),
    )
    message = "1. Your file is 120 DPI, so it will print blurry. Upload 3,000 x 600 pixels."
    assert problems(message, [finding], wide) == []


def test_a_problem_that_was_found_must_be_mentioned():
    found = problems("Thanks for your order! Please upload a new file.")
    assert any("does not mention the RES_DPI" in p for p in found)


def test_a_problem_that_was_not_found_must_not_be_mentioned():
    found = problems(GOOD + " Also make sure the background is transparent.")
    assert any("transparency" in p for p in found)


@pytest.mark.parametrize("promise", ["we'll fix it for you", "you'll get a refund", "for free"])
def test_the_message_may_not_promise_anything(promise):
    assert problems(f"{GOOD} Or {promise}.")


def test_an_empty_or_rambling_message_is_caught():
    assert problems("   ") == ["the message is empty"]
    assert any("words long" in p for p in problems(GOOD + " blurry" * 200))


def test_the_template_passes_the_guard_on_every_labelled_file(tmp_path):
    """The fallback is what gets sent when a model fails, so it must never
    fail the guard itself."""
    build(tmp_path)
    drafted = 0
    for case in CASES:
        result = run(tmp_path / case.filename, case.product, draft=True)
        if result.message is None:
            continue
        drafted += 1
        facts = facts_for(result.findings, case.product)
        assert check(result.message.text, facts, case.product) == [], case.name
    assert drafted == sum(c.expect_route is State.NEEDS_CUSTOMER_FIX for c in CASES)


# --- asking a writer ----------------------------------------------------


def test_no_writer_means_the_template():
    draft = explain((LOW_RES,), THREE_IN, writer=None)
    assert draft.source == "template"
    assert draft.text == template(facts_for((LOW_RES,), THREE_IN), THREE_IN)


def test_a_good_first_draft_is_used_as_written():
    writer = Scripted(GOOD)
    draft = explain((LOW_RES,), THREE_IN, writer)
    assert (draft.source, draft.text, writer.calls) == ("model", GOOD, 1)


def test_a_rejected_draft_gets_one_retry_with_the_reasons():
    writer = Scripted(GOOD.replace("96", "72"), GOOD)
    draft = explain((LOW_RES,), THREE_IN, writer)
    assert draft.source == "model"
    assert draft.text == GOOD
    assert [a.passed for a in draft.attempts] == [False, True]
    assert any('"72"' in p for p in draft.attempts[0].problems)


def test_two_rejected_drafts_fall_back_to_the_template():
    bad = GOOD.replace("96", "72")
    draft = explain((LOW_RES,), THREE_IN, Scripted(bad, bad))
    assert draft.source == "fallback"
    assert "72" not in draft.text
    assert len(draft.attempts) == 2


def test_an_unreachable_model_falls_back_without_retrying():
    draft = explain((LOW_RES,), THREE_IN, Down())
    assert draft.source == "fallback"
    assert draft.attempts[0].error == "connection refused"
    assert len(draft.attempts) == 1


def test_nothing_to_say_means_no_message():
    passing = Finding(code="RES_DPI", severity=Severity.PASS, summary="300 DPI")
    assert explain((passing,), THREE_IN, Scripted(GOOD)) is None


# --- inside the pipeline ------------------------------------------------


def _case(name, tmp_path):
    build(tmp_path)
    case = next(c for c in CASES if c.name == name)
    return tmp_path / case.filename, case.product


def test_a_lying_model_cannot_change_the_route(tmp_path):
    path, product = _case("res_fail_96", tmp_path)
    liar = Scripted("Great news, your file is approved!", "Approved, 600 DPI.")
    result = run(path, product, draft=True, writer=liar)
    assert result.state is State.NEEDS_CUSTOMER_FIX
    assert result.message.source == "fallback"


def test_an_approved_file_never_reaches_the_writer(tmp_path):
    path, product = _case("clean_die_cut_300", tmp_path)
    writer = Scripted()
    result = run(path, product, draft=True, writer=writer)
    assert result.state is State.APPROVED
    assert result.message is None and writer.calls == 0


def test_the_draft_is_logged_between_checked_and_the_route(tmp_path):
    path, product = _case("res_fail_96", tmp_path)
    log = EventLog(tmp_path / "log.jsonl")
    run(path, product, log=log, draft=True, writer=Scripted(GOOD.replace("3 x 3", "3x3")))
    edges = [(e["from"], e["to"]) for e in log.read()]
    assert edges[-2:] == [("CHECKED", "DRAFTED"), ("DRAFTED", "NEEDS_CUSTOMER_FIX")]
