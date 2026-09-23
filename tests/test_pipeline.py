"""End to end: a file goes in, a route and an audit trail come out."""

from __future__ import annotations

from preflight import run
from preflight.eventlog import EventLog
from preflight.types import ProductSpec, Severity, Shape, State

from .factories import clean_die_cut, inset_content, opaque_design, save


def test_a_good_die_cut_file_is_approved(tmp_path, config):
    path = save(clean_die_cut(900), tmp_path / "logo.png")
    log = EventLog(tmp_path / "run.jsonl")

    result = run(path, ProductSpec(3.0, 3.0, Shape.DIE_CUT), config=config, log=log)

    assert result.state is State.APPROVED
    assert result.dpi == 300.0
    assert all(f.severity is Severity.PASS for f in result.findings)


def test_an_opaque_die_cut_upload_goes_back_to_the_customer(tmp_path, config):
    path = save(opaque_design(900), tmp_path / "logo.png")

    result = run(path, ProductSpec(3.0, 3.0, Shape.DIE_CUT), config=config)

    assert result.state is State.NEEDS_CUSTOMER_FIX
    assert "ALPHA_MISSING" in result.reason


def test_low_resolution_and_a_tight_edge_both_reported(tmp_path, config):
    """The customer gets every problem at once, not one round trip each."""
    path = save(inset_content(200, 3), tmp_path / "small.png")

    result = run(path, ProductSpec(2.0, 2.0, Shape.SQUARE), config=config)

    codes = {f.code for f in result.findings if f.is_fail}
    assert codes == {"RES_DPI", "SAFE_ZONE", "DETAIL_AT_CUT"}
    assert result.state is State.NEEDS_CUSTOMER_FIX


def test_an_undecodable_file_is_rejected_not_crashed(tmp_path, config):
    path = tmp_path / "invoice.png"
    path.write_bytes(b"%PDF-1.7\n% not a png at all\n")

    result = run(path, ProductSpec(2.0, 2.0, Shape.SQUARE), config=config)

    assert result.state is State.REJECTED
    assert "pdf" in result.reason


def test_every_transition_is_logged(tmp_path, config):
    path = save(clean_die_cut(900), tmp_path / "logo.png")
    log = EventLog(tmp_path / "run.jsonl")

    result = run(path, ProductSpec(3.0, 3.0, Shape.DIE_CUT), config=config, log=log)
    events = log.read()

    assert [e["to"] for e in events] == [
        "RECEIVED",
        "NORMALIZED",
        "ANALYZED",
        "CHECKED",
        "APPROVED",
    ]
    assert {e["run_id"] for e in events} == {result.run_id}
    assert [e["actor"] for e in events] == [
        "intake",
        "normalize",
        "analyze",
        "checks",
        "decide",
    ]
    assert all(e["payload_sha"] for e in events)


def test_a_rejected_run_still_leaves_a_trail(tmp_path, config):
    path = tmp_path / "empty.png"
    path.write_bytes(b"")
    log = EventLog(tmp_path / "run.jsonl")

    run(path, ProductSpec(2.0, 2.0, Shape.SQUARE), config=config, log=log)

    assert [e["to"] for e in log.read()] == ["RECEIVED", "REJECTED"]
