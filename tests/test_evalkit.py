"""The evaluation harness is part of the agent, so it is tested like one.

The last test here is the one that matters: it fails the build if the agent
ever approves a file that should have gone back to the customer.
"""

from __future__ import annotations

import yaml

from evalkit.dataset import CASES, build
from evalkit.harness import run_set
from evalkit.report import render


def test_every_case_has_a_unique_name():
    names = [c.name for c in CASES]
    assert len(names) == len(set(names))


def test_build_writes_a_file_and_a_label_for_every_case(tmp_path):
    labels_path = build(tmp_path)
    labels = yaml.safe_load(labels_path.read_text(encoding="utf-8"))

    assert len(labels["cases"]) == len(CASES)
    for case in CASES:
        assert (tmp_path / case.filename).is_file()


def test_ground_truth_routes_are_consistent_with_the_labels():
    """A case expecting a FAIL finding must expect the customer-fix route.

    The labels are hand-written; this keeps them honest against the routing
    rules they are supposed to be testing.
    """
    from preflight.types import Severity, State

    for case in CASES:
        has_fail = any(s is Severity.FAIL for s in case.expect.values())
        if has_fail:
            assert case.expect_route is State.NEEDS_CUSTOMER_FIX, case.name
        elif case.expect_route is State.NEEDS_CUSTOMER_FIX:
            raise AssertionError(f"{case.name} expects a fix route but no FAIL finding")


def test_the_agent_never_approves_what_should_go_back(tmp_path, config):
    report = run_set(tmp_path, config=config)

    assert report.unsafe_approvals == [], [
        r.case.name for r in report.unsafe_approvals
    ]


def test_report_always_carries_its_caveats(tmp_path, config):
    """The numbers must never be publishable without the paragraph that
    qualifies them, so it is generated, not written by hand."""
    markdown = render(run_set(tmp_path, config=config))

    assert "How to read these numbers" in markdown
    assert "The set is synthetic" in markdown
    assert "in-sample" in markdown
