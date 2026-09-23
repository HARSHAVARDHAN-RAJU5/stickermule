"""Run the labelled set and score the agent against it.

Two scores, because they answer different questions:

  per check    precision and recall on FAIL severity. A miss and a false alarm
               cost completely different things — a miss reaches the press, a
               false alarm annoys a customer who did nothing wrong — so one
               blended accuracy number would hide the only distinction that
               matters here.

  per route    a confusion matrix over the three routes plus REJECTED. One cell
               in it decides whether this is allowed to run unsupervised:
               should have gone back to the customer, was approved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from preflight.config import Config, load_config
from preflight.eventlog import EventLog
from preflight.pipeline import run
from preflight.types import Severity, State

from .dataset import CASES, Case, build

ROUTES = (State.APPROVED, State.NEEDS_CUSTOMER_FIX, State.HUMAN_REVIEW, State.REJECTED)


@dataclass(frozen=True)
class CaseResult:
    case: Case
    actual_route: State
    actual: dict[str, Severity]
    elapsed_ms: float
    reason: str

    @property
    def route_ok(self) -> bool:
        return self.actual_route is self.case.expect_route

    @property
    def findings_ok(self) -> bool:
        return self.actual == self.case.expect

    def disagreements(self) -> list[tuple[str, str, str]]:
        """(code, expected, actual) for every check that did not match."""
        codes = sorted(set(self.case.expect) | set(self.actual))
        out = []
        for code in codes:
            want = self.case.expect.get(code)
            got = self.actual.get(code)
            if want is not got:
                out.append(
                    (code, want.value if want else "absent", got.value if got else "absent")
                )
        return out


@dataclass
class CheckScore:
    code: str
    tp: int = 0  # should fail, did fail
    fp: int = 0  # should not fail, did fail
    fn: int = 0  # should fail, did not
    exact: int = 0  # severity matched exactly
    seen: int = 0

    @property
    def precision(self) -> float | None:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else None

    @property
    def recall(self) -> float | None:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else None

    @property
    def exact_rate(self) -> float:
        return self.exact / self.seen if self.seen else 0.0


@dataclass
class Report:
    results: list[CaseResult]
    checks: dict[str, CheckScore] = field(default_factory=dict)
    confusion: dict[tuple[State, State], int] = field(default_factory=dict)

    @property
    def route_accuracy(self) -> float:
        return sum(r.route_ok for r in self.results) / len(self.results)

    @property
    def unsafe_approvals(self) -> list[CaseResult]:
        """The cell that decides whether the agent may run unsupervised."""
        return [
            r
            for r in self.results
            if r.actual_route is State.APPROVED
            and r.case.expect_route is State.NEEDS_CUSTOMER_FIX
        ]

    @property
    def latencies(self) -> np.ndarray:
        return np.array([r.elapsed_ms for r in self.results])

    def percentile(self, p: float) -> float:
        return float(np.percentile(self.latencies, p))


def run_set(
    out_dir: str | Path = "evalset",
    config: Config | None = None,
    log_path: str | Path | None = None,
    rebuild: bool = True,
) -> Report:
    out_dir = Path(out_dir)
    if rebuild:
        build(out_dir)
    config = config or load_config()
    log = EventLog(log_path) if log_path else None

    results = [_run_case(case, out_dir, config, log) for case in CASES]
    return _score(results)


def _run_case(case: Case, out_dir: Path, config: Config, log: EventLog | None) -> CaseResult:
    result = run(out_dir / case.filename, case.product, config=config, log=log)
    return CaseResult(
        case=case,
        actual_route=result.state,
        actual={f.code: f.severity for f in result.findings},
        elapsed_ms=result.elapsed_ms,
        reason=result.reason,
    )


def _score(results: list[CaseResult]) -> Report:
    report = Report(results=results)

    for result in results:
        key = (result.case.expect_route, result.actual_route)
        report.confusion[key] = report.confusion.get(key, 0) + 1

        codes = set(result.case.expect) | set(result.actual)
        for code in codes:
            score = report.checks.setdefault(code, CheckScore(code))
            want = result.case.expect.get(code)
            got = result.actual.get(code)

            score.seen += 1
            if want is got:
                score.exact += 1

            should_fail = want is Severity.FAIL
            did_fail = got is Severity.FAIL
            if should_fail and did_fail:
                score.tp += 1
            elif should_fail and not did_fail:
                score.fn += 1
            elif did_fail and not should_fail:
                score.fp += 1

    return report
