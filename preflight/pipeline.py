"""The state machine that drives a job from upload to routed decision.

Stages 05 (AUTO_FIX) and 06 (EXPLAIN) are not built yet, so a run currently
goes RECEIVED -> NORMALIZED -> ANALYZED -> CHECKED -> terminal. The hooks are
marked; nothing else has to move when they land.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from .analyze import analyze
from .checks import run_checks
from .config import Config, load_config
from .decide import decide
from .eventlog import EventLog, NullLog, new_run_id
from .normalize import normalize
from .types import Finding, ProductSpec, State, UndecodableFile, Verdict


@dataclass(frozen=True)
class Result:
    run_id: str
    state: State
    findings: tuple[Finding, ...]
    reason: str
    elapsed_ms: float
    dpi: float | None = None

    @property
    def approved(self) -> bool:
        return self.state is State.APPROVED


def run(
    path: str | Path,
    product: ProductSpec,
    config: Config | None = None,
    log: EventLog | None = None,
    run_id: str | None = None,
) -> Result:
    """Run one artwork file through the pipeline."""
    config = config or load_config()
    log = log if log is not None else NullLog()
    run_id = run_id or new_run_id()
    started = time.perf_counter()

    def elapsed_ms() -> float:
        return round((time.perf_counter() - started) * 1000, 2)

    path = Path(path)
    state = State.RECEIVED
    log.transition(
        run_id,
        None,
        state,
        actor="intake",
        file=path.name,
        product={
            "width_in": product.width_in,
            "height_in": product.height_in,
            "shape": product.shape.value,
            "material": product.material.value,
        },
    )

    # --- 01 NORMALIZE ---------------------------------------------------
    try:
        canvas = normalize(path)
    except UndecodableFile as exc:
        log.transition(run_id, state, State.REJECTED, actor="normalize", error=str(exc))
        return Result(
            run_id=run_id,
            state=State.REJECTED,
            findings=(),
            reason=str(exc),
            elapsed_ms=elapsed_ms(),
        )

    state = State.NORMALIZED
    log.transition(
        run_id,
        State.RECEIVED,
        state,
        actor="normalize",
        format=canvas.source_format,
        px=[canvas.px_width, canvas.px_height],
        metadata_dpi=canvas.metadata_dpi,
    )

    # --- 02 ANALYZE -----------------------------------------------------
    features = analyze(canvas, product)
    log.transition(
        run_id,
        state,
        State.ANALYZED,
        actor="analyze",
        dpi=round(features.dpi, 1),
        content_source=features.content_source,
        alpha_coverage_pct=round(features.alpha_coverage_pct, 2),
    )
    state = State.ANALYZED

    # --- 03 CHECK -------------------------------------------------------
    findings = run_checks(features, config)
    log.transition(
        run_id,
        state,
        State.CHECKED,
        actor="checks",
        findings=[f.to_dict() for f in findings],
    )
    state = State.CHECKED

    # --- 04 DECIDE ------------------------------------------------------
    verdict: Verdict = decide(findings, config)

    # --- 05 AUTO_FIX / 06 EXPLAIN --------------------------------------
    # Not built yet. When they land they sit here, between CHECKED and the
    # terminal state, and neither is allowed to change `verdict.route`.

    log.transition(
        run_id,
        state,
        verdict.route,
        actor="decide",
        reason=verdict.reason,
        elapsed_ms=elapsed_ms(),
    )

    return Result(
        run_id=run_id,
        state=verdict.route,
        findings=verdict.findings,
        reason=verdict.reason,
        elapsed_ms=elapsed_ms(),
        dpi=round(features.dpi, 1),
    )
