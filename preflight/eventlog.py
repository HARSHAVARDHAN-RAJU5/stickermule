"""Append-only event log.

Every state transition is one line of JSON. This is the part that makes the
evaluation possible at all: without it there is no accuracy table, no
cost-per-file figure, and no way to argue the agent deserves to keep running.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .types import State


def new_run_id() -> str:
    return uuid.uuid4().hex[:12]


def _canonical(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def payload_sha(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()[:16]


class EventLog:
    """JSONL writer. One file per day, or an explicit path for a test run."""

    def __init__(self, path: str | Path | None = None) -> None:
        if path is None:
            day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            path = Path("runs") / f"{day}.jsonl"
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def transition(
        self,
        run_id: str,
        from_state: State | None,
        to_state: State,
        actor: str,
        **payload: Any,
    ) -> dict[str, Any]:
        """Record one edge of the state machine.

        `actor` is a check id, a model id, or a human — whoever caused the
        transition. Knowing which is what lets the eval separate the
        deterministic half of the pipeline from the model half.
        """
        record = {
            "run_id": run_id,
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "from": from_state.value if from_state else None,
            "to": to_state.value,
            "actor": actor,
            "payload": payload,
            "payload_sha": payload_sha(payload),
        }
        line = json.dumps(record, default=str)
        # Line-buffered append: a crash mid-run still leaves every prior
        # transition on disk, which is the whole point of an audit log.
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        return record

    def read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]


class NullLog(EventLog):
    """Drops events. For unit tests that are not testing the log itself."""

    def __init__(self) -> None:  # noqa: D107 - deliberately skips EventLog.__init__
        self.path = Path(os.devnull)

    def transition(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {}
