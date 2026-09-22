"""Threshold configuration.

Checks read their numbers from here, never from literals in their own body,
so the whole agent can be tightened or loosened by editing one YAML file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "thresholds.yaml"


@dataclass(frozen=True)
class Config:
    checks: dict[str, dict[str, Any]]
    auto_approvable_warns: frozenset[str]

    def for_check(self, code: str) -> dict[str, Any]:
        try:
            return self.checks[code]
        except KeyError:
            raise KeyError(
                f"no thresholds configured for check {code!r}; add it to thresholds.yaml"
            ) from None


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    decide = raw.get("decide", {}) or {}
    return Config(
        checks=raw.get("checks", {}) or {},
        auto_approvable_warns=frozenset(decide.get("auto_approvable_warns", []) or []),
    )
