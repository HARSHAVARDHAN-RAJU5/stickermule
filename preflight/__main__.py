"""Command line entry point.

    python -m preflight artwork.png --size 3x3 --shape die_cut
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_config
from .eventlog import EventLog
from .pipeline import run
from .types import Material, ProductSpec, Severity, Shape

MARK = {Severity.PASS: "pass", Severity.WARN: "warn", Severity.FAIL: "FAIL"}


def parse_size(text: str) -> tuple[float, float]:
    try:
        width, height = (float(part) for part in text.lower().split("x", 1))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"size must look like 3x2 (inches), got {text!r}"
        ) from None
    return width, height


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="preflight", description=__doc__)
    parser.add_argument("artwork", type=Path)
    parser.add_argument(
        "--size", required=True, type=parse_size, help="ordered size in inches, e.g. 3x2"
    )
    parser.add_argument(
        "--shape", default=Shape.DIE_CUT.value, choices=[s.value for s in Shape]
    )
    parser.add_argument(
        "--material", default=Material.WHITE_VINYL.value, choices=[m.value for m in Material]
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--log", type=Path, default=None, help="JSONL event log path")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    width, height = args.size
    product = ProductSpec(width, height, Shape(args.shape), Material(args.material))

    result = run(
        args.artwork,
        product,
        config=load_config(args.config),
        log=EventLog(args.log) if args.log else None,
    )

    if args.json:
        print(
            json.dumps(
                {
                    "run_id": result.run_id,
                    "state": result.state.value,
                    "reason": result.reason,
                    "dpi": result.dpi,
                    "elapsed_ms": result.elapsed_ms,
                    "findings": [f.to_dict() for f in result.findings],
                },
                indent=2,
            )
        )
    else:
        print(f"{args.artwork.name} -> {result.state.value}  ({result.elapsed_ms:.0f} ms)")
        print(f"  {result.reason}")
        for finding in result.findings:
            print(f"  [{MARK[finding.severity]:>4}] {finding.code:<14} {finding.summary}")

    # Exit code carries the route, so the CLI composes in a shell pipeline.
    return 0 if result.approved else 1


if __name__ == "__main__":
    sys.exit(main())
