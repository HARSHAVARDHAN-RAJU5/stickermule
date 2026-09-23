"""Build the labelled set, or run the agent against it.

    python -m evalkit build
    python -m evalkit run --write docs/results.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from preflight.config import load_config

from .dataset import CASES, build
from .harness import run_set
from .report import render


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evalkit", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="generate the labelled files and labels.yaml")
    p_build.add_argument("--out", type=Path, default=Path("evalset"))

    p_run = sub.add_parser("run", help="score the agent against the labelled set")
    p_run.add_argument("--out", type=Path, default=Path("evalset"))
    p_run.add_argument("--config", type=Path, default=None)
    p_run.add_argument("--log", type=Path, default=None, help="JSONL event log path")
    p_run.add_argument("--write", type=Path, default=None, help="write markdown here")
    p_run.add_argument(
        "--no-rebuild", action="store_true", help="use the files already on disk"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "build":
        labels = build(args.out)
        print(f"built {len(CASES)} cases in {args.out}/")
        print(f"ground truth: {labels}")
        return 0

    report = run_set(
        args.out,
        config=load_config(args.config),
        log_path=args.log,
        rebuild=not args.no_rebuild,
    )
    markdown = render(report)
    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(markdown + "\n", encoding="utf-8")
        print(f"wrote {args.write}")
    print(markdown)

    # Non-zero when the agent approved something it should have sent back.
    # That is the failure that must never pass CI quietly.
    return 1 if report.unsafe_approvals else 0


if __name__ == "__main__":
    sys.exit(main())
