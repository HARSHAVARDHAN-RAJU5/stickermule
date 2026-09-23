"""Evaluation harness for the artwork pre-flight agent.

Built in the same weekend as the agent, not after it. An agent whose accuracy
nobody has measured cannot be argued for, and cannot be honestly removed
either.
"""

from .dataset import CASES, build
from .harness import Report, run_set
from .report import render

__all__ = ["CASES", "Report", "build", "render", "run_set"]
