"""Artwork pre-flight agent.

Deterministic checks decide. A language model only explains.
"""

from .config import Config, load_config
from .pipeline import Result, run
from .types import (
    Canvas,
    Finding,
    Material,
    ProductSpec,
    Severity,
    Shape,
    State,
    UndecodableFile,
    Verdict,
)

__all__ = [
    "Canvas",
    "Config",
    "Finding",
    "Material",
    "ProductSpec",
    "Result",
    "Severity",
    "Shape",
    "State",
    "UndecodableFile",
    "Verdict",
    "load_config",
    "run",
]
