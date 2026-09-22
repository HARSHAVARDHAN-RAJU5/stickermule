from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preflight.analyze import Features, analyze  # noqa: E402
from preflight.config import load_config  # noqa: E402
from preflight.normalize import normalize  # noqa: E402
from preflight.types import Material, ProductSpec, Shape  # noqa: E402


@pytest.fixture(scope="session")
def config():
    return load_config()


@pytest.fixture
def make_features(tmp_path):
    """Write an RGBA array to disk, decode it, and analyze it.

    Round-tripping through a real file rather than constructing a Canvas by
    hand means the tests exercise stage 01 too — including the EXIF and
    magic-byte handling that everything downstream silently depends on.
    """

    from .factories import save

    def _make(rgba, product: ProductSpec, name: str = "art.png") -> Features:
        path = save(rgba, tmp_path / name)
        return analyze(normalize(path), product)

    return _make


@pytest.fixture
def die_cut_2in():
    return ProductSpec(3.0, 3.0, Shape.DIE_CUT, Material.WHITE_VINYL)


@pytest.fixture
def square_2in():
    return ProductSpec(2.0, 2.0, Shape.SQUARE, Material.WHITE_VINYL)
