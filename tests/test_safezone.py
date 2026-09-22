"""SAFE_ZONE: how close the design comes to the blade."""

from __future__ import annotations

import pytest

from preflight.checks.safezone import safe_zone
from preflight.types import ProductSpec, Severity, Shape

from .factories import content_near_bottom, inset_content

# 600 px over 2 in -> 300 dpi, so 1/16" is 18.75 px and 1/8" is 37.5 px.
SQUARE = ProductSpec(2.0, 2.0, Shape.SQUARE)
DIE_CUT = ProductSpec(2.0, 2.0, Shape.DIE_CUT)


@pytest.mark.parametrize(
    "inset_px, expected",
    [
        (60, Severity.PASS),  # 0.200 in
        (45, Severity.PASS),  # 0.150 in
        (30, Severity.WARN),  # 0.100 in — inside 1/8", outside 1/16"
        (20, Severity.WARN),  # 0.065 in
        (10, Severity.FAIL),  # 0.032 in
        (2, Severity.FAIL),  # all but touching the cut line
    ],
)
def test_severity_follows_clearance(make_features, config, inset_px, expected):
    features = make_features(inset_content(600, inset_px), SQUARE)
    finding = safe_zone(features, config)

    assert finding.severity is expected
    assert finding.measured == pytest.approx(inset_px / 300, abs=0.005)
    assert finding.unit == "in"


def test_names_the_edge_at_risk(make_features, config):
    """"Near the bottom" is the difference between a useful message and a shrug."""
    features = make_features(content_near_bottom(600, gap_px=4), SQUARE)
    finding = safe_zone(features, config)

    assert finding.severity is Severity.FAIL
    assert finding.detail["nearest_edge"] == "bottom"
    assert "bottom" in finding.summary


def test_overlay_carries_the_worst_point(make_features, config):
    features = make_features(content_near_bottom(600, gap_px=4), SQUARE)
    finding = safe_zone(features, config)

    x, y = finding.overlay["worst_point_px"]
    assert 0 <= x < 600
    assert y > 500  # near the bottom, where the ink runs out


def test_content_running_off_the_edge_fails(make_features, config):
    features = make_features(content_near_bottom(600, gap_px=0), SQUARE)
    finding = safe_zone(features, config)

    assert finding.severity is Severity.FAIL
    assert finding.measured == pytest.approx(0.0, abs=0.002)


def test_full_bleed_defeats_the_background_heuristic(make_features, config):
    """A known limitation, pinned here so it cannot regress into a silent pass.

    With no alpha channel the background is found by flooding inward from the
    four corners. A design whose corners are all ink leaves nothing to flood,
    so no content is detected and the check declines to report rather than
    inventing a measurement. A customer-supplied die line, or a bleed-aware
    background model, is what would fix it.
    """
    features = make_features(inset_content(600, 0), SQUARE)

    assert not features.has_content
    assert safe_zone(features, config) is None


def test_skipped_for_die_cut(make_features, config):
    """The cut path is derived from the art, so the clearance is arithmetic.

    Measuring it would only confirm our own offset. A customer-supplied die
    line is what would make this check meaningful for die-cut orders.
    """
    features = make_features(inset_content(600, 5), DIE_CUT)

    assert safe_zone(features, config) is None
