"""DETAIL_AT_CUT: is anything that matters about to be cut off?

The check that replaced an escalation. The point of every test here is the
difference between an edge and fine detail — flat colour running off the edge
is deliberate, lettering running off the edge is a mistake, and both are edges.
"""

from __future__ import annotations

import pytest

from evalkit.art import RED, WHITE, blank, rect, text_block, two_tone
from preflight.checks.detail import detail_at_cut
from preflight.types import ProductSpec, Severity, Shape

# 600 px over 2 in -> 300 dpi, so 1/16" is 18.75 px and 1/8" is 37.5 px.
SQUARE = ProductSpec(2.0, 2.0, Shape.SQUARE)
DIE_CUT = ProductSpec(2.0, 2.0, Shape.DIE_CUT)


def test_lettering_running_off_the_edge_fails(make_features, config):
    art = text_block(blank(600, fill=WHITE), 10, 10, 590, 590)
    finding = detail_at_cut(make_features(art, SQUARE), config)

    assert finding.severity is Severity.FAIL
    assert finding.measured < 0.0625


def test_lettering_well_inside_passes(make_features, config):
    art = text_block(blank(600, fill=WHITE), 120, 120, 480, 480)
    finding = detail_at_cut(make_features(art, SQUARE), config)

    assert finding.severity is Severity.PASS
    assert finding.measured > 0.125


def test_flat_colour_to_all_four_edges_passes(make_features, config):
    """A full-bleed sticker is a normal order, not a problem.

    This is the file that used to be escalated to a person, because the agent
    could not separate a background and gave up rather than approving on a
    measurement it never made. Now there is something real to measure.
    """
    art = rect(blank(600, fill=WHITE), 0, 0, 599, 599, RED)
    finding = detail_at_cut(make_features(art, SQUARE), config)

    assert finding.severity is Severity.PASS
    assert finding.detail["detail_area_pct"] == 0.0


def test_a_hard_boundary_between_two_flat_colours_is_not_fine_detail(
    make_features, config
):
    """The distinction the whole check rests on.

    Two colours meeting in a line produce a strong edge that runs right off the
    sticker. It is still not detail, and failing it would be a false alarm on a
    perfectly ordinary design.
    """
    finding = detail_at_cut(make_features(two_tone(600), SQUARE), config)

    assert finding.severity is Severity.PASS


def test_lettering_on_a_full_bleed_field_still_fails(make_features, config):
    """Detail near the cut is caught whether or not a background exists."""
    art = text_block(blank(600, fill=RED), 5, 5, 595, 595, colour=WHITE)
    finding = detail_at_cut(make_features(art, SQUARE), config)

    assert finding.severity is Severity.FAIL


def test_names_the_edge_at_risk(make_features, config):
    art = text_block(blank(600, fill=WHITE), 40, 480, 560, 596)
    finding = detail_at_cut(make_features(art, SQUARE), config)

    assert finding.detail["nearest_edge"] == "bottom"


@pytest.mark.parametrize(
    "inset_px, expected",
    [
        (60, Severity.PASS),  # 0.200 in
        (30, Severity.WARN),  # 0.100 in
        (10, Severity.FAIL),  # 0.033 in
    ],
)
def test_severity_follows_distance(make_features, config, inset_px, expected):
    art = text_block(
        blank(600, fill=WHITE), inset_px, inset_px, 599 - inset_px, 599 - inset_px
    )
    finding = detail_at_cut(make_features(art, SQUARE), config)

    assert finding.severity is expected


def test_skipped_for_die_cut(make_features, config):
    """The cut path is derived from the art, offset outward by a fixed amount,
    so nothing can be closer to it than that offset."""
    art = text_block(blank(600), 10, 10, 590, 590)

    assert detail_at_cut(make_features(art, DIE_CUT), config) is None


def test_a_full_bleed_photograph_is_not_rejected(make_features, config):
    """The false alarm that used to reject an ordinary order.

    A photograph running to all four edges has texture everywhere, which the
    detail measure reads as lettering at the cut. The gate is whether a
    background could be separated at all: on a photograph the flood fill
    returns scattered speckle, not one region, so the artwork is
    continuous-tone imagery and full bleed is the point of it.
    """
    from evalkit.art import photo_texture

    finding = detail_at_cut(make_features(photo_texture(900), SQUARE), config)

    assert finding.severity is Severity.PASS
    assert finding.detail["continuous_tone"] is True
