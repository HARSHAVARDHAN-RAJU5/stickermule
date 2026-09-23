"""ALPHA_MISSING and ALPHA_FAKE: two mistakes, two sentences, two codes."""

from __future__ import annotations

import pytest

from preflight.checks.transparency import alpha_fake, alpha_missing
from preflight.types import ProductSpec, Severity, Shape

from .factories import clean_die_cut, opaque_design, white_box_with_alpha

DIE_CUT = ProductSpec(3.0, 3.0, Shape.DIE_CUT)
SQUARE = ProductSpec(3.0, 3.0, Shape.SQUARE)


class TestAlphaMissing:
    def test_opaque_file_ordered_as_die_cut_fails(self, make_features, config):
        finding = alpha_missing(make_features(opaque_design(), DIE_CUT), config)

        assert finding.severity is Severity.FAIL
        assert finding.measured == pytest.approx(0.0)
        assert finding.unit == "%"

    def test_real_transparency_passes(self, make_features, config):
        finding = alpha_missing(make_features(clean_die_cut(), DIE_CUT), config)

        assert finding.severity is Severity.PASS
        assert finding.measured > 1.0

    def test_skipped_for_fixed_shapes(self, make_features, config):
        """An opaque background is exactly right for a square sticker."""
        assert alpha_missing(make_features(opaque_design(), SQUARE), config) is None


class TestAlphaFake:
    def test_white_box_behind_the_design_fails(self, make_features, config):
        features = make_features(white_box_with_alpha(), DIE_CUT)
        finding = alpha_fake(features, config)

        assert finding.severity is Severity.FAIL
        assert finding.measured == pytest.approx(100.0, abs=0.5)
        assert finding.overlay["type"] == "bbox"

    def test_a_real_cut_out_passes(self, make_features, config):
        """A disc leaves most of its bounding box empty; a box does not."""
        finding = alpha_fake(make_features(clean_die_cut(), DIE_CUT), config)

        assert finding.severity is Severity.PASS
        assert finding.measured < 97.0

    def test_defers_to_alpha_missing_when_there_is_no_alpha(self, make_features, config):
        """One mistake must not produce two findings about the same thing."""
        assert alpha_fake(make_features(opaque_design(), DIE_CUT), config) is None

    def test_skipped_for_fixed_shapes(self, make_features, config):
        assert alpha_fake(make_features(white_box_with_alpha(), SQUARE), config) is None

    def test_a_gradient_background_is_still_a_box(self, make_features, config):
        """The case that four corner samples missed.

        A background ramping from 255 to 225 is not uniform corner to corner,
        so pairwise colour difference cleared it. It is still a white box, and
        it still prints as one. The check now measures the median lightness,
        chroma and spread of the whole perimeter band instead.
        """
        from evalkit.art import blank, disc, horizontal_gradient

        art = disc(horizontal_gradient(blank(900), 255, 225, (40, 40, 859, 859)), 0.25)

        assert alpha_fake(make_features(art, DIE_CUT), config).severity is Severity.FAIL

    def test_a_dark_opaque_block_is_not_flagged_as_a_white_box(
        self, make_features, config
    ):
        """The check is about a light background, not about any filled shape.

        A solid dark design that genuinely fills its bounding box is a real
        design, and flagging it would be a false alarm at the customer's
        expense.
        """
        from .factories import INK, blank, rect

        art = blank(600)
        rect(art, 100, 100, 499, 499, INK)

        finding = alpha_fake(make_features(art, DIE_CUT), config)

        assert finding.severity is Severity.PASS
