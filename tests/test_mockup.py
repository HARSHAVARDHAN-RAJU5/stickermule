"""STICKER_MOCKUP: the file is a picture of a finished sticker.

The failure this catches is silent. A product-page screenshot is sharp, well
inside the cut line and has no transparency problem for a square order, so
every other check passes it and the customer receives a sticker with a fake
white rim and a photograph of a shadow printed on it.
"""

from __future__ import annotations

from evalkit.art import RED, WHITE, blank, disc, photo_texture, sticker_mockup
from preflight.checks.mockup import sticker_mockup as check_mockup
from preflight.types import ProductSpec, Severity, Shape

SQUARE = ProductSpec(3.0, 3.0, Shape.SQUARE)


def test_a_rendered_sticker_is_flagged(make_features, config):
    finding = check_mockup(make_features(sticker_mockup(900), SQUARE), config)

    assert finding.severity is Severity.WARN
    assert finding.detail["rim_lightness"] > 235
    assert finding.detail["shadow_halo"] > 2


def test_it_warns_rather_than_failing(make_features, config):
    """A designer may legitimately draw a white outline and a soft shadow.

    The agent cannot tell that apart from a mockup, so it must not reject the
    customer's artwork on a guess. It reports what it sees and a person spends
    two seconds on it.
    """
    finding = check_mockup(make_features(sticker_mockup(900), SQUARE), config)

    assert finding.severity is not Severity.FAIL


def test_a_genuine_white_outline_without_a_shadow_passes(make_features, config):
    """The white rim alone is not enough; the shadow is what gives it away."""
    art = disc(disc(blank(900, fill=WHITE), 0.34, WHITE), 0.29, RED)
    finding = check_mockup(make_features(art, SQUARE), config)

    assert finding.severity is Severity.PASS


def test_a_dark_design_is_never_flagged(make_features, config):
    art = disc(blank(900, fill=WHITE), 0.4)
    finding = check_mockup(make_features(art, SQUARE), config)

    assert finding.severity is Severity.PASS
    assert finding.detail["rim_lightness"] < 235


def test_skipped_when_the_file_has_a_real_cut_out(make_features, config):
    """A file with genuine transparency is artwork, not a photograph of one."""
    assert check_mockup(make_features(disc(blank(900)), SQUARE), config) is None


def test_skipped_on_a_photograph(make_features, config):
    """No trustworthy backdrop, so there is no shadow to measure against."""
    assert check_mockup(make_features(photo_texture(900), SQUARE), config) is None


def test_the_warning_is_not_auto_approvable(config):
    """If this ever became auto-approvable the whole check would be pointless."""
    assert "STICKER_MOCKUP" not in config.auto_approvable_warns
