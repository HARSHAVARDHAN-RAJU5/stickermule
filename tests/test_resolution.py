"""RES_DPI: the check is pixels over ordered inches, and nothing else."""

from __future__ import annotations

import pytest

from preflight.checks.resolution import res_dpi
from preflight.types import Material, ProductSpec, Severity, Shape

from .factories import blank, disc


def art(size_px: int):
    return disc(blank(size_px))


@pytest.mark.parametrize(
    "size_px, expected, expected_dpi",
    [
        (900, Severity.PASS, 300.0),  # exactly at target
        (1200, Severity.PASS, 400.0),
        (600, Severity.PASS, 200.0),  # below target, but a smooth shape
        (899, Severity.PASS, 299.7),  # one pixel under the bar
        (288, Severity.FAIL, 96.0),  # the classic web-logo upload
        (300, Severity.FAIL, 100.0),
    ],
)
def test_severity_follows_effective_dpi(
    make_features, config, size_px, expected, expected_dpi
):
    product = ProductSpec(3.0, 3.0, Shape.DIE_CUT)
    finding = res_dpi(make_features(art(size_px), product), config)

    assert finding.severity is expected
    assert finding.measured == pytest.approx(expected_dpi, abs=0.1)
    assert finding.unit == "dpi"


def test_the_middle_band_is_decided_by_the_artwork(make_features, config):
    """Between 150 and 300 dpi, what is in the picture decides.

    Both files are 200 dpi. One is a single large shape and prints fine. The
    other is covered in small lettering and does not. Sending both to a person
    was the old behaviour and it wasted the person's time on the first one.
    """
    from evalkit.art import blank, text_block

    product = ProductSpec(3.0, 3.0, Shape.DIE_CUT)
    smooth = res_dpi(make_features(art(600), product), config)
    lettered = res_dpi(make_features(text_block(blank(600), 60, 60, 540, 540), product), config)

    assert smooth.measured == lettered.measured == pytest.approx(200.0)
    assert smooth.severity is Severity.PASS
    assert lettered.severity is Severity.FAIL


def test_the_smaller_axis_decides(make_features, config):
    """A wide, short file prints soft on the axis that has fewer pixels."""
    product = ProductSpec(4.0, 2.0, Shape.SQUARE)
    art_ = blank(1200)  # 1200x1200 -> 300 dpi across, 600 dpi down

    finding = res_dpi(make_features(art_, product), config)

    assert finding.measured == pytest.approx(300.0)


def test_reports_the_pixels_the_customer_needs(make_features, config):
    """The message writer can only be specific if the check hands it numbers."""
    product = ProductSpec(3.0, 2.0, Shape.SQUARE)
    finding = res_dpi(make_features(art(288), product), config)

    assert finding.detail["required_px"] == [900, 600]
    assert 900.0 in finding.suggested
    assert 300.0 in finding.suggested


def test_embedded_dpi_tag_never_changes_the_verdict(make_features, config, tmp_path):
    """A file tagged 300 DPI that is 96 DPI at the ordered size still fails."""
    from PIL import Image

    from .factories import save

    path = save(art(288), tmp_path / "tagged.png")
    Image.open(path).save(path, dpi=(300, 300))  # a confident lie

    from preflight.analyze import analyze
    from preflight.normalize import normalize

    features = analyze(normalize(path), ProductSpec(3.0, 3.0, Shape.SQUARE))
    finding = res_dpi(features, config)

    # PNG stores resolution as integer pixels-per-metre, so the tag does not
    # round-trip exactly. Which is itself a reason not to trust it.
    assert features.canvas.metadata_dpi == pytest.approx((300.0, 300.0), abs=0.01)
    assert finding.severity is Severity.FAIL
    assert finding.measured == pytest.approx(96.0)


def test_numbers_exposed_for_the_grounding_guard(make_features, config):
    product = ProductSpec(3.0, 3.0, Shape.DIE_CUT, Material.CLEAR)
    finding = res_dpi(make_features(art(288), product), config)

    numbers = finding.numbers()
    assert 96.0 in numbers  # measured
    assert 150.0 in numbers  # threshold it failed
    assert 900.0 in numbers  # what to upload instead
