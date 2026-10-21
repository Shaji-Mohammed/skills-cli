import pytest

from router.stats import wilson


def test_wilson_known_values() -> None:
    # Reference values from the closed-form Wilson interval at z = 1.96.
    lo, hi = wilson(8, 10)
    assert lo == pytest.approx(0.4902, abs=1e-4)
    assert hi == pytest.approx(0.9433, abs=1e-4)


def test_wilson_extremes_stay_in_bounds() -> None:
    lo, hi = wilson(0, 10)
    assert lo == pytest.approx(0.0, abs=1e-12)
    assert hi == pytest.approx(0.2775, abs=1e-4)
    lo, hi = wilson(10, 10)
    assert lo == pytest.approx(0.7225, abs=1e-4)
    assert hi == pytest.approx(1.0, abs=1e-12)
    assert 0.0 <= lo <= hi <= 1.0


def test_wilson_narrows_with_n() -> None:
    small = wilson(5, 10)
    large = wilson(500, 1000)
    assert (large[1] - large[0]) < (small[1] - small[0])


def test_wilson_empty_and_invalid() -> None:
    assert wilson(0, 0) == (0.0, 1.0)
    with pytest.raises(ValueError):
        wilson(11, 10)
