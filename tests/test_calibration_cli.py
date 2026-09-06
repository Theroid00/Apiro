import pytest

from scripts.run_safety_calibration_eval import reporting_thresholds


def test_custom_tau_is_added_to_reporting_thresholds():
    assert reporting_thresholds(0.7) == (0.5, 0.65, 0.7, 0.8)


@pytest.mark.parametrize("tau", [-0.01, 1.01])
def test_out_of_range_tau_is_rejected(tau):
    with pytest.raises(ValueError, match="between 0 and 1"):
        reporting_thresholds(tau)
