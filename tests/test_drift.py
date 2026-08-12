import numpy as np
import pandas as pd
import pytest

from loadfc.evaluation.drift import (
    feature_drift_report,
    page_hinkley,
    population_stability_index,
)


def test_psi_is_small_for_same_distribution_and_large_for_shift():
    reference = np.linspace(0, 100, 1000)

    stable = population_stability_index(reference, reference.copy())
    shifted = population_stability_index(reference, reference + 100)

    assert stable == pytest.approx(0.0)
    assert shifted > 0.25


def test_psi_detects_shift_from_constant_reference():
    assert population_stability_index(np.ones(100), np.full(100, 2.0)) > 0.25


def test_feature_report_assigns_operating_status():
    reference = pd.DataFrame({"temperature": np.linspace(0, 10, 100)})
    current = pd.DataFrame({"temperature": np.linspace(100, 110, 100)})

    report = feature_drift_report(reference, current, ["temperature"])

    assert report.loc["temperature", "status"] == "critical"


def test_page_hinkley_alerts_after_residual_mean_increases():
    residual_magnitude = np.r_[np.ones(100), np.full(30, 10.0)]

    report = page_hinkley(residual_magnitude, threshold=20.0)

    assert not report.iloc[:100]["alert"].any()
    assert report.iloc[100:]["alert"].any()


def test_page_hinkley_can_monitor_against_calibration_mean():
    report = page_hinkley(
        np.full(30, 5.0),
        reference_mean=10.0,
        threshold=20.0,
    )
    assert not report["alert"].any()


@pytest.mark.parametrize(
    ("values", "kwargs", "message"),
    [
        ([], {}, "finite observations"),
        ([1.0], {"threshold": 0.0}, "threshold"),
        ([1.0], {"delta": -1.0}, "delta"),
    ],
)
def test_page_hinkley_validates_inputs(values, kwargs, message):
    with pytest.raises(ValueError, match=message):
        page_hinkley(values, **kwargs)
