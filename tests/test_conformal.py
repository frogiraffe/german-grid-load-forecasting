import numpy as np
import pandas as pd
import pytest

from loadfc.evaluation.conformal import (
    adaptive_conformal_interval,
    conformal_halfwidth,
    coverage_by_horizon,
    empirical_coverage,
    horizon_conformal_intervals,
    horizon_cqr_intervals,
    interval_evidence,
    split_conformal_interval,
)


def test_halfwidth_is_the_conformal_quantile():
    scores = np.arange(1, 101, dtype="float64")
    hw = conformal_halfwidth(scores, alpha=0.1)
    assert hw == 91.0


def test_halfwidth_grows_as_confidence_rises():
    rng = np.random.default_rng(0)
    scores = np.abs(rng.normal(size=500))
    assert conformal_halfwidth(scores, alpha=0.20) < conformal_halfwidth(scores, alpha=0.05)


def test_split_conformal_covers_near_nominal():
    rng = np.random.default_rng(1)
    cal_pred = rng.normal(500, 50, size=1000)
    cal_true = cal_pred + rng.normal(0, 30, size=1000)
    test_pred = rng.normal(500, 50, size=1000)
    test_true = test_pred + rng.normal(0, 30, size=1000)
    lower, upper = split_conformal_interval(cal_true, cal_pred, test_pred, alpha=0.1)
    covered = np.mean((test_true >= lower) & (test_true <= upper))
    assert 0.86 <= covered <= 0.96
    assert np.all(upper > lower)


def test_halfwidth_rejects_empty_calibration():
    with pytest.raises(ValueError, match="calibration"):
        conformal_halfwidth([], alpha=0.1)


@pytest.mark.parametrize("alpha", [0.0, 1.0])
def test_halfwidth_rejects_invalid_alpha(alpha):
    with pytest.raises(ValueError, match="alpha"):
        conformal_halfwidth([1.0], alpha=alpha)


def test_halfwidth_rejects_nonfinite_residuals():
    with pytest.raises(ValueError, match="finite"):
        conformal_halfwidth([1.0, np.nan], alpha=0.1)


def test_empirical_coverage_includes_interval_edges():
    assert empirical_coverage([1, 2, 4], [1, 1, 1], [1, 2, 3]) == pytest.approx(2 / 3)


def test_interval_evidence_reports_one_validated_central_interval_contract():
    """Catch coverage-only evidence that hides a miss outside otherwise equal-width intervals."""

    evidence = interval_evidence([1.0, 3.0], [0.0, 0.0], [2.0, 2.0], alpha=0.2)

    assert evidence == {
        "nominal": pytest.approx(0.8),
        "empirical_coverage": pytest.approx(0.5),
        "mean_width": pytest.approx(2.0),
        "interval_score": pytest.approx(7.0),
        "n": 2,
    }


@pytest.mark.parametrize(
    ("actual", "lower", "upper", "alpha", "message"),
    [
        ([1.0], [0.0, 0.0], [2.0, 2.0], 0.2, "matching"),
        ([], [], [], 0.2, "non-empty"),
        ([np.nan], [0.0], [2.0], 0.2, "finite"),
        ([1.0], [2.0], [0.0], 0.2, "must not cross"),
        ([1.0], [0.0], [2.0], 1.0, "alpha"),
    ],
)
def test_interval_evidence_rejects_malformed_scoring_inputs(actual, lower, upper, alpha, message):
    with pytest.raises(ValueError, match=message):
        interval_evidence(actual, lower, upper, alpha)


def test_adaptive_interval_mutation_changes_only_later_interval_state():
    calibration = np.arange(1.0, 101.0)
    prediction = np.zeros(3)
    actual_a = np.zeros(3)
    actual_b = np.array([0.0, 1000.0, 0.0])

    lower_a, upper_a, alpha_a = adaptive_conformal_interval(
        calibration, np.zeros(100), actual_a, prediction, 0.1
    )
    lower_b, upper_b, alpha_b = adaptive_conformal_interval(
        calibration, np.zeros(100), actual_b, prediction, 0.1
    )

    np.testing.assert_allclose(lower_a[:2], lower_b[:2])
    np.testing.assert_allclose(upper_a[:2], upper_b[:2])
    np.testing.assert_allclose(alpha_a[:2], alpha_b[:2])
    assert lower_a[2] != lower_b[2]
    assert upper_a[2] != upper_b[2]
    assert alpha_a[2] != alpha_b[2]


def test_adaptive_alpha_falls_after_a_miss_and_widens_next_interval():
    calibration_true = np.arange(1.0, 101.0)
    calibration_pred = np.zeros(100)
    predictions = np.array([0.0, 0.0])
    actual = np.array([1000.0, 0.0])
    lower, upper, alpha_history = adaptive_conformal_interval(
        calibration_true,
        calibration_pred,
        actual,
        predictions,
        alpha=0.1,
        gamma=0.05,
    )
    assert alpha_history[1] < alpha_history[0]
    assert (upper[1] - lower[1]) >= (upper[0] - lower[0])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"alpha": 0.0}, "alpha"),
        ({"alpha": 0.1, "gamma": 0.0}, "gamma"),
        ({"alpha": 0.1, "window": 0}, "window"),
    ],
)
def test_adaptive_interval_validates_parameters(kwargs, message):
    with pytest.raises(ValueError, match=message):
        adaptive_conformal_interval([1], [0], [1], [0], **kwargs)


def test_adaptive_interval_requires_matching_shapes():
    with pytest.raises(ValueError, match="matching"):
        adaptive_conformal_interval([1, 2], [0], [1], [0], alpha=0.1)


def test_horizon_conformal_uses_different_error_scales():
    calibration = pd.DataFrame(
        {
            "horizon": [1] * 20 + [24] * 20,
            "actual": [1.0] * 20 + [10.0] * 20,
            "prediction": [0.0] * 40,
        }
    )
    predictions = pd.DataFrame({"horizon": [1, 24], "prediction": [100.0, 100.0]})

    intervals = horizon_conformal_intervals(calibration, predictions, alpha=0.1)

    assert list(intervals["halfwidth"]) == [1.0, 10.0]
    assert intervals.loc[1, "upper"] - intervals.loc[1, "lower"] == 20.0


def test_coverage_by_horizon_reports_conditional_coverage():
    intervals = pd.DataFrame(
        {
            "horizon": [1, 1, 24],
            "actual": [0.0, 2.0, 10.0],
            "lower": [-1.0, -1.0, 0.0],
            "upper": [1.0, 1.0, 20.0],
        }
    )

    report = coverage_by_horizon(intervals)

    assert report.loc[1, "coverage"] == 0.5
    assert report.loc[24, "coverage"] == 1.0


def test_coverage_by_horizon_accepts_identity_index_and_horizon_column():
    index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2024-01-01", tz="UTC"), 24)],
        names=["valid_time", "horizon"],
    )
    intervals = pd.DataFrame(
        {"horizon": [24], "actual": [1.0], "lower": [0.0], "upper": [2.0]},
        index=index,
    )

    report = coverage_by_horizon(intervals)

    assert report.loc[24, "coverage"] == 1.0


def test_horizon_cqr_uses_asymmetric_quantiles_and_horizon_scores():
    calibration = pd.DataFrame(
        {
            "horizon": [1] * 20 + [24] * 20,
            "actual": [0.0] * 20 + [10.0] * 20,
            "lower_quantile": [-1.0] * 20 + [0.0] * 20,
            "upper_quantile": [2.0] * 20 + [5.0] * 20,
        }
    )
    predictions = pd.DataFrame(
        {
            "horizon": [1, 24],
            "lower_quantile": [90.0, 80.0],
            "upper_quantile": [110.0, 120.0],
        }
    )

    intervals = horizon_cqr_intervals(calibration, predictions, alpha=0.1)

    assert list(intervals["conformal_adjustment"]) == [0.0, 5.0]
    assert list(intervals["lower"]) == [90.0, 75.0]
    assert list(intervals["upper"]) == [110.0, 125.0]


def test_horizon_cqr_rejects_crossed_quantiles():
    frame = pd.DataFrame(
        {
            "horizon": [24],
            "actual": [100.0],
            "lower_quantile": [110.0],
            "upper_quantile": [90.0],
        }
    )

    with pytest.raises(ValueError, match="must not cross"):
        horizon_cqr_intervals(frame, frame, alpha=0.1)


def test_horizon_cqr_accepts_horizon_as_index_level_and_column():
    index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2024-01-01", tz="UTC"), 24)],
        names=["valid_time", "horizon"],
    )
    calibration = pd.DataFrame(
        {
            "horizon": [24],
            "actual": [100.0],
            "lower_quantile": [90.0],
            "upper_quantile": [110.0],
        },
        index=index,
    )

    result = horizon_cqr_intervals(calibration, calibration, alpha=0.1)

    assert result["lower"].iloc[0] == 90.0
    assert result["upper"].iloc[0] == 110.0
