"""Fixed and sequentially adaptive conformal prediction intervals."""

from __future__ import annotations

import numpy as np
import pandas as pd


def conformal_halfwidth(cal_residuals, alpha: float) -> float:
    """Split-conformal interval half-width at miscoverage level ``alpha``.

    Returns the standard finite-sample rank of the absolute calibration
    residuals: ``ceil((n+1)(1-alpha))``.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    s = np.sort(np.abs(np.asarray(cal_residuals, dtype="float64")))
    n = s.size
    if n == 0:
        raise ValueError("need at least one calibration residual")
    if not np.isfinite(s).all():
        raise ValueError("calibration residuals must be finite")
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    return float(s[min(k, n) - 1])


def split_conformal_interval(cal_true, cal_pred, test_pred, alpha: float):
    """Lower/upper interval arrays for ``test_pred`` at level ``alpha``.

    Half-width is calibrated on the calibration slice (``cal_true`` vs
    ``cal_pred``) and applied symmetrically to the test forecasts.
    """
    residuals = np.asarray(cal_true, dtype="float64") - np.asarray(cal_pred, dtype="float64")
    hw = conformal_halfwidth(residuals, alpha)
    p = np.asarray(test_pred, dtype="float64")
    return p - hw, p + hw


def empirical_coverage(y_true, lower, upper) -> float:
    """Fraction of ``y_true`` falling inside ``[lower, upper]``."""
    y = np.asarray(y_true, dtype="float64")
    return float(np.mean((y >= np.asarray(lower)) & (y <= np.asarray(upper))))


def interval_evidence(actual, lower, upper, alpha: float) -> dict[str, float | int]:
    """Aggregate validated central-interval evidence using the standard interval score."""

    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    y = np.asarray(actual, dtype="float64")
    lo = np.asarray(lower, dtype="float64")
    hi = np.asarray(upper, dtype="float64")
    if y.shape != lo.shape or y.shape != hi.shape:
        raise ValueError("actual and interval arrays must have matching shapes")
    if y.size == 0:
        raise ValueError("interval evidence requires non-empty inputs")
    if not np.isfinite(y).all() or not np.isfinite(lo).all() or not np.isfinite(hi).all():
        raise ValueError("interval evidence inputs must be finite")
    if (lo > hi).any():
        raise ValueError("interval bounds must not cross")

    width = hi - lo
    score = width + (2.0 / alpha) * np.maximum(lo - y, 0.0) + (2.0 / alpha) * np.maximum(y - hi, 0.0)
    return {
        "nominal": 1.0 - alpha,
        "empirical_coverage": empirical_coverage(y, lo, hi),
        "mean_width": float(width.mean()),
        "interval_score": float(score.mean()),
        "n": int(y.size),
    }


def horizon_conformal_intervals(
    calibration: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    alpha: float,
) -> pd.DataFrame:
    """Calibrate a separate symmetric interval for each forecast horizon."""

    required_calibration = {"horizon", "actual", "prediction"}
    required_predictions = {"horizon", "prediction"}
    if not required_calibration <= set(calibration):
        raise ValueError("calibration requires horizon, actual and prediction columns")
    if not required_predictions <= set(predictions):
        raise ValueError("predictions require horizon and prediction columns")

    widths = (
        calibration.assign(residual=calibration["actual"] - calibration["prediction"])
        .groupby("horizon")["residual"]
        .apply(lambda values: conformal_halfwidth(values, alpha))
    )
    missing = sorted(set(predictions["horizon"]) - set(widths.index))
    if missing:
        raise ValueError(f"missing calibration residuals for horizons: {missing}")
    out = predictions.copy()
    out["halfwidth"] = out["horizon"].map(widths)
    out["lower"] = out["prediction"] - out["halfwidth"]
    out["upper"] = out["prediction"] + out["halfwidth"]
    return out


def horizon_cqr_intervals(
    calibration: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    alpha: float,
) -> pd.DataFrame:
    """Conformalize lower/upper quantile predictions separately by horizon."""

    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    required_calibration = {"horizon", "actual", "lower_quantile", "upper_quantile"}
    required_predictions = {"horizon", "lower_quantile", "upper_quantile"}
    if not required_calibration <= set(calibration):
        raise ValueError("CQR calibration requires horizon, actual and quantile columns")
    if not required_predictions <= set(predictions):
        raise ValueError("CQR predictions require horizon and quantile columns")
    if (calibration["lower_quantile"] > calibration["upper_quantile"]).any():
        raise ValueError("CQR calibration quantiles must not cross")
    if (predictions["lower_quantile"] > predictions["upper_quantile"]).any():
        raise ValueError("CQR prediction quantiles must not cross")

    scored = calibration.assign(
        conformity_score=np.maximum(
            np.maximum(
                calibration["lower_quantile"] - calibration["actual"],
                calibration["actual"] - calibration["upper_quantile"],
            ),
            0.0,
        )
    )

    def _finite_sample_quantile(values: pd.Series) -> float:
        scores = np.sort(values.to_numpy(dtype="float64"))
        if scores.size == 0 or not np.isfinite(scores).all():
            raise ValueError("CQR conformity scores must be finite and non-empty")
        rank = int(np.ceil((scores.size + 1) * (1.0 - alpha)))
        return float(scores[min(rank, scores.size) - 1])

    adjustments = scored.groupby(scored["horizon"])["conformity_score"].apply(
        _finite_sample_quantile
    )
    missing = sorted(set(predictions["horizon"]) - set(adjustments.index))
    if missing:
        raise ValueError(f"missing CQR calibration scores for horizons: {missing}")

    out = predictions.copy()
    out["conformal_adjustment"] = out["horizon"].map(adjustments)
    out["lower"] = out["lower_quantile"] - out["conformal_adjustment"]
    out["upper"] = out["upper_quantile"] + out["conformal_adjustment"]
    if (out["lower"] > out["upper"]).any():
        raise ValueError("conformal adjustment produced crossed CQR intervals")
    return out


def coverage_by_horizon(intervals: pd.DataFrame) -> pd.DataFrame:
    """Report empirical coverage and mean width for each horizon."""

    required = {"horizon", "actual", "lower", "upper"}
    if not required <= set(intervals):
        raise ValueError("intervals require horizon, actual, lower and upper columns")
    frame = intervals.assign(
        covered=(intervals["actual"] >= intervals["lower"])
        & (intervals["actual"] <= intervals["upper"]),
        width=intervals["upper"] - intervals["lower"],
    )
    return frame.groupby(frame["horizon"]).agg(
        coverage=("covered", "mean"),
        mean_width=("width", "mean"),
        n=("covered", "size"),
    )


def adaptive_conformal_interval(
    cal_true,
    cal_pred,
    test_true,
    test_pred,
    alpha: float,
    *,
    gamma: float = 0.01,
    window: int = 365,
):
    """Prequential Adaptive Conformal Inference intervals.

    The interval for step ``t`` uses calibration errors and test errors observed
    only through ``t-1``. After the actual value arrives, the effective
    miscoverage rate is updated by

    ``alpha_t+1 = alpha_t + gamma * (alpha - miss_t)``.

    A miss lowers the effective alpha and widens the next interval; a hit moves
    it upward. The returned alpha history records the value used for each
    forecast.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    if not 0 < gamma < 1:
        raise ValueError("gamma must be between 0 and 1")
    if window < 1:
        raise ValueError("window must be positive")

    cal_y = np.asarray(cal_true, dtype="float64")
    cal_p = np.asarray(cal_pred, dtype="float64")
    y = np.asarray(test_true, dtype="float64")
    p = np.asarray(test_pred, dtype="float64")
    if cal_y.shape != cal_p.shape or y.shape != p.shape:
        raise ValueError("actual and forecast arrays must have matching shapes")

    scores = list(np.abs(cal_y - cal_p)[-window:])
    if not scores:
        raise ValueError("need at least one calibration residual")

    lower = np.empty_like(p)
    upper = np.empty_like(p)
    alpha_history = np.empty_like(p)
    effective_alpha = float(alpha)
    for i, (actual, forecast) in enumerate(zip(y, p, strict=True)):
        minimum = 1.0 / (len(scores) + 1)
        bounded_alpha = float(np.clip(effective_alpha, minimum, 0.999))
        halfwidth = conformal_halfwidth(scores, bounded_alpha)
        lower[i] = forecast - halfwidth
        upper[i] = forecast + halfwidth
        alpha_history[i] = bounded_alpha

        missed = float(actual < lower[i] or actual > upper[i])
        effective_alpha = float(np.clip(effective_alpha + gamma * (alpha - missed), 0.001, 0.999))
        scores.append(abs(actual - forecast))
        if len(scores) > window:
            scores.pop(0)

    return lower, upper, alpha_history
