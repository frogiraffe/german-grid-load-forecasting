# German Grid-Load Forecasting

## Technical report

**Author:** chiki

<!-- loadfc:generated-start -->
Release results are generated from the checked source revision.
<!-- loadfc:generated-end -->

## 1. Purpose

The system forecasts German electricity grid load. It produces daily and
hourly day-ahead forecasts.

The daily model estimates the next daily mean. The hourly model estimates the
intraday profile. Reconciliation combines these two outputs.

## 2. Data

SMARD supplies realized grid-load data. Open-Meteo supplies archived weather
forecasts.

The target is realized electricity consumption in MW. Residual load is outside
the report scope.

The weather features use forecasts from 24 hours before valid time. Data before
20 January 2024 uses previous-day observed weather.

The configured chronological stages are training, validation, calibration, and
retrospective final. Validation selects models before later evidence is inspected.

## 3. Hourly models

The evaluation compares three hourly models:

- residual hybrid;
- direct LightGBM;
- direct Ridge.

Each model predicts valid hours directly. The feature set includes horizon,
lag-24, lag-168, calendar, weather, and Fourier terms.

The direct method prevents recursive error propagation. The lag features
represent daily and weekly load patterns.

## 4. Daily model

Validation MAE selects the daily anchor. The candidates are Random Forest and
LightGBM.

The final fit uses only data before calibration.

## 5. Reconciliation

The reconciliation operation uses this equation:

```math
\hat{y}^{rec}_{d,h}
=
\hat{y}^{hourly}_{d,h}
\times
\frac{\hat{y}^{daily}_{d}}
{\frac{1}{|H_d|}\sum_{j \in H_d}\hat{y}^{hourly}_{d,j}}
```

The operation keeps the relative hourly shape. The reconciled mean equals the
daily forecast.

The code supports 23-hour, 24-hour, and 25-hour local days. It requires a
complete local day.

The generated release evidence reports the raw and reconciled results from the
same retrospective-final forecast rows.

## 6. Model selection

Validation uses reconciled hourly MAE.

The release keeps the validation-selected model. Later candidate comparisons
are retrospective evidence and do not change that selection.

The paired bootstrap uses one error difference for each local day and resamples
the paired day blocks. The generated release evidence reports its result.

## 7. Prediction intervals

The point model uses data before the calibration period. The daily model and
quantile models use the same boundary.

The frozen models predict calibration and retrospective-final data.
Reconciliation occurs before conformal score calculation.

Both interval methods use the same reconciled point forecasts.

The current implementation keeps the models frozen. Coverage is empirical; adaptive intervals are prequential monitoring without an unconditional time-series guarantee.

## 8. MLflow

MLflow stores `ReconciledForecaster`. The object contains these components:

- fitted hourly model;
- fitted daily model;
- hourly feature schema;
- daily feature schema;
- reconciliation operation.

Artifact replay is checked against the persisted evaluation predictions.

## 9. Data validation

Pandera validates load and weather data.

- Load values must be finite and positive.
- Weather values must be within configured ranges.
- UTC timestamps must be timezone-aware.
- UTC timestamps must be unique and complete.
- Each timestamp must be on the hourly grid.

The telemetry process compares indexes before each join. It rejects missing or
nonfinite interval bounds.

## 10. Datetime calculation

Pandas can store datetime values in nanoseconds or microseconds. Raw `.asi8`
values depend on this storage unit.

The feature code calculates hours with `pd.Timedelta(hours=1)`. The calculation
produces the same scale for both storage units.

A regression test uses `datetime64[us]`. It checks hourly increments and the
weekly Fourier phase.

## 11. Verification

The release uses these commands:

```bash
uv run ruff check src scripts tests
uv run mypy src
uv run pytest --cov=loadfc --cov-report=term-missing --cov-fail-under=80
uv run python scripts/validate_results.py
```

## 12. Limits

The model comparison is retrospective and has uncertainty.

Five population-weighted cities represent national weather exposure. Regional
effects remain outside the model scope.

Conformal coverage can change during distribution shift. The local MLflow
registry has no remote promotion controls.

A production service also requires scheduling, access control, service-level
objectives, and rollback rules.

## References

- Bundesnetzagentur, SMARD: <https://www.smard.de/en>
- Open-Meteo Previous Runs API:
  <https://open-meteo.com/en/docs/previous-runs-api>
- Hyndman et al., *Optimal Combination Forecasts for Hierarchical Time Series*
- Romano et al., *Conformalized Quantile Regression*
- Gibbs and Candès, *Adaptive Conformal Inference Under Distribution Shift*
- Ke et al., *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*
