# German Grid-Load Forecasting

## Technical report

**Author:** chiki

<!-- loadfc:generated-start -->
### Release results

- **Period:** 01 Jan 2026 to 04 Aug 2026; n=216 days.
- **Daily forecast:** Ensemble. MAE 1003.6 MW, MAPE 1.867%, and MASE 0.445. Reference models: Naive (t-1) MAPE 7.025% (n=216); Seasonal naive (t-7) MAPE 5.575% (n=216).
- **Hourly forecast:** Residual hybrid. Daily-total alignment reduced MAE from 1627.6 MW to 1557.7 MW (-4.30%). The result contains n=5183 hourly values. The daily model was LightGBM.
- **Model comparison:** Validation MAE was 1465.3 MW for Residual hybrid and 1471.0 MW for Direct LightGBM. The paired difference on the final data was +1.9 MW. Its 95% range was [-48.8, 52.1] MW across 216 days. This range includes zero, so the data do not show a clear winner.
- **Uncertainty ranges:** Symmetric 90%: target 90%, measured coverage 84.89%, mean width 5573.1 MW, interval score 8831.6 MW, n=5183; Adaptive 90%: target 90%, measured coverage 89.95%, mean width 6529.3 MW, interval score 8250.5 MW, n=5183; CQR 90%: target 90%, measured coverage 86.51%, mean width 7550.2 MW, interval score 10567.7 MW, n=5183.
- **Release check:** source `4dacfc64c5bff25b66263ced93d6c14cbc1257b1`; daily protocol `ef5d669f14abd4e8b58a35cbd0c4c500fb070817dd4e12a227e9b218b1719247`; hourly protocol `bf649cb8b18e90d447fef2a69314fe61f526d6ab91313c3c5b32755d30221258`; bundle `bdfbd6a7b98efa7b8f2279206bd17e0a400e7740086e5026a6d063e6913d3588`.
- **Scope:** The final period was already examined. These results describe this data period and do not state future accuracy.
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

## 8. Data validation

Pandera validates load and weather data.

- Load values must be finite and positive.
- Weather values must be within configured ranges.
- UTC timestamps must be timezone-aware.
- UTC timestamps must be unique and complete.
- Each timestamp must be on the hourly grid.

The evaluation process compares indexes before each join. It rejects missing
or nonfinite interval bounds.

## 9. Datetime calculation

Pandas can store datetime values in nanoseconds or microseconds. Raw `.asi8`
values depend on this storage unit.

The feature code calculates hours with `pd.Timedelta(hours=1)`. The calculation
produces the same scale for both storage units.

A regression test uses `datetime64[us]`. It checks hourly increments and the
weekly Fourier phase.

## 10. Verification

The release uses these commands:

```bash
uv run ruff check src scripts tests
uv run mypy src
uv run pytest --cov=loadfc --cov-report=term-missing --cov-fail-under=80
uv run python scripts/validate_results.py
```

## 11. Limits

The model comparison is retrospective and has uncertainty.

Five population-weighted cities represent national weather exposure. Regional
effects remain outside the model scope.

Conformal coverage can change during distribution shift.

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
