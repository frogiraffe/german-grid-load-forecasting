# German Grid-Load Forecasting

## Empirical study

**Author:** chiki

<!-- loadfc:generated-start -->
### Release results

- **Period:** 01 Jan 2026 to 04 Aug 2026; n=216 days.
- **Daily forecast:** Ensemble. MAE 1003.6 MW, MAPE 1.867%, and MASE 0.445. Reference models: Naive (t-1) MAPE 7.025% (n=216); Seasonal naive (t-7) MAPE 5.575% (n=216).
- **Hourly forecast:** Residual hybrid. Daily-total alignment reduced MAE from 1627.6 MW to 1557.7 MW (-4.30%). The result contains n=5183 hourly values. The daily model was LightGBM.
- **Model comparison:** Validation MAE was 1465.3 MW for Residual hybrid and 1471.0 MW for Direct LightGBM. The paired difference on the final data was +1.9 MW. Its 95% range was [-48.8, 52.1] MW across 216 days. This range includes zero, so the data do not show a clear winner.
- **Uncertainty ranges:** Symmetric 90%: target 90%, measured coverage 84.89%, mean width 5573.1 MW, interval score 8831.6 MW, n=5183; Adaptive 90%: target 90%, measured coverage 89.95%, mean width 6529.3 MW, interval score 8250.5 MW, n=5183; CQR 90%: target 90%, measured coverage 86.51%, mean width 7550.2 MW, interval score 10567.7 MW, n=5183.
<!-- provenance: source `75bdc9e063dcd91625eb61cb084cc7c236494985`; daily protocol `6617bd74797ed102fb43d4c05e311d4752103aff13eb12cfbd0394cf859cc385`; hourly protocol `f766013085ed941324b3a97fd1806b9e30f59f04cf7692a1f43876151f3a9352`; bundle `4a9c208191dbc32dbcda36873b6c978df98754b8a79724279e0f4e6a86ba77bb` -->
- **Scope:** The final period was already examined. These results describe this data period and do not state future accuracy.
<!-- loadfc:generated-end -->

## Abstract

This study asks how accurately German electricity demand can be forecast from a restricted
forecast-origin information set. It compares one-day-ahead daily models, fixed 24-hour-ahead hourly
forecasts, daily–hourly reconciliation, and three
forms of uncertainty quantification. Model choice is fixed on a chronological validation period and
a separate calibration period is used for interval construction. The final period is retrospective
and was previously inspected; its results describe this experiment rather than future accuracy.

## 1. Research question

Retrospective load forecasts can look stronger than they are when they use weather observations that
were unavailable at forecast time. The analysis asks two related questions: how accurately the next
daily mean can be predicted and how accurately each valid hour can be predicted exactly 24 hours
earlier. The hourly result is not a common-origin profile for the next local day.

## 2. Data

SMARD supplies realized German electricity consumption (filter 410). Open-Meteo supplies
temperature and wind for Berlin, Hamburg, Munich, Cologne, and Frankfurt, combined with population
weights as a national weather proxy. The dataset contains 2,760 daily observations from 14 January
2019 through 4 August 2026.

The target is realized electricity consumption in MW. Residual load is outside
the report scope.

From 20 January 2024 onward, weather features use archived forecasts issued 24 hours before the
target day. Earlier dates use the previous day's observation because an equivalent forecast archive
is unavailable. This creates a documented change in information quality.

Training ends in 2023. Validation runs from January 2024 through June 2025, calibration from July
through December 2025, and final evaluation from January through 4 August 2026. Validation selects
models before later evidence is inspected; calibration sets interval state without selecting the
point model.

## 3. Forecasting methods

Daily candidates are SARIMAX, XGBoost, LightGBM, Random Forest, two persistence baselines, and a
fixed ensemble of SARIMAX, XGBoost, and LightGBM. Predictors include one- and seven-day load lags,
calendar effects, weekly and annual Fourier terms, forecast-origin weather, and degree-day variables.

The hourly evaluation compares three models:

- residual hybrid;
- direct LightGBM;
- direct Ridge.

Each model predicts valid hours directly. The feature set includes horizon, lag-24, lag-168,
calendar, weather, and Fourier terms. The released hourly metrics use horizon 24 only: the origin
for each row is its own valid time minus 24 hours.

The direct method prevents recursive error propagation. The lag features
represent daily and weekly load patterns.

## 4. Daily model selection

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

The current implementation keeps the models frozen. Coverage is empirical; adaptive intervals are
prequential monitoring without an unconditional time-series guarantee. Coverage is interpreted
together with mean width and interval score, since coverage alone rewards arbitrarily wide ranges.

## 8. Results and interpretation

The generated results above show that the daily ensemble led both validation and final-period MAPE.
It also substantially outperformed the previous-day and weekly seasonal baselines, so the reported
choice follows the selection period rather than the final ranking.

The residual hybrid was fixed from validation. Reconciliation lowered its final hourly MAE. Direct
LightGBM finished slightly lower, but the paired day-block interval for their MAE difference crossed
zero. The data therefore do not establish a clear winner between the two hourly models.

For nominal 90% hourly intervals, the adaptive method came closest to nominal coverage and achieved
the lowest interval score, although it was wider than the fixed symmetric interval. Replacing
forecast-origin weather with a previous-day proxy modestly increased ensemble MAPE. Weather added
measurable value, but recent and seasonal load structure carried most of the signal.

## 9. Data validation

Pandera validates load and weather data.

- Load values must be finite and positive.
- Weather values must be within configured ranges.
- UTC timestamps must be timezone-aware.
- UTC timestamps must be unique and complete.
- Each timestamp must be on the hourly grid.

The evaluation process compares indexes before each join. It rejects missing
or nonfinite interval bounds.

## 10. Reproducibility

Split dates, feature definitions, parameters, and the random seed are fixed in `config.yaml`. A clean
run rebuilds public inputs, evaluates the frozen protocols, compiles the report, validates artifact
hashes, and emits the bounded dashboard payload. Detailed CSV tables are generated on demand rather
than stored in Git. Machine-readable source and protocol hashes remain in the result manifests; they
do not belong in the scientific narrative. Tree-model parameters are frozen legacy inputs. The
complete historical search trials are unavailable, so the study is reproducible from the committed
parameters onward but does not claim reproducibility of the original search.

The verification commands are:

```bash
uv run ruff check src scripts tests
uv run mypy src
uv run pytest --cov=loadfc --cov-report=term-missing --cov-fail-under=80
uv run python scripts/validate_results.py
```

## 11. Limitations

The final period was inspected during development, so its metrics are descriptive rather than
prospective claims. The pre-2024 weather fallback is weaker than an archived issue-time forecast.

Five population-weighted cities represent national weather exposure. Regional
effects remain outside the model scope.

The hourly headline is fixed 24-hour-ahead evaluation. It does not represent a market-style
day-ahead profile issued once for all hours of the next local day.

Conformal coverage can change during distribution shift.

The study evaluates an offline forecasting pipeline. Scheduling, data-arrival latency, access
control, service-level objectives, and rollback rules remain deployment questions.

## References

- Bundesnetzagentur, SMARD: <https://www.smard.de/en>
- Open-Meteo Previous Runs API:
  <https://open-meteo.com/en/docs/previous-runs-api>
- Hyndman et al., *Optimal Combination Forecasts for Hierarchical Time Series*
- Romano et al., *Conformalized Quantile Regression*
- Gibbs and Candès, *Adaptive Conformal Inference Under Distribution Shift*
- Ke et al., *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*
