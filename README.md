# German Grid-Load Forecasting (`loadfc`)

A daily and 24-hour German electricity-load forecasting case study built around a practical
question: how accurately can grid demand be predicted using only information available when the
forecast is issued?

## Key findings

- The daily ensemble reached 1.87% MAPE and reduced error by 73% against the previous-day baseline.
- Aligning hourly forecasts with the daily total reduced the selected hourly model's MAE by 4.3%.
- The selected residual hybrid and direct LightGBM were statistically comparable in the final
  evaluation; the evidence does not establish a clear winner.
- The adaptive 90% interval achieved 89.95% empirical coverage, the closest tested method to its
  stated level.

The dashboard connects the forecasts, error patterns, uncertainty intervals, and model comparisons
in one place.

[Explore the dashboard](https://frogiraffe.github.io/german-grid-load-forecasting/) · [Read the model card](./docs/MODEL_CARD.md) ·
[Read the technical report](./report/technical-report-en.pdf)

## Features

- **Data** — Fetches German net load from SMARD (`filter: 410`, hourly, `data/raw/`) and weather from Open-Meteo archive and previous-runs forecast APIs. Uses a day-ahead weather strategy: archived forecasts issued 24 hours before the target day from 2024 onward; earlier rows fall back to the previous day's observation (`config.yaml` -> `features.weather_strategy`).
- **Features** — Calendar features (holidays via the `holidays` package), Fourier terms (weekly and yearly), HDD/CDD from temperature, lagged load (`lags: [1, 7]` daily, `[24, 168]` hourly), and structural-break indicators (COVID 2020, energy crisis 2022/23).
- **Models** — Daily models: SARIMAX, XGBoost, LightGBM, RandomForest, and a frozen ensemble of SARIMAX + XGBoost + LightGBM selected on the 2024-2025 validation period. Hourly models: a residual-hybrid forecaster (linear trend plus LightGBM residual model), a direct multi-horizon LightGBM model, and a direct ridge model (`src/loadfc/models/`).
- **Uncertainty quantification** — Fixed and sequentially adaptive conformal prediction intervals: symmetric split conformal per horizon, CQR, and adaptive conformal with calibration window and gamma drift (`src/loadfc/evaluation/conformal.py`).
- **Evaluation** — Metrics (MAE, RMSE, MAPE, MASE), baselines (naive 1-day, seasonal naive 7-day), day-block bootstrap ablations (rolling origin), Diebold-Mariano tests, error slices, weather ablations, and drift analysis (`src/loadfc/evaluation/`).
- **Provenance** — Release manifests bind the source revision, evaluation protocols, generated result tables, dashboard payload, and report artifacts with SHA-256 hashes (`src/loadfc/tracking.py`).
- **Report** — LaTeX technical report generated from run outputs (`report/technical-report-en.tex`, compiled with Tectonic).

## Quick start

Requires Python `>=3.12,<3.14`, [uv](https://docs.astral.sh/uv/), and Tectonic 0.16.9.

```bash
uv sync --extra explain
uv run python scripts/run_pipeline.py --config config.yaml
```

`scripts/run_pipeline.py` runs the full chain: fetch data, build features, fit and evaluate models, compute intervals and analyses, compile the LaTeX report, validate the staged bundle, and atomically promote it. Add `--refresh-data` to re-download raw data or `--config <path>` to use another config file. The release command requires a clean committed source tree.

Detailed CSV evidence is generated on demand rather than stored in Git. To reproduce it without a local setup, run the manual **reproduce release evidence** workflow in GitHub Actions. A successful run uploads `results/` and the regenerated report PDF as `release-evidence-<commit>`, retained for 30 days.

To inspect the local static dashboard after regeneration:

```bash
npm ci --prefix dashboard
npm --prefix dashboard run build:docs
python -m http.server --directory docs 8000
```

Visit `http://127.0.0.1:8000/`.

## Project layout

```
config.yaml                  All run settings: data ranges, split dates, model hyperparameters
src/loadfc/
  data/                      SMARD and Open-Meteo fetching, dataset building, validation
  features/                  Calendar, Fourier, lags, weather features, feature assembly
  models/                    SARIMAX, ML (XGBoost/LightGBM/RF), hybrid and direct hourly models
  evaluation/                Metrics, baselines, conformal intervals, rolling origin, DM tests, drift
  tracking.py                Release provenance and artifact hashing
scripts/                     One runnable step per concern (run_fetch, run_features, run_evaluate,
                             run_intervals, run_error_analysis, run_analysis, run_shap,
                             run_hourly, validate_results)
tests/                       pytest suite (framework, features, models, evaluation, scripts)
results/                     Compact tracked manifests plus locally generated evidence tables
report/                      LaTeX technical report (source, markdown, PDF)
docs/                        Technical design notes
```

Data split (`config.yaml`): train through 2023-12-31, validation 2024-01-01 to 2025-06-30, calibration 2025-07-01 to 2025-12-31, and retrospective final evaluation 2026-01-01 to 2026-08-04.

## Results

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

## Documentation

- `docs/TECHNICAL.md` — Technical design: system flow, validation rules, and protocol constraints.
- `report/technical-report-en.md` — Full technical report (also `.tex` and PDF).
- `DATA_LICENSE.md` — License notes: code is MIT; data files retain their provider licenses.

## License

MIT. See [LICENSE](LICENSE). Code is MIT; data files retain their provider licenses (see [DATA_LICENSE.md](DATA_LICENSE.md)).
