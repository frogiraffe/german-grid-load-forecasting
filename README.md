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

[Explore the dashboard](./docs/) · [Read the model card](./docs/MODEL_CARD.md) ·
[Read the technical report](./report/technical-report-en.pdf)

## Features

- **Data** — Fetches German net load from SMARD (`filter: 410`, hourly, `data/raw/`) and weather from Open-Meteo archive and previous-runs forecast APIs. Uses a day-ahead weather strategy: archived forecasts issued 24 hours before the target day from 2024 onward; earlier rows fall back to the previous day's observation (`config.yaml` -> `features.weather_strategy`).
- **Features** — Calendar features (holidays via the `holidays` package), Fourier terms (weekly and yearly), HDD/CDD from temperature, lagged load (`lags: [1, 7]` daily, `[24, 168]` hourly), and structural-break indicators (COVID 2020, energy crisis 2022/23).
- **Models** — Daily models: SARIMAX, XGBoost, LightGBM, RandomForest, and a frozen ensemble of SARIMAX + XGBoost + LightGBM selected on the 2024-2025 validation period. Hourly models: a residual-hybrid forecaster (linear trend plus LightGBM residual model), a direct multi-horizon LightGBM model, and a direct ridge model (`src/loadfc/models/`).
- **Uncertainty quantification** — Fixed and sequentially adaptive conformal prediction intervals: symmetric split conformal per horizon, CQR, and adaptive conformal with calibration window and gamma drift (`src/loadfc/evaluation/conformal.py`).
- **Evaluation** — Metrics (MAE, RMSE, MAPE, MASE), baselines (naive 1-day, seasonal naive 7-day), day-block bootstrap ablations (rolling origin), Diebold-Mariano tests, error slices, weather ablations, drift and backtest telemetry (`src/loadfc/evaluation/`).
- **Tracking** — Opt-in local MLflow tracking and model registration (`src/loadfc/tracking.py`). Hyperparameter tuning via Optuna (`src/loadfc/tuning/`).
- **Report** — LaTeX technical report generated from run outputs (`report/technical-report-en.tex`, compiled with Tectonic).

## Quick start

Requires Python `>=3.12,<3.14` and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run python scripts/run_pipeline.py --config config.yaml
```

`run_pipeline.py` runs the full chain: fetch data, build features, fit and evaluate models, compute intervals, run telemetry and analysis, render plots, compile the LaTeX report with Tectonic 0.16.9, validate the staged bundle, and atomically promote it. Add `--refresh-data` to re-download raw data or `--config <path>` to use another config file.

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
  evaluation/                Metrics, baselines, conformal intervals, rolling origin, DM tests, drift, telemetry
  tuning/                    Optuna search spaces and study management
  viz/                       Plotting
  tracking.py                MLflow tracking
scripts/                     One runnable step per concern (run_fetch, run_features, run_evaluate,
                             run_intervals, run_telemetry, run_error_analysis, run_analysis,
                             run_plots, run_shap, run_hourly, run_tune, validate_results)
tests/                       pytest suite (framework, features, models, evaluation, scripts)
results/                     Metrics CSVs, predictions, figures, run_summary.json
report/                      LaTeX technical report (source, markdown, PDF)
docs/                        Technical design notes
```

Data split (`config.yaml`): train through 2023-12-31, validation 2024-01-01 to 2025-06-30, calibration 2025-07-01 to 2025-12-31, and retrospective final evaluation 2026-01-01 to 2026-08-04.

## Results

<!-- loadfc:generated-start -->
Release results are generated from the checked source revision.
<!-- loadfc:generated-end -->

## Documentation

- `docs/TECHNICAL.md` — Technical design: system flow, validation rules, and protocol constraints.
- `report/technical-report-en.md` — Full technical report (also `.tex` and PDF).
- `DATA_LICENSE.md` — License notes: code is MIT; data files retain their provider licenses.

## License

MIT. See [LICENSE](LICENSE). Code is MIT; data files retain their provider licenses (see [DATA_LICENSE.md](DATA_LICENSE.md)).
