# Configuration

The project is configured by a single YAML file, `config.yaml` at the repository
root. It is loaded by `Config.from_yaml()` in `src/loadfc/config.py`, which parses
the file into a frozen `Config` dataclass and runs a validation pass
(`Config.validate()`) that raises `ValueError` on invalid values.

All paths in `paths` are resolved relative to the directory that contains the
config file (the repository root for the default `config.yaml`), not relative to
the current working directory (`Config.path()`, `Config.root`).

## File location and loading

| Item | Value |
| --- | --- |
| File | `config.yaml` (repository root) |
| Loader | `Config.from_yaml(path)` in `src/loadfc/config.py` |
| Alternate configs | any script accepts `--config <path>` (default `config.yaml`) |
| Data model | frozen dataclasses: `Config`, `SplitConfig`, `City` |

To run with an alternate config, pass `--config` to any script, for example:

```bash
python scripts/run_pipeline.py --config experiments/experiment-a.yaml
```

All scripts in `scripts/` accept `--config` with the same default:
`run_fetch.py`, `run_features.py`, `run_pipeline.py`, `run_evaluate.py`,
`run_intervals.py`, `run_hourly.py`, `run_analysis.py`, `run_plots.py`,
`run_tune.py`, `run_shap.py`, `run_telemetry.py`, `run_error_analysis.py`,
`write_run_summary.py`, `validate_results.py`, `render_report_data.py`.

## Top-level sections

| Section | Required | Content |
| --- | --- | --- |
| `project` | yes | date windows for raw and dataset data |
| `split` | yes | train/validation/calibration/test boundaries |
| `cities` | yes | cities aggregated into the national forecast |
| `smard` | yes | SMARD load-data source |
| `weather` | yes | Open-Meteo weather source |
| `features` | yes | feature engineering settings |
| `models` | yes | model hyperparameters (frozen from tuning) |
| `tuning` | yes | Optuna tuning budget |
| `ensemble` | optional | ensemble members (default `["SARIMAX", "xgboost", "lightgbm"]`) |
| `uncertainty` | optional | conformal calibration settings |
| `monitoring` | optional | drift/quality alert thresholds |
| `seed` | yes | global random seed |
| `paths` | yes | output directories |

## `project`

| Key | Type | Current value | Effect |
| --- | --- | --- | --- |
| `raw_start` | date | `2019-01-01` | earliest date of raw fetched data |
| `raw_end` | date | `2026-08-04` | latest date of raw fetched data; `split.test_end` must not exceed it |
| `dataset_start` | date | `2019-01-14` | start of the cleaned dataset (after warm-up rows are dropped) |

## `split`

| Key | Type | Current value | Effect |
| --- | --- | --- | --- |
| `train_end` | date | `2023-12-31` | end of the training window (model fitting) |
| `val_end` | date | `2025-06-30` | end of the validation window (model/architecture selection) |
| `calibration_start` | date | `2025-07-01` | start of the conformal calibration window |
| `calibration_end` | date | `2025-12-31` | end of the calibration window |
| `test_start` | date | `2026-01-01` | start of the held-out test window |
| `test_end` | date | `2026-08-04` | end of the test window; the last day of available raw data |

Validation enforces that the eight dates are strictly chronological:
`raw_start <= dataset_start <= train_end <= val_end <= calibration_start <=
calibration_end <= test_start <= test_end`, and that `test_end <= raw_end`.

## `cities`

List of cities whose loads are aggregated (population-weighted) into the
national forecast. Each entry:

| Key | Type | Example value | Effect |
| --- | --- | --- | --- |
| `name` | string | `Berlin` | city identifier |
| `lat` | float | `52.5200` | latitude for weather fetch |
| `lon` | float | `13.4050` | longitude for weather fetch |
| `population` | int | `3700000` | weight in `Config.city_weights()` (share of total population) |

Current cities: Berlin (52.5200, 13.4050, 3.7M), Hamburg (53.5511, 9.9937, 1.9M),
Munich (48.1351, 11.5820, 1.5M), Cologne (50.9375, 6.9603, 1.1M), Frankfurt
(50.1109, 8.6821, 770k).

Validation: at least one city required; all populations must be positive.

## `smard`

| Key | Type | Current value | Effect |
| --- | --- | --- | --- |
| `base_url` | string | `https://www.smard.de/app/chart_data` | SMARD chart-data endpoint for historical load |
| `filter` | int | `410` | SMARD filter id for real-time electricity consumption / grid load |
| `region` | string | `DE` | SMARD region code (Germany) |
| `resolution` | string | `hour` | time resolution of the fetched series |

## `weather`

| Key | Type | Current value | Effect |
| --- | --- | --- | --- |
| `base_url` | string | `https://archive-api.open-meteo.com/v1/archive` | Open-Meteo archive endpoint (historical observations) |
| `previous_runs_url` | string | `https://previous-runs-api.open-meteo.com/v1/forecast` | Open-Meteo previous-runs endpoint (archived forecasts) |
| `operational_start` | date | `2024-01-20` | from this date onward, archived day-ahead forecasts are used as weather features |
| `hourly_vars` | list | `["temperature_2m", "wind_speed_10m"]` | observed hourly variables fetched from the archive |
| `operational_hourly_vars` | list | `["temperature_2m_previous_day1", "wind_speed_10m_previous_day1"]` | forecast variables fetched from previous-runs for the operational period |
| `timezone` | string | `Europe/Berlin` | timezone used for all date handling |

## `features`

| Key | Type | Current value | Effect |
| --- | --- | --- | --- |
| `weather_strategy` | string | `available_day_ahead` | how weather enters features: `persistence`, `oracle`, or `available_day_ahead` (archived forecast issued 24 h before the target day from 2024 onward; earlier rows use the previous day's observation, see `src/loadfc/data/build_dataset.py`). Default when key absent: `persistence` |
| `hdd_threshold` | float | `18.0` | heating-degree-day base: `HDD = max(0, 18.0 - T)` (`src/loadfc/features/weather_features.py`) |
| `cdd_threshold` | float | `22.0` | cooling-degree-day base: `CDD = max(0, T - 22.0)` |
| `fourier` | list | see below | Fourier seasonal terms added as features |
| `structural_breaks` | map | see below | one-hot window flags for regime changes |
| `lags` | list | `[1, 7]` | daily lag steps added as features |
| `hourly_lags` | list | `[24, 168]` | hourly lag steps (24 h and 168 h) for the hourly model. Default when key absent: `[24, 168]` |

`fourier` entries (each: `name`, `period`, `harmonics`):

| name | period | harmonics | Effect |
| --- | --- | --- | --- |
| `week` | `7` | `1` | weekly seasonality (sin/cos pair) |
| `year` | `365.25` | `1` | annual seasonality (sin/cos pair) |

`structural_breaks` flags:

| Key | start | end | Effect |
| --- | --- | --- | --- |
| `is_covid` | `2020-03-15` | `2020-06-15` | 1 in the COVID lockdown window, else 0 |
| `is_energy_crisis` | `2022-09-01` | `2023-03-31` | 1 in the energy-crisis window, else 0 |

Validation: `weather_strategy` must be one of `persistence`, `oracle`,
`available_day_ahead`, otherwise `ValueError`.

## `models`

Hyperparameters for the four model families. The tree-model values come from
Optuna tuning and are frozen in `config.yaml` (see `tuning` below). Scripts
apply `cfg.seed` as `random_state` when a model accepts it.

### `models.sarimax`

| Key | Type | Current value | Effect |
| --- | --- | --- | --- |
| `order` | list | `[2, 1, 1]` | SARIMAX `(p, d, q)` order |
| `seasonal_order` | list | `[1, 0, 1, 7]` | SARIMAX seasonal `(P, D, Q, s)` order with weekly seasonality |
| `refit` | string/bool | `false` | refit strategy: `false`, `true`, or `periodic` (see `src/loadfc/models/sarimax.py`) |
| `refit_period` | int | `90` | days between refits when `refit: periodic`; default `90` when absent |

### `models.xgboost`

| Key | Type | Current value | Effect |
| --- | --- | --- | --- |
| `n_estimators` | int | `425` | number of boosting rounds |
| `max_depth` | int | `5` | maximum tree depth |
| `learning_rate` | float | `0.04133507869002662` | shrinkage per boosting round |
| `subsample` | float | `0.7265537839619253` | row sampling ratio per round |
| `colsample_bytree` | float | `0.8845748582218896` | column sampling ratio per tree |
| `min_child_weight` | int | `2` | minimum sum of instance weights in a child |
| `gamma` | float | `3.2780443858087294` | minimum loss reduction for a split |

### `models.lightgbm`

| Key | Type | Current value | Effect |
| --- | --- | --- | --- |
| `n_estimators` | int | `1137` | number of boosting rounds |
| `max_depth` | int | `5` | maximum tree depth |
| `learning_rate` | float | `0.0068141987644542885` | shrinkage per boosting round |
| `subsample` | float | `0.8491138046425559` | row sampling ratio per round |
| `subsample_freq` | int | `1` | subsample every N rounds (1 = every round) |
| `colsample_bytree` | float | `0.8139882925367778` | column sampling ratio per tree |
| `min_child_samples` | int | `5` | minimum samples per leaf |
| `num_leaves` | int | `18` | maximum number of leaves per tree |

### `models.random_forest`

| Key | Type | Current value | Effect |
| --- | --- | --- | --- |
| `n_estimators` | int | `463` | number of trees |
| `max_depth` | int | `16` | maximum tree depth |
| `min_samples_split` | int | `2` | minimum samples to split a node |
| `min_samples_leaf` | int | `4` | minimum samples per leaf |
| `max_features` | float | `0.9987985103162023` | fraction of features considered per split |

## `tuning`

| Key | Type | Current value | Effect |
| --- | --- | --- | --- |
| `n_trials` | int | `50` | Optuna trials per model in `scripts/run_tune.py` |
| `cv_splits` | int | `4` | cross-validation folds for tuning evaluation |

Tuning results are written to `results/tuning/best_params.json`; `config.yaml`
is never overwritten by `run_tune.py` — values must be copied in by hand to
adopt them (`scripts/run_tune.py`).

## `ensemble`

| Key | Type | Current value | Effect |
| --- | --- | --- | --- |
| `members` | list | `["SARIMAX", "xgboost", "lightgbm"]` | models combined into the ensemble forecast; selected on the 2024-2025 validation period |

Default when key absent: `["SARIMAX", "xgboost", "lightgbm"]`. Validation:
`members` must not be empty.

## `uncertainty`

| Key | Type | Current value | Default (key absent) | Effect |
| --- | --- | --- | --- | --- |
| `calibration_days` | int | `184` | `181` | trailing days of the calibration window used for conformal scores (`scripts/run_intervals.py`, overridable with `--calib`) |
| `adaptive_gamma` | float | `0.01` | `0.01` | smoothing factor for adaptive conformal adjustment; must satisfy `0 < gamma < 1` |
| `adaptive_window` | int | `365` | `365` | trailing window (days) over which the adaptive factor is estimated; must be >= 1 |

## `monitoring`

| Key | Type | Current value | Default (key absent) | Effect |
| --- | --- | --- | --- | --- |
| `rolling_window` | int | `14` | `14` | trailing days for rolling MAPE and coverage in telemetry; must be >= 2 |
| `mape_warning` | float | `5.0` | `5.0` | rolling MAPE above this (percent) raises a `mape_alert`; must be > 0 |
| `coverage_floor` | float | `0.85` | `0.85` | rolling interval coverage below this raises a `coverage_alert`; must satisfy `0 < floor < 1` |

Alerts are computed in `src/loadfc/evaluation/telemetry.py`.

## `seed`

| Key | Type | Current value | Effect |
| --- | --- | --- | --- |
| `seed` | int | `42` | global random seed; passed to Optuna trials (`run_tune.py`) and applied as `random_state` for tree models (`run_evaluate.py`, `run_hourly.py`, `run_shap.py`) |

## `paths`

Relative to the config file's directory. Resolved via `Config.path(key)`.

| Key | Type | Current value | Effect |
| --- | --- | --- | --- |
| `raw_dir` | string | `data/raw` | downloaded raw SMARD/weather data |
| `processed_dir` | string | `data/processed` | cleaned, feature-engineered datasets |
| `results_dir` | string | `results` | evaluation output, plots, tuning artifacts |

## MLflow tracking

There is no `mlflow` section in `config.yaml`. Tracking is configured in
`src/loadfc/tracking.py`:

- default tracking/registry URI: `sqlite:///<repo-root>/mlflow.db`
  (`local_tracking_uri()`)
- artifact store: `<repo-root>/mlartifacts`
- both can be overridden only in code via the `tracking_uri` argument of
  `track_sklearn_run()`; there is no environment-variable or config-file
  override
- `GITHUB_SHA` (if set) is used as the tracked source revision instead of the
  local `git rev-parse HEAD`

## Environment variables

There is no `.env` file, `.env.example`, or dotenv loading anywhere in the
repository. The only environment variable read by the codebase is `GITHUB_SHA`
(in `src/loadfc/tracking.py`, optional).

## Validation behavior (summary)

`Config.validate()` raises `ValueError` with the following messages on invalid
configuration:

| Rule | Error message |
| --- | --- |
| dates out of chronological order | `dates must be chronological: raw_start <= dataset_start <= train_end <= val_end <= calibration_start <= calibration_end <= test_start <= test_end` |
| `test_end > raw_end` | `test_end must not exceed raw_end` |
| no cities | `at least one city is required` |
| non-positive population | `city populations must be positive` |
| unknown `weather_strategy` | `weather_strategy must be 'persistence', 'oracle' or 'available_day_ahead'` |
| empty `ensemble.members` | `ensemble.members must not be empty` |
| `adaptive_gamma` out of (0, 1) | `uncertainty.adaptive_gamma must be between 0 and 1` |
| `adaptive_window` < 1 | `uncertainty.adaptive_window must be positive` |
| `rolling_window` < 2 | `monitoring.rolling_window must be at least 2` |
| `mape_warning` <= 0 | `monitoring.mape_warning must be positive` |
| `coverage_floor` out of (0, 1) | `monitoring.coverage_floor must be between 0 and 1` |

`Config.from_yaml()` also fails with `KeyError` if any required section or key
(`project`, `split`, `cities`, `smard`, `weather`, `features`, `models`,
`tuning`, `seed`, `paths`) is missing. The optional sections `ensemble`,
`uncertainty`, and `monitoring` fall back to the defaults listed above.

## Testing

Config behavior is covered by `tests/test_config.py`, which verifies date/city
parsing, population-weighted city weights, root-relative path resolution, and
the chronological and monitoring validation rules.
