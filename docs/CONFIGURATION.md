# Configuration

The project uses one YAML file, [`config.yaml`](../config.yaml). `Config.from_yaml()`
in `src/loadfc/config.py` parses it into frozen dataclasses, resolves paths relative
to the YAML file, and calls `Config.validate()`.

## Config file format

Run the full release pipeline with the repository configuration:

```bash
uv run python scripts/run_pipeline.py --config config.yaml
```

To use another configuration, pass its path through `--config`. Relative paths
inside that file are resolved from the file's directory, not from the shell's
current directory:

```bash
uv run python scripts/run_pipeline.py --config experiments/experiment-a.yaml
```

Every current script in `scripts/` accepts `--config` and defaults to
`config.yaml`: `build_dashboard_data.py`, `render_report_data.py`,
`run_analysis.py`, `run_comparison.py`, `run_error_analysis.py`,
`run_evaluate.py`, `run_features.py`, `run_fetch.py`, `run_hourly.py`,
`run_intervals.py`, `run_pipeline.py`, `run_shap.py`, `validate_results.py`, and
`write_run_summary.py`.

A minimal shape is:

```yaml
project:
  raw_start: "2019-01-01"
  raw_end: "2026-08-04"
  dataset_start: "2019-01-14"
split:
  train_end: "2023-12-31"
  val_end: "2025-06-30"
  calibration_start: "2025-07-01"
  calibration_end: "2025-12-31"
  test_start: "2026-01-01"
  test_end: "2026-08-04"
cities:
  - {name: "Berlin", lat: 52.52, lon: 13.405, population: 3700000}
smard: {base_url: "https://www.smard.de/app/chart_data", filter: 410, region: "DE", resolution: "hour"}
weather:
  base_url: "https://archive-api.open-meteo.com/v1/archive"
  previous_runs_url: "https://previous-runs-api.open-meteo.com/v1/forecast"
  operational_start: "2024-01-20"
  hourly_vars: ["temperature_2m", "wind_speed_10m"]
  operational_hourly_vars: ["temperature_2m_previous_day1", "wind_speed_10m_previous_day1"]
  timezone: "Europe/Berlin"
features:
  weather_strategy: "available_day_ahead"
  hdd_threshold: 18.0
  cdd_threshold: 22.0
  fourier: [{name: "week", period: 7, harmonics: 1}]
  structural_breaks: {}
  lags: [1, 7]
  hourly_lags: [24, 168]
models:
  sarimax: {order: [2, 1, 1], seasonal_order: [1, 0, 1, 7], refit: false}
  xgboost: {}
  lightgbm: {}
  random_forest: {}
ensemble: {members: ["SARIMAX", "xgboost", "lightgbm"]}
uncertainty: {calibration_days: 184, adaptive_gamma: 0.01, adaptive_window: 365}
seed: 42
paths: {raw_dir: "data/raw", processed_dir: "data/processed", results_dir: "results"}
```

The empty model maps above only illustrate the schema. The shipped release uses
the concrete model parameters documented below.

## Environment variables

The application does not use dotenv files and needs no secrets for the public
data sources in `config.yaml`.

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `GITHUB_SHA` | No | current `git rev-parse HEAD` result | Overrides the source revision recorded in release provenance when CI supplies it. |

`SOURCE_DATE_EPOCH=0` is set internally by `run_pipeline.py` while compiling the
report; users do not need to configure it.

## Settings reference

### `project` and `split`

| Key | Required | Current value | Purpose |
| --- | --- | --- | --- |
| `project.raw_start` | Yes | `2019-01-01` | First date fetched from the raw providers. |
| `project.raw_end` | Yes | `2026-08-04` | Last raw-data date. |
| `project.dataset_start` | Yes | `2019-01-14` | First retained daily or local-calendar hourly date. |
| `split.train_end` | Yes | `2023-12-31` | Last training date. |
| `split.val_end` | Yes | `2025-06-30` | Last validation/model-selection date. |
| `split.calibration_start` | Yes | `2025-07-01` | First conformal-calibration date. |
| `split.calibration_end` | Yes | `2025-12-31` | Last conformal-calibration date. |
| `split.test_start` | Yes | `2026-01-01` | First retrospective final-evaluation date. |
| `split.test_end` | Yes | `2026-08-04` | Last retrospective final-evaluation date. |

Dates use `YYYY-MM-DD`. Validation requires
`raw_start <= dataset_start <= train_end <= val_end <= calibration_start <=
calibration_end <= test_start <= test_end` and also requires
`split.test_end <= project.raw_end`.

### `cities`

`cities` must contain at least one entry. Weather series are fetched per city
and combined using `population / total_population` from `Config.city_weights()`.

| Entry key | Required | Type | Purpose |
| --- | --- | --- | --- |
| `name` | Yes | string | City identifier used in fetched frames. |
| `lat` | Yes | float | Weather API latitude. |
| `lon` | Yes | float | Weather API longitude. |
| `population` | Yes | positive integer | Population weight for national weather aggregation. |

The shipped configuration contains Berlin, Hamburg, Munich, Cologne, and
Frankfurt.

### `smard`

`src/loadfc/data/smard.py` uses all four values to construct SMARD index and
chunk URLs.

| Key | Required | Current value | Purpose |
| --- | --- | --- | --- |
| `base_url` | Yes | `https://www.smard.de/app/chart_data` | Base URL for chart-data requests. |
| `filter` | Yes | `410` | Filter identifier inserted into request paths. |
| `region` | Yes | `DE` | Region inserted into request paths. |
| `resolution` | Yes | `hour` | Resolution inserted into index and chunk filenames. |

### `weather`

| Key | Required | Current value | Purpose |
| --- | --- | --- | --- |
| `base_url` | Yes | `https://archive-api.open-meteo.com/v1/archive` | Historical weather request endpoint. |
| `previous_runs_url` | For `available_day_ahead` | `https://previous-runs-api.open-meteo.com/v1/forecast` | Archived forecast request endpoint. |
| `operational_start` | For `available_day_ahead` | `2024-01-20` | First target date requested from the previous-runs archive. |
| `hourly_vars` | Yes | `temperature_2m`, `wind_speed_10m` | Variables requested for observed weather. |
| `operational_hourly_vars` | For `available_day_ahead` | `temperature_2m_previous_day1`, `wind_speed_10m_previous_day1` | Day-ahead variables requested from previous runs. |
| `timezone` | Yes | `Europe/Berlin` | Timezone sent for daily weather requests and used for local-date slicing of cached hourly data; live hourly requests use UTC. |

### `features`

| Key | Required | Current value | Default if omitted | Purpose |
| --- | --- | --- | --- | --- |
| `weather_strategy` | No | `available_day_ahead` | `persistence` | Selects `persistence`, `oracle`, or `available_day_ahead` weather features. |
| `hdd_threshold` | Yes | `18.0` | none | Heating-degree threshold. |
| `cdd_threshold` | Yes | `22.0` | none | Cooling-degree threshold. |
| `fourier` | Yes | weekly and yearly entries | none | Seasonal terms; each entry requires `name`, `period`, and `harmonics`. |
| `structural_breaks` | No | COVID and energy-crisis windows | built-in windows | Overrides inclusive calendar-indicator windows by `start` and `end`. |
| `lags` | Yes | `[1, 7]` | none | Daily target lags. |
| `hourly_lags` | No | `[24, 168]` | `[24, 168]` | Hourly target lags. |

Current Fourier entries are `week` (`period: 7`, `harmonics: 1`) and `year`
(`period: 365.25`, `harmonics: 1`). Current structural-break entries are
`is_covid` (`2020-03-15` through `2020-06-15`) and `is_energy_crisis`
(`2022-09-01` through `2023-03-31`).

### `models`

All four model maps are required by `run_evaluate.py`. Except for SARIMAX's
explicit control fields, their values are passed to the corresponding installed
estimator.

| Model | Keys and current values |
| --- | --- |
| `sarimax` | `order: [2, 1, 1]`; `seasonal_order: [1, 0, 1, 7]`; `refit: false`; `refit_period: 90` |
| `xgboost` | `n_estimators: 425`; `max_depth: 5`; `learning_rate: 0.04133507869002662`; `subsample: 0.7265537839619253`; `colsample_bytree: 0.8845748582218896`; `min_child_weight: 2`; `gamma: 3.2780443858087294` |
| `lightgbm` | `n_estimators: 1137`; `max_depth: 5`; `learning_rate: 0.0068141987644542885`; `subsample: 0.8491138046425559`; `subsample_freq: 1`; `colsample_bytree: 0.8139882925367778`; `min_child_samples: 5`; `num_leaves: 18` |
| `random_forest` | `n_estimators: 463`; `max_depth: 16`; `min_samples_split: 2`; `min_samples_leaf: 4`; `max_features: 0.9987985103162023` |

`models.sarimax.refit` accepts YAML `false`/`true` or the string `periodic`.
`refit_period` defaults to `90` when omitted. `seed` is added as
`random_state` for tree models unless that parameter is already present.

### `ensemble`, `uncertainty`, `seed`, and `paths`

| Key | Required | Current value | Default if omitted | Purpose |
| --- | --- | --- | --- | --- |
| `ensemble.members` | No | `SARIMAX`, `xgboost`, `lightgbm` | same list | Members averaged by the daily ensemble; the list cannot be empty. |
| `uncertainty.calibration_days` | No | `184` | `181` | Trailing calibration rows used by `run_intervals.py`; `--calib` overrides it. |
| `uncertainty.adaptive_gamma` | No | `0.01` | `0.01` | Adaptive conformal update factor; must be strictly between 0 and 1. |
| `uncertainty.adaptive_window` | No | `365` | `365` | Adaptive conformal trailing window; must be positive. |
| `seed` | Yes | `42` | none | Random state supplied to tree models. |
| `paths.raw_dir` | Yes | `data/raw` | none | Cached downloaded data. |
| `paths.processed_dir` | Yes | `data/processed` | none | Built daily/hourly datasets and feature matrices. |
| `paths.results_dir` | Yes | `results` | none | Evaluation and release artifacts. |

The `ensemble` and `uncertainty` sections are optional as complete sections;
if present, their listed keys are read directly. All path values in the shipped
configuration are relative. The canonical pipeline rejects an absolute
`paths.results_dir` and any relative non-result staging path that resolves
outside the staging directory; absolute non-result paths are left as external inputs.

## Required vs optional settings

`Config.from_yaml()` directly requires `project`, `split`, `cities`, `smard`,
`weather`, `features`, `models`, `seed`, and `paths`. It reads the project,
split, and city fields immediately; other nested keys marked "Yes" are required
by their downstream data, feature, model, or path consumer. Missing accessed
keys raise `KeyError`. `ensemble` and `uncertainty` have whole-section defaults.

`Config.validate()` raises `ValueError` for out-of-order dates, a test end after
the raw end, no cities, non-positive populations, an unsupported weather
strategy, an empty ensemble, an adaptive gamma outside `(0, 1)`, or a
non-positive adaptive window.

Some nested values are conditionally required by the code that uses them. For
example, previous-runs weather settings are required when
`features.weather_strategy` is `available_day_ahead`, and all four model maps are
required by the full evaluation pipeline.

## Defaults

Source-defined defaults are:

| Setting | Default |
| --- | --- |
| `features.weather_strategy` | `persistence` |
| `features.hourly_lags` | `[24, 168]` |
| `features.structural_breaks.is_covid` | `2020-03-15` through `2020-06-15` |
| `features.structural_breaks.is_energy_crisis` | `2022-09-01` through `2023-03-31` |
| `models.sarimax.refit_period` | `90` |
| `ensemble` | `{members: [SARIMAX, xgboost, lightgbm]}` |
| `uncertainty` | `{calibration_days: 181, adaptive_gamma: 0.01, adaptive_window: 365}` |

There are no source defaults for other required settings.

## Per-environment overrides

The repository has no `.env`, `.env.example`, environment-specific YAML files,
or automatic development/staging/production merge logic. Use a separate complete
YAML file and pass it explicitly with `--config`. Keep relative path values next
to that YAML file because `Config.root` is the file's parent directory.
