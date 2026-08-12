# Evaluation Ledger

This ledger is the human-readable decision record for the frozen evaluation
protocol. Machine-readable stream records live in
[`results/evaluation_protocol.json`](../results/evaluation_protocol.json); use a
stream ID there rather than copying a fingerprint that changes with source or
configuration identity.

## Frozen chronology

| Role | Inclusive period | Permitted use |
| --- | --- | --- |
| train | 2019-01-14–2023-12-31 | Fit candidate models and feature transforms. |
| validation | 2024-01-01–2025-06-30 | Choose model/candidate and feature configuration. |
| calibration | 2025-07-01–2025-12-31 | Calibration supplies interval residuals only. |
| retrospective_final | 2026-01-01–2026-08-04 | Report frozen point and interval evidence; never tune or select. |

The current final role is `retrospective_final`, not a prospective holdout. The
expanded 2026-01-01–2026-08-04 block has already been inspected, so it cannot
be presented as an untouched final evaluation.

## Prior inspection history

| Earlier protocol role | Inclusive period | Status now |
| --- | --- | --- |
| validation | 2024-01-01–2025-12-31 | Previously used selection window; already inspected and superseded by the frozen validation boundary. |
| calibration | 2026-01-01–2026-06-30 | Previously used interval-calibration window; already inspected. |
| final/test | 2026-07-01–2026-07-22 | Previously used final/test window; already inspected. |

## Selection boundary

Model and feature choices use **training/validation evidence only**. Calibration
supplies interval residuals; calibration and retrospective-final metrics do not
tune, rank, or select candidates. The validation evidence path is
`results/metrics/validation_metrics.csv`.

## Frozen decision and state policy

| Field | Recorded policy |
| --- | --- |
| Weather strategy | `available_day_ahead`: archived day-ahead forecast weather where available, otherwise the documented prior-day observation fallback. |
| Seed | Seed: `42` |
| Ordered feature schema | The exact ordered feature schema is the `ordered_feature_columns` field of each stream record; order is part of model input compatibility. |
| Point-state policy | Fit the validation-approved point model through 2025-06-30, then run one continuous calibration-plus-retrospective stream under the configured refit/update policy (`sarimax.refit: false`; actuals update rolling state only). |
| Interval-state policy | Fixed/CQR interval residual state is calibrated separately; adaptive interval state updates only after an actual is observed and never changes point-model fitting or features. |
| Compatibility record | Source revision, config SHA-256, stream/model identity, ordered schema, model parameters, seed, all four periods, weather strategy, point-state policy, interval-state policy, validation evidence, and rationale are fingerprinted in `results/evaluation_protocol.json`. |

## Validation-only candidate decisions

| Path | Candidate set | Frozen choice and rationale | Machine evidence |
| --- | --- | --- | --- |
| Daily point forecasts | SARIMAX, XGBoost, LightGBM, RandomForest, deterministic naive baselines, and the configured ensemble | Preserve the configured SARIMAX + XGBoost + LightGBM ensemble and member streams because the 2024-01-01–2025-06-30 validation results supplied the decision. Calibration/final outcomes cannot reselect it. | `results/metrics/validation_metrics.csv`; `results/evaluation_protocol.json` stream IDs `daily/*` |
| Hourly point forecasts | residual hybrid, direct LightGBM, and direct ridge | Preserve the validation-selected residual-hybrid choice; the existing hourly candidates remain comparison baselines, not final-block selection inputs. | Validation evidence and, once registered, `results/evaluation_protocol.json` stream IDs `hourly/*` |
| Weather feature candidate | operational day-ahead weather vs. persistence fallback | The persistence candidate is scored on 2024-01-01–2025-06-30 only. Acceptance requires a deterministic ensemble-MAPE improvement beyond the practical tie threshold and intact protocol checks; otherwise the configured operational weather/model is preserved. | `results/metrics/persistence_weather_validation_metrics.csv`; `results/metrics/low_risk_improvement_decision.json` |

The weather-candidate decision is intentionally validation-only. Retrospective
weather differences remain descriptive evidence and cannot change the selected
model. This single candidate does not establish causal weather value, and the
time-series uncertainty and realized-weather limitations below still apply.

## Interval evidence limits

- Fixed split-conformal and CQR results are **empirical retrospective coverage**
  over eligible rows from the frozen protocol. They are not a prospective claim
  or an unconditional time-series coverage guarantee.
- Adaptive conformal is **prequential monitoring only**: its interval state
  changes after observed actuals and it has **no unconditional time-series coverage guarantee**.
- Hourly aggregate, horizon, or Berlin-local-day interval metrics require the
  **23-row threshold**, chosen from the shortest valid Berlin DST day; sparse
  groups are not reported as complete evidence.

## How to review a stream

1. Start with the stream ID in `results/evaluation_protocol.json`.
2. Confirm its fingerprinted split, schema, source/config identity, weather
   strategy, model parameters, seed, and point/interval policies.
3. Confirm calibration rows are labelled `calibration` and final rows are
   labelled `retrospective_final`; the pair must share the same stream ID and
   protocol fingerprint before interval or result evidence is consumed.
