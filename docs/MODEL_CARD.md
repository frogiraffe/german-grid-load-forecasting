# German Grid-Load Forecasting Model Card

## Intended use

Portfolio evaluation of daily and 24-step hourly German load forecasts from SMARD load and forecast-origin weather data.

## Evaluation

Model and feature choices are frozen from training/validation evidence.

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

## Limitations

Weather provenance and availability assumptions are attached to prediction artifacts. Results should not be interpreted as operational guarantees, causal effects, or future accuracy promises.

## Run the study

Install the explanation extra and run the clean-source release pipeline:

```bash
uv sync --extra explain
uv run python scripts/run_pipeline.py --config config.yaml
```

The generated `results/` tables are uploaded by the manual **reproduce release evidence** GitHub Actions workflow instead of being stored in Git. The bounded dashboard and this model card remain available in the repository.
