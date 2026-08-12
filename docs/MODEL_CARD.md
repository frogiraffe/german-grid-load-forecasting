# German Grid-Load Forecasting Model Card

## Intended use

Portfolio evaluation of daily and 24-step hourly German load forecasts from SMARD load and forecast-origin weather data.

## Evaluation

Model and feature choices are frozen from training/validation evidence.

<!-- loadfc:generated-start -->
Release results are generated from the checked source revision.
<!-- loadfc:generated-end -->

## Limitations

Weather provenance and availability assumptions are attached to prediction artifacts. Results should not be interpreted as operational guarantees, causal effects, or future accuracy promises.

## Run the study

Run `uv run python scripts/run_pipeline.py --config config.yaml`, then inspect `results/run_summary.json` and `results/evaluation_protocol.json`.
