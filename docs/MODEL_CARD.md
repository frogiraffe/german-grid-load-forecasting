# German Grid-Load Forecasting Model Card

## Intended use

Portfolio evaluation of daily and fixed 24-hour-ahead German load forecasts from SMARD load and forecast-origin weather data.

## Evaluation

Model and feature choices are frozen from training/validation evidence. Current numerical results
are reported once in the [README](../README.md#results) and in full in the
[technical report](../report/technical-report-en.pdf).

## Limitations

Weather provenance and availability assumptions are attached to prediction artifacts. The hourly
headline is a fixed 24-hour-ahead evaluation, not a 24-hour profile issued from one common origin.
Results should not be interpreted as operational guarantees, causal effects, or future accuracy promises.

## Run the study

Install the explanation extra and run the clean-source release pipeline:

```bash
uv sync --extra explain
uv run python scripts/run_pipeline.py --config config.yaml
```

The generated `results/` tables are uploaded by the manual **reproduce release evidence** GitHub Actions workflow instead of being stored in Git. The bounded dashboard and this model card remain available in the repository.
