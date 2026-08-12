# Technical report

The repository contains an English report in Markdown, TeX, and PDF formats.

The report generator reads the result CSV files in `results`. It writes
numerical LaTeX macros to `generated_results.tex`. Run the pipeline before you
build the report. The pipeline creates `results`.

Build the complete evaluated release, including the report:

```bash
uv run python scripts/run_pipeline.py
```

To rebuild only the report from an existing local `results/` bundle:

```bash
uv run python scripts/render_report_data.py
tectonic -X compile report/technical-report-en.tex
```

Generated files:

- `technical-report-en.pdf`

Source files:

- `technical-report-en.md`
- `technical-report-en.tex`
