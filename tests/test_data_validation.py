from __future__ import annotations

import numpy as np
import pandas as pd
import pandera.errors
import pytest

from loadfc.data.validation import (
    validate_load_series,
    validate_model_panel,
    validate_weather_frame,
)


def test_load_schema_rejects_nonpositive_values():
    load = pd.Series([50_000.0, 0.0], name="hourly_load")

    with pytest.raises(pandera.errors.SchemaErrors, match="grid load must be positive"):
        validate_load_series(load)


def test_load_schema_rejects_infinite_values():
    load = pd.Series([50_000.0, np.inf], name="hourly_load")

    with pytest.raises(pandera.errors.SchemaErrors, match="grid load must be finite"):
        validate_load_series(load)


def test_model_panel_rejects_infinite_load():
    panel = pd.DataFrame(
        {
            "hourly_load": [50_000.0, np.inf],
            "Temp_t": [15.0, 16.0],
            "Wind_t": [5.0, 6.0],
        }
    )

    with pytest.raises(pandera.errors.SchemaErrors, match="grid load must be finite"):
        validate_model_panel(panel, load_column="hourly_load", operational=False)


def test_weather_schema_rejects_physical_range_violation():
    weather = pd.DataFrame(
        {
            "temperature_2m": [15.0, 100.0],
            "wind_speed_10m": [5.0, 10.0],
        }
    )

    with pytest.raises(pandera.errors.SchemaErrors, match="temperature_2m"):
        validate_weather_frame(
            weather,
            ["temperature_2m", "wind_speed_10m"],
        )


def test_weather_schema_rejects_missing_required_column():
    weather = pd.DataFrame({"Temp_t": [15.0]})

    with pytest.raises(pandera.errors.SchemaErrors, match="Wind_t"):
        validate_weather_frame(weather, ["Temp_t", "Wind_t"])


def test_model_panel_allows_unavailable_operational_history():
    panel = pd.DataFrame(
        {
            "hourly_load": [50_000.0],
            "Temp_t": [15.0],
            "Wind_t": [5.0],
            "Temp_operational": [None],
            "Wind_operational": [None],
        }
    )

    validated = validate_model_panel(
        panel,
        load_column="hourly_load",
        operational=True,
    )

    assert validated.isna().sum().to_dict() == {
        "hourly_load": 0,
        "Temp_t": 0,
        "Wind_t": 0,
        "Temp_operational": 1,
        "Wind_operational": 1,
    }
