"""Pandera schemas for external load and weather inputs."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import pandera.pandas as pa

_TEMPERATURE_RANGE = (-80.0, 60.0)
_WIND_RANGE = (0.0, 200.0)


def _load_column() -> pa.Column:
    return pa.Column(
        float,
        checks=[
            pa.Check(
                lambda values: np.isfinite(values),
                element_wise=True,
                error="grid load must be finite",
            ),
            pa.Check.gt(0, error="grid load must be positive"),
        ],
        nullable=False,
        coerce=True,
    )


def validate_load_series(series: pd.Series) -> pd.Series:
    """Require finite, positive grid-load observations."""

    name = series.name or "load"
    schema = pa.DataFrameSchema(
        {name: _load_column()},
        strict=True,
    )
    return schema.validate(series.rename(name).to_frame(), lazy=True)[name]


def _weather_column(name: str, *, nullable: bool) -> pa.Column:
    normalized = name.lower()
    if "temperature" in normalized or normalized.startswith("temp"):
        checks = pa.Check.in_range(
            *_TEMPERATURE_RANGE,
            error=f"{name} must be within {_TEMPERATURE_RANGE}",
        )
    elif "wind" in normalized:
        checks = pa.Check.in_range(
            *_WIND_RANGE,
            error=f"{name} must be within {_WIND_RANGE}",
        )
    else:
        raise ValueError(f"no weather validation rule for column: {name}")
    return pa.Column(float, checks=checks, nullable=nullable, coerce=True)


def validate_weather_frame(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    nullable: bool = False,
    strict: bool = True,
) -> pd.DataFrame:
    """Validate required weather columns and physical ranges."""

    schema = pa.DataFrameSchema(
        {column: _weather_column(column, nullable=nullable) for column in columns},
        strict=strict,
    )
    return schema.validate(frame, lazy=True)


def validate_model_panel(
    frame: pd.DataFrame,
    *,
    load_column: str,
    operational: bool,
) -> pd.DataFrame:
    """Validate the joined modelling panel before feature construction."""

    columns: dict[str, pa.Column] = {
        load_column: _load_column(),
        "Temp_t": _weather_column("Temp_t", nullable=False),
        "Wind_t": _weather_column("Wind_t", nullable=False),
    }
    if operational:
        columns.update(
            {
                "Temp_operational": _weather_column(
                    "Temp_operational",
                    nullable=True,
                ),
                "Wind_operational": _weather_column(
                    "Wind_operational",
                    nullable=True,
                ),
            }
        )
    return pa.DataFrameSchema(columns, strict=True).validate(frame, lazy=True)
