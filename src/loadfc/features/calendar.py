"""Calendar, holiday, bridge-day and structural-break indicator features."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import date, timedelta

import holidays
import pandas as pd

# Fixed-date nationwide secular holidays; everything else nationwide is religious.
_NATIONAL = {(1, 1), (5, 1), (10, 3)}
_ONE_DAY = timedelta(days=1)

# Structural-break windows (inclusive).
_COVID = (date(2020, 3, 15), date(2020, 6, 15))
_ENERGY_CRISIS = (date(2022, 9, 1), date(2023, 3, 31))


def german_holidays(years: Iterable[int]) -> holidays.HolidayBase:
    return holidays.country_holidays("DE", years=list(years))


def weekend(index: Iterable[date]) -> pd.Series:
    idx = pd.Index(index)
    return pd.Series([int(d.weekday() >= 5) for d in idx], index=idx, name="Weekend")


def holiday_national(d: date, hol: holidays.HolidayBase) -> int:
    return int(d in hol and (d.month, d.day) in _NATIONAL)


def holiday_religious(d: date, hol: holidays.HolidayBase) -> int:
    return int(d in hol and (d.month, d.day) not in _NATIONAL)


def bridge_day(d: date, hol: holidays.HolidayBase) -> int:
    if d.weekday() >= 5 or d in hol:
        return 0
    if d.weekday() == 4 and (d - _ONE_DAY) in hol:  # Friday after Thursday holiday
        return 1
    if d.weekday() == 0 and (d + _ONE_DAY) in hol:  # Monday before Tuesday holiday
        return 1
    return 0


def pre_holiday(d: date, hol: holidays.HolidayBase) -> int:
    return int(d.weekday() < 5 and d not in hol and (d + _ONE_DAY) in hol)


def post_holiday(d: date, hol: holidays.HolidayBase) -> int:
    return int(d.weekday() < 5 and d not in hol and (d - _ONE_DAY) in hol)


def _in_window(d: date, window: tuple[date, date]) -> int:
    return int(window[0] <= d <= window[1])


def _window_from_config(
    structural_breaks: Mapping | None,
    name: str,
    default: tuple[date, date],
) -> tuple[date, date]:
    if not structural_breaks or name not in structural_breaks:
        return default
    raw = structural_breaks[name]
    return (date.fromisoformat(raw["start"]), date.fromisoformat(raw["end"]))


def calendar_features(
    index: Iterable[date], structural_breaks: Mapping | None = None
) -> pd.DataFrame:
    idx = pd.Index(index)
    hol = german_holidays({d.year for d in idx})
    covid = _window_from_config(structural_breaks, "is_covid", _COVID)
    energy_crisis = _window_from_config(structural_breaks, "is_energy_crisis", _ENERGY_CRISIS)
    data = {
        "Weekend": [int(d.weekday() >= 5) for d in idx],
        "holiday_national": [holiday_national(d, hol) for d in idx],
        "holiday_religious": [holiday_religious(d, hol) for d in idx],
        "bridge_day": [bridge_day(d, hol) for d in idx],
        "pre_holiday": [pre_holiday(d, hol) for d in idx],
        "post_holiday": [post_holiday(d, hol) for d in idx],
        "is_covid": [_in_window(d, covid) for d in idx],
        "is_energy_crisis": [_in_window(d, energy_crisis) for d in idx],
    }
    return pd.DataFrame(data, index=idx)
