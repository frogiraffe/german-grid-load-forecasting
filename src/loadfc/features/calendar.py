"""Calendar, holiday, and bridge-day features."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date, timedelta

import holidays
import pandas as pd

# Fixed-date nationwide secular holidays; everything else nationwide is religious.
_NATIONAL = {(1, 1), (5, 1), (10, 3)}
_ONE_DAY = timedelta(days=1)

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


def calendar_features(index: Iterable[date]) -> pd.DataFrame:
    idx = pd.Index(index)
    hol = german_holidays({d.year for d in idx})
    data = {
        "Weekend": [int(d.weekday() >= 5) for d in idx],
        "holiday_national": [holiday_national(d, hol) for d in idx],
        "holiday_religious": [holiday_religious(d, hol) for d in idx],
        "bridge_day": [bridge_day(d, hol) for d in idx],
        "pre_holiday": [pre_holiday(d, hol) for d in idx],
        "post_holiday": [post_holiday(d, hol) for d in idx],
    }
    return pd.DataFrame(data, index=idx)
