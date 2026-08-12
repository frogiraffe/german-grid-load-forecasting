from datetime import date

import pandas as pd

from loadfc.features import calendar


def test_weekend_flag():
    idx = [date(2024, 6, 28), date(2024, 6, 29), date(2024, 6, 30)]  # Fri, Sat, Sun
    s = calendar.weekend(idx)
    assert list(s) == [0, 1, 1]


def test_holiday_classification():
    hol = calendar.german_holidays([2024])
    assert calendar.holiday_national(date(2024, 1, 1), hol) == 1  # New Year
    assert calendar.holiday_national(date(2024, 10, 3), hol) == 1  # Unity Day
    assert calendar.holiday_national(date(2024, 12, 25), hol) == 0  # Christmas is religious
    assert calendar.holiday_religious(date(2024, 12, 25), hol) == 1
    assert calendar.holiday_religious(date(2024, 3, 29), hol) == 1  # Good Friday 2024
    assert calendar.holiday_religious(date(2024, 1, 1), hol) == 0
    assert calendar.holiday_national(date(2024, 7, 15), hol) == 0  # ordinary day


def test_bridge_day_friday_after_thursday_holiday():
    hol = calendar.german_holidays([2024])
    # Ascension 2024 = Thu 2024-05-09 -> Fri 2024-05-10 is a bridge day
    assert calendar.bridge_day(date(2024, 5, 10), hol) == 1
    assert calendar.bridge_day(date(2024, 5, 9), hol) == 0  # the holiday itself
    assert calendar.bridge_day(date(2024, 5, 13), hol) == 0  # ordinary Monday


def test_calendar_features_frame_has_all_columns():
    idx = pd.Index([date(2020, 4, 1), date(2022, 10, 1), date(2024, 7, 15)])
    feats = calendar.calendar_features(idx)
    assert list(feats.columns) == [
        "Weekend",
        "holiday_national",
        "holiday_religious",
        "bridge_day",
        "pre_holiday",
        "post_holiday",
        "is_covid",
        "is_energy_crisis",
    ]
    assert feats.loc[date(2020, 4, 1), "is_covid"] == 1
    assert feats.loc[date(2022, 10, 1), "is_energy_crisis"] == 1
    assert feats.loc[date(2024, 7, 15), "is_covid"] == 0


def test_pre_and_post_holiday_indicators():
    hol = calendar.german_holidays([2024])
    # Ascension 2024 = Thu 2024-05-09. Wednesday 2024-05-08 is pre_holiday.
    assert calendar.pre_holiday(date(2024, 5, 8), hol) == 1
    # Friday 2024-05-10 is also the day after the holiday (and a bridge day).
    assert calendar.post_holiday(date(2024, 5, 10), hol) == 1
    # Unity Day 2024 = Thu 2024-10-03. Friday 2024-10-04 is post_holiday.
    assert calendar.post_holiday(date(2024, 10, 4), hol) == 1
    assert calendar.pre_holiday(date(2024, 7, 15), hol) == 0
    assert calendar.post_holiday(date(2024, 7, 15), hol) == 0
