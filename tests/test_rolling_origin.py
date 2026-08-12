from datetime import date

from loadfc.evaluation.rolling_origin import six_month_windows


def test_six_month_windows_split_2024_2025():
    wins = six_month_windows(date(2024, 1, 1), date(2025, 12, 31))
    labels = [w[0] for w in wins]
    assert labels == ["2024-H1", "2024-H2", "2025-H1", "2025-H2"]
    assert wins[0][1] == date(2024, 1, 1) and wins[0][2] == date(2024, 6, 30)
    assert wins[3][1] == date(2025, 7, 1) and wins[3][2] == date(2025, 12, 31)
