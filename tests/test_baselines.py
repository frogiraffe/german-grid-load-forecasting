from datetime import date, timedelta

import pandas as pd

from loadfc.evaluation.baselines import baseline_predictions


def test_baseline_predictions_use_only_declared_lags():
    index = [date(2024, 1, 1) + timedelta(days=i) for i in range(10)]
    frame = pd.DataFrame(
        {
            "daily_load": range(10),
            "L_t-1": [None, *range(9)],
            "L_t-7": [None] * 7 + list(range(3)),
        },
        index=index,
    )
    predictions = baseline_predictions(frame, index[0], index[-1])
    assert predictions["naive_1d"].iloc[0] == 0
    assert predictions["seasonal_naive_7d"].tolist() == [0, 1, 2]
    assert predictions["naive_1d"].name == "naive_1d"
