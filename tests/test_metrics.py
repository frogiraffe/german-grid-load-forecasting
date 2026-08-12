import pandas as pd
import pytest

from loadfc.evaluation import metrics


def test_basic_metrics():
    y = pd.Series([100.0, 200.0, 300.0])
    yhat = pd.Series([110.0, 190.0, 300.0])  # errors: -10, +10, 0
    assert metrics.mae(y, yhat) == pytest.approx(20.0 / 3)
    assert metrics.rmse(y, yhat) == pytest.approx((200.0 / 3) ** 0.5)
    assert metrics.mape(y, yhat) == pytest.approx(5.0)  # 100*mean(.1,.05,0)


def test_seasonal_naive_mae_and_mase():
    train = pd.Series([10.0, 12.0, 14.0, 16.0, 18.0])  # |y_t - y_{t-2}| == 4
    naive = metrics.seasonal_naive_mae(train, season=2)
    assert naive == pytest.approx(4.0)
    y = pd.Series([100.0, 100.0])
    yhat = pd.Series([102.0, 98.0])  # mae == 2
    assert metrics.mase(y, yhat, naive) == pytest.approx(0.5)


def test_all_metrics_keys():
    y = pd.Series([1.0, 2.0])
    out = metrics.all_metrics(y, y, naive_mae=1.0)
    assert set(out) == {"RMSE", "MAE", "MAPE", "MASE"}
