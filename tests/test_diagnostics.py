import numpy as np

from loadfc.evaluation.diagnostics import arch_lm, ljung_box


def test_ljung_box_white_noise_not_rejected():
    rng = np.random.default_rng(0)
    resid = rng.normal(size=500)
    stat, p = ljung_box(resid, lags=10)
    assert p > 0.05


def test_arch_lm_detects_volatility_clustering():
    rng = np.random.default_rng(0)
    n = 800
    e = np.zeros(n)
    for t in range(1, n):
        scale = np.sqrt(0.2 + 0.8 * e[t - 1] ** 2)
        e[t] = rng.normal(scale=scale)
    stat, p = arch_lm(e, lags=5)
    assert p < 0.05
