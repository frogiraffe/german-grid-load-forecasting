import numpy as np

from loadfc.evaluation.diebold_mariano import dm_pvalue_matrix, dm_test


def test_dm_detects_clear_winner():
    rng = np.random.default_rng(0)
    e_good = rng.normal(scale=1.0, size=400)
    e_bad = rng.normal(scale=5.0, size=400)
    stat, p = dm_test(e_bad, e_good)  # model 1 worse than model 2
    assert p < 0.05


def test_dm_equal_models_not_significant():
    rng = np.random.default_rng(0)
    e = rng.normal(size=400)
    stat, p = dm_test(e, e.copy())
    assert p > 0.5


def test_pvalue_matrix_shape_and_bonferroni():
    rng = np.random.default_rng(0)
    errs = {
        "A": rng.normal(scale=1.0, size=300),
        "B": rng.normal(scale=1.0, size=300),
        "C": rng.normal(scale=8.0, size=300),
    }
    m = dm_pvalue_matrix(errs)
    assert list(m.index) == ["A", "B", "C"] and list(m.columns) == ["A", "B", "C"]
    assert np.isnan(m.loc["A", "A"])
    assert m.loc["A", "C"] == m.loc["C", "A"]
    vals = m.to_numpy()
    assert (vals[~np.isnan(vals)] <= 1.0).all()
