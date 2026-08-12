import numpy as np
import pandas as pd

from loadfc.evaluation.vif import compute_vif


def test_vif_flags_collinear_columns():
    rng = np.random.default_rng(0)
    x1 = rng.normal(size=300)
    df = pd.DataFrame(
        {
            "x1": x1,
            "x2": 2.0 * x1 + rng.normal(scale=1e-3, size=300),
            "x3": rng.normal(size=300),
        }
    )
    vif = compute_vif(df)
    assert vif["x1"] > 10 and vif["x2"] > 10
    assert vif["x3"] < 5
    assert list(vif.index) == list(vif.sort_values(ascending=False).index)
