"""Declarative search spaces for the tree-based forecasters.

Each entry is ``name -> (kind, low, high)`` where ``kind`` is one of ``int``,
``float`` (uniform) or ``float_log`` (log-uniform). Keeping the ranges as data,
not buried in ``trial.suggest_*`` calls, lets the tests assert two things: that
the keys match ``config.yaml`` exactly, and that every committed value sits
inside its range (so a fresh study lands near the blessed hyperparameters).
"""

from __future__ import annotations

from typing import Any

SPACES: dict[str, dict[str, tuple[str, float, float]]] = {
    "xgboost": {
        "n_estimators": ("int", 200, 1200),
        "max_depth": ("int", 3, 8),
        "learning_rate": ("float_log", 0.005, 0.3),
        "subsample": ("float", 0.5, 1.0),
        "colsample_bytree": ("float", 0.5, 1.0),
        "min_child_weight": ("int", 1, 10),
        "gamma": ("float", 0.0, 5.0),
    },
    "lightgbm": {
        "n_estimators": ("int", 200, 1200),
        "max_depth": ("int", 3, 8),
        "learning_rate": ("float_log", 0.005, 0.3),
        "subsample": ("float", 0.5, 1.0),
        "subsample_freq": ("int", 1, 1),
        "colsample_bytree": ("float", 0.5, 1.0),
        "min_child_samples": ("int", 5, 50),
        "num_leaves": ("int", 4, 256),
    },
    "random_forest": {
        "n_estimators": ("int", 100, 600),
        "max_depth": ("int", 4, 16),
        "min_samples_split": ("int", 2, 10),
        "min_samples_leaf": ("int", 1, 10),
        "max_features": ("float", 0.3, 1.0),
    },
}


def suggest_params(kind: str, trial: Any) -> dict[str, Any]:
    """Draw one hyperparameter set for `kind` from the given Optuna trial."""
    if kind not in SPACES:
        raise ValueError(f"unknown ML model: {kind!r}")
    params: dict[str, Any] = {}
    for name, (space, lo, hi) in SPACES[kind].items():
        if kind == "lightgbm" and name == "num_leaves":
            hi = min(hi, 2 ** int(params["max_depth"]))
        if space == "int":
            params[name] = trial.suggest_int(name, int(lo), int(hi))
        elif space == "float":
            params[name] = trial.suggest_float(name, lo, hi)
        elif space == "float_log":
            params[name] = trial.suggest_float(name, lo, hi, log=True)
        else:  # pragma: no cover - guarded by the spec above
            raise ValueError(f"unknown space kind: {space!r}")
    return params
