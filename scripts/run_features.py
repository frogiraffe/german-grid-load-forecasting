"""Build the modelling feature matrix from the processed dataset and save it.

Usage:
    python scripts/run_features.py [--config config.yaml]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from loadfc.config import Config
from loadfc.features.assemble import build_features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = Config.from_yaml(Path(args.config))
    dataset = pd.read_parquet(cfg.path("processed_dir") / "dataset.parquet")
    feats = build_features(dataset, cfg)

    out_dir = cfg.path("processed_dir")
    feats.to_parquet(out_dir / "features.parquet")

    print(f"features: {feats.shape[0]} rows x {feats.shape[1]} cols")
    print("columns:", list(feats.columns))
    print("holiday rows:", int(feats[["holiday_national", "holiday_religious"]].sum().sum()))
    print(
        "covid rows:",
        int(feats["is_covid"].sum()),
        "| energy-crisis rows:",
        int(feats["is_energy_crisis"].sum()),
    )


if __name__ == "__main__":
    main()
