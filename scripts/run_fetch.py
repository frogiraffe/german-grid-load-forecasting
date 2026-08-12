"""Fetch SMARD load + Open-Meteo weather and build the processed dataset.

Usage:
    python scripts/run_fetch.py [--config config.yaml] [--refresh]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from loadfc.config import Config
from loadfc.data.build_dataset import build


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--refresh", action="store_true", help="ignore cached raw data")
    args = parser.parse_args()

    cfg = Config.from_yaml(Path(args.config))
    df = build(cfg, refresh=args.refresh)

    print(f"dataset: {len(df)} rows, {df.index.min()} -> {df.index.max()}")
    print(df.describe().round(1).to_string())


if __name__ == "__main__":
    main()
