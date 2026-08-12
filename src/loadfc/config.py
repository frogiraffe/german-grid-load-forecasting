from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml


def _to_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


@dataclass(frozen=True)
class City:
    name: str
    lat: float
    lon: float
    population: int


@dataclass(frozen=True)
class SplitConfig:
    train_end: date
    val_end: date
    calibration_start: date
    calibration_end: date
    test_start: date
    test_end: date


@dataclass(frozen=True)
class Config:
    raw_start: date
    raw_end: date
    dataset_start: date
    split: SplitConfig
    cities: list[City]
    smard: dict[str, Any]
    weather: dict[str, Any]
    features: dict[str, Any]
    models: dict[str, Any]
    ensemble: dict[str, Any]
    uncertainty: dict[str, Any]
    seed: int
    paths: dict[str, str]
    root: Path

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        path = Path(path).resolve()
        root = path.parent
        raw = yaml.safe_load(path.read_text())

        split = SplitConfig(
            train_end=_to_date(raw["split"]["train_end"]),
            val_end=_to_date(raw["split"]["val_end"]),
            calibration_start=_to_date(raw["split"]["calibration_start"]),
            calibration_end=_to_date(raw["split"]["calibration_end"]),
            test_start=_to_date(raw["split"]["test_start"]),
            test_end=_to_date(raw["split"]["test_end"]),
        )
        cities = [
            City(
                name=c["name"],
                lat=float(c["lat"]),
                lon=float(c["lon"]),
                population=int(c["population"]),
            )
            for c in raw["cities"]
        ]
        cfg = cls(
            raw_start=_to_date(raw["project"]["raw_start"]),
            raw_end=_to_date(raw["project"]["raw_end"]),
            dataset_start=_to_date(raw["project"]["dataset_start"]),
            split=split,
            cities=cities,
            smard=raw["smard"],
            weather=raw["weather"],
            features=raw["features"],
            models=raw["models"],
            ensemble=raw.get("ensemble", {"members": ["SARIMAX", "xgboost", "lightgbm"]}),
            uncertainty=raw.get(
                "uncertainty",
                {"calibration_days": 181, "adaptive_gamma": 0.01, "adaptive_window": 365},
            ),
            seed=int(raw["seed"]),
            paths=raw["paths"],
            root=root,
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        ordered = [
            self.raw_start,
            self.dataset_start,
            self.split.train_end,
            self.split.val_end,
            self.split.calibration_start,
            self.split.calibration_end,
            self.split.test_start,
            self.split.test_end,
        ]
        if ordered != sorted(ordered):
            raise ValueError(
                "dates must be chronological: raw_start <= dataset_start <= "
                "train_end <= val_end <= calibration_start <= calibration_end "
                "<= test_start <= test_end"
            )
        if self.split.test_end > self.raw_end:
            raise ValueError("test_end must not exceed raw_end")
        if not self.cities:
            raise ValueError("at least one city is required")
        if any(c.population <= 0 for c in self.cities):
            raise ValueError("city populations must be positive")
        if self.features.get("weather_strategy", "persistence") not in {
            "persistence",
            "oracle",
            "available_day_ahead",
        }:
            raise ValueError(
                "weather_strategy must be 'persistence', 'oracle' or 'available_day_ahead'"
            )
        if not self.ensemble.get("members"):
            raise ValueError("ensemble.members must not be empty")
        gamma = float(self.uncertainty["adaptive_gamma"])
        if not 0 < gamma < 1:
            raise ValueError("uncertainty.adaptive_gamma must be between 0 and 1")
        if int(self.uncertainty["adaptive_window"]) < 1:
            raise ValueError("uncertainty.adaptive_window must be positive")

    def city_weights(self) -> dict[str, float]:
        total = sum(c.population for c in self.cities)
        return {c.name: c.population / total for c in self.cities}

    def path(self, key: str) -> Path:
        return (self.root / self.paths[key]).resolve()
