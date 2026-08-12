from datetime import date
from pathlib import Path

import pytest

from loadfc.config import Config

FIXTURE = Path(__file__).parent / "fixtures" / "config_test.yaml"


def test_loads_dates_and_cities():
    cfg = Config.from_yaml(FIXTURE)
    assert cfg.raw_start == date(2019, 1, 1)
    assert cfg.dataset_start == date(2019, 1, 14)
    assert cfg.split.test_end == date(2019, 12, 31)
    assert [c.name for c in cfg.cities] == ["Berlin", "Hamburg"]


def test_city_weights_normalize_to_one():
    cfg = Config.from_yaml(FIXTURE)
    weights = cfg.city_weights()
    assert pytest.approx(sum(weights.values())) == 1.0
    assert weights["Berlin"] == pytest.approx(0.75)  # 3M of 4M total


def test_path_resolves_relative_to_repo_root():
    cfg = Config.from_yaml(FIXTURE)
    p = cfg.path("raw_dir")
    assert p.name == "raw"
    assert p.is_absolute()


def test_validate_rejects_out_of_order_split(tmp_path):
    import yaml

    raw = yaml.safe_load(FIXTURE.read_text())
    raw["split"]["train_end"] = "2019-11-30"  # after val_end -> invalid
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="chronological"):
        Config.from_yaml(bad)


def test_validate_rejects_dataset_start_before_raw_start(tmp_path):
    import yaml

    raw = yaml.safe_load(FIXTURE.read_text())
    raw["project"]["dataset_start"] = "2018-12-01"  # before raw_start
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="chronological"):
        Config.from_yaml(bad)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("rolling_window", 1, "at least 2"),
        ("mape_warning", 0, "must be positive"),
        ("coverage_floor", 1.0, "between 0 and 1"),
    ],
)
def test_validate_rejects_invalid_monitoring_thresholds(tmp_path, key, value, message):
    import yaml

    raw = yaml.safe_load(FIXTURE.read_text())
    raw["monitoring"] = {
        "rolling_window": 14,
        "mape_warning": 5.0,
        "coverage_floor": 0.85,
        key: value,
    }
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(raw))

    with pytest.raises(ValueError, match=message):
        Config.from_yaml(bad)
