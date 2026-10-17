from __future__ import annotations

from pathlib import Path

import pytest
from medvision_ml.config import DEFAULT_CONFIG_PATH, ConfigError, load_config


def write(tmp_path: Path, text: str) -> Path:
    configs = tmp_path / "configs"
    configs.mkdir(parents=True)
    path = configs / "test.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_default_config_loads_with_expected_values() -> None:
    config = load_config(DEFAULT_CONFIG_PATH, environ={})
    assert config.model.backbone == "efficientnetb0"
    assert config.data.image_size == 224
    assert config.data.split_ratios == (0.8, 0.1, 0.1)
    assert config.training.class_weighting == "sqrt"
    assert "Melanoma" in config.evaluation.malignant_classes


def test_relative_paths_resolve_against_config_root(tmp_path: Path) -> None:
    config = load_config(write(tmp_path, "paths:\n  data_dir: data/x\n"), environ={})
    assert config.data_dir == tmp_path / "data" / "x"
    assert config.output_dir == tmp_path / "outputs"


def test_absolute_paths_are_kept(tmp_path: Path) -> None:
    config = load_config(write(tmp_path, f"paths:\n  data_dir: {tmp_path / 'abs'}\n"), environ={})
    assert config.data_dir == tmp_path / "abs"


def test_environment_overrides_are_typed(tmp_path: Path) -> None:
    env = {
        "MEDVISION__TRAINING__EPOCHS": "3",
        "MEDVISION__TRAINING__MIXED_PRECISION": "false",
        "MEDVISION__MODEL__BACKBONE": "MobileNetV2",
        "MEDVISION__SEED": "7",
        "MEDVISION__DATA__CACHE_DIR": "/tmp/cache",
        "UNRELATED": "ignored",
    }
    config = load_config(write(tmp_path, "training:\n  epochs: 50\n"), environ=env)
    assert config.training.epochs == 3
    assert config.training.mixed_precision is False
    assert config.model.backbone == "mobilenetv2"
    assert config.seed == 7
    assert config.data.cache_dir == Path("/tmp/cache")
    assert load_config(write(tmp_path / "x", "{}\n"), environ={}).data.cache_dir is None
    relative = load_config(write(tmp_path / "y", "data:\n  cache_dir: c\n"), environ={})
    assert relative.data.cache_dir == tmp_path / "y" / "c"


@pytest.mark.parametrize(
    "text",
    [
        "training:\n  epochs: 0\n",
        "model:\n  backbone: vgg16\n",
        "model:\n  dropout: 1.5\n",
        "data:\n  split_ratios: [0.5, 0.2, 0.2]\n",
        "data:\n  batch_size: 2.5\n",
        "data:\n  cache: maybe\n",
        "unknown_section: {}\n",
        "training:\n  epochs_typo: 3\n",
        "- not a mapping\n",
        "training: [1, 2]\n",
        "evaluation:\n  target_in_distribution_acceptance: 0\n",
        "evaluation:\n  ood_shrinkage: 1.5\n",
    ],
)
def test_invalid_configs_are_rejected(tmp_path: Path, text: str) -> None:
    with pytest.raises(ConfigError):
        load_config(write(tmp_path, text), environ={})


def test_missing_file_raises_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_config(tmp_path / "missing.yaml", environ={})


def test_loading_has_no_filesystem_side_effects(tmp_path: Path) -> None:
    path = write(tmp_path, "{}\n")
    before = sorted(tmp_path.rglob("*"))
    load_config(path, environ={})
    assert sorted(tmp_path.rglob("*")) == before
