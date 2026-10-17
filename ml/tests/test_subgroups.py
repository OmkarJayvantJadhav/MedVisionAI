from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from medvision_ml.evaluation.subgroups import (
    MIN_IMAGES,
    MIN_MALIGNANT,
    age_band,
    group_metrics,
    read_metadata,
    subgroup_report,
)


@pytest.mark.parametrize(
    ("value", "band"),
    [("5", "under 30"), ("30", "30-49"), ("65", "50-69"), ("85", "70 and over"), ("", "unknown")],
)
def test_age_band(value: str, band: str) -> None:
    assert age_band(value) == band


def test_small_groups_report_counts_only() -> None:
    probs = np.array([[0.9, 0.1]] * 3)
    result = group_metrics(probs, [0, 0, 1], [0], 0.5)
    assert result == {
        "n_images": 3,
        "n_malignant": 2,
        "accuracy": None,
        "malignant_recall": None,
        "malignant_precision": None,
    }


def test_group_metrics_on_a_large_group() -> None:
    n = max(MIN_IMAGES, 2 * MIN_MALIGNANT)
    labels = np.array([0] * (n // 2) + [1] * (n - n // 2))
    probs = np.tile([0.2, 0.8], (n, 1))
    probs[: n // 4] = [0.9, 0.1]  # half of the malignant images are flagged
    result = group_metrics(probs, labels, [0], 0.5)
    assert result["malignant_recall"] == pytest.approx(0.5, abs=0.05)
    assert result["malignant_precision"] == 1.0
    assert result["accuracy"] == pytest.approx(0.75, abs=0.05)


def test_report_joins_metadata_and_marks_missing_images_unknown(tmp_path: Path) -> None:
    csv = tmp_path / "meta.csv"
    csv.write_text(
        "image,age_approx,anatom_site_general,lesion_id,sex\na,45,head/neck,,female\nb,,,,\n",
        encoding="utf-8",
    )
    metadata = read_metadata(csv)
    probs = np.array([[0.9, 0.1], [0.1, 0.9], [0.5, 0.5]])
    report = subgroup_report(["a", "b", "c"], probs, [0, 1, 0], metadata, [0], 0.5)
    assert set(report) == {"sex", "age", "site"}
    assert report["sex"]["female"]["n_images"] == 1
    assert report["sex"]["unknown"]["n_images"] == 2
    assert report["age"]["30-49"]["n_images"] == 1
    assert report["site"]["head/neck"]["n_malignant"] == 1
