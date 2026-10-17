from __future__ import annotations

import csv
import hashlib
import io
from pathlib import Path

import pytest
from medvision_ml.data import non_skin
from PIL import Image

from tests.conftest import smooth_image


def _row(image_id: str, licence: str = non_skin.LICENCE) -> dict[str, str]:
    return {
        "ImageID": image_id,
        "License": licence,
        "Author": f"Author {image_id}",
        "Title": f"Title, {image_id}",
        "OriginalLandingURL": f"https://example.org/{image_id}",
    }


def test_select_is_deterministic_and_filters_the_licence() -> None:
    rows = [_row(f"id{i}") for i in range(20)] + [_row("other", "https://example.org/nc")]
    chosen = non_skin.select(reversed(rows), 5)
    assert chosen == non_skin.select(rows, 5)
    keys = [hashlib.sha256(r["ImageID"].encode()).hexdigest() for r in chosen]
    assert keys == sorted(keys)
    assert all(r["License"] == non_skin.LICENCE for r in non_skin.select(rows, 100))
    assert len(non_skin.select(rows, 100)) == 20


def test_download_saves_valid_images_with_attribution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [_row("good"), _row("broken"), _row("offline")]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    jpeg = io.BytesIO()
    smooth_image(0).save(jpeg, format="JPEG")

    def fake_fetch(url: str) -> bytes:
        if url == non_skin.METADATA_URL:
            return buffer.getvalue().encode()
        if "offline" in url:
            raise OSError("unreachable")
        return jpeg.getvalue() if "good" in url else b"<Error/>"

    monkeypatch.setattr(non_skin, "_fetch", fake_fetch)
    assert non_skin.download_non_skin(tmp_path, 3) == 1
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ATTRIBUTION.csv", "good.jpg"]
    with Image.open(tmp_path / "good.jpg") as image:
        assert image.format == "JPEG"
    with (tmp_path / "ATTRIBUTION.csv").open(encoding="utf-8") as handle:
        attribution = list(csv.DictReader(handle))
    assert attribution == [
        {
            "image": "good.jpg",
            "author": "Author good",
            "title": "Title, good",
            "source": "https://example.org/good",
            "licence": non_skin.LICENCE,
        }
    ]


def test_download_rejects_a_non_positive_count(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        non_skin.download_non_skin(tmp_path, 0)
