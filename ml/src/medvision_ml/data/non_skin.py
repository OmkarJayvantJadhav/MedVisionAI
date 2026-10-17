"""Download a reproducible set of ordinary photos for the out-of-distribution report.

The photos come from the Open Images V4 validation set: every image there is
listed under CC BY 2.0 with its author and source page. The selection is the
first ``count`` image IDs ordered by SHA-256, so it is the same on every
machine. An ``ATTRIBUTION.csv`` with the author, title, source and licence of
each image is written next to them.
"""

from __future__ import annotations

import csv
import hashlib
import io
import logging
import urllib.request
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

LOGGER = logging.getLogger(__name__)

METADATA_URL = (
    "https://storage.googleapis.com/openimages/2018_04/validation/"
    "validation-images-with-rotation.csv"
)
IMAGE_URL = "https://open-images-dataset.s3.amazonaws.com/validation/{image_id}.jpg"
LICENCE = "https://creativecommons.org/licenses/by/2.0/"
ATTRIBUTION_FILE = "ATTRIBUTION.csv"
ATTRIBUTION_FIELDS = ("image", "author", "title", "source", "licence")
TIMEOUT_SECONDS = 60


def select(rows: Iterable[dict[str, str]], count: int) -> list[dict[str, str]]:
    """Return ``count`` CC BY 2.0 rows, ordered by the SHA-256 of their image ID."""
    eligible = [row for row in rows if row.get("License") == LICENCE]
    eligible.sort(key=lambda row: hashlib.sha256(row["ImageID"].encode()).hexdigest())
    return eligible[:count]


def _fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=TIMEOUT_SECONDS) as response:
        data: bytes = response.read()
        return data


def _is_jpeg(data: bytes) -> bool:
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
            return image.format == "JPEG"
    except (OSError, SyntaxError):
        return False


def write_attribution(dest: Path, rows: Sequence[dict[str, str]]) -> Path:
    """Write ``ATTRIBUTION.csv`` for the downloaded images in ``rows``."""
    path = dest / ATTRIBUTION_FILE
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(ATTRIBUTION_FIELDS)
        for row in rows:
            writer.writerow(
                (
                    f"{row['ImageID']}.jpg",
                    row["Author"],
                    row["Title"],
                    row["OriginalLandingURL"],
                    row["License"],
                )
            )
    return path


def download_non_skin(dest: Path, count: int, *, workers: int = 16) -> int:
    """Download ``count`` selected photos into ``dest`` and return how many were saved.

    Files that already exist are kept; images that fail to download or decode
    as JPEG are skipped and logged.

    Raises:
        ValueError: If ``count`` is not positive.
    """
    if count < 1:
        raise ValueError("count must be positive.")
    dest.mkdir(parents=True, exist_ok=True)
    text = _fetch(METADATA_URL).decode("utf-8")
    rows = select(csv.DictReader(io.StringIO(text)), count)

    def download(row: dict[str, str]) -> dict[str, str] | None:
        path = dest / f"{row['ImageID']}.jpg"
        if path.is_file():
            return row
        try:
            data = _fetch(IMAGE_URL.format(image_id=row["ImageID"]))
        except OSError as error:
            LOGGER.warning("Skipping %s: %s", row["ImageID"], error)
            return None
        if not _is_jpeg(data):
            LOGGER.warning("Skipping %s: not a JPEG image", row["ImageID"])
            return None
        path.write_bytes(data)
        return row

    with ThreadPoolExecutor(max_workers=workers) as pool:
        saved = [row for row in pool.map(download, rows) if row is not None]
    write_attribution(dest, saved)
    LOGGER.info("Saved %d of %d photos to %s", len(saved), len(rows), dest)
    return len(saved)
