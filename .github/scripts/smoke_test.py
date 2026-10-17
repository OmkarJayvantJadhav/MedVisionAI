"""Smoke-test a running MedVision AI deployment (a local container or the live Space).

Usage: python smoke_test.py BASE_URL IMAGE [--wait SECONDS]

Waits for /api/v1/ready, checks the home page, sends IMAGE to /api/v1/predict
and /api/v1/explain, checks that a generated non-skin image is rejected when the
model ships an out-of-distribution gate, checks that a generated grainy image is
rejected when it ships a noise check, and reports p95 /predict latency over
20 warm requests. Exits non-zero on any failure. Uses only the standard library.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import struct
import sys
import time
import urllib.error
import urllib.request
import uuid
import zlib
from collections.abc import Callable
from pathlib import Path

LATENCY_BUDGET_MS = 150  # plan §11: p95 /predict, warning only


def get(url: str, timeout: float = 10) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def png(size: int, pixel: Callable[[int, int], tuple[int, ...]]) -> bytes:
    """Encode an RGB image given as ``pixel(x, y) -> (r, g, b)``."""
    rows = b"".join(
        b"\x00" + bytes(c for x in range(size) for c in pixel(x, y)) for y in range(size)
    )

    def chunk(kind: bytes, data: bytes) -> bytes:
        crc = struct.pack(">I", zlib.crc32(kind + data))
        return struct.pack(">I", len(data)) + kind + data + crc

    header = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)  # 8-bit RGB
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


def stripes_png(size: int = 256) -> bytes:
    """Coloured diagonal stripes: an image nothing like a dermoscopy photo."""
    palette = [(230, 60, 40), (40, 90, 220), (250, 220, 40), (30, 160, 80)]
    return png(size, lambda x, y: palette[((x + y) // 16) % 4])


def grainy_png(size: int = 256) -> bytes:
    """A smooth skin-coloured gradient with heavy sensor-like noise."""
    rng = random.Random(0)

    def pixel(x: int, y: int) -> tuple[int, ...]:
        grain = rng.gauss(0, 30)
        base = (200 - y // 4, 150 - x // 6, 130 - x // 8)
        return tuple(max(0, min(255, round(c + grain))) for c in base)

    return png(size, pixel)


def post_image(url: str, image: Path | bytes, timeout: float = 60) -> dict:
    boundary = uuid.uuid4().hex
    if isinstance(image, Path):
        name, kind, data = image.name, "image/jpeg", image.read_bytes()
    else:
        name, kind, data = "generated.png", "image/png", image
    body = (
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
            f'filename="{name}"\r\nContent-Type: {kind}\r\n\r\n'
        ).encode()
        + data
        + f"\r\n--{boundary}--\r\n".encode()
    )
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def wait_until_ready(base: str, seconds: float) -> dict:
    deadline = time.monotonic() + seconds
    last = "no response"
    while time.monotonic() < deadline:
        try:
            status, body = get(f"{base}/api/v1/ready")
            if status == 200:
                return json.loads(body)
            last = f"HTTP {status}"
        except (urllib.error.URLError, OSError) as error:
            last = str(error)
        time.sleep(10)
    raise SystemExit(f"Not ready after {seconds:.0f} s ({last}).")


def check(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"FAIL: {message}")
    print(f"ok   {message}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("base_url")
    parser.add_argument("image", type=Path)
    parser.add_argument("--wait", type=float, default=120, help="seconds to wait for /ready")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    ready = wait_until_ready(base, args.wait)
    check(ready.get("ready") is True, f"ready, model {ready.get('model_version')}")

    status, page = get(f"{base}/")
    check(status == 200 and b'<div id="root">' in page, "home page served")

    prediction = post_image(f"{base}/api/v1/predict", args.image)
    check(prediction["status"] in {"ok", "low_confidence", "rejected"}, "predict returns a status")
    check(bool(prediction.get("disclaimer")), "predict includes the disclaimer")
    check(prediction["calibrated"] and prediction["thresholds_fitted"], "model is calibrated")
    print(
        f"     {prediction['predicted_class']} {prediction['confidence']:.2f}, "
        f"{prediction['risk_flag']}"
    )

    explanation = post_image(f"{base}/api/v1/explain", args.image)
    check(len(explanation.get("heatmap_png", "")) > 100, "explain returns a heatmap")

    _, meta = get(f"{base}/api/v1/meta")
    if json.loads(meta)["thresholds"].get("ood_mahalanobis") is not None:
        unrelated = post_image(f"{base}/api/v1/predict", stripes_png())
        check(
            unrelated["status"] == "rejected" and "out_of_distribution" in unrelated["reasons"],
            "non-skin image rejected as out of distribution",
        )
    else:
        print("     no out-of-distribution gate in this model")
    if json.loads(meta)["thresholds"].get("max_noise") is not None:
        grainy = post_image(f"{base}/api/v1/predict", grainy_png())
        check(
            grainy["status"] == "rejected" and "too_noisy" in grainy["reasons"],
            "grainy image rejected as too noisy",
        )
    else:
        print("     no noise check in this model")

    timings = []
    for _ in range(20):
        start = time.perf_counter()
        post_image(f"{base}/api/v1/predict", args.image)
        timings.append((time.perf_counter() - start) * 1000)
    p95 = statistics.quantiles(timings, n=20)[18]
    print(f"     /predict p95 {p95:.0f} ms over {len(timings)} requests")
    if p95 > LATENCY_BUDGET_MS:
        print(f"::warning::/predict p95 {p95:.0f} ms is above the {LATENCY_BUDGET_MS} ms budget")


if __name__ == "__main__":
    try:
        main()
    except (urllib.error.URLError, KeyError, json.JSONDecodeError) as error:
        sys.exit(f"FAIL: {error}")
