"""Deploy the app to a Hugging Face Space (Docker SDK).

Usage: HF_TOKEN=... python deploy_space.py OWNER/SPACE [--dry-run DIR]

Copies only what the Docker build needs (Dockerfile, backend, frontend) into a
staging folder with a Space README, creates the Space if it does not exist,
uploads the folder as one commit (removing files that are no longer present)
and prints the Space URL. With --dry-run the staging folder is written to DIR
and nothing is uploaded.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO_URL = "https://github.com/OmkarJayvantJadhav/MedVisionAI"

# The Space runs behind Hugging Face's proxy, so every visitor shares one client
# address; the limit protects the free CPU as a whole (D43).
SPACE_VARIABLES = {"MEDVISION_RATE_LIMIT_PER_MINUTE": "60"}

SPACE_README = """---
title: MedVision AI
emoji: 🩺
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
license: cc-by-nc-4.0
short_description: Explainable skin-lesion screening research demo
---

# MedVision AI

Explainable skin-lesion classification for education and screening research.
**Not a medical diagnosis.** Please consult a qualified dermatologist about any
skin concern.

This Space is deployed automatically from [{repo}]({repo}) (commit `{sha}`).
Source code, model card, test results and licences live there.
"""


def tracked_files(folder: str) -> list[Path]:
    """Files under ``folder`` that git tracks, so local build outputs never ship."""
    output = subprocess.run(
        ["git", "ls-files", folder], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout
    return [ROOT / line for line in output.splitlines() if line]


def stage(destination: Path, sha: str) -> list[Path]:
    """Write the Space contents to ``destination`` and return the staged files."""
    if destination.exists():
        shutil.rmtree(destination)
    files = [ROOT / "Dockerfile", ROOT / ".dockerignore"]
    files += tracked_files("backend") + tracked_files("frontend")
    for source in files:
        target = destination / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    (destination / "README.md").write_text(
        SPACE_README.format(repo=REPO_URL, sha=sha[:12]), encoding="utf-8"
    )
    return files


FAILED_STAGES = {"BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR", "NO_APP_FILE"}


def wait_for_rebuild(api: object, space: str, timeout: float = 1800) -> None:
    """Wait until the Space has rebuilt and runs the new commit.

    Right after an upload the previous build may still answer requests, so the
    Space must be seen building (or starting) before RUNNING counts.
    """
    deadline = time.monotonic() + timeout
    rebuilt = False
    while time.monotonic() < deadline:
        stage = api.get_space_runtime(space).stage  # type: ignore[attr-defined]
        print(f"Space stage: {stage}", flush=True)
        if stage in FAILED_STAGES:
            raise SystemExit(f"The Space failed with stage {stage}; see its build logs.")
        if stage == "RUNNING" and rebuilt:
            return
        rebuilt = rebuilt or stage in {"BUILDING", "APP_STARTING", "RUNNING_BUILDING"}
        time.sleep(15)
    raise SystemExit(f"The Space did not finish rebuilding within {timeout:.0f} s.")


def space_url(space: str) -> str:
    owner, name = space.split("/")
    subdomain = f"{owner}-{name}".lower().replace("_", "-").replace(".", "-")
    return f"https://{subdomain}.hf.space"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("space", help="OWNER/SPACE")
    parser.add_argument("--dry-run", type=Path, default=None, metavar="DIR")
    args = parser.parse_args()
    if args.space.count("/") != 1:
        parser.error("space must look like OWNER/SPACE")

    sha = (
        os.environ.get("GITHUB_SHA")
        or subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    )
    staging = args.dry_run or ROOT / ".space-staging"
    files = stage(staging, sha)
    print(f"Staged {len(files) + 1} files in {staging}")
    if args.dry_run:
        return 0

    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN is not set.", file=sys.stderr)
        return 1
    api = HfApi(token=token)
    api.create_repo(args.space, repo_type="space", space_sdk="docker", exist_ok=True)
    for key, value in SPACE_VARIABLES.items():
        api.add_space_variable(args.space, key, value)
    api.upload_folder(
        repo_id=args.space,
        repo_type="space",
        folder_path=staging,
        commit_message=f"Deploy {sha[:12]}",
        delete_patterns="*",
    )
    wait_for_rebuild(api, args.space)
    url = space_url(args.space)
    print(f"Deployed {sha[:12]} to {url}")
    if output := os.environ.get("GITHUB_OUTPUT"):
        with Path(output).open("a", encoding="utf-8") as handle:
            handle.write(f"url={url}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
