"""Write the OpenAPI schema for the frontend: ``python -m app.openapi [PATH]``.

The frontend generates its TypeScript API types from this file, and a backend
test fails when the committed copy is out of date.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from app.core.settings import REPO_ROOT, Settings
from app.main import create_app

DEFAULT_PATH = REPO_ROOT / "frontend" / "src" / "api" / "openapi.json"


def schema() -> dict[str, Any]:
    """Return the API schema; independent of the environment and of model files."""
    settings = Settings(_env_file=None, model_dir=Path("/nonexistent"))  # type: ignore[call-arg]
    result: dict[str, Any] = create_app(settings).openapi()
    return result


def render() -> str:
    return json.dumps(schema(), indent=2, sort_keys=True) + "\n"


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(), encoding="utf-8")
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
