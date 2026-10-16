from __future__ import annotations

from app.openapi import DEFAULT_PATH, render


def test_frontend_schema_is_up_to_date() -> None:
    assert DEFAULT_PATH.is_file(), "Run `make api-schema` to generate the frontend schema."
    assert DEFAULT_PATH.read_text(encoding="utf-8") == render(), (
        "The API changed: run `make api-schema` and commit the result."
    )
