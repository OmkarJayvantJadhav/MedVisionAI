"""Service configuration from environment variables (prefix ``MEDVISION_``)."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """All runtime settings; every field can be set as ``MEDVISION_<FIELD>``."""

    model_config = SettingsConfigDict(env_prefix="MEDVISION_", env_file=".env", extra="ignore")

    model_dir: Path = REPO_ROOT / "models"
    verify_checksums: bool = True
    static_dir: Path | None = Field(
        default=None, description="Built frontend to serve at '/', if present."
    )
    allowed_origins: list[str] = Field(default_factory=list, description="CORS origins.")
    max_upload_mb: float = Field(default=8.0, gt=0, le=32)
    max_image_side: int = Field(default=4096, ge=256)
    rate_limit_per_minute: int = Field(default=30, ge=0, description="0 disables the limit.")
    trust_forwarded_for: bool = Field(
        default=False, description="Use X-Forwarded-For for client IPs (behind a proxy)."
    )
    log_level: str = "INFO"

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)
