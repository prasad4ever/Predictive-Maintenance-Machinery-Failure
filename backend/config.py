"""Runtime settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE) if ENV_FILE.exists() else None,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    plant_name: str = "Coastal Refinery Unit 2"
    app_host: str = "127.0.0.1"
    app_port: int = 8000
    model_path: Path = Path("models/best.joblib")
    telemetry_path: Path = Path("data/telemetry.csv")
    comparison_path: Path = Path("models/comparison.json")
    replay_interval_seconds: float = 0.8

    def resolve(self, path: Path) -> Path:
        return path if path.is_absolute() else ROOT / path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
