"""Settings loaded from environment variables (12-factor)."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ANTIFRAUDE_", env_file=".env", extra="ignore")

    database_url: str = "sqlite:///./antifraude.db"
    model_dir: Path = Path("models")
    # Scores >= block_threshold are blocked; >= review_threshold go to manual review.
    review_threshold: float = 0.5
    block_threshold: float = 0.85
    # Promotion gate used by the retraining pipeline.
    min_roc_auc: float = 0.85
    min_recall: float = 0.6


@lru_cache
def get_settings() -> Settings:
    return Settings()
