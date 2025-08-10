from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables.

    Uses pydantic-settings to support .env files and environment overrides.
    """

    app_name: str = Field(default="Participatory Governance & Cultural Heritage Agent")
    environment: str = Field(default="development")

    # Directories
    base_dir: Path = Field(default=Path("."))
    log_dir: Path = Field(default=Path("logs"))
    static_dir: Path = Field(default=Path("static"))
    maps_dir: Path = Field(default=Path("static/maps"))
    data_dir: Path = Field(default=Path("data"))
    models_dir: Path = Field(default=Path("models"))

    # Hugging Face / Models
    hf_model_name: str = Field(
        default="google/flan-t5-small", validation_alias="HF_MODEL_NAME"
    )
    sentiment_model_name: str = Field(
        default="cardiffnlp/twitter-roberta-base-sentiment-latest",
        validation_alias="SENTIMENT_MODEL_NAME",
    )
    huggingface_token: Optional[str] = Field(default=None, validation_alias="HUGGINGFACE_TOKEN")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    def ensure_directories(self) -> None:
        for directory in [self.log_dir, self.static_dir, self.maps_dir, self.data_dir, self.models_dir]:
            Path(directory).mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_directories()


