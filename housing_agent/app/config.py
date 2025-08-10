from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv


load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    Attributes:
        API_KEY: Secret key required in `X-API-Key` header.
        MODEL_DIR: Optional local directory for ML artifacts.
    """

    MODEL_DIR: str | None = None

    # Pydantic v2 settings config
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()


