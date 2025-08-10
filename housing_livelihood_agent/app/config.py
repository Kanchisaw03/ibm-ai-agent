from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    """
    Application settings loaded from the .env file.
    
    Attributes:
        API_KEY: The secret key for API authentication.
        HF_TOKEN: The Hugging Face Hub token for write access.
        HF_DATASET_REPO: The repository ID of the Hugging Face Dataset.
    """
    API_KEY: str
    HF_TOKEN: str
    HF_DATASET_REPO: str

    class Config:
        # Pydantic V2+ uses model_config instead of class Config
        # However, for BaseSettings, class Config is still used for env config.
        env_file = ".env"
        env_file_encoding = 'utf-8'

# Create a single, reusable instance of the settings
settings = Settings()
