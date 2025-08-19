from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # External source API base (e.g., https://api.example.com/programs)
    API_BASE: Optional[str] = None

    # Embeddings model + key
    OPENAI_API_KEY: str = Field(..., description="OpenAI API key (no default)")
    OPENAI_EMBED_MODEL: str = "text-embedding-3-small"

    # Storage
    DATA_DIR: str = "data"
    MAX_ITEMS: int = 5000

    # CORS
    CORS_ORIGINS: str = "*"  # Adjust in production

    # Jobs
    REDIS_URL: str = "redis://redis:6379/0"

settings = Settings()
