import os
from functools import lru_cache
from typing import List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Application
    app_name: str = "Enterprise RAG"
    app_version: str = "0.1.0"
    debug: bool = False

    # Security
    api_key: Optional[str] = None
    api_key_header: str = "X-API-Key"
    cors_origins: str = "*"

    # Rate Limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 60

    # Qdrant
    qdrant_url: Optional[str] = None
    qdrant_host: str = "qdrant"
    qdrant_port: int = 6333
    qdrant_api_key: Optional[str] = os.getenv("QDRANT_API_KEY", "")
    qdrant_collection_name: str = "documents"
    qdrant_timeout: float = 30.0

    # LLM Provider Configuration
    llm_provider: str = os.getenv("LLM_PROVIDER", "groq")  # openai, anthropic, google, groq,
    llm_model: Optional[str] = None  # Uses provider default if not set
    llm_temperature: float = 0.3

    # Provider API Keys
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    google_api_key: Optional[str] = os.getenv("GOOGLE_API_KEY", "")
    groq_api_key: Optional[str] = os.getenv("GROQ_API_KEY", "")

    # Embedding
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # File upload
    max_file_size: int = 50 * 1024 * 1024  # 50MB

    def get_qdrant_url(self) -> str:
        """Get Qdrant URL - prioritize cloud URL if set."""
        if self.qdrant_url:
            return self.qdrant_url
        return f"http://{self.qdrant_host}:{self.qdrant_port}"

    @property
    def is_qdrant_cloud(self) -> bool:
        """Check if using Qdrant Cloud."""
        return self.qdrant_url is not None and self.qdrant_api_key is not None

    @property
    def cors_origins_list(self) -> List[str]:
        """Parse CORS origins from comma-separated string."""
        if self.cors_origins == "*":
            return ["*"]
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
