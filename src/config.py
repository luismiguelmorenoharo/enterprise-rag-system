"""
Application configuration with environment variables.
Production-ready settings management using Pydantic.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache
from pathlib import Path
from typing import Optional


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    Uses pydantic-settings for validation and type safety.
    Critical for production: validates config on startup.
    """
    
    # API Keys
    OPENAI_API_KEY: str
    ANTHROPIC_API_KEY: Optional[str] = None
    PINECONE_API_KEY: str
    
    # Pinecone Config
    PINECONE_ENVIRONMENT: str = "gcp-starter"
    PINECONE_INDEX_NAME: str = "enterprise-rag"
    
    # LLM Settings
    LLM_MODEL: str = "gpt-4o-mini"  # Cost-effective for demo
    EMBEDDING_MODEL: str = "text-embedding-3-large"
    MAX_TOKENS: int = 2000
    TEMPERATURE: float = 0.1
    
    # RAG Config
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    TOP_K_RETRIEVAL: int = 5
    SIMILARITY_THRESHOLD: float = 0.7
    
    # Database & Caching
    REDIS_URL: str = "redis://localhost:6379"
    
    # Paths
    UPLOAD_DIR: Path = Path("data/uploads")
    PROCESSED_DIR: Path = Path("data/processed")
    
    # Model Config
    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Using lru_cache ensures we only load .env once,
    improving performance in production.
    """
    return Settings()


# Convenience function for quick access
def get_config() -> Settings:
    """Alias for get_settings()"""
    return get_settings()
