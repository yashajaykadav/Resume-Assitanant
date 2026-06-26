"""
app/config.py
Centralised settings — loaded once at startup from .env
"""
from pydantic_settings import BaseSettings
from functools import lru_cache
import os
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    groq_api_key: str = os.getenv("GROQ_API_KEY")
    embedding_model: str = os.getenv("EMBEDDING_MODEL")
    model_name: str = os.getenv("MODEL_NAME")
    chroma_path: str = os.getenv("CHROMA_PATH")
    upload_dir: str = os.getenv("UPLOAD_DIR")
    app_env: str = os.getenv("APP_ENV")
    max_upload_size_mb: int = os.getenv("MAX_UPLOAD_SIZE_MB")
    chunk_size: int = os.getenv("CHUNK_SIZE")
    chunk_overlap: int = os.getenv("CHUNK_OVERLAP")
    top_k_results: int = os.getenv("TOP_K_RESULTS")

    class Config:
        env_file = ".env"
        extra = "ignore"
        protected_namespaces = ("settings_",)


@lru_cache()
def get_settings() -> Settings:
    return Settings()
