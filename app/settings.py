from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    HF_EMBED_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    HF_TEXT_GEN_MODEL: str = "google/flan-t5-base"  # small CPU-friendly model
    INDEX_DIR: str = str(Path(__file__).resolve().parents[1] / "indexes")
    DATA_DIR: str  = str(Path(__file__).resolve().parents[1] / "data")
    TOP_K: int = 4

settings = Settings()
