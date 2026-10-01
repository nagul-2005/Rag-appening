import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# Base directory paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Document source
    PDF_URL: str = "https://konverge.ai/pdf/Ebook-Agentic-AI.pdf"
    PDF_FILENAME: str = "Ebook-Agentic-AI.pdf"

    @property
    def PDF_PATH(self) -> Path:
        return DATA_DIR / self.PDF_FILENAME

    # Chunking settings
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 150

    # Vector store
    COLLECTION_NAME: str = "agentic_ai_ebook"
    CHROMA_DIR: Path = DATA_DIR / "chroma_db"

    # Models
    EMBEDDING_MODEL: str = "gemini-embedding-001"
    LLM_MODEL: str = "gemini-2.5-flash"
    TEMPERATURE: float = 0.0

    # Retrieval settings
    TOP_K: int = 4
    MIN_SCORE: float = 0.25

    # API Keys
    GOOGLE_API_KEY: str | None = None
    OPENAI_API_KEY: str | None = None

    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    BACKEND_URL: str = "http://localhost:8000"


settings = Settings()
