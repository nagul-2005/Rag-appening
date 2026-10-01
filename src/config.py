"""
config.py
---------
Central configuration for the RAG application.
All settings are read from the .env file using pydantic-settings.
This means you never hardcode API keys or paths in the source code.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# The project root is two levels up from this file (src/config.py → root)
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    """
    Application settings loaded automatically from the .env file.
    Any variable defined here can be overridden by setting an environment variable.
    """

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",        # Ignore unknown keys in .env
    )

    # --- PDF Source ---
    PDF_URL: str      = "https://konverge.ai/pdf/Ebook-Agentic-AI.pdf"
    PDF_FILENAME: str = "Ebook-Agentic-AI.pdf"

    @property
    def PDF_PATH(self) -> Path:
        return DATA_DIR / self.PDF_FILENAME

    # --- Vector Database ---
    COLLECTION_NAME: str  = "agentic_ai_ebook"
    CHROMA_PERSIST_DIR: Path = DATA_DIR / "chroma_db"

    # --- Embedding Model (Google) ---
    EMBEDDING_MODEL: str = "gemini-embedding-001"

    # --- LLM ---
    LLM_MODEL: str   = "gemini-2.5-flash"
    TEMPERATURE: float = 0.0

    # --- Retrieval ---
    CHUNK_SIZE: int    = 800    # Characters per chunk
    CHUNK_OVERLAP: int = 150    # Characters shared between adjacent chunks
    TOP_K: int         = 4      # Number of chunks to retrieve per query
    MIN_RELEVANCE: float = 0.25 # Chunks below this score are filtered out

    # --- API Keys ---
    GOOGLE_API_KEY: str | None = None

    # --- Server ---
    FASTAPI_HOST: str = "0.0.0.0"
    FASTAPI_PORT: int = 8000
    BACKEND_URL: str  = "http://localhost:8000"


# Single shared settings instance used everywhere in the project
settings = Settings()
