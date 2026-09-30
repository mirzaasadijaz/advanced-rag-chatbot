"""
Centralized app settings, loaded from environment variables (see .env.example).
Everything here has a sane default except GOOGLE_API_KEY, which is required.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _bool_env(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Settings:
    # --- LLM (Google Gemini) ---
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-3.1-pro")

    # --- Optional web-search tool (agentic fallback) ---
    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")

    # --- Embeddings (local, free, no API key - runs via ONNX/fastembed) ---
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

    # --- Storage ---
    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
    docstore_path: str = os.getenv("DOCSTORE_PATH", "./data/docstore.jsonl")

    # --- Chunking ---
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))

    # --- Retrieval / advanced RAG knobs ---
    retrieval_k: int = int(os.getenv("RETRIEVAL_K", "6"))
    rerank_top_n: int = int(os.getenv("RERANK_TOP_N", "4"))
    max_self_correct_retries: int = int(os.getenv("MAX_SELF_CORRECT_RETRIES", "2"))
    enable_reranker: bool = _bool_env("ENABLE_RERANKER", True)

    # --- Web scraping ---
    scrape_max_pages_default: int = int(os.getenv("SCRAPE_MAX_PAGES_DEFAULT", "5"))
    scrape_max_pages_limit: int = int(os.getenv("SCRAPE_MAX_PAGES_LIMIT", "25"))

    # --- API ---
    allowed_origins: list = field(
        default_factory=lambda: [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",")]
    )

    @property
    def web_search_enabled(self) -> bool:
        return bool(self.tavily_api_key)

    def ensure_dirs(self) -> None:
        Path(self.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
        Path(self.docstore_path).parent.mkdir(parents=True, exist_ok=True)


settings = Settings()