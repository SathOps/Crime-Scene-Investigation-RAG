import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

# Data directories
DATA_DIR = BASE_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"
SAMPLE_DATA_DIR = BASE_DIR / "sample_data"

# Ensure directories exist
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Default RAG Configurations
DEFAULT_CHUNK_SIZE = 1000  # characters / token approximation
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_TOP_K = 5
DEFAULT_VECTOR_STORE_BACKEND = os.getenv("VECTOR_STORE_BACKEND", "chroma").strip().lower()

# LLM & Embedding Settings
DEFAULT_LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # options: ollama, gemini, openai, local
DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL", os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b"))
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
DEFAULT_EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local")  # options: local, gemini, openai

# Ollama Settings
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")

# Helper to avoid treating placeholder template values as valid keys
def _clean_env_key(val: str) -> str:
    cleaned = (val or "").strip()
    if cleaned.lower().startswith("your_") or "here" in cleaned.lower():
        return ""
    return cleaned

def _get_secret_or_env(key: str, default: str = "") -> str:
    # 1. Try environment variable
    val = os.getenv(key, "")
    if not val:
        # 2. Try Streamlit secrets if running on Streamlit Community Cloud
        try:
            import streamlit as st
            if hasattr(st, "secrets") and key in st.secrets:
                val = str(st.secrets[key])
        except Exception:
            pass
    return _clean_env_key(val) or default

# API Keys (support both local .env and Streamlit Community Cloud secrets)
GEMINI_API_KEY = _get_secret_or_env("GEMINI_API_KEY", _get_secret_or_env("GOOGLE_API_KEY", ""))
OPENAI_API_KEY = _get_secret_or_env("OPENAI_API_KEY", "")

# Default LLM Provider: auto-detect best provider based on available keys or local daemon
_explicit_provider = _get_secret_or_env("LLM_PROVIDER", "")
if _explicit_provider:
    DEFAULT_LLM_PROVIDER = _explicit_provider.lower()
elif GEMINI_API_KEY:
    DEFAULT_LLM_PROVIDER = "gemini"
elif OPENAI_API_KEY:
    DEFAULT_LLM_PROVIDER = "openai"
else:
    DEFAULT_LLM_PROVIDER = "ollama"
