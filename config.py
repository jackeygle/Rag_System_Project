"""
RAG System Configuration
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent
STORAGE_DIR = Path(os.getenv("RAG_STORAGE_DIR", str(BASE_DIR)))
DATA_DIR = STORAGE_DIR / "data" / "documents"
DB_DIR = STORAGE_DIR / "db" / "chroma"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_DIR.mkdir(parents=True, exist_ok=True)

# API Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Model Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")

# LLM Configuration (using Groq for faster response)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

# Text Splitting Configuration
CHUNK_SIZE = 1000  # Characters per chunk
CHUNK_OVERLAP = 200  # Overlap between chunks

# Retrieval Configuration
TOP_K = 4  # Number of documents to retrieve
MIN_RELEVANCE = float(os.getenv("MIN_RELEVANCE", "0.25"))
if not 0 <= MIN_RELEVANCE <= 1:
    raise ValueError("MIN_RELEVANCE must be between 0 and 1")

# Vector Store Configuration
COLLECTION_NAME = "rag_documents"


def validate_api_keys() -> tuple[bool, list[str]]:
    """Validate that all required API keys are configured.
    
    Returns:
        tuple: (is_valid, list of missing key names)
    """
    missing = []
    if not GOOGLE_API_KEY:
        missing.append("GOOGLE_API_KEY")
    if not GROQ_API_KEY:
        missing.append("GROQ_API_KEY")
    return len(missing) == 0, missing


def get_api_key_help() -> str:
    """Return help text for configuring API keys."""
    return """
To configure API keys, create a .env file with:

GOOGLE_API_KEY=your_gemini_api_key
GROQ_API_KEY=your_groq_api_key

Get your keys from:
- Gemini: https://aistudio.google.com/apikey
- Groq: https://console.groq.com/keys
"""

