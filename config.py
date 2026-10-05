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

# Azure OpenAI deployments (resource endpoint, not a chat/completions URL).
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
AZURE_OPENAI_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "")
# Actual embedding model name is used for token counting / index identity.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MODEL = AZURE_OPENAI_CHAT_DEPLOYMENT or "Azure OpenAI (not configured)"
EMBEDDING_IDENTITY = ["azure-openai", AZURE_OPENAI_ENDPOINT,
                      AZURE_OPENAI_EMBEDDING_DEPLOYMENT, EMBEDDING_MODEL]


def azure_client_settings(deployment):
    """Validate provider configuration before constructing an SDK client."""
    from urllib.parse import urlsplit
    if not AZURE_OPENAI_API_KEY or not AZURE_OPENAI_ENDPOINT or not deployment:
        raise ValueError("Azure OpenAI configuration is missing. Set the API key, resource endpoint and deployment names.")
    url = urlsplit(AZURE_OPENAI_ENDPOINT)
    if (url.scheme != "https" or not url.hostname or url.username or url.password
            or url.path not in ("", "/") or url.query or url.fragment):
        raise ValueError("AZURE_OPENAI_ENDPOINT must be the HTTPS resource root, such as https://your-resource.openai.azure.com.")
    return dict(api_key=AZURE_OPENAI_API_KEY, azure_endpoint=AZURE_OPENAI_ENDPOINT,
                api_version=AZURE_OPENAI_API_VERSION, azure_deployment=deployment,
                max_retries=2)

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
    """Validate required Azure settings; keep the existing UI entrypoint."""
    names = ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT",
             "AZURE_OPENAI_CHAT_DEPLOYMENT", "AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]
    missing = [name for name in names if not globals()[name]]
    return not missing, missing


def get_api_key_help() -> str:
    return """Configure Azure OpenAI in .env or your hosting provider's secrets/settings:
AZURE_OPENAI_API_KEY=your_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_CHAT_DEPLOYMENT=your_chat_deployment_name
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=your_embedding_deployment_name
AZURE_OPENAI_API_VERSION=2024-10-21
EMBEDDING_MODEL=text-embedding-3-small
Use deployment names from Azure, not just model catalog names.
"""
