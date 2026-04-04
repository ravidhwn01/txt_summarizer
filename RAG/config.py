"""
RAG Configuration Settings
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Get RAG folder path (where this config file is located)
RAG_FOLDER = Path(__file__).parent.absolute()


def _resolve_path(env_value: str, default_path: Path) -> str:
    """Resolve relative paths against the RAG folder for consistent file handling."""
    path = Path(env_value) if env_value else default_path
    if not path.is_absolute():
        path = RAG_FOLDER / path
    return str(path.resolve())

# Model Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
VISION_MODEL = os.getenv("VISION_MODEL", "gemini-1.5-flash")
IMAGE_CAPTION_MODEL = os.getenv("IMAGE_CAPTION_MODEL", "Salesforce/blip-image-captioning-base")
ENABLE_LOCAL_IMAGE_CAPTIONING = os.getenv("ENABLE_LOCAL_IMAGE_CAPTIONING", "false").lower() in {"1", "true", "yes", "on"}

# Backward compatibility for older environment files
if LLM_MODEL == "mixtral-8x7b-32768":
    LLM_MODEL = "llama-3.1-8b-instant"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Vector Store Configuration
VECTOR_STORE_TYPE = os.getenv("VECTOR_STORE_TYPE", "faiss")  # faiss or chroma
VECTOR_STORE_PATH = _resolve_path(os.getenv("VECTOR_STORE_PATH", "vector_store"), RAG_FOLDER / "vector_store")
CHROMA_PERSIST_DIR = _resolve_path(os.getenv("CHROMA_PERSIST_DIR", "chroma_db"), RAG_FOLDER / "chroma_db")

# PDF Configuration
PDF_UPLOAD_FOLDER = _resolve_path(os.getenv("PDF_UPLOAD_FOLDER", "uploaded_pdfs"), RAG_FOLDER / "uploaded_pdfs")
USER_UPLOAD_FOLDER = _resolve_path(os.getenv("USER_UPLOAD_FOLDER", "user_uploads"), RAG_FOLDER / "user_uploads")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# Retrieval Configuration
TOP_K = int(os.getenv("TOP_K", "3"))  # Number of documents to retrieve
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.5"))

# Conversation Memory Configuration
CONVERSATION_MEMORY_TURNS = int(os.getenv("CONVERSATION_MEMORY_TURNS", "6"))

# LLM Configuration
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1000"))

# Create necessary directories
try:
    os.makedirs(PDF_UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(USER_UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
except Exception as e:
    print(f"⚠ Warning creating directories: {e}")
