"""
RAG Configuration Settings
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Get RAG folder path (where this config file is located)
RAG_FOLDER = Path(__file__).parent.absolute()

# Model Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")

# Backward compatibility for older environment files
if LLM_MODEL == "mixtral-8x7b-32768":
    LLM_MODEL = "llama-3.1-8b-instant"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Vector Store Configuration
VECTOR_STORE_TYPE = os.getenv("VECTOR_STORE_TYPE", "faiss")  # faiss or chroma
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", str(RAG_FOLDER / "vector_store"))
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", str(RAG_FOLDER / "chroma_db"))

# PDF Configuration
PDF_UPLOAD_FOLDER = os.getenv("PDF_UPLOAD_FOLDER", str(RAG_FOLDER / "uploaded_pdfs"))
USER_UPLOAD_FOLDER = os.getenv("USER_UPLOAD_FOLDER", str(RAG_FOLDER / "user_uploads"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# Retrieval Configuration
TOP_K = int(os.getenv("TOP_K", "3"))  # Number of documents to retrieve
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.5"))

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
