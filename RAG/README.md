# 🤖 AI Research Assistant with RAG

A complete Retrieval Augmented Generation (RAG) system for building intelligent research assistants that can answer questions based on uploaded PDF documents with grounded, factual answers.

## 🎯 Features

- **📄 PDF Processing**: Upload and automatically process research papers
- **🔍 Smart Retrieval**: Use FAISS or Chroma vector databases for fast document retrieval
- **🧠 Intelligent Answers**: Leverage Groq LLM with retrieved context for accurate responses
- **⚡ Fast & Scalable**: Efficient chunking and embedding strategies
- **🎨 Web Interface**: Streamlit app for easy interaction
- **🔄 Reload & Update**: Add new documents dynamically to the knowledge base

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        User Query                           │
└────────────────────────────┬────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│              Embedding Model (HuggingFace)                  │
│        sentence-transformers/all-MiniLM-L6-v2              │
└────────────────────────────┬────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│          Vector Database (FAISS / Chroma)                   │
│         Semantic Search & Similarity Matching               │
└────────────────────────────┬────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│           Relevant Documents Retrieved (Top-K)              │
│                  (Context Formatting)                       │
└────────────────────────────┬────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│      LLM with Retrieved Context (Groq - Mixtral)            │
│           (Answer Generation from Context)                  │
└────────────────────────────┬────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│              Final Grounded Answer                          │
│    (Not Hallucinated - Based on Retrieved Documents)        │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
RAG/
├── config.py                 # Configuration settings
├── pdf_loader.py            # PDF processing and chunking
├── vector_store.py          # FAISS/Chroma integration
├── retrieval.py             # Document retrieval logic
├── rag_pipeline.py          # Main RAG orchestrator
├── rag_app.py              # Streamlit web interface
├── test_rag.py             # Testing and demo script
├── __init__.py             # Package initialization
└── README.md               # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Setup Environment Variables

Create a `.env` file in your project root:

```env
# Groq API
GROQ_API_KEY=your_groq_api_key_here

# Model Configuration
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
LLM_MODEL=mixtral-8x7b-32768
VECTOR_STORE_TYPE=faiss

# Document Processing
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K=3

# LLM Settings
LLM_TEMPERATURE=0.7
MAX_TOKENS=1000
```

### 3. Add PDF Documents

Place your research papers in the `uploaded_pdfs/` folder (created automatically).

### 4. Run the System

#### Option A: Interactive Session
```bash
cd RAG
python test_rag.py interactive
```

#### Option B: Streamlit Web App
```bash
cd RAG
streamlit run rag_app.py
```

#### Option C: Batch Processing
```bash
cd RAG
python test_rag.py batch
```

## 📚 Components Overview

### 1. **config.py** - Configuration Management
Centralized configuration for all RAG components:
- Embedding model selection
- Vector store type (FAISS/Chroma)
- Chunk size and overlap settings
- LLM parameters
- Retrieval settings

```python
from RAG.config import EMBEDDING_MODEL, LLM_MODEL, TOP_K
```

### 2. **pdf_loader.py** - PDF Processing
Handles PDF loading, text extraction, and chunking:

```python
from RAG.pdf_loader import PDFLoader

loader = PDFLoader(chunk_size=1000, chunk_overlap=200)
documents = loader.load_multiple_pdfs("./uploaded_pdfs")
```

**Features:**
- Extract text from multiple pages
- Intelligent text chunking with overlap
- Preserve source metadata
- Handle multiple PDFs

### 3. **vector_store.py** - Vector Database Management
Seamless integration with FAISS or Chroma:

```python
from RAG.vector_store import VectorStoreManager

vs = VectorStoreManager(store_type="faiss")
vs.create_vector_store(documents)
vs.save_faiss_index()
results = vs.search("your query", top_k=3)
```

**Capabilities:**
- Create/load vector stores
- Add new documents dynamically
- Semantic search with similarity scores
- Support for both FAISS and Chroma

### 4. **retrieval.py** - Smart Document Retrieval
Advanced retrieval with filtering and ranking:

```python
from RAG.retrieval import Retriever

retriever = Retriever(vector_store, top_k=3)
relevant_docs = retriever.retrieve("your question")
formatted_context = retriever.format_retrieved_documents(relevant_docs)
```

**Features:**
- Relevance filtering
- Score-based ranking
- Context formatting for LLM
- Retrieval statistics

### 5. **rag_pipeline.py** - Main Orchestrator
Complete RAG pipeline end-to-end:

```python
from RAG.rag_pipeline import RAGPipeline

rag = RAGPipeline(vector_store_type="faiss")
rag.ingest_documents()
answer = rag.query("What are the main findings?")
```

**Operations:**
- Document ingestion
- Query processing
- Interactive sessions
- Batch processing
- Statistics reporting

### 6. **rag_app.py** - Streamlit Interface
User-friendly web interface:

```bash
streamlit run RAG/rag_app.py
```

**Features:**
- PDF upload and management
- Interactive Q&A interface
- Document retrieval visualization
- Pipeline statistics
- Example queries

## 💻 Usage Examples

### Example 1: Simple Query
```python
from RAG.rag_pipeline import RAGPipeline

rag = RAGPipeline()
rag.ingest_documents()

answer = rag.query("What is the methodology used in this research?")
print(answer)
```

### Example 2: Batch Processing
```python
rag = RAGPipeline()
rag.ingest_documents()

questions = [
    "What are the main conclusions?",
    "Who are the authors?",
    "What datasets were used?"
]

results = rag.batch_query(questions)
for q, a in results:
    print(f"Q: {q}\nA: {a}\n")
```

### Example 3: Interactive Session
```python
from RAG.rag_pipeline import RAGPipeline

rag = RAGPipeline()
rag.ingest_documents()
rag.interactive_session()
```

### Example 4: Custom Retrieval
```python
from RAG.pdf_loader import PDFLoader
from RAG.vector_store import VectorStoreManager
from RAG.retrieval import Retriever

# Load documents
loader = PDFLoader()
docs = loader.load_multiple_pdfs()

# Create vector store
vs = VectorStoreManager()
vs.create_vector_store(docs)

# Retrieve with custom settings
retriever = Retriever(vs, top_k=5)
relevant_docs = retriever.retrieve("custom query")

for doc in relevant_docs:
    print(f"From {doc.metadata['filename']}:")
    print(doc.page_content[:200])
```

## 🔧 Configuration Examples

### FAISS Configuration (Default)
```env
VECTOR_STORE_TYPE=faiss
VECTOR_STORE_PATH=./vector_store
```

### Chroma Configuration
```env
VECTOR_STORE_TYPE=chroma
CHROMA_PERSIST_DIR=./chroma_db
```

### Custom Chunking
```env
CHUNK_SIZE=2000        # Larger chunks for context
CHUNK_OVERLAP=400      # More overlap for continuity
```

## 📊 Performance Tips

1. **Chunk Size**: Larger chunks (1500-2000) for better context, smaller (500-1000) for speed
2. **Top-K**: Use 3-5 for balance between relevance and context size
3. **Temperature**: 0.3-0.5 for factual answers, 0.7-0.9 for creative responses
4. **Model Selection**: Mixtral 8x7b for best quality, faster models for speed

## 🐛 Troubleshooting

### Issue: "No documents found"
- **Solution**: Ensure PDFs are in `uploaded_pdfs/` folder and run reload

### Issue: FAISS index errors
- **Solution**: Delete `vector_store/` folder and recreate index

### Issue: Memory issues with large PDFs
- **Solution**: Reduce `CHUNK_SIZE` or use Chroma (more memory efficient)

### Issue: Poor answer quality
- **Solution**: Increase `TOP_K`, adjust `CHUNK_OVERLAP`, or use temperature 0.5

## 📈 Roadmap

- [ ] Multi-language support
- [ ] Citation tracking
- [ ] Hybrid search (semantic + keyword)
- [ ] Answer ranking and confidence scores
- [ ] Document summarization
- [ ] Web scraping for documents
- [ ] Fine-tuned retrieval models
- [ ] Query expansion and reformulation

## 📝 License

MIT License

## 🤝 Contributing

Contributions welcome! Please feel free to submit pull requests.

## 📞 Support

For issues and questions, please contact the development team.

---

**Built with ❤️ using LangChain, Groq, and HuggingFace**
