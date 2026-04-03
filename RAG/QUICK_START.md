# ⚡ RAG System - Quick Start Guide

## 5-Minute Setup

### Step 1: Install Dependencies (2 min)
```bash
cd c:\Users\Ravi\aiworkspace\langchain
pip install -r requirements.txt
```

### Step 2: Configure Environment (1 min)
```bash
cd RAG
copy .env.example .env
```

Then edit `.env` and add your Groq API key:
```
GROQ_API_KEY=your_actual_api_key_here
```

**Get Groq API Key:**
- Visit https://console.groq.com/keys
- Create a new API key
- Copy and paste into `.env`

### Step 3: Add PDF Documents (1 min)
```bash
# Create folder if it doesn't exist (auto-created)
mkdir uploaded_pdfs

# Copy your research papers here
# Example: copy "my_research_paper.pdf" uploaded_pdfs/
```

### Step 4: Run the Web Interface (1 min)
```bash
streamlit run rag_app.py
```

The app will open at `http://localhost:8501`

## 🎯 Common Workflows

### Workflow 1: Web Interface (Easiest)
```bash
cd RAG
streamlit run rag_app.py
```
- Open browser → Upload PDFs → Ask questions → Get answers

### Workflow 2: Interactive Terminal
```bash
cd RAG
python test_rag.py interactive
```
- Same functionality, command-line interface
- Type `quit` to exit
- Type `stats` for pipeline statistics

### Workflow 3: Python Script
```python
from RAG.rag_pipeline import RAGPipeline

rag = RAGPipeline()
rag.ingest_documents()

answer = rag.query("What are the main findings?")
print(answer)
```

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| "No PDFs found" | 1. Copy PDFs to `RAG/uploaded_pdfs/` folder<br>2. Click "📤 Load PDFs" in Streamlit app |
| API key error | Check `.env` file has correct `GROQ_API_KEY` |
| Memory error | Reduce `CHUNK_SIZE` in `.env` (1000 → 500) |
| Slow performance | Use FAISS instead of Chroma (default) |
| Poor answers | Increase `TOP_K` value (3 → 5) in `.env` |

## 📚 File Structure
```
RAG/
├── rag_app.py              ← Streamlit web interface
├── rag_pipeline.py         ← Main RAG logic
├── test_rag.py            ← Testing & CLI
├── config.py              ← Configuration
├── pdf_loader.py          ← PDF processing
├── vector_store.py        ← Vector database
├── retrieval.py           ← Document retrieval
├── .env.example           ← Copy to .env
└── uploaded_pdfs/         ← Put your PDFs here
```

## 🔑 Key Configuration Values

```env
# Most important
GROQ_API_KEY=your_key_here

# Adjust for quality vs speed
CHUNK_SIZE=1000          # Larger = more context
TOP_K=3                  # More = slower but better
LLM_TEMPERATURE=0.7      # 0.3 = factual, 0.9 = creative
```

## 📊 Performance Tips

| Action | Effect |
|--------|--------|
| Increase `CHUNK_SIZE` | Better context but fewer chunks |
| Increase `TOP_K` | Better accuracy but slower |
| Decrease `LLM_TEMPERATURE` | More factual answers |
| Use FAISS | Much faster than Chroma |

## 🆘 Getting Help

### Check Logs
```bash
# Look for error messages in terminal
# StreamLit logs appear in console
```

### Debug Mode
```python
from RAG.rag_pipeline import RAGPipeline

rag = RAGPipeline()
rag.print_statistics()  # Shows current state
```

### Reload Everything
```bash
# Delete vector store and reload
rm -r vector_store
python test_rag.py
```

## 📝 Next Steps

1. ✅ Complete quick start above
2. 📖 Read [RAG/README.md](README.md) for detailed documentation
3. 🔧 Explore [integration_examples.py](integration_examples.py) for advanced use
4. 🚀 Integrate into your application

## 💡 Example Queries to Try

Put a research paper in `uploaded_pdfs/` then ask:

- "What is this paper about?"
- "Summarize the methodology"
- "What are the main conclusions?"
- "Who are the authors?"
- "What datasets were used?"
- "What are the limitations?"

## 🎓 Learn More

- [LangChain Documentation](https://python.langchain.com)
- [FAISS Documentation](https://faiss.ai)
- [Groq API Docs](https://console.groq.com/docs)
- [Streamlit Docs](https://docs.streamlit.io)

---

**Ready to go!** 🚀 Your RAG system is now set up and ready to use.
