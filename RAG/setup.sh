#!/bin/bash
# RAG System Setup Script
# Automates RAG system installation and configuration

echo "🚀 RAG System Setup"
echo "===================="

# Check Python
python --version || { echo "Python not found"; exit 1; }

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p uploaded_pdfs
mkdir -p vector_store
mkdir -p chroma_db

# Copy environment template
if [ ! -f .env ]; then
    echo "⚙️  Creating .env file..."
    cp .env.example .env
    echo "⚠️  Edit .env and add your GROQ_API_KEY"
else
    echo "✓ .env already exists"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env and add your GROQ_API_KEY"
echo "2. Copy your PDF files to uploaded_pdfs/"
echo "3. Run: streamlit run rag_app.py"
echo ""
