@echo off
REM RAG System Setup Script for Windows
REM Automates RAG system installation and configuration

echo.
echo ========================================
echo   RAG System Setup
echo ========================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo Python not found. Please install Python first.
    exit /b 1
)

REM Install dependencies
echo [1/4] Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo Failed to install dependencies
    exit /b 1
)

REM Create necessary directories
echo [2/4] Creating directories...
if not exist "uploaded_pdfs" mkdir uploaded_pdfs
if not exist "vector_store" mkdir vector_store
if not exist "chroma_db" mkdir chroma_db

REM Copy environment template
if not exist ".env" (
    echo [3/4] Creating .env file...
    copy .env.example .env
    echo.
    echo [WARNING] Edit .env and add your GROQ_API_KEY
    echo Visit: https://console.groq.com/keys
) else (
    echo [3/4] .env already exists (skipping)
)

echo [4/4] Cleanup...

echo.
echo ========================================
echo   Setup Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Edit .env and add your GROQ_API_KEY
echo 2. Copy your PDF files to uploaded_pdfs/
echo 3. Run: streamlit run rag_app.py
echo.
pause
