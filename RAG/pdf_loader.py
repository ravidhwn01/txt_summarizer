"""
PDF Loader and Document Processing
Handles uploading, loading, and chunking PDF documents
"""
import os
from pathlib import Path
from typing import List
from PyPDF2 import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP, PDF_UPLOAD_FOLDER


class PDFLoader:
    """Handles PDF loading and text extraction"""
    
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        """
        Initialize PDF Loader
        
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        self.documents = []
    
    def load_pdf(self, file_path: str) -> List[Document]:
        """
        Load and extract text from a PDF file
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of Document objects
        """
        try:
            pdf_reader = PdfReader(file_path)
            text = ""
            metadata = {
                "source": file_path,
                "filename": os.path.basename(file_path),
                "page_count": len(pdf_reader.pages)
            }
            
            # Extract text from all pages
            for page_num, page in enumerate(pdf_reader.pages):
                text += f"\n--- Page {page_num + 1} ---\n"
                text += page.extract_text()
            
            # Create a base document
            doc = Document(page_content=text, metadata=metadata)
            
            # Split into chunks
            chunks = self.text_splitter.split_documents([doc])
            
            print(f"✓ Loaded {file_path}: {len(pdf_reader.pages)} pages -> {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            print(f"✗ Error loading PDF {file_path}: {str(e)}")
            return []
    
    def load_multiple_pdfs(self, folder_path: str = PDF_UPLOAD_FOLDER) -> List[Document]:
        """
        Load all PDF files from a folder
        
        Args:
            folder_path: Path to folder containing PDFs
            
        Returns:
            Combined list of Document chunks
        """
        all_documents = []
        
        # Ensure folder exists
        os.makedirs(folder_path, exist_ok=True)
        
        # Convert to absolute path if relative
        folder_path = os.path.abspath(folder_path)
        print(f"📁 Looking for PDFs in: {folder_path}")
        
        pdf_files = list(Path(folder_path).glob("*.pdf"))
        
        if not pdf_files:
            print(f"⚠ No PDF files found in {folder_path}")
            print(f"📝 Please place PDF files (.pdf) in this folder")
            return all_documents
        
        print(f"Loading {len(pdf_files)} PDF files...")
        for pdf_file in pdf_files:
            documents = self.load_pdf(str(pdf_file))
            all_documents.extend(documents)
        
        self.documents = all_documents
        print(f"Total chunks: {len(all_documents)}")
        return all_documents
    
    def save_uploaded_pdf(self, file_content: bytes, filename: str) -> str:
        """
        Save uploaded PDF file
        
        Args:
            file_content: Binary content of the file
            filename: Name of the file
            
        Returns:
            Path to saved file
        """
        file_path = os.path.join(PDF_UPLOAD_FOLDER, filename)
        
        # Ensure filename is safe
        if not filename.endswith('.pdf'):
            filename = filename + '.pdf'
            file_path = os.path.join(PDF_UPLOAD_FOLDER, filename)
        
        with open(file_path, 'wb') as f:
            f.write(file_content)
        
        print(f"✓ Saved PDF: {file_path}")
        return file_path
    
    def get_document_count(self) -> int:
        """Get number of loaded document chunks"""
        return len(self.documents)
