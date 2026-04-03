"""
PDF Loader and Document Processing
Handles uploading, loading, and chunking PDF documents
"""
import os
from pathlib import Path
from typing import List, Optional
from PyPDF2 import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP, PDF_UPLOAD_FOLDER, USER_UPLOAD_FOLDER

try:
    from PIL import Image
except ImportError:  # pragma: no cover - optional dependency
    Image = None

try:
    import pytesseract
except ImportError:  # pragma: no cover - optional dependency
    pytesseract = None


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

    def load_image(self, file_path: str) -> List[Document]:
        """
        Load and OCR text from an image file.

        Args:
            file_path: Path to the image file

        Returns:
            List of Document objects
        """
        if Image is None or pytesseract is None:
            print(f"⚠ OCR dependencies are missing; cannot process image {file_path}")
            return []

        try:
            image = Image.open(file_path)
            extracted_text = pytesseract.image_to_string(image)
            extracted_text = extracted_text.strip()

            if not extracted_text:
                print(f"⚠ No text detected in image {file_path}")
                return []

            metadata = {
                "source": file_path,
                "filename": os.path.basename(file_path),
                "file_type": "image",
            }
            document = Document(page_content=extracted_text, metadata=metadata)
            chunks = self.text_splitter.split_documents([document])

            print(f"✓ Loaded image {file_path}: {len(chunks)} chunks")
            return chunks

        except Exception as e:
            print(f"✗ Error loading image {file_path}: {str(e)}")
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

    def load_multiple_files(self, folder_path: str = PDF_UPLOAD_FOLDER) -> List[Document]:
        """
        Load all supported files from a folder, including PDFs and images.

        Args:
            folder_path: Path to folder containing files

        Returns:
            Combined list of Document chunks
        """
        all_documents = []
        os.makedirs(folder_path, exist_ok=True)
        folder_path = os.path.abspath(folder_path)
        print(f"📁 Looking for files in: {folder_path}")

        supported_extensions = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
        file_paths = [path for path in Path(folder_path).iterdir() if path.is_file() and path.suffix.lower() in supported_extensions]

        if not file_paths:
            print(f"⚠ No supported files found in {folder_path}")
            print("📝 Supported file types: PDF, PNG, JPG, JPEG, WEBP, BMP, TIFF")
            return all_documents

        print(f"Loading {len(file_paths)} supported file(s)...")
        for file_path in file_paths:
            suffix = file_path.suffix.lower()
            if suffix == ".pdf":
                documents = self.load_pdf(str(file_path))
            else:
                documents = self.load_image(str(file_path))
            all_documents.extend(documents)

        self.documents = all_documents
        print(f"Total chunks: {len(all_documents)}")
        return all_documents

    def load_uploaded_file(self, file_content: bytes, filename: str, folder_path: str = USER_UPLOAD_FOLDER) -> List[Document]:
        """
        Save and load a single uploaded file based on its extension.

        Args:
            file_content: Binary content of the file
            filename: Original filename

        Returns:
            List of Document objects
        """
        saved_path = self.save_uploaded_file(file_content, filename, folder_path=folder_path)
        suffix = Path(saved_path).suffix.lower()

        if suffix == ".pdf":
            return self.load_pdf(saved_path)

        return self.load_image(saved_path)

    def save_uploaded_file(self, file_content: bytes, filename: str, folder_path: str = USER_UPLOAD_FOLDER) -> str:
        """
        Save an uploaded PDF or image file.

        Args:
            file_content: Binary content of the file
            filename: Name of the file

        Returns:
            Path to saved file
        """
        os.makedirs(folder_path, exist_ok=True)
        file_path = os.path.join(folder_path, filename)

        with open(file_path, 'wb') as file_handle:
            file_handle.write(file_content)

        print(f"✓ Saved file: {file_path}")
        return file_path
    
    def save_uploaded_pdf(self, file_content: bytes, filename: str, folder_path: str = USER_UPLOAD_FOLDER) -> str:
        """
        Save uploaded PDF file
        
        Args:
            file_content: Binary content of the file
            filename: Name of the file
            
        Returns:
            Path to saved file
        """
        os.makedirs(folder_path, exist_ok=True)
        file_path = os.path.join(folder_path, filename)
        
        # Ensure filename is safe
        if not filename.endswith('.pdf'):
            filename = filename + '.pdf'
            file_path = os.path.join(folder_path, filename)
        
        with open(file_path, 'wb') as f:
            f.write(file_content)
        
        print(f"✓ Saved PDF: {file_path}")
        return file_path
    
    def get_document_count(self) -> int:
        """Get number of loaded document chunks"""
        return len(self.documents)
