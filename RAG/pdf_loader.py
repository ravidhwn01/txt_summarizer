"""
PDF Loader and Document Processing
Handles uploading, loading, and chunking PDF documents
"""
import base64
import importlib
import os
import mimetypes
import shutil
from pathlib import Path
from typing import List, Optional
from PyPDF2 import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    PDF_UPLOAD_FOLDER,
    USER_UPLOAD_FOLDER,
    GOOGLE_API_KEY,
    VISION_MODEL,
    IMAGE_CAPTION_MODEL,
    ENABLE_LOCAL_IMAGE_CAPTIONING,
)

try:
    from PIL import Image
except ImportError:  # pragma: no cover - optional dependency
    Image = None

try:
    pytesseract = importlib.import_module("pytesseract")
except ImportError:  # pragma: no cover - optional dependency
    pytesseract = None

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import HumanMessage
except ImportError:  # pragma: no cover - optional dependency
    ChatGoogleGenerativeAI = None
    HumanMessage = None

try:
    import torch
    from transformers import BlipForConditionalGeneration, BlipProcessor
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    BlipForConditionalGeneration = None
    BlipProcessor = None

try:
    import easyocr
except ImportError:  # pragma: no cover - optional dependency
    easyocr = None


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
        self.vision_model_cache = {}
        self.image_caption_processor = None
        self.image_caption_model = None
        self.easyocr_reader = None
        self.enable_local_image_captioning = ENABLE_LOCAL_IMAGE_CAPTIONING
        self.tesseract_path = shutil.which("tesseract")

        self.vision_model_candidates = self._build_vision_model_candidates()

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
        Load and summarize text from an image file.

        Args:
            file_path: Path to the image file

        Returns:
            List of Document objects
        """
        image_summary = self._describe_image(file_path)

        if image_summary:
            metadata = {
                "source": file_path,
                "filename": os.path.basename(file_path),
                "file_type": "image",
                "analysis_method": image_summary["method"],
            }
            document = Document(page_content=image_summary["text"], metadata=metadata)
            chunks = self.text_splitter.split_documents([document])

            print(f"✓ Loaded image {file_path}: {len(chunks)} chunks via {image_summary['method']}")
            return chunks

        print(f"⚠ Image summarization unavailable for {file_path}; storing fallback text instead")
        return [self._create_image_fallback_document(file_path, self._image_summary_failure_reason())]

    def _describe_image(self, file_path: str) -> Optional[dict]:
        """Use OCR, a vision API, or a local captioner to produce text for an image."""
        if Image is None:
            return None

        try:
            image = Image.open(file_path)
        except Exception as e:
            print(f"⚠ Could not open image {file_path}: {str(e)}")
            return None

        # Prefer OCR-like extraction first so screenshots with visible text are handled locally.
        ocr_text = self._extract_text_from_image(image, file_path)
        if ocr_text:
            return {
                "text": ocr_text,
                "method": "ocr",
            }

        # Fall back to vision or captioning when OCR does not find usable text.
        vision_summary = self._describe_with_vision(file_path, image)
        if vision_summary:
            return vision_summary

        caption_text = self._caption_image_locally(image, file_path)
        if caption_text:
            return {
                "text": f"Image summary for {os.path.basename(file_path)}: {caption_text}",
                "method": "local_captioning",
            }

        return None

    def _extract_text_from_image(self, image, file_path: str) -> Optional[str]:
        """Extract text using Tesseract when available, otherwise EasyOCR."""
        if pytesseract is not None and self.tesseract_path:
            try:
                extracted_text = pytesseract.image_to_string(image).strip()
                if extracted_text:
                    return extracted_text
            except Exception as e:
                print(f"⚠ OCR failed for {file_path}: {str(e)}")

        if easyocr is not None:
            try:
                if self.easyocr_reader is None:
                    self.easyocr_reader = easyocr.Reader(["en"], gpu=False)

                result = self.easyocr_reader.readtext(file_path, detail=0, paragraph=True)
                extracted_text = " ".join(part.strip() for part in result if part and part.strip()).strip()
                if extracted_text:
                    return extracted_text
            except Exception as e:
                print(f"⚠ EasyOCR failed for {file_path}: {str(e)}")

        return None

    def _caption_image_locally(self, image, file_path: str) -> Optional[str]:
        """Generate a caption using a local BLIP model when available."""
        if not self.enable_local_image_captioning:
            return None

        if torch is None or BlipProcessor is None or BlipForConditionalGeneration is None:
            return None

        try:
            if self.image_caption_processor is None or self.image_caption_model is None:
                self.image_caption_processor = BlipProcessor.from_pretrained(IMAGE_CAPTION_MODEL)
                self.image_caption_model = BlipForConditionalGeneration.from_pretrained(IMAGE_CAPTION_MODEL)
                self.image_caption_model.eval()

            inputs = self.image_caption_processor(images=image, return_tensors="pt")
            with torch.no_grad():
                generated_tokens = self.image_caption_model.generate(**inputs, max_new_tokens=30)

            caption = self.image_caption_processor.decode(generated_tokens[0], skip_special_tokens=True).strip()
            return caption or None
        except Exception as e:
            print(f"⚠ Local image captioning failed for {file_path}: {str(e)}")
            return None

    def _build_vision_model_candidates(self) -> List[str]:
        """Build a prioritized, de-duplicated list of vision model names."""
        candidates = [VISION_MODEL, "gemini-2.0-flash", "gemini-1.5-flash-latest"]
        seen = set()
        unique_candidates = []

        for candidate in candidates:
            if candidate and candidate not in seen:
                seen.add(candidate)
                unique_candidates.append(candidate)

        return unique_candidates

    def _get_vision_model(self, model_name: str):
        """Create or reuse a Gemini vision model instance."""
        if model_name in self.vision_model_cache:
            return self.vision_model_cache[model_name]

        if GOOGLE_API_KEY and ChatGoogleGenerativeAI is not None:
            try:
                model = ChatGoogleGenerativeAI(
                    model=model_name,
                    google_api_key=GOOGLE_API_KEY,
                    temperature=0.2,
                )
                self.vision_model_cache[model_name] = model
                return model
            except Exception as e:
                print(f"⚠ Vision model unavailable for {model_name}: {str(e)}")

        self.vision_model_cache[model_name] = None
        return None

    def _describe_with_vision(self, file_path: str, image) -> Optional[dict]:
        """Try multiple Gemini vision models until one works."""
        if HumanMessage is None:
            return None

        mime_type = mimetypes.guess_type(file_path)[0] or "image/png"
        with open(file_path, "rb") as file_handle:
            encoded_image = base64.b64encode(file_handle.read()).decode("utf-8")

        prompt = (
            "Describe this image for a knowledge base. "
            "If it contains visible text, extract the key text first. "
            "Then summarize the image content in 3-6 concise sentences."
        )

        for model_name in self.vision_model_candidates:
            vision_model = self._get_vision_model(model_name)
            if vision_model is None:
                continue

            try:
                response = vision_model.invoke([
                    HumanMessage(content=[
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{encoded_image}"
                            },
                        },
                    ])
                ])
                summary_text = getattr(response, "content", "").strip()
                if summary_text:
                    return {
                        "text": summary_text,
                        "method": f"vision:{model_name}",
                    }
            except Exception as e:
                print(f"⚠ Vision model failed for {model_name} on {file_path}: {str(e)}")
                self.vision_model_cache[model_name] = None

        return None

    def _image_summary_failure_reason(self) -> str:
        """Explain why image summarization could not run."""
        missing_backends = []

        if pytesseract is None:
            missing_backends.append("pytesseract")
        if not self.tesseract_path:
            missing_backends.append("tesseract_executable")
        if GOOGLE_API_KEY is None or ChatGoogleGenerativeAI is None:
            missing_backends.append("vision_api")
        if self.enable_local_image_captioning and (torch is None or BlipProcessor is None or BlipForConditionalGeneration is None):
            missing_backends.append("local_captioner")

        if not missing_backends:
            return "Image summarization failed unexpectedly"

        return (
            "No image summarization backend is available (missing: "
            + ", ".join(missing_backends)
            + "). Install OCR or local captioning dependencies, or configure the vision API."
        )

    def _create_image_fallback_document(self, file_path: str, reason: str) -> Document:
        """Create a lightweight fallback document when image summarization is unavailable."""
        return Document(
            page_content=(
                f"Image uploaded: {os.path.basename(file_path)}\n"
                f"Image summarization was unavailable. Reason: {reason}.\n"
                "This image is stored in the knowledge base, but it does not contain extracted text or a generated summary."
            ),
            metadata={
                "source": file_path,
                "filename": os.path.basename(file_path),
                "file_type": "image",
                "analysis_method": "fallback",
                "fallback_reason": reason,
            },
        )
    
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
