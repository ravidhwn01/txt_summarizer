"""
Main RAG Pipeline
Orchestrates the complete RAG workflow: PDF loading -> embedding -> retrieval -> LLM generation
"""
import os
from pathlib import Path
from typing import List, Optional, Tuple
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

from pdf_loader import PDFLoader
from vector_store import VectorStoreManager
from retrieval import Retriever
from config import (
    PDF_UPLOAD_FOLDER, 
    VECTOR_STORE_TYPE, 
    LLM_MODEL,
    LLM_TEMPERATURE,
    MAX_TOKENS,
    TOP_K,
    CONVERSATION_MEMORY_TURNS
)

load_dotenv()


class RAGPipeline:
    """Main RAG Pipeline - Orchestrates the end-to-end RAG workflow"""
    
    def __init__(self, vector_store_type: str = VECTOR_STORE_TYPE):
        """
        Initialize RAG Pipeline
        
        Args:
            vector_store_type: Type of vector store ('faiss' or 'chroma')
        """
        print("🚀 Initializing RAG Pipeline...\n")
        
        # Initialize components
        self.pdf_loader = PDFLoader()
        self.vector_store = VectorStoreManager(store_type=vector_store_type)
        self.retriever = Retriever(self.vector_store, top_k=TOP_K)
        self.conversation_history = []
        self.conversation_memory_turns = CONVERSATION_MEMORY_TURNS
        
        # Initialize LLM
        self.llm = ChatGroq(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            max_tokens=MAX_TOKENS,
            api_key=os.getenv("GROQ_API_KEY")
        )
        
        # Define prompt template for RAG
        self.rag_prompt = PromptTemplate(
            input_variables=["context", "question", "conversation_history"],
            template="""You are an AI Research Assistant. Use the provided documents and the conversation history to answer the question accurately.

CONVERSATION HISTORY:
{conversation_history}

DOCUMENTS:
{context}

QUESTION: {question}

INSTRUCTIONS:
- Use the conversation history only to understand references like "it", "that", or "the previous answer"
- Answer based primarily on the provided documents
- If the answer is not in the documents, say "I couldn't find this information in the provided documents"
- Be specific and cite which document(s) you're referring to
- Provide clear, well-structured answers

ANSWER:"""
        )
        
        print("✓ RAG Pipeline initialized successfully\n")

    def clear_conversation_memory(self) -> None:
        """Clear stored conversation history."""
        self.conversation_history = []

    def get_conversation_history(self) -> List[dict]:
        """Return the stored conversation history."""
        return list(self.conversation_history)

    def _format_conversation_history(self, max_turns: Optional[int] = None) -> str:
        """Format recent conversation turns for prompt context."""
        if not self.conversation_history:
            return "No prior conversation yet."

        if max_turns is None:
            max_turns = self.conversation_memory_turns

        recent_turns = self.conversation_history[-max_turns:]
        formatted_turns = []

        for turn in recent_turns:
            formatted_turns.append(f"User: {turn['question']}")
            formatted_turns.append(f"Assistant: {turn['answer']}")

        return "\n".join(formatted_turns)

    def _build_memory_aware_query(self, question: str) -> str:
        """Combine recent history with the current question for retrieval."""
        if not self.conversation_history:
            return question

        recent_turns = self.conversation_history[-self.conversation_memory_turns:]
        history_lines = []

        for turn in recent_turns:
            history_lines.append(f"User: {turn['question']}")
            history_lines.append(f"Assistant: {turn['answer']}")

        history_text = "\n".join(history_lines)
        return f"Conversation history:\n{history_text}\n\nCurrent question: {question}"

    def add_conversation_turn(self, question: str, answer: str) -> None:
        """Store a completed question and answer pair."""
        self.conversation_history.append({"question": question, "answer": answer})

    def _rehydrate_image_documents(self, documents: List[Document]) -> List[Document]:
        """Replace fallback image documents with freshly summarized image chunks when possible."""
        refreshed_documents = []

        for document in documents:
            metadata = document.metadata or {}
            file_type = metadata.get("file_type")
            analysis_method = metadata.get("analysis_method")
            source_path = metadata.get("source")

            if file_type == "image" and analysis_method == "fallback" and source_path:
                try:
                    refreshed_documents.extend(self.pdf_loader.load_image(source_path))
                    continue
                except Exception as e:
                    print(f"⚠ Could not refresh image {source_path}: {str(e)}")

            refreshed_documents.append(document)

        return refreshed_documents
    
    def ingest_documents(self, folder_path: str = PDF_UPLOAD_FOLDER, 
                        reload: bool = False) -> bool:
        """
        Ingest PDF documents and images into the vector store
        
        Args:
            folder_path: Path to folder containing PDFs
            reload: Whether to reload from scratch (delete existing index)
            
        Returns:
            True if successful
        """
        print("\n" + "=" * 60)
        print("DOCUMENT INGESTION")
        print("=" * 60)
        
        # Ensure folder path is absolute
        folder_path = os.path.abspath(folder_path)
        
        # Load supported files
        try:
            documents = self.pdf_loader.load_multiple_files(folder_path)
        except Exception as e:
            print(f"✗ Error loading files: {str(e)}")
            return False
            
        if not documents:
            print("\n✗ No documents loaded")
            print(f"📁 Folder path: {folder_path}")
            print(f"📝 Action: Place PDF or image files in the folder above")
            return False
        
        # Check if vector store exists
        if reload or not self._vector_store_exists():
            self.vector_store.create_vector_store(documents)
        else:
            self.vector_store.load_vector_store()
            self.vector_store.add_documents(documents)
        
        print("\n✓ Document ingestion complete\n")
        return True

    def ingest_uploaded_files(self, uploaded_files, reload: bool = False) -> bool:
        """
        Ingest user-uploaded PDFs and images into the vector store.

        Args:
            uploaded_files: Iterable of uploaded file objects from Streamlit
            reload: Whether to rebuild the vector store from scratch

        Returns:
            True if successful
        """
        print("\n" + "=" * 60)
        print("UPLOADED FILE INGESTION")
        print("=" * 60)

        if not uploaded_files:
            print("✗ No files were uploaded")
            return False

        os.makedirs(PDF_UPLOAD_FOLDER, exist_ok=True)

        documents = []
        supported_extensions = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}

        for uploaded_file in uploaded_files:
            suffix = Path(uploaded_file.name).suffix.lower()
            if suffix not in supported_extensions:
                print(f"⚠ Skipping unsupported file type: {uploaded_file.name}")
                continue

            try:
                file_content = uploaded_file.getvalue()
                loaded_documents = self.pdf_loader.load_uploaded_file(file_content, uploaded_file.name)
                documents.extend(loaded_documents)
            except Exception as e:
                print(f"✗ Error processing uploaded file {uploaded_file.name}: {str(e)}")

        if not documents:
            print("✗ No documents were extracted from uploaded files")
            return False

        if reload or not self._vector_store_exists():
            self.vector_store.create_vector_store(documents)
        else:
            if not self.vector_store.load_vector_store():
                print("⚠ Could not load existing vector store; creating a new one")
                self.vector_store.create_vector_store(documents)
            else:
                self.vector_store.add_documents(documents)

        print("\n✓ Uploaded file ingestion complete\n")
        return True
    
    def query(self, question: str, top_k: Optional[int] = None, remember: bool = True) -> str:
        """
        Process a user query through the RAG pipeline
        
        Args:
            question: User's question
            top_k: Number of documents to retrieve
            remember: Whether to store the question and answer in conversation memory
            
        Returns:
            Generated answer
        """
        print("\n" + "=" * 60)
        print("RAG QUERY PROCESSING")
        print("=" * 60)
        
        # Step 1: Load vector store if not already loaded
        if self.vector_store.vector_store is None:
            if not self.vector_store.load_vector_store():
                print("⚠ Vector store not found. Please ingest documents first.")
                return "No documents available. Please upload PDFs first."
        
        memory_aware_query = self._build_memory_aware_query(question)

        # Step 2: Retrieve relevant documents
        retrieved_docs = self.retriever.retrieve(memory_aware_query, top_k=top_k)
        
        if not retrieved_docs:
            print("⚠ No documents found in vector store")
            return "No documents found. Please load PDF files first using the 'Load PDFs' button."

        retrieved_docs = self._rehydrate_image_documents(retrieved_docs)
        
        # Step 3: Format context
        context = self.retriever.format_retrieved_documents(retrieved_docs)
        
        # Step 4: Generate answer using LLM
        print("\n🤖 Generating answer with Groq...")
        try:
            # Format prompt - handle both old and new langchain versions
            try:
                prompt = self.rag_prompt.format_prompt(
                    context=context,
                    question=question,
                    conversation_history=self._format_conversation_history()
                )
            except (AttributeError, TypeError):
                # For newer versions
                prompt = self.rag_prompt.format(
                    context=context,
                    question=question,
                    conversation_history=self._format_conversation_history()
                )
            
            response = self.llm.invoke(prompt)
            answer = response.content

            if remember:
                self.add_conversation_turn(question, answer)
            
            print("✓ Answer generated successfully\n")
            return answer
            
        except Exception as e:
            print(f"✗ Error generating answer: {str(e)}")
            import traceback
            traceback.print_exc()
            return f"Error generating answer: {str(e)}"
    
    def interactive_session(self):
        """Start an interactive RAG session"""
        print("\n" + "=" * 60)
        print("INTERACTIVE RAG SESSION")
        print("=" * 60)
        print("Type 'quit' to exit, 'reload' to reload documents, 'stats' for statistics\n")
        
        while True:
            try:
                question = input("\n📝 Ask a question: ").strip()
                
                if question.lower() == 'quit':
                    print("\nGoodbye! 👋")
                    break
                
                elif question.lower() == 'reload':
                    self.ingest_documents(reload=True)
                    continue
                
                elif question.lower() == 'stats':
                    self.print_statistics()
                    continue
                
                elif not question:
                    continue
                
                # Process query
                answer = self.query(question)
                print("\n" + "-" * 60)
                print("ANSWER:")
                print("-" * 60)
                print(answer)
                print("-" * 60)
                print(f"Conversation turns stored: {len(self.conversation_history)}")
                
            except KeyboardInterrupt:
                print("\nInterrupted. Exiting...")
                break
            except Exception as e:
                print(f"Error: {str(e)}")
    
    def batch_query(self, questions: List[str]) -> List[Tuple[str, str]]:
        """
        Process multiple questions in batch
        
        Args:
            questions: List of questions
            
        Returns:
            List of (question, answer) tuples
        """
        print(f"\n📊 Processing {len(questions)} questions in batch...\n")
        
        results = []
        for i, question in enumerate(questions, 1):
            print(f"[{i}/{len(questions)}] {question}")
            answer = self.query(question)
            results.append((question, answer))
        
        return results
    
    def print_statistics(self):
        """Print RAG pipeline statistics"""
        print("\n" + "=" * 60)
        print("RAG PIPELINE STATISTICS")
        print("=" * 60)
        
        vs_stats = self.vector_store.get_vector_store_stats()
        print(f"Vector Store Type: {vs_stats.get('type', 'N/A')}")
        print(f"Vector Store Status: {vs_stats.get('status', 'N/A')}")
        
        retrieval_stats = self.retriever.get_retrieval_stats()
        print(f"Top-K Documents: {retrieval_stats['top_k']}")
        print(f"Relevance Threshold: {retrieval_stats['relevance_threshold']}")
        print(f"Conversation Turns Stored: {len(self.conversation_history)}")
        
        print("\n" + "=" * 60 + "\n")
    
    def _vector_store_exists(self) -> bool:
        """Check if vector store exists"""
        if self.vector_store.store_type == "faiss":
            return os.path.exists(os.path.join(self.vector_store.store_path))
        else:
            return os.path.exists(self.vector_store.store_path)
