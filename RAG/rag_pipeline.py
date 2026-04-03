"""
Main RAG Pipeline
Orchestrates the complete RAG workflow: PDF loading -> embedding -> retrieval -> LLM generation
"""
import os
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
    TOP_K
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
        
        # Initialize LLM
        self.llm = ChatGroq(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            max_tokens=MAX_TOKENS,
            api_key=os.getenv("GROQ_API_KEY")
        )
        
        # Define prompt template for RAG
        self.rag_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are an AI Research Assistant. Use the provided documents to answer the question accurately.

DOCUMENTS:
{context}

QUESTION: {question}

INSTRUCTIONS:
- Answer based only on the provided documents
- If the answer is not in the documents, say "I couldn't find this information in the provided documents"
- Be specific and cite which document(s) you're referring to
- Provide clear, well-structured answers

ANSWER:"""
        )
        
        print("✓ RAG Pipeline initialized successfully\n")
    
    def ingest_documents(self, folder_path: str = PDF_UPLOAD_FOLDER, 
                        reload: bool = False) -> bool:
        """
        Ingest PDF documents into the vector store
        
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
        
        # Load PDFs
        try:
            documents = self.pdf_loader.load_multiple_pdfs(folder_path)
        except Exception as e:
            print(f"✗ Error loading PDFs: {str(e)}")
            return False
            
        if not documents:
            print("\n✗ No documents loaded")
            print(f"📁 Folder path: {folder_path}")
            print(f"📝 Action: Place your PDF files in the folder above")
            return False
        
        # Check if vector store exists
        if reload or not self._vector_store_exists():
            self.vector_store.create_vector_store(documents)
        else:
            self.vector_store.load_vector_store()
            self.vector_store.add_documents(documents)
        
        print("\n✓ Document ingestion complete\n")
        return True
    
    def query(self, question: str, top_k: Optional[int] = None) -> str:
        """
        Process a user query through the RAG pipeline
        
        Args:
            question: User's question
            top_k: Number of documents to retrieve
            
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
        
        # Step 2: Retrieve relevant documents
        retrieved_docs = self.retriever.retrieve(question, top_k=top_k)
        
        if not retrieved_docs:
            print("⚠ No documents found in vector store")
            return "No documents found. Please load PDF files first using the 'Load PDFs' button."
        
        # Step 3: Format context
        context = self.retriever.format_retrieved_documents(retrieved_docs)
        
        # Step 4: Generate answer using LLM
        print("\n🤖 Generating answer with Groq...")
        try:
            # Format prompt - handle both old and new langchain versions
            try:
                prompt = self.rag_prompt.format_prompt(
                    context=context,
                    question=question
                )
            except (AttributeError, TypeError):
                # For newer versions
                prompt = self.rag_prompt.format(
                    context=context,
                    question=question
                )
            
            response = self.llm.invoke(prompt)
            answer = response.content
            
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
        
        print("\n" + "=" * 60 + "\n")
    
    def _vector_store_exists(self) -> bool:
        """Check if vector store exists"""
        if self.vector_store.store_type == "faiss":
            return os.path.exists(os.path.join(self.vector_store.store_path))
        else:
            return os.path.exists(self.vector_store.store_path)
