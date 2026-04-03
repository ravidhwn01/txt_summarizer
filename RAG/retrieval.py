"""
Retrieval Module
Handles document retrieval from vector store with re-ranking and filtering
"""
from typing import List, Tuple, Optional
from langchain_core.documents import Document
from vector_store import VectorStoreManager
from config import TOP_K, RELEVANCE_THRESHOLD


class Retriever:
    """Handles document retrieval and ranking"""
    
    def __init__(self, vector_store_manager: VectorStoreManager, top_k: int = TOP_K):
        """
        Initialize Retriever
        
        Args:
            vector_store_manager: Vector store manager instance
            top_k: Number of documents to retrieve
        """
        self.vector_store = vector_store_manager
        self.top_k = top_k
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Document]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: User query
            top_k: Number of results to return (uses default if None)
            
        Returns:
            List of relevant Document objects
        """
        if top_k is None:
            top_k = self.top_k
        
        print(f"\n🔍 Retrieving documents for query: '{query}'")
        
        # Search in vector store
        results = self.vector_store.search(query, top_k=top_k)
        
        if not results:
            print("⚠ No documents found in vector store")
            return []
        
        # Extract documents - accept all retrieved results
        documents = []
        for i, (doc, score) in enumerate(results, 1):
            # FAISS returns distance scores (lower is better), show all results
            print(f"  [{i}] Distance: {score:.4f} | Source: {doc.metadata.get('filename', 'unknown')}")
            documents.append(doc)
        
        print(f"✓ Retrieved {len(documents)} documents\n")
        return documents
    
    def retrieve_with_scores(self, query: str, top_k: Optional[int] = None) -> List[Tuple[Document, float]]:
        """
        Retrieve documents with relevance scores
        
        Args:
            query: User query
            top_k: Number of results to return
            
        Returns:
            List of (Document, score) tuples
        """
        if top_k is None:
            top_k = self.top_k
        
        return self.vector_store.search(query, top_k=top_k)
    
    def format_retrieved_documents(self, documents: List[Document]) -> str:
        """
        Format retrieved documents for LLM context
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            Formatted string for LLM context
        """
        if not documents:
            return "No relevant documents found."
        
        formatted = "Retrieved Documents:\n" + "=" * 50 + "\n"
        
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("filename", "unknown")
            formatted += f"\n[Document {i}] from {source}\n"
            formatted += "-" * 40 + "\n"
            # Limit content length to avoid context overload
            content = doc.page_content[:1000]
            if len(doc.page_content) > 1000:
                content += "... [truncated]"
            formatted += content + "\n"
        
        formatted += "\n" + "=" * 50 + "\n"
        return formatted
    
    def get_retrieval_stats(self) -> dict:
        """Get retrieval statistics"""
        return {
            "top_k": self.top_k,
            "relevance_threshold": RELEVANCE_THRESHOLD,
            "vector_store_stats": self.vector_store.get_vector_store_stats()
        }
