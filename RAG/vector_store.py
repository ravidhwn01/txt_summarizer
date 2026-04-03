"""
Vector Store Management
Handles FAISS and Chroma vector databases
"""
import os
import pickle
from typing import List, Optional, Tuple
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from config import (
    VECTOR_STORE_TYPE, 
    VECTOR_STORE_PATH, 
    CHROMA_PERSIST_DIR,
    EMBEDDING_MODEL,
    TOP_K
)


class VectorStoreManager:
    """Manages vector database operations"""
    
    def __init__(self, store_type: str = VECTOR_STORE_TYPE):
        """
        Initialize Vector Store Manager
        
        Args:
            store_type: Type of vector store ('faiss' or 'chroma')
        """
        self.store_type = store_type.lower()
        self.vector_store = None
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        
        if self.store_type == "faiss":
            self.store_path = os.path.join(VECTOR_STORE_PATH, "faiss_index")
        elif self.store_type == "chroma":
            self.store_path = CHROMA_PERSIST_DIR
        else:
            raise ValueError(f"Unsupported store type: {store_type}")
    
    def create_vector_store(self, documents: List[Document]) -> None:
        """
        Create a new vector store from documents
        
        Args:
            documents: List of Document objects to embed
        """
        if not documents:
            print("No documents to create vector store")
            return
        
        print(f"Creating {self.store_type.upper()} vector store with {len(documents)} documents...")
        
        try:
            if self.store_type == "faiss":
                self.vector_store = FAISS.from_documents(
                    documents=documents,
                    embedding=self.embeddings
                )
                self._save_faiss_index()
            
            elif self.store_type == "chroma":
                self.vector_store = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    persist_directory=self.store_path
                )
                self.vector_store.persist()
            
            print(f"✓ Vector store created successfully")
            
        except Exception as e:
            print(f"✗ Error creating vector store: {str(e)}")
            raise
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add new documents to existing vector store
        
        Args:
            documents: List of Document objects to add
        """
        if self.vector_store is None:
            print("No vector store loaded. Creating new one...")
            self.create_vector_store(documents)
            return
        
        print(f"Adding {len(documents)} documents to vector store...")
        
        try:
            if self.store_type == "faiss":
                self.vector_store.add_documents(documents)
                self._save_faiss_index()
            
            elif self.store_type == "chroma":
                self.vector_store.add_documents(documents)
                self.vector_store.persist()
            
            print(f"✓ Documents added successfully")
            
        except Exception as e:
            print(f"✗ Error adding documents: {str(e)}")
            raise
    
    def load_vector_store(self) -> bool:
        """
        Load existing vector store
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.store_type == "faiss":
                if not os.path.exists(self.store_path):
                    print(f"FAISS index not found at {self.store_path}")
                    return False
                
                self.vector_store = FAISS.load_local(
                    folder_path=VECTOR_STORE_PATH,
                    embeddings=self.embeddings,
                    index_name="faiss_index"
                )
            
            elif self.store_type == "chroma":
                if not os.path.exists(self.store_path):
                    print(f"Chroma directory not found at {self.store_path}")
                    return False
                
                self.vector_store = Chroma(
                    persist_directory=self.store_path,
                    embedding_function=self.embeddings
                )
            
            print(f"✓ {self.store_type.upper()} vector store loaded successfully")
            return True
            
        except Exception as e:
            print(f"✗ Error loading vector store: {str(e)}")
            return False
    
    def search(self, query: str, top_k: int = TOP_K) -> List[Tuple[Document, float]]:
        """
        Search for relevant documents
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of (Document, similarity_score) tuples
        """
        if self.vector_store is None:
            print("Vector store not loaded")
            return []
        
        try:
            # LangChain uses the singular method name here.
            if self.store_type == "faiss":
                results = self.vector_store.similarity_search_with_score(query, k=top_k)
            else:
                # Chroma uses the same singular method name in current LangChain versions.
                results = self.vector_store.similarity_search_with_score(query, k=top_k)
            
            return results
            
        except Exception as e:
            print(f"✗ Error during search: {str(e)}")
            return []
    
    def delete_vector_store(self) -> None:
        """Delete the vector store"""
        try:
            if self.store_type == "faiss" and os.path.exists(self.store_path):
                import shutil
                shutil.rmtree(os.path.dirname(self.store_path))
                print(f"✓ FAISS index deleted")
            
            elif self.store_type == "chroma" and os.path.exists(self.store_path):
                import shutil
                shutil.rmtree(self.store_path)
                print(f"✓ Chroma database deleted")
            
            self.vector_store = None
            
        except Exception as e:
            print(f"✗ Error deleting vector store: {str(e)}")
    
    def _save_faiss_index(self) -> None:
        """Save FAISS index to disk"""
        os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
        self.vector_store.save_local(
            folder_path=VECTOR_STORE_PATH,
            index_name="faiss_index"
        )
    
    def get_vector_store_stats(self) -> dict:
        """Get statistics about the vector store"""
        if self.vector_store is None:
            return {"status": "not_loaded"}
        
        stats = {
            "type": self.store_type,
            "path": self.store_path,
            "status": "loaded"
        }
        
        if self.store_type == "faiss":
            stats["index_size"] = self.vector_store.index.ntotal
        
        return stats
