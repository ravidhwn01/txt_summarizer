"""
Test and Demo Script for RAG Pipeline
Run this to test the complete RAG workflow
"""
import os
from dotenv import load_dotenv
from rag_pipeline import RAGPipeline

load_dotenv()


def demo_rag_pipeline():
    """Demonstrate the RAG pipeline with sample queries"""
    
    print("\n" + "=" * 70)
    print("RAG PIPELINE DEMONSTRATION")
    print("=" * 70)
    
    # Initialize RAG pipeline
    rag = RAGPipeline(vector_store_type="faiss")
    
    # Step 1: Ingest documents
    print("\nStep 1: Loading and ingesting PDF documents...")
    print("-" * 70)
    
    success = rag.ingest_documents(reload=True)
    
    if not success:
        print("\n⚠ No PDFs found. Please add PDF files to the 'uploaded_pdfs' folder first.")
        return
    
    # Step 2: Test queries
    print("\nStep 2: Testing queries...")
    print("-" * 70)
    
    test_questions = [
        "What is the main topic discussed in the documents?",
        "Summarize the key findings.",
        "What methodology was used?",
        "What are the limitations?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\nQuery {i}: {question}")
        answer = rag.query(question)
        print(f"\nAnswer:\n{answer}")
        print("-" * 70)
    
    # Step 3: Print statistics
    print("\nStep 3: Pipeline Statistics")
    print("-" * 70)
    rag.print_statistics()
    
    print("\n✓ Demo completed successfully!")


def interactive_demo():
    """Start an interactive RAG session"""
    
    print("\n" + "=" * 70)
    print("INTERACTIVE RAG SESSION")
    print("=" * 70)
    
    # Initialize RAG pipeline
    rag = RAGPipeline(vector_store_type="faiss")
    
    # Load documents
    print("\nLoading documents...")
    success = rag.ingest_documents()
    
    if not success:
        print("⚠ No PDFs found. Please add PDF files first.")
        return
    
    # Start interactive session
    rag.interactive_session()


def batch_query_demo():
    """Demonstrate batch query processing"""
    
    print("\n" + "=" * 70)
    print("BATCH QUERY DEMONSTRATION")
    print("=" * 70)
    
    # Initialize RAG pipeline
    rag = RAGPipeline(vector_store_type="faiss")
    
    # Load documents
    print("\nLoading documents...")
    success = rag.ingest_documents()
    
    if not success:
        print("⚠ No PDFs found. Please add PDF files first.")
        return
    
    # Batch queries
    questions = [
        "What is the research about?",
        "Who conducted this research?",
        "What are the results?",
        "What recommendations are made?",
    ]
    
    results = rag.batch_query(questions)
    
    # Print results
    print("\n" + "=" * 70)
    print("BATCH QUERY RESULTS")
    print("=" * 70)
    
    for question, answer in results:
        print(f"\nQ: {question}")
        print(f"A: {answer[:200]}...")
        print("-" * 70)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == "interactive":
            interactive_demo()
        elif mode == "batch":
            batch_query_demo()
        else:
            demo_rag_pipeline()
    else:
        print("""
RAG Pipeline Test Script
Usage:
    python test_rag.py              # Run demo with test queries
    python test_rag.py interactive  # Start interactive session
    python test_rag.py batch        # Run batch queries
        """)
        demo_rag_pipeline()
