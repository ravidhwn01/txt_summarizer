"""
Simple Streamlit App for RAG System
"""
import streamlit as st
import os
from pathlib import Path
from rag_pipeline import RAGPipeline
from config import PDF_UPLOAD_FOLDER

# Page configuration
st.set_page_config(
    page_title="AI Research Assistant with RAG",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        padding: 0.5rem;
        margin: 0.25rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'rag_pipeline' not in st.session_state:
    st.session_state.rag_pipeline = RAGPipeline()
    st.session_state.documents_loaded = False
    st.session_state.vector_store_initialized = False

def main():
    st.title("🤖 AI Research Assistant with RAG")
    st.markdown("Upload research papers and ask intelligent questions to get grounded answers")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Vector Store Type
        store_type = st.radio("Vector Store Type", ["FAISS", "Chroma"])
        
        # Document Management
        st.subheader("📄 Document Management")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📤 Load PDFs", use_container_width=True, key="load_pdfs"):
                with st.spinner("Loading documents..."):
                    success = st.session_state.rag_pipeline.ingest_documents(reload=False)
                    if success:
                        st.session_state.documents_loaded = True
                        st.session_state.vector_store_initialized = True
                        st.success("✓ Documents loaded successfully!")
                        st.balloons()
                    else:
                        st.error("✗ No PDFs found - please check the uploaded_pdfs folder")
        
        with col2:
            if st.button("🔄 Reload PDFs", use_container_width=True, key="reload_pdfs"):
                with st.spinner("Reloading documents..."):
                    success = st.session_state.rag_pipeline.ingest_documents(reload=True)
                    if success:
                        st.session_state.documents_loaded = True
                        st.session_state.vector_store_initialized = True
                        st.success("✓ Documents reloaded!")
                        st.balloons()
                    else:
                        st.error("✗ Failed to reload documents")
        
        # Display statistics
        st.subheader("📊 Statistics")
        try:
            if st.session_state.vector_store_initialized:
                stats = st.session_state.rag_pipeline.vector_store.get_vector_store_stats()
                st.metric("Store Type", stats.get('type', 'N/A').upper())
                st.metric("Status", "✓ Ready" if stats.get('status') == 'loaded' else "Pending")
            else:
                st.info("ℹ️ Load documents to see statistics")
        except Exception as e:
            st.info("ℹ️ No vector store loaded yet")
        
        # PDF Upload Folder
        st.subheader("📁 Upload Folder")
        pdf_files = list(Path(PDF_UPLOAD_FOLDER).glob("*.pdf"))
        
        # Create folder if it doesn't exist
        os.makedirs(PDF_UPLOAD_FOLDER, exist_ok=True)
        
        st.info(f"PDFs in folder: {len(pdf_files)}")
        st.caption(f"Location: {os.path.abspath(PDF_UPLOAD_FOLDER)}")
        
        if not pdf_files:
            st.warning("⚠️ No PDFs found! Copy your PDF files to the folder above.")
        
        if pdf_files:
            with st.expander("View PDF files"):
                for pdf in pdf_files:
                    st.text(f"✓ {pdf.name}")
    
    # Main content
    if not st.session_state.documents_loaded:
        st.info("👈 Click '📤 Load PDFs' in the sidebar to start")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            ### 📋 How to use:
            1. **Add PDFs**: Copy your research papers to:
               ```
               RAG/uploaded_pdfs/
               ```
            2. **Load**: Click '📤 Load PDFs' button
            3. **Ask**: Query the documents
            """)
        
        with col2:
            st.markdown(f"""
            ### 📁 PDF Location:
            ```
            {os.path.abspath(PDF_UPLOAD_FOLDER)}
            ```
            
            ### 📄 Supported Format:
            - PDF files (.pdf)
            
            ### 💡 Example files:
            - research_paper.pdf
            - document.pdf
            - whitepaper.pdf
            """)
    else:
        # Query section
        st.subheader("❓ Ask a Question")
        
        # Show status
        st.info(f"📚 Vector store status: {'✓ Ready' if st.session_state.vector_store_initialized else '⚠ Not ready'}")
        
        question = st.text_area(
            "Enter your question:",
            height=100,
            placeholder="e.g., What are the main findings of this research?"
        )
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            if st.button("🔍 Search", use_container_width=True, key="search_btn"):
                if question.strip():
                    with st.spinner("Processing your question..."):
                        try:
                            answer = st.session_state.rag_pipeline.query(question)
                            st.subheader("📋 Answer")
                            st.markdown(answer)
                            
                            # Always show retrieved documents
                            with st.expander("📚 Show Retrieved Documents", expanded=True):
                                retrieved_docs = st.session_state.rag_pipeline.retriever.retrieve(question)
                                if retrieved_docs:
                                    st.success(f"✓ Found {len(retrieved_docs)} relevant document(s)")
                                    for i, doc in enumerate(retrieved_docs, 1):
                                        with st.expander(f"📄 Document {i} - {doc.metadata.get('filename', 'Unknown')}"):
                                            content = doc.page_content[:800]
                                            if len(doc.page_content) > 800:
                                                content += "\n\n... [truncated]"
                                            st.text(content)
                                else:
                                    st.warning("No documents retrieved")
                        except Exception as e:
                            st.error(f"Error processing query: {str(e)}")
                else:
                    st.warning("Please enter a question")
        
        with col2:
            if st.button("📊 Stats", use_container_width=True, key="stats_btn"):
                try:
                    st.session_state.rag_pipeline.print_statistics()
                    st.success("Statistics printed to console")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        with col3:
            if st.button("🔄 Reinit", use_container_width=True, key="reinit_btn"):
                st.session_state.documents_loaded = False
                st.session_state.vector_store_initialized = False
                st.rerun()
        
        # Example queries
        st.subheader("💡 Example Queries")
        st.caption("Click any example to try it:")
        
        examples = [
            "What are the main conclusions of this research?",
            "Summarize the methodology used in this paper",
            "What datasets were used in this study?",
            "What are the limitations of this research?"
        ]
        
        for idx, example in enumerate(examples):
            if st.button(f"Try: {example}", use_container_width=True, key=f"example_{idx}"):
                with st.spinner("Processing..."):
                    try:
                        answer = st.session_state.rag_pipeline.query(example)
                        st.subheader("📋 Answer")
                        st.markdown(answer)
                        
                        with st.expander("📚 Retrieved Documents"):
                            retrieved_docs = st.session_state.rag_pipeline.retriever.retrieve(example)
                            if retrieved_docs:
                                st.success(f"✓ Found {len(retrieved_docs)} document(s)")
                                for i, doc in enumerate(retrieved_docs, 1):
                                    st.text(f"**Doc {i}: {doc.metadata.get('filename', 'Unknown')}**")
                                    st.text(doc.page_content[:400] + "...")
                            else:
                                st.warning("No documents found")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
