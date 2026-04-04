"""
Simple Streamlit App for RAG System
"""
import streamlit as st
import os
from pathlib import Path
from rag_pipeline import RAGPipeline
from config import PDF_UPLOAD_FOLDER

SUPPORTED_UPLOAD_TYPES = ["pdf", "png", "jpg", "jpeg", "webp", "bmp", "tiff"]

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
if 'documents_loaded' not in st.session_state:
    st.session_state.documents_loaded = False
if 'vector_store_initialized' not in st.session_state:
    st.session_state.vector_store_initialized = False
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []

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

        st.caption("Upload PDFs or images from your computer, or use the existing folder-based workflow below.")

        uploaded_files = st.file_uploader(
            "Upload PDFs or images",
            type=SUPPORTED_UPLOAD_TYPES,
            accept_multiple_files=True,
            key="rag_file_uploader",
        )

        upload_mode_col1, upload_mode_col2 = st.columns(2)
        with upload_mode_col1:
            upload_reload = st.checkbox("Rebuild index", value=False, key="rebuild_index_checkbox")
        with upload_mode_col2:
            st.caption("Images are OCR-processed when possible.")

        if st.button("📥 Upload & Ingest", use_container_width=True, key="upload_and_ingest_btn"):
            with st.spinner("Uploading and ingesting files..."):
                success = st.session_state.rag_pipeline.ingest_uploaded_files(
                    uploaded_files,
                    reload=upload_reload,
                )
                if success:
                    st.session_state.documents_loaded = True
                    st.session_state.vector_store_initialized = True
                    st.success("✓ Uploaded files ingested successfully!")
                    st.balloons()
                else:
                    st.error("✗ Upload ingestion failed. Check file type, PDF validity, or OCR dependencies.")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📤 Load Files", use_container_width=True, key="load_pdfs"):
                with st.spinner("Loading documents and images..."):
                    success = st.session_state.rag_pipeline.ingest_documents(reload=False)
                    if success:
                        st.session_state.documents_loaded = True
                        st.session_state.vector_store_initialized = True
                        st.success("✓ Documents loaded successfully!")
                        st.balloons()
                    else:
                        st.error("✗ No PDF or image files found - please check the uploaded_pdfs folder")
        
        with col2:
            if st.button("🔄 Reload Files", use_container_width=True, key="reload_pdfs"):
                with st.spinner("Reloading documents and images..."):
                    success = st.session_state.rag_pipeline.ingest_documents(reload=True)
                    if success:
                        st.session_state.documents_loaded = True
                        st.session_state.vector_store_initialized = True
                        st.success("✓ Documents reloaded!")
                        st.balloons()
                    else:
                        st.error("✗ Failed to reload documents or images")

        st.subheader("🧠 Conversation Memory")
        st.caption("Recent turns are kept so follow-up questions can reuse prior context.")
        st.metric("Stored turns", len(st.session_state.rag_pipeline.get_conversation_history()))
        if st.button("🧹 Clear Conversation", use_container_width=True, key="clear_memory_btn"):
            st.session_state.rag_pipeline.clear_conversation_memory()
            st.session_state.chat_messages = []
            st.success("Conversation memory cleared")
            st.rerun()
        
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
        image_files = [path for path in Path(PDF_UPLOAD_FOLDER).iterdir() if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}]
        
        # Create folder if it doesn't exist
        os.makedirs(PDF_UPLOAD_FOLDER, exist_ok=True)
        
        st.info(f"PDFs in folder: {len(pdf_files)} | Images in folder: {len(image_files)}")
        st.caption(f"Location: {os.path.abspath(PDF_UPLOAD_FOLDER)}")
        
        if not pdf_files and not image_files:
            st.warning("⚠️ No PDFs or images found! Copy your files to the folder above or upload them using the uploader.")
        
        if pdf_files or image_files:
            with st.expander("View files"):
                for pdf in pdf_files:
                    st.text(f"✓ PDF: {pdf.name}")
                for image_file in image_files:
                    st.text(f"🖼️ Image: {image_file.name}")
    
    # Main content
    if not st.session_state.documents_loaded:
        st.info("👈 Click '📤 Load PDFs' in the sidebar to start")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            ### 📋 How to use:
            1. **Upload files**: Use the uploader in the sidebar for PDFs or images
            2. **Or add PDFs**: Copy your research papers to:
               ```
               RAG/uploaded_pdfs/
               ```
            3. **Load**: Click '📤 Load PDFs' or '📥 Upload & Ingest'
            4. **Ask**: Query the documents
            """)
        
        with col2:
            st.markdown(f"""
            ### 📁 PDF Location:
            ```
            {os.path.abspath(PDF_UPLOAD_FOLDER)}
            ```
            
            ### 📄 Supported Format:
            - PDF files (.pdf)
            - Images (.png, .jpg, .jpeg, .webp, .bmp, .tiff)
            
            ### 💡 Example files:
            - research_paper.pdf
            - document.pdf
            - screenshot.png
            - whitepaper.pdf
            """)
    else:
        # Query section
        st.subheader("❓ Ask a Question")
        
        # Show status
        st.info(f"📚 Vector store status: {'✓ Ready' if st.session_state.vector_store_initialized else '⚠ Not ready'}")

        if not st.session_state.chat_messages:
            st.caption("Start a conversation. Follow-up questions will reuse the recent turns above.")

        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        question = st.chat_input("Ask about the uploaded documents...")

        if question:
            st.session_state.chat_messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking with conversation memory..."):
                    try:
                        answer = st.session_state.rag_pipeline.query(question)
                        st.markdown(answer)
                        st.session_state.chat_messages.append({"role": "assistant", "content": answer})

                        with st.expander("📚 Show Retrieved Documents", expanded=True):
                            memory_aware_query = st.session_state.rag_pipeline._build_memory_aware_query(question)
                            retrieved_docs = st.session_state.rag_pipeline.retriever.retrieve(memory_aware_query)
                            retrieved_docs = st.session_state.rag_pipeline._rehydrate_image_documents(retrieved_docs)
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
                        error_message = f"Error processing query: {str(e)}"
                        st.error(error_message)
                        st.session_state.chat_messages.append({"role": "assistant", "content": error_message})

        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            if st.button("📊 Stats", use_container_width=True, key="stats_btn"):
                try:
                    st.session_state.rag_pipeline.print_statistics()
                    st.success("Statistics printed to console")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

        with col2:
            if st.button("🔄 Reinit", use_container_width=True, key="reinit_btn"):
                st.session_state.documents_loaded = False
                st.session_state.vector_store_initialized = False
                st.session_state.chat_messages = []
                st.session_state.rag_pipeline.clear_conversation_memory()
                st.rerun()

        with col3:
            if st.button("🧾 Show Memory", use_container_width=True, key="show_memory_btn"):
                history = st.session_state.rag_pipeline.get_conversation_history()
                if history:
                    st.json(history)
                else:
                    st.info("No conversation memory stored yet")

        st.subheader("💡 Example Queries")
        st.caption("Use these as starting points for a conversation:")
        
        examples = [
            "What are the main conclusions of this research?",
            "Can you explain the methodology in more detail?",
            "What datasets were used in this study?",
            "What are the limitations of this research?"
        ]

        example_cols = st.columns(2)
        for idx, example in enumerate(examples):
            with example_cols[idx % 2]:
                if st.button(example, key=f"example_{idx}", use_container_width=True):
                    st.session_state.chat_messages.append({"role": "user", "content": example})
                    answer = st.session_state.rag_pipeline.query(example)
                    st.session_state.chat_messages.append({"role": "assistant", "content": answer})
                    st.rerun()

if __name__ == "__main__":
    main()
