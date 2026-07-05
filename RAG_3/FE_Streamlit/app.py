import streamlit as st
import pickle
import os
import sys
from dotenv import load_dotenv

# Ensure the parent directory is in sys.path to import src and config modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Load configuration and project environment variables
load_dotenv(os.path.join(parent_dir, ".env"))

from src.embedding import load_embedding_model
from src.pipeline import rag_pipeline
from config.Config import MODEL_NAME, TOP_K, CHUNK_SIZE

# Page configuration
st.set_page_config(
    page_title="RAG Chatbot Assistant",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling using CSS (ChatGPT-like clean layout and font)
st.markdown("""
<style>
    /* Import ChatGPT-like font and apply it globally */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Gradient Title */
    .title-container {
        background: linear-gradient(90deg, #1f2937 0%, #111827 100%);
        padding: 2.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        border: 1px solid #374151;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .title-gradient {
        background: linear-gradient(45deg, #10B981, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 2.8rem;
        margin: 0;
    }
    .subtitle {
        color: #9CA3AF;
        font-size: 1.1rem;
        font-weight: 400;
        margin-top: 0.5rem;
    }
    /* Info cards */
    .info-card {
        background-color: #1f2937;
        border: 1px solid #374151;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    /* Custom status indicators */
    .status-ok {
        color: #10B981;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- Caching Resources -----------------
@st.cache_resource(show_spinner="Đang tải embedding model...")
def get_embedding_model():
    return load_embedding_model(MODEL_NAME)

@st.cache_resource(show_spinner="Đang tải dữ liệu vector store (Semantic Chunks)...")
def get_vector_store():
    # Loading semantic chunking database
    vector_store_path = os.path.join(parent_dir, "data/vector_store/embedding_semantic.pkl")
    if not os.path.exists(vector_store_path):
        raise FileNotFoundError(f"Không tìm thấy file vector store tại: {vector_store_path}")
    with open(vector_store_path, "rb") as f:
        return pickle.load(f)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar info
with st.sidebar:
    st.markdown("### ⚙️ Cấu hình hệ thống RAG")
    
    # Model details (Updated: Removed overlap and renamed chunk_size to Max Chunk Size)
    st.markdown("<div class='info-card'>", unsafe_allow_html=True)
    st.markdown(f"**Model Name:** `{MODEL_NAME}`")
    st.markdown(f"**Top K Retrieval:** `{TOP_K}`")
    st.markdown(f"**Max Chunk Size:** `{CHUNK_SIZE}`")
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Ingestion section for PDF/DOCX
    st.markdown("### 📁 Nạp tài liệu mới (.pdf, .docx)")
    uploaded_file = st.file_uploader("Tải tài liệu lên", type=["pdf", "docx"])
    if uploaded_file is not None:
        if st.button("🚀 Chạy Ingestion", use_container_width=True):
            with st.spinner("Đang xử lý nạp tài liệu..."):
                try:
                    raw_dir = os.path.join(parent_dir, "data/raw")
                    os.makedirs(raw_dir, exist_ok=True)
                    file_path = os.path.join(raw_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    from src.ingestion import process_and_add_document
                    model = get_embedding_model()
                    vector_store_path = os.path.join(parent_dir, "data/vector_store/embedding_semantic.pkl")
                    num_chunks = process_and_add_document(file_path, vector_store_path, model)
                    
                    st.cache_resource.clear()
                    st.success(f"Nạp thành công! Tạo ra {num_chunks} chunks.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Lỗi: {str(e)}")
                    
    st.markdown("---")
    
    # Status checker
    st.markdown("### 📊 Trạng thái kết nối")
    try:
        embedding_model = get_embedding_model()
        records = get_vector_store()
        st.markdown("🔹 Model Embedding: <span class='status-ok'>Đã tải</span>", unsafe_allow_html=True)
        st.markdown("🔹 Vector Store: <span class='status-ok'>Đã tải</span>", unsafe_allow_html=True)
        st.markdown(f"🔹 Tổng số chunks: **{len(records)}**")
    except Exception as e:
        st.error(f"Lỗi khởi tạo tài nguyên: {str(e)}")
        
    st.markdown("---")
    
    # Control actions
    if st.button("🗑️ Xoá lịch sử chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Header banner
st.markdown("""
<div class="title-container">
    <h1 class="title-gradient">RAG Chatbot Assistant</h1>
    <div class="subtitle">Giao diện Chatbot thông minh tích hợp Semantic RAG</div>
</div>
""", unsafe_allow_html=True)

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt_input := st.chat_input("Hỏi tôi về tài liệu của bạn..."):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt_input)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    
    # Generate chatbot response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Đang suy nghĩ và truy vấn thông tin..."):
            try:
                # Load assets cached
                model = get_embedding_model()
                data_records = get_vector_store()
                
                # Get RAG response
                response = rag_pipeline(prompt_input, data_records, model)
                
                message_placeholder.markdown(response)
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                err_msg = f"Đã xảy ra lỗi khi tạo câu trả lời: {str(e)}"
                message_placeholder.error(err_msg)
                st.session_state.messages.append({"role": "assistant", "content": err_msg})

