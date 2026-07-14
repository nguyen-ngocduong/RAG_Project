import os
import pickle
from src.loader import load_document
from src.clean_vietnamese_text import text_processing
from src.chunker.sematic import chunk_semantic
from config.Config import CHUNK_SIZE, VECTOR_DB_TYPE, VECTOR_DB_PERSIST_DIR, VECTOR_DB_COLLECTION, VECTOR_DB_SIZE
from src.vectorstore.chroma_store import ChromaStore
from src.vectorstore.qdrant_store import QdrantStore

def process_and_add_document(file_path, embedding_model):
    """
    Quy quy trinh tu dong hoa nạp tai lieu (.pdf, .docx):
    Doc file -> Lam sach -> Semantic Chunking -> Embedding -> Cap nhat/Gop vao database.
    """
    # 1. Doc du lieu tu file (tu dong nhan dien .pdf hoac .docx)
    raw_blocks = load_document(file_path)
    full_text = "\n\n".join(raw_blocks)
    
    # 2. Tien xu ly / lam sach van ban
    cleaned_text = text_processing(full_text)
    
    # 3. Chia nho theo ngu nghia (Semantic Chunking)
    chunks = chunk_semantic(cleaned_text, embedding_model=embedding_model, max_chunk_size=CHUNK_SIZE)
    
    if not chunks:
        raise ValueError("Tài liệu rỗng hoặc không trích xuất được chunk nào.")
        
    # 4. Sinh vector embedding cho cac chunks moi
    doc_id = int(os.path.getmtime(file_path)) # dung timestamp lam ID doc nhat
    embeddings = embedding_model.encode(chunks, show_progress_bar=False)
    
    # 5. Khởi tạo VectorStore dựa trên Config
    if VECTOR_DB_TYPE.lower() == "chroma":
        db = ChromaStore(persist_directory=VECTOR_DB_PERSIST_DIR, collection_name=VECTOR_DB_COLLECTION)
    elif VECTOR_DB_TYPE.lower() == "qdrant":
        db = QdrantStore(persist_directory=VECTOR_DB_PERSIST_DIR, collection_name=VECTOR_DB_COLLECTION, vector_size=VECTOR_DB_SIZE)
    else:
        raise ValueError(f"Loại VectorDB không được hỗ trợ: {VECTOR_DB_TYPE}")

    # Tạo metadata cho từng chunk
    metadata = [
        {
            "document_id": doc_id,
            "source": os.path.basename(file_path),
            "chunk_size": CHUNK_SIZE,
        }
        for _ in chunks
    ]
    
    # 6. Thêm vào vector database
    db.add_documents(chunks=chunks, embeddings=embeddings, metadata=metadata)
        
    return len(chunks)
