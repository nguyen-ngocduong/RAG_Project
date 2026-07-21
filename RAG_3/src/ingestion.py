import os
from config.Config import (
    MAX_CHUNK_SIZE, VECTOR_DB_TYPE, VECTOR_DB_PERSIST_DIR, 
    VECTOR_DB_COLLECTION, VECTOR_DB_SIZE, MIN_CHUNK_SIZE, MODEL_NAME
)
from src.vectorstore.chroma_store import ChromaStore
from src.vectorstore.qdrant_store import QdrantStore
from src.chunker.sematic import SemanticChunk


def process_and_add_document(file_path, embedding_model):
    """
    Quy quy trinh tu dong hoa nạp tai lieu (.pdf, .docx):
    Doc file -> Lam sach -> Semantic Chunking -> Embedding -> Cap nhat/Gop vao database.
    """
    chunker = SemanticChunk(
        output_path="data/processed",
        input_path="data/raw",
    )
    chunk_records = chunker.run(save_file=True)
    source_name = os.path.basename(file_path)
    chunk_records = [
        chunk for chunk in chunk_records
        if chunk["source"] == source_name
    ]
    if not chunk_records:
        raise ValueError("Tài liệu rỗng hoặc không trích xuất được chunk nào.")

    chunks = [chunk["text"] for chunk in chunk_records]

    # 4. Sinh vector embedding cho cac chunks moi
    doc_id = int(os.path.getmtime(file_path)) # dung timestamp lam ID doc nhat
    embeddings = embedding_model.encode(chunks, show_progress_bar=True)
    
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
            "source": chunk["source"],
            "chunk_id": chunk["chunk_id"],
            "max_chunk_size": MAX_CHUNK_SIZE,
            "min_chunk_size": MIN_CHUNK_SIZE,
            "model_name": MODEL_NAME,
        }
        for chunk in chunk_records
    ]
    
    # 6. Thêm vào vector database
    db.add_documents(chunks=chunks, embeddings=embeddings, metadata=metadata)
        
    return len(chunks)
