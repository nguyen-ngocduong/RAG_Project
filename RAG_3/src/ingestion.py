import os
import pickle
from src.loader import load_document
from src.clean_vietnamese_text import text_processing
from src.chunker.sematic import chunk_semantic
from config.Config import CHUNK_SIZE

def process_and_add_document(file_path, vector_store_path, embedding_model):
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
    
    new_embedded_record = {
        "document_id": doc_id,
        "source": os.path.basename(file_path),
        "chunk_size": CHUNK_SIZE,
        "chunks": chunks,
        "embeddings": embeddings
    }
    
    # 5. Gop vao co so du lieu vector store da co
    existing_records = []
    if os.path.exists(vector_store_path):
        try:
            with open(vector_store_path, "rb") as f:
                existing_records = pickle.load(f)
                if not isinstance(existing_records, list):
                    existing_records = [existing_records]
        except Exception as e:
            # Truong hop loi khi doc thi xem nhu bat dau moi
            existing_records = []
            
    # Them record moi vao va ghi de file vector store
    existing_records.append(new_embedded_record)
    
    with open(vector_store_path, "wb") as f:
        pickle.dump(existing_records, f)
        
    return len(chunks)
