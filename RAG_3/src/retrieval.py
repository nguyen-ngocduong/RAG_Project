import numpy as np
import logging
from config.Config import TOP_K, MODEL_NAME, VECTOR_DB_TYPE, VECTOR_DB_PERSIST_DIR, VECTOR_DB_COLLECTION, VECTOR_DB_SIZE
from src.embedding import load_embedding_model
from src.vectorstore.chroma_store import ChromaStore
from src.vectorstore.qdrant_store import QdrantStore
from src.ingestion import process_and_add_document
logger = logging.getLogger(__name__)

def retrieval(query, embedding_model, top_k = TOP_K):
    """
    - query: user question
    - embedding_model: Vietnamese embedding model.
    - top_k: TOP_K (config)
    """
    query_embedding = embedding_model.encode(query)

    if VECTOR_DB_TYPE.lower() == "chroma":
        db = ChromaStore(persist_directory=VECTOR_DB_PERSIST_DIR, collection_name=VECTOR_DB_COLLECTION)
    elif VECTOR_DB_TYPE.lower() == "qdrant":
        db = QdrantStore(persist_directory=VECTOR_DB_PERSIST_DIR, collection_name=VECTOR_DB_COLLECTION, vector_size=VECTOR_DB_SIZE)
    else:
        raise ValueError(f"Loại VectorDB không được hỗ trợ: {VECTOR_DB_TYPE}")

    # Kiểm tra thông tin collection (hỗ trợ debug)
    if hasattr(db, 'get_collection_info'):
        info = db.get_collection_info()
        logger.info("Collection '%s' có %d điểm dữ liệu", info['collection_name'], info['total_points'])
        if info['total_points'] == 0:
            logger.warning("Collection rỗng! Bạn cần chạy ingestion trước.")
    
    results = db.search(query_embedding=query_embedding, top_k=top_k)

    if not results:
        logger.warning("Không tìm thấy kết quả nào cho query: '%s'", query)
    else:
        for i, r in enumerate(results):
            logger.info("  [%d] score=%.4f | source=%s", i, r['score'], r['source'])

    return results

if __name__ == "__main__":
    logging.basicConfig(
        #filename = "logs/app.log",
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    logging.info("="*20 + " Logging retrieval.py " + "="*20)
    query = "hay neu hien tuong deadlock"
    logging.info("Query: '%s'", query)
    embedding_model = load_embedding_model(MODEL_NAME)
    process_and_add_document(file_path="data/raw/AIO2205_Grafana-Prometheus-Monitoring.pdf", embedding_model=embedding_model)
    results = retrieval(query, embedding_model)
    logging.info("Số kết quả: %d", len(results))
    if results:
        for i, r in enumerate(results):
            logging.info("  [%d] score=%.4f | chunk=%.100s...", i, r['score'], r['chunk'])
    else:
        logging.warning("Không tìm thấy kết quả! Kiểm tra lại collection.")