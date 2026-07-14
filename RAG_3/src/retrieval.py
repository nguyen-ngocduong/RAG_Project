import numpy as np
import logging
from config.Config import TOP_K, MODEL_NAME, VECTOR_DB_TYPE, VECTOR_DB_PERSIST_DIR, VECTOR_DB_COLLECTION, VECTOR_DB_SIZE
from src.embedding import load_embedding_model
from src.vectorstore.chroma_store import ChromaStore
from src.vectorstore.qdrant_store import QdrantStore

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

    results = db.search(query_embedding=query_embedding, top_k=top_k)
    return results

if __name__ == "__main__":
    logging.basicConfig(
        filename = "logs/app.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logging.info("="*20 + "Logging retrieval.py" + "="*20)
    query = "hay neu hien tuong deadlock"
    embedding_model = load_embedding_model(MODEL_NAME)
    
    results = retrieval(query, embedding_model)
    logging.info(results)