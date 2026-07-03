import numpy as np
import pickle
import logging
from src.similarity import cosine_similarity
from config.Config import TOP_K, MODEL_NAME
from src.embedding import load_embedding_model

def retrieval(query, embedding_model,records, top_k = TOP_K):
    """
    - query: user question
    - embedding_model: Vietnamese embedding model.
    - records: embedding.pkl
    - top_k: TOP_K (config)
    """
    query_embedding = embedding_model.encode(query)
    corpus = []
    metadata = []

    for record in records:

        for chunk_idx, embedding in enumerate(record["embeddings"]):
            corpus.append(embedding) #embedding cua records[i]
            metadata.append({
                "source": record["source"],
                "chunk": record["chunks"][chunk_idx]
            })
    scores = [cosine_similarity(query_embedding, cor) for cor in corpus]
    top_indices = np.argsort(scores)[::-1][:top_k]
    results = []
    for i in top_indices:
        results.append({
            "chunk": metadata[i]["chunk"],
            "score": scores[i],
            "source": metadata[i]["source"]
        })
    return results

# if __name__ == "__main__":
#     logging.basicConfig(
#         filename = "logs/app.log",
#         level=logging.INFO,
#         format="%(asctime)s - %(levelname)s - %(message)s"
#     )
#     logging.info("="*20 + "Logging retrieval.py" + "="*20)
#     query = "Server phân bổ tài nguyên như thế nào"
#     vector_store = "data/vector_store/embedding.pkl"
#     embedding_model = load_embedding_model(MODEL_NAME)
#     with open(vector_store, "rb") as file:
#         records = pickle.load(file)
#     logging.info(retrieval(query, embedding_model, records))