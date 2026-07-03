from sentence_transformers import SentenceTransformer
import pickle

def load_embedding_model(model_name):
  model = SentenceTransformer(model_name)
  return model

def embedding_chunk(embedding_model, chunks):
  embedded_fixed = []
  for doc in chunks:
    embeddings = embedding_model.encode(
      doc["chunks"], show_progress_bar =  True
    )
    embedded_fixed.append({
        "document_id": doc["document_id"],
        "source": doc["source"],
        "chunk_size": doc["chunk_size"],
        "chunk_overlap": doc["chunk_overlap"],
        "chunks": doc["chunks"],
        "embeddings": embeddings
    })
  return embedded_fixed

# if __name__== "__main__":
#   # Load model embedding -> embedding
#   processed_path = "data/processed/recursive_chunk.pkl"
#   vector_store = "data/vector_store/embedding.pkl"
#   with open(processed_path, "rb") as file:
#     recursive_chunk = pickle.load(file)
#   print("Đang tải model embedding...")
#   model_name = 'BAAI/bge-m3'
#   embedding_model = load_embedding_model(model_name)
#   #embedding_model.max_seq_length = 256
#   embedded_fixed = embedding_chunk(embedding_model, recursive_chunk)
#   with open(vector_store, "wb") as f:
#     pickle.dump(embedded_fixed, f)

#   print("!Done")