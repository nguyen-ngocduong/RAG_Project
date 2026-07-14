from sentence_transformers import SentenceTransformer
from config.Config import MODEL_NAME
from pyvi.ViTokenizer import tokenize
import pickle

def load_embedding_model(model_name):
  model = SentenceTransformer(model_name)
  return model

def embedding_chunk(embedding_model, chunks):
  embedded_chunks = []
  for doc in chunks:
    tokenized_chunks = [tokenize(sentence) for sentence in doc["chunks"]]
    embeddings = embedding_model.encode(
      tokenized_chunks, 
      show_progress_bar=True
    )
    embedded_chunks.append({
        "document_id": doc["document_id"],
        "source": doc["source"],
        "chunk_size": doc["max_chunk_size"],
        #"chunk_overlap": doc["chunk_overlap"],
        "chunks": doc["chunks"],
        "embeddings": embeddings
    })
  return embedded_chunks

if __name__ == "__main__":
  # Load chunks đã xử lý
  processed_path = "data/processed/semantic_chunk.pkl"
  vector_store = "data/vector_store/embedding_semantic.pkl"
  
  print("Đang load chunks...")
  with open(processed_path, "rb") as file:
    semantic_chunk = pickle.load(file)
  
  print(f"Đã load {len(semantic_chunk)} documents")
  print("Đang tải model embedding...")
  embedding_model = load_embedding_model(MODEL_NAME)
  
  print("Đang embedding chunks...")
  embedded_chunks = embedding_chunk(embedding_model, semantic_chunk)
  
  print("Đang lưu embeddings...")
  with open(vector_store, "wb") as f:
    pickle.dump(embedded_chunks, f)

  print("Done!")