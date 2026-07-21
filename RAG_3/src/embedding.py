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