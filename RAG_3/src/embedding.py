from sentence_transformers import SentenceTransformer

def load_embedding_model(model_name):
  model = SentenceTransformer(model_name)
  return model