import pickle
from src.embedding import load_embedding_model
from config.Config import MODEL_NAME
from src.similarity import cosine_similarity
import logging

logging.basicConfig(
    filename ="logs/app.log",
    level = logging.INFO,
    format = "%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("="*20 + "Logging test.py" + "="*20)

embedding_path = "data/vector_store/embedding.pkl"
with open(embedding_path, "rb") as file:
    records = pickle.load(file)
print("Đang load model...")
query = "Server phân bổ tài nguyên như thế nào"
embedding_model = load_embedding_model(MODEL_NAME)
query_embedding = embedding_model.encode(query)
print("In ra vector embedding cua query\n")
print(query_embedding.shape)
print("In ra list records\n")
print(records[0]["embeddings"][0].shape)

logging.info(cosine_similarity(query_embedding, records[0]["embeddings"][0]))