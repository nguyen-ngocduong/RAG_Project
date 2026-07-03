import pickle
from src.embedding import load_embedding_model
from src.pipeline import rag_pipeline
from config.Config import MODEL_NAME
vector_store = "data/vector_store/embedding.pkl"
embedding_model = load_embedding_model(MODEL_NAME)
with open(vector_store,"rb") as f:
    records = pickle.load(f)

while True:
    query = input("Nhập câu hỏi: ")
    if query.lower() == "exit":
        break
    answer = rag_pipeline(query,records,embedding_model)
    print(answer)