from src.embedding import load_embedding_model
from src.pipeline import rag_pipeline
from config.Config import MODEL_NAME

embedding_model = load_embedding_model(MODEL_NAME)

while True:
    query = input("Nhập câu hỏi: ")
    if query.lower() == "exit":
        break
    answer = rag_pipeline(query, embedding_model)
    print(answer)