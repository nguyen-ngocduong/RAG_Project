from sentence_transformers import SentenceTransformer
# Sử dụng mô hình đa ngôn ngữ hoặc chuyên Việt
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

def get_embedding(text):
    return model.encode(text)
