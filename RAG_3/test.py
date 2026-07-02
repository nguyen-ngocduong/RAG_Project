from sentence_transformers import SentenceTransformer
from pyvi.ViTokenizer import tokenize
from src.embedding import load_embedding_model
sentences = ["Hà Nội là thủ đô của Việt Nam", "Đà Nẵng là thành phố du lịch"]
tokenizer_sent = [tokenize(sent) for sent in sentences]

embedding_model = load_embedding_model("dangvantuan/vietnamese-embedding")
embeddings = embedding_model.encode(tokenizer_sent)
print(embeddings)
