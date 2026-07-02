from src.loader import load_pdf
from src.clean_vietnamese_text import text_processing
from src.chunker.fixed_size import chunk_fixed_size
from src.chunker.recursive import chunk_recursive
from config.Config import CHUNK_OVERLAP, CHUNK_SIZE
from src.embedding import load_embedding_model
from pyvi.ViTokenizer import tokenize
import logging
import pickle

logging.basicConfig(
    filename = "logs/app.log",
    level=logging.INFO,
    format = "%(asctime)s - %(levelname)s - %(message)s"
    )
logging.info("="*20 + "Logging Start" + "="*20)
pdf_path = ["data/raw/AIO2205_Grafana-Prometheus-Monitoring.pdf", "data/raw/M6W1D5_Evaluation_Metrics.pdf"]
vector_store = "data/vector_store/embedding.pkl"
all_docs = [load_pdf(path) for path in pdf_path]
print(len(all_docs[0]))
# load recursive_chunk.pkl
# with open (processed_path, "rb") as file:
#     recursive_chunk = pickle.load(file)

#print(recursive_chunk[0]["chunks"])
#embeddings
with open(vector_store, "rb") as file:
    embedding_chunk = pickle.load(file)

logging.info(embedding_chunk)