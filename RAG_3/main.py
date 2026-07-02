from src.loader import load_pdf
from src.clean_vietnamese_text import text_processing
from src.chunker.fixed_size import chunk_fixed_size
from src.chunker.recursive import chunk_recursive
from config.Config import CHUNK_OVERLAP, CHUNK_SIZE
import logging

logging.basicConfig(
    filename = "logs/app.log",
    level=logging.INFO,
    format = "%(asctime)s - %(levelname)s - %(message)s"
    )
logging.info("="*10 + "Logging Start" + "="*10)
pdf_path = ["data/raw/AIO2205_Grafana-Prometheus-Monitoring.pdf", "data/raw/M6W1D5_Evaluation_Metrics.pdf"]
all_docs = [load_pdf(path) for path in pdf_path]
print(len(all_docs[0]))
processed_text = []
for i in range(len(all_docs)):
    if i == 0:
        docs = [page for j, page in enumerate(all_docs[i]) if j != 2]
    else: docs = all_docs[i]
    text = "\n\n".join(docs)
    processed_text.append(text_processing(text))
# print(len(processed_text)) # 2
#print(len(processed_text[0].split())) #4592 
# ================================
# Chunking
# ================================

# Su dung fixed size
chunks1 = []
for text in processed_text:
    chunks_fixed_size = chunk_fixed_size(text, CHUNK_SIZE, CHUNK_OVERLAP)
    chunks1.append(chunks_fixed_size)
# Su dung recursive chunk
chunks2 = []
for text in processed_text:
    chunks_recursive = chunk_recursive(text)
    chunks2.append(chunks_recursive)
# print(len(chunks1[0])) 29
# print(len(chunks2[0])) 31