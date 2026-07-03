from src.loader import load_pdf
from config.Config import CHUNK_OVERLAP, CHUNK_SIZE
from src.clean_vietnamese_text import text_processing
import logging
import pickle

def chunk_fixed_size(text, chunk_size = CHUNK_SIZE, overlap = CHUNK_OVERLAP):
    """
    Chúng ta chia văn bản thành các đoạn có kích thước đồng nhất 
    dựa trên số lượng ký tự, từ hoặc token được xác định trước.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap  # Di chuyển vị trí bắt đầu cho đoạn tiếp theo
    return chunks

# if __name__ == "__main__":
#     processed_path = "data/processed/fixed_chunk.pkl"
#     logging.basicConfig(
#         filename = "logs/app.log",
#         level=logging.INFO,
#         format = "%(asctime)s - %(levelname)s - %(message)s"
#         )
#     logging.info("="*20 + "Logging file fixed_size.py" + "="*20)
#     pdf_path = ["data/raw/AIO2205_Grafana-Prometheus-Monitoring.pdf", "data/raw/M6W1D5_Evaluation_Metrics.pdf"]
#     all_docs = [load_pdf(path) for path in pdf_path]
#     processed_text = []
#     for i in range(len(all_docs)):
#         if i == 0:
#             docs = [page for j, page in enumerate(all_docs[i]) if j != 2]
#         else: docs = all_docs[i]
#         text = "\n\n".join(docs)
#         processed_text.append(text_processing(text))
#     chunks = []
#     for text in processed_text:
#         chunks_recursive = chunk_fixed_size(text)
#         chunks.append(chunks_recursive)
#     # print(len(chunks))
#     # print(len(chunks[0]))
#     # logging.info(chunks[0])
#     fixed_chunk = []
#     for i in range(len(pdf_path)):
#         fixed_chunk.append(
#             {
#                 "document_id": i,
#                 "source": pdf_path[i],
#                 "chunk_size": CHUNK_SIZE,
#                 "chunk_overlap": CHUNK_OVERLAP,
#                 "chunks": chunks[i]
#             }
#         )
#     with open(processed_path, "wb") as file:
#         pickle.dump(fixed_chunk, file)