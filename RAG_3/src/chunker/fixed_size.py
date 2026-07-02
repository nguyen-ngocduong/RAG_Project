from src.loader import load_pdf
from config.Config import CHUNK_OVERLAP, CHUNK_SIZE
import logging
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
    
#     logging.basicConfig(
#         filename = "logs/app.log",
#         level = logging.INFO,
#         format="%(asctime)s - %(levelname)s - %(message)s"
#     )
#     logging.info("="*20 + "Log file fixed_size.py" + "="*20) 
#     pdf_path = ["data/raw/AIO2205_Grafana-Prometheus-Monitoring.pdf", "data/raw/M6W1D5_Evaluation_Metrics.pdf"]
#     all_documents = [load_pdf(path) for path in pdf_path]
#     print(len(all_documents[0]))
#     #logging.info("\n" + docs[0][0])
#     chunks = [chunk_fixed_size(document) for documents in all_documents for document in documents]
#     print(len(chunks))
#     logging.info(chunks[0:5])