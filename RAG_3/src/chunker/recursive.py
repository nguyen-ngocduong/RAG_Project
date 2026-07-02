from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.clean_vietnamese_text import text_processing
from config.Config import CHUNK_SIZE, CHUNK_OVERLAP
from src.loader import load_pdf
import logging
import pickle

def splitter(separators):
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index = True, # Lưu lại vị trí ban đầu mỗi Chunk
        strip_whitespace = True, # loại bỏ khoảng trắng ở đầu, cuối
        separators = separators # Các separator để chia nhỏ tài liệu
    )

def chunk_recursive(docs):
    """
    - Đầu tiên, cố gắng chia văn bản dựa trên các dấu phân tách có sẵn 
    - Nếu một chunk được tạo ra vẫn lớn hơn giới hạn kích thước định trước, 
      tiếp tục áp dụng quy trình chia nhỏ một cách đệ quy với các dấu phân 
      tách cấp thấp hơn cho đến khi chunk đó đạt kích thước mong muốn.
    - Nếu một chunk đã nhỏ hơn giới hạn kích thước, giữ nguyên nó.
    """
    separators = ["\n\n", "\n", " ", ""] # "\n\n" (đoạn văn), "\n" (dòng)
    text_splitter = splitter(separators)
    chunks = text_splitter.split_text(docs)
    return chunks

if __name__ == "__main__":

    processed_path = "data/processed/recursive_chunk.pkl"
    logging.basicConfig(
        filename = "logs/app.log",
        level=logging.INFO,
        format = "%(asctime)s - %(levelname)s - %(message)s"
        )
    logging.info("="*20 + "Logging file recursive.py" + "="*20)
    pdf_path = ["data/raw/AIO2205_Grafana-Prometheus-Monitoring.pdf", "data/raw/M6W1D5_Evaluation_Metrics.pdf"]
    all_docs = [load_pdf(path) for path in pdf_path]
    processed_text = []
    for i in range(len(all_docs)):
        if i == 0:
            docs = [page for j, page in enumerate(all_docs[i]) if j != 2]
        else: docs = all_docs[i]
        text = "\n\n".join(docs)
        processed_text.append(text_processing(text))
    chunks = []
    for text in processed_text:
        chunks_recursive = chunk_recursive(text)
        chunks.append(chunks_recursive)
    # print(len(chunks))
    # print(len(chunks[0]))
    # logging.info(chunks[0])
    fixed_chunk = []
    for i in range(len(pdf_path)):
        fixed_chunk.append(
            {
                "document_id": i,
                "source": pdf_path[i],
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
                "chunks": chunks[i]
            }
        )
    with open(processed_path, "wb") as file:
        pickle.dump(fixed_chunk, file)