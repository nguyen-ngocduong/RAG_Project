from src.loader import load_pdf
import logging
def chunk_semantic(text):
    """
    Thay vì cắt văn bản theo 1000 ký tự hoặc 500 token, 
    ta cắt theo ý nghĩa cua cau hoac cat theo tung paragraph
    """
    chunks = [p.strip() for p in text.split("\n\n") if p.strip()]
    return chunks
if __name__ == "__main__":
        
    logging.basicConfig(
        filename = "logs/app.log",
        level = logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logging.info("="*20 + "Log file sematic.py" + "="*20) 
    pdf_path = ["data/raw/AIO2205_Grafana-Prometheus-Monitoring.pdf", "data/raw/M6W1D5_Evaluation_Metrics.pdf"]
    all_documents = [load_pdf(path) for path in pdf_path]
    print(len(all_documents[0]))
    #logging.info("\n" + docs[0][0])
    chunks = [chunk_semantic(document) for documents in all_documents for document in documents]
    print(len(chunks))
    logging.info(chunks[0:5])