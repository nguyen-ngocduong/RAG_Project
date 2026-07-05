import re
import logging
import numpy as np
import pickle
from src.loader import load_pdf
from src.clean_vietnamese_text import text_processing
from src.embedding import load_embedding_model
from src.similarity import cosine_similarity
from config.Config import MODEL_NAME,CHUNK_SIZE

def split_into_sentences(text):
    """
    Bước 1: Phân đoạn văn bản thành các câu.
    - Chia dựa trên dấu câu kết thúc: . ? !
    - Loại bỏ câu rỗng hoặc chỉ chứa khoảng trắng.
    """
    raw_sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in raw_sentences if s.strip()]
    return sentences


def split_long_sentence(sentence, max_chunk_size, length_fn):
    """
    Cắt nhỏ MỘT câu quá dài (dài hơn max_chunk_size) thành nhiều mảnh nhỏ hơn,
    cắt theo từ (word) để không phá vỡ từ giữa chừng.

    Đây là lớp an toàn cho trường hợp: 1 "câu" (theo dấu . ? !) vẫn vượt quá
    CHUNK_SIZE - ví dụ câu liệt kê dài, câu thiếu dấu ngắt, bảng số liệu dính
    liền thành 1 câu, v.v.
    """
    words = sentence.split()
    pieces = []
    current_words = []
    current_len = 0

    for w in words:
        w_len = length_fn(w) + 1  # +1 cho khoảng trắng nối từ
        if current_words and current_len + w_len > max_chunk_size:
            pieces.append(" ".join(current_words))
            current_words = [w]
            current_len = length_fn(w)
        else:
            current_words.append(w)
            current_len += w_len

    if current_words:
        pieces.append(" ".join(current_words))

    return pieces


def enforce_max_size(sentences, max_chunk_size, length_fn):
    """
    Tiền xử lý: duyệt qua danh sách câu, câu nào vượt quá max_chunk_size thì
    cắt nhỏ ra thành nhiều đơn vị con trước khi đưa vào thuật toán chunking.
    """
    if max_chunk_size is None:
        return sentences

    result = []
    for s in sentences:
        if length_fn(s) > max_chunk_size:
            result.extend(split_long_sentence(s, max_chunk_size, length_fn))
        else:
            result.append(s)
    return result


def get_chunk_embedding(chunk_sentences, chunk_sentence_embeddings, embedding_model, method="mean"):
    """
    Tạo/cập nhật embedding đại diện cho chunk đã tích lũy.

    method="mean": trung bình cộng embedding các câu trong chunk (nhanh, mặc định).
    method="reencode": encode lại toàn bộ đoạn văn đã gộp (sát ngữ nghĩa hơn, chậm hơn).
    """
    if method == "mean":
        return np.mean(chunk_sentence_embeddings, axis=0)
    elif method == "reencode":
        full_text = " ".join(chunk_sentences)
        return embedding_model.encode([full_text], show_progress_bar=False)[0]
    else:
        raise ValueError(f"Unknown method: {method}")


def chunk_semantic(
    text,
    embedding_model=None,
    threshold=0.6,
    max_chunk_sentences=None,
    max_chunk_size=CHUNK_SIZE,
    length_fn=len,
    embedding_method="mean",
):
    """
    Semantic Chunking - so sánh CHUNK ĐÃ TÍCH LŨY với CÂU MỚI.

    Args:
        text (str): văn bản đầu vào.
        embedding_model: model dùng để encode. None -> tự động load bằng MODEL_NAME.
        threshold (float): ngưỡng cosine similarity để gộp/tách chunk (0.7-0.85 khuyến nghị,
                            mặc định giữ 0.6 theo pipeline hiện tại).
        max_chunk_sentences (int|None): giới hạn số câu tối đa trong 1 chunk.
        max_chunk_size (int|None): giới hạn kích thước tối đa của 1 chunk, đơn vị theo
                            length_fn (mặc định là số KÝ TỰ). Mặc định lấy từ CHUNK_SIZE
                            trong config. Truyền None để tắt kiểm soát này.
        length_fn (callable): hàm đo "độ dài" của 1 chuỗi. Mặc định `len` (đếm ký tự).
                            Muốn đo theo TOKEN thay vì ký tự, truyền vào 1 hàm đếm token,
                            ví dụ: `lambda s: len(embedding_model.tokenizer.encode(s))`
                            hoặc dùng thư viện `tiktoken`.
        embedding_method (str): "mean" hoặc "reencode" - cách tính embedding chunk tích lũy.

    Returns:
        List[str]: danh sách các chunk văn bản, không chunk nào vượt quá max_chunk_size.
    """
    # ----- Bước 1: Phân đoạn văn bản thành các câu -----
    sentences = split_into_sentences(text)

    if not sentences:
        return []

    # ----- Lớp an toàn: đảm bảo không câu nào tự nó đã vượt quá max_chunk_size -----
    sentences = enforce_max_size(sentences, max_chunk_size, length_fn)

    if len(sentences) == 1:
        return [sentences[0]]

    if embedding_model is None:
        embedding_model = load_embedding_model(MODEL_NAME)

    sentence_embeddings = embedding_model.encode(sentences, show_progress_bar=False)

    # ----- Bước 2: Khởi tạo chunk đầu tiên -----
    current_chunk = [sentences[0]]
    current_chunk_embeddings = [sentence_embeddings[0]]
    current_chunk_embedding = get_chunk_embedding(
        current_chunk, current_chunk_embeddings, embedding_model, embedding_method
    )

    chunks = []

    # ----- Bước 3 & 4: Duyệt từng câu tiếp theo -----
    for i in range(1, len(sentences)):
        sentence_embedding = sentence_embeddings[i]
        sentence_text = sentences[i]

        # So sánh embedding của CHUNK hiện tại với embedding của CÂU mới
        similarity = cosine_similarity(current_chunk_embedding, sentence_embedding)

        # Điều kiện dừng do đầy số câu
        chunk_is_full_by_count = (
            max_chunk_sentences is not None
            and len(current_chunk) >= max_chunk_sentences
        )

        # Điều kiện dừng do vượt quá kích thước cho phép (ký tự/token)
        would_be_text = " ".join(current_chunk + [sentence_text])
        chunk_would_exceed_size = (
            max_chunk_size is not None
            and length_fn(would_be_text) > max_chunk_size
        )

        if similarity >= threshold and not chunk_is_full_by_count and not chunk_would_exceed_size:
            # --- Gộp câu mới vào chunk hiện tại ---
            current_chunk.append(sentence_text)
            current_chunk_embeddings.append(sentence_embedding)
            current_chunk_embedding = get_chunk_embedding(
                current_chunk, current_chunk_embeddings, embedding_model, embedding_method
            )
        else:
            # --- Kết thúc chunk hiện tại (do lệch ngữ nghĩa HOẶC đầy kích thước) ---
            chunks.append(" ".join(current_chunk))
            # --- Bắt đầu chunk mới với câu hiện tại ---
            current_chunk = [sentence_text]
            current_chunk_embeddings = [sentence_embedding]
            current_chunk_embedding = get_chunk_embedding(
                current_chunk, current_chunk_embeddings, embedding_model, embedding_method
            )

    # ----- Bước 5: Xử lý chunk cuối cùng còn dang dở -----
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


if __name__ == "__main__":
    processed_path = "data/processed/semantic_chunk.pkl"
    logging.basicConfig(
        filename = "logs/app.log",
        level=logging.INFO,
        format = "%(asctime)s - %(levelname)s - %(message)s"
        )
    logging.info("="*20 + "Logging file sematic.py" + "="*20)
    pdf_path = ["data/raw/AIO2205_Grafana-Prometheus-Monitoring.pdf", "data/raw/M6W1D5_Evaluation_Metrics.pdf"]
    all_docs = [load_pdf(path) for path in pdf_path]
    processed_text = []
    for i in range(len(all_docs)):
        if i == 0:
            docs = [page for j, page in enumerate(all_docs[i]) if j != 2]
        else: docs = all_docs[i]
        text = "\n\n".join(docs)
        processed_text.append(text_processing(text))
    print(str(len(processed_text[0])) + " | " + str(len(processed_text[1])))
    chunks = []

    # Sử dụng default threshold = 0.6
    for text in processed_text:
        chunks_recursive = chunk_semantic(text)
        chunks.append(chunks_recursive)

    print(str(len(chunks[0])) + " | " + str(len(chunks[1])))
    logging.info(chunks[0])
    fixed_chunk = []
    for i in range(len(pdf_path)):
        fixed_chunk.append(
            {
                "document_id": i,
                "source": pdf_path[i],
                "max_chunk_size": CHUNK_SIZE,
                "chunks": chunks[i]
            }
        )
    with open(processed_path, "wb") as file:
        pickle.dump(fixed_chunk, file)