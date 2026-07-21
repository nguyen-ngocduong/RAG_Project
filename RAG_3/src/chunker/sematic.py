"""
Chiến lược chia chunk được thể hiện ở file chunker.py
- Chia nhỏ văn bản thành các semantic chunks tối ưu cho Embedding & Vector Search, đảm bảo mỗi chunk giữ mạch nghĩa tự nhiên thay vì cắt cứng theo độ dài.
1. Split (đơn vị gốc trước)
- Tách văn bản thành các đơn vị nhỏ ban đầu (thường là câu/đoạn ngắn).
- Mỗi đơn vị được embedding độc lập để chuẩn bị cho bước gom ngữ nghĩa.
- Mục tiêu: tạo “hạt cơ sở” đủ nhỏ để hệ thống quyết định ranh giới theo meaning.
2. Merge Cộng dồn theo ngữ nghĩa (Cumulative Semantic Merge)
- Duyệt tuần tự từng đơn vị.
- So sánh độ tương đồng ngữ nghĩa giữa:
    + chunk hiện tại (đã cộng dồn), và đơn vị kế tiếp.
- Nếu similarity còn đủ cao (>= ngưỡng), tiếp tục gộp vào cùng chunk.
- Nếu similarity giảm dưới ngưỡng, đóng chunk hiện tại và mở chunk mới.
=> Mục tiêu: chunk dài/ngắn linh hoạt theo dòng ý thực tế của nội dung.
3. Split lại khi vượt ngưỡng kích thước (Guardrail Split)
- Nếu chunk sau khi cộng dồn quá dài (> max_chunk_size), tách nhỏ lại bằng bộ tách đệ quy (ví dụ RecursiveCharacterTextSplitter) hoặc quy tắc phân đoạn mềm.
- Có thể dùng overlap nhỏ để giữ ngữ cảnh liên tục.
=> Mục tiêu: giữ chunk trong giới hạn phù hợp với model embedding và truy hồi.
"""
from config.Config import (
    MAX_CHUNK_SIZE, MIN_CHUNK_SIZE, 
    THRESHOLD, CHUNK_OVERLAP, MODEL_NAME
    )
import logging
import os
import pickle
import re
from pathlib import Path

from src.loader import load_document
from src.clean_vietnamese_text import text_processing
from src.similarity import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Can chay: pip install sentence-transformers "
        "de su dung buoc embedding trong SemanticChunk."
    ) from exc

logging.basicConfig(
    filename= "logs/app.log",
    level=logging.INFO,
    format= "%(asctime)s | %(levelname)s | %(message)s"
)

class SemanticChunk:
    # Model embedding mac dinh danh cho tieng Viet.
    # Co the thay bang model embedding khac tuy nhu cau (vd: OpenAI, BGE-M3, ...).
    DEFAULT_EMBEDDING_MODEL = MODEL_NAME

    def __init__(
        self,
        output_path: str | Path,
        input_path: str | Path = "data/raw",
        max_chunk_size: int = MAX_CHUNK_SIZE,
        min_chunk_size: int = MIN_CHUNK_SIZE,
        threshold: float = THRESHOLD,
        chunk_overlap: int = CHUNK_OVERLAP,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        self.output_path = str(output_path)
        self.input_path = str(input_path)
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.threshold = threshold
        self.chunk_overlap = chunk_overlap

        os.makedirs(self.output_path, exist_ok=True)

        logging.info("Dang load embedding model: %s", embedding_model_name)
        self.embedding_model = SentenceTransformer(embedding_model_name)

    # ------------------------------------------------------------------
    # Load & tien xu ly
    # ------------------------------------------------------------------
    def load(self) -> list[tuple[str, str]]:
        """
        Load tat ca file tu input_path, tra ve list (ten_file, van_ban_da_lam_sach).
        """
        file_paths = [
            os.path.join(self.input_path, filename)
            for filename in os.listdir(self.input_path)
            if os.path.isfile(os.path.join(self.input_path, filename))
        ]

        documents = []
        for file_path in file_paths:
            raw = load_document(file_path)
            # load_document co the tra ve list (cac trang pdf / cac block docx)
            # => noi lai thanh 1 van ban duy nhat truoc khi lam sach.
            full_text = "\n".join(raw) if isinstance(raw, list) else raw
            clean_text = text_processing(full_text)
            documents.append((os.path.basename(file_path), clean_text))

        return documents

    # ------------------------------------------------------------------
    # Buoc 1: Split thanh cau
    # ------------------------------------------------------------------
    def split_into_sentence(self, text: str) -> list[str]:
        """
        Buoc 1: Phan doan van ban thanh cac cau.
            - Chia dua tren dau cau ket thuc: . ? !
            - Loai bo cau rong hoac chi chua khoang trang.
        """
        if not text:
            return []

        # Tach theo dau cau ket thuc (. ? !), giu lai dau cau o cuoi moi cau.
        raw_sentences = re.split(r"(?<=[.?!])\s+", text.strip())

        sentences = [s.strip() for s in raw_sentences if s and s.strip()]
        return sentences

    # ------------------------------------------------------------------
    # Buoc 2: Cumulative Semantic Merge
    # ------------------------------------------------------------------
    def _embed(self, sentences: list[str]):
        """Embedding hang loat cho danh sach cau (tra ve numpy array)."""
        return self.embedding_model.encode(
            sentences, convert_to_numpy=True, show_progress_bar=False
        )

    def merge_short_sentences(self, sentences: list[str]) -> list[str]:
        """
        Buoc 2: Merge Cong don theo ngu nghia (Cumulative Semantic Merge)
            - Duyet tuan tu tung don vi (cau).
            - So sanh do tuong dong ngu nghia giua chunk hien tai (da cong don)
              va don vi ke tiep.
            - Neu similarity >= threshold: gop vao chunk hien tai.
            - Neu similarity < threshold: dong chunk hien tai, mo chunk moi.
            - Ngoai le: neu chunk hien tai van con qua ngan (< min_chunk_size),
              van tiep tuc gop du similarity thap, de tranh sinh ra cac chunk
              qua nho khong co gia tri truy hoi.
        """
        if not sentences:
            return []

        if len(sentences) == 1:
            return sentences

        embeddings = self._embed(sentences)

        merged_chunks: list[str] = []
        current_sentences = [sentences[0]]
        # Vector dai dien cho chunk dang cong don = trung binh embedding cac cau trong chunk.
        current_embedding_sum = embeddings[0].copy()
        current_count = 1

        for i in range(1, len(sentences)):
            current_avg_embedding = current_embedding_sum / current_count
            sim = cosine_similarity(current_avg_embedding, embeddings[i])

            current_text_len = len(" ".join(current_sentences))
            force_merge = current_text_len < self.min_chunk_size

            if sim >= self.threshold or force_merge:
                # Gop cau tiep theo vao chunk hien tai
                current_sentences.append(sentences[i])
                current_embedding_sum += embeddings[i]
                current_count += 1
            else:
                # Dong chunk hien tai, mo chunk moi
                merged_chunks.append(" ".join(current_sentences))
                current_sentences = [sentences[i]]
                current_embedding_sum = embeddings[i].copy()
                current_count = 1

        # Dong chunk cuoi cung con lai
        if current_sentences:
            merged_chunks.append(" ".join(current_sentences))

        return merged_chunks

    # ------------------------------------------------------------------
    # Buoc 3: Guardrail Split
    # ------------------------------------------------------------------
    def _recursive_split(self, text: str, separators: list[str] | None = None) -> list[str]:
        """
        Bo tach de quy don gian (thay the cho RecursiveCharacterTextSplitter):
        thu tach theo tung loai separator tu "tho" (doan) den "min" (ky tu),
        gop lai theo max_chunk_size va giu overlap giua cac phan.
        """
        if separators is None:
            separators = ["\n\n", "\n", ". ", " ", ""]

        if len(text) <= self.max_chunk_size:
            return [text] if text.strip() else []

        separator = separators[0]
        remaining_separators = separators[1:]

        if separator == "":
            # Truong hop cuoi: cat cung theo ky tu, co overlap.
            pieces = []
            step = max(self.max_chunk_size - self.chunk_overlap, 1)
            for start in range(0, len(text), step):
                piece = text[start:start + self.max_chunk_size]
                if piece.strip():
                    pieces.append(piece)
                if start + self.max_chunk_size >= len(text):
                    break
            return pieces

        parts = [p for p in text.split(separator) if p.strip()] if separator else [text]
        if len(parts) <= 1 and remaining_separators:
            return self._recursive_split(text, remaining_separators)

        # Gop cac phan nho lai thanh cac khoi <= max_chunk_size, co overlap nho.
        chunks: list[str] = []
        current = ""
        for part in parts:
            candidate = (current + separator + part) if current else part
            if len(candidate) <= self.max_chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                if len(part) > self.max_chunk_size:
                    # Phan nay van con qua dai => de quy tiep voi separator nho hon.
                    chunks.extend(self._recursive_split(part, remaining_separators))
                    current = ""
                else:
                    current = part

        if current:
            chunks.append(current)

        # Them overlap giua cac chunk lien tiep de giu ngu canh.
        if self.chunk_overlap > 0 and len(chunks) > 1:
            overlapped = [chunks[0]]
            for prev, curr in zip(chunks, chunks[1:]):
                overlap_text = prev[-self.chunk_overlap:]
                overlapped.append((overlap_text + " " + curr).strip())
            return overlapped

        return chunks

    def split_long_sentence(self, chunks: list[str]) -> list[str]:
        """
        Buoc 3: Split lai khi vuot nguong kich thuoc (Guardrail Split)
            - Neu chunk sau khi cong don qua dai (> max_chunk_size), tach nho lai
              bang bo tach de quy, co overlap nho de giu ngu canh lien tuc.
        """
        final_chunks: list[str] = []
        for chunk in chunks:
            if len(chunk) > self.max_chunk_size:
                final_chunks.extend(self._recursive_split(chunk))
            elif chunk.strip():
                final_chunks.append(chunk)
        return final_chunks

    # ------------------------------------------------------------------
    # Save & Run
    # ------------------------------------------------------------------
    def save(self, chunks: list, file_name: str) -> None:
        """
        Save chunks ra output_path duoi dang file .pkl
        """
        os.makedirs(self.output_path, exist_ok=True)
        output_file = os.path.join(self.output_path, file_name)
        with open(output_file, "wb") as file:
            pickle.dump(chunks, file)
        logging.info("Da luu %d chunks vao %s", len(chunks), output_file)

    def run(self, save_file: bool = True) -> list[dict]:
        """
        Chay toan bo pipeline chia chunk:
            load -> split_into_sentence -> merge_short_sentences ->
            split_long_sentence -> save
        Tra ve danh sach tat ca chunks (kem metadata nguon) cua toan bo dataset.
        """
        documents = self.load()
        all_chunks: list[dict] = []

        for source_name, text in documents:
            logging.info("Dang xu ly file: %s", source_name)

            sentences = self.split_into_sentence(text)
            merged_chunks = self.merge_short_sentences(sentences)
            final_chunks = self.split_long_sentence(merged_chunks)

            for idx, chunk_text in enumerate(final_chunks):
                all_chunks.append(
                    {
                        "source": source_name,
                        "chunk_id": idx,
                        "text": chunk_text,
                    }
                )
            # for i in range(len(final_chunks)):
            #     logging.info(f"Chunk_{i} - {final_chunks[i]}")
            print("DONE!")
            if save_file:
                file_stem = Path(source_name).stem
                self.save(final_chunks, f"{file_stem}_chunks.pkl")
        logging.info("Hoan tat: tong cong %d chunks tu %d file.", len(all_chunks), len(documents))
        return all_chunks


if __name__ == "__main__":
    chunker = SemanticChunk(
        output_path="data/processed",
        input_path="data/raw",
    )
    chunker.run(save_file=False)
