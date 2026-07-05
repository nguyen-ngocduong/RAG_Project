from src.loader import load_pdf
import re
import unicodedata
from config.Config import CHUNK_SIZE, CHUNK_OVERLAP

def text_processing(text: str) -> str:
    """
    Tiền xử lý văn bản PDF cho RAG

    - Chuẩn hóa Unicode về NFC
    - Loại bỏ ký tự điều khiển (giữ \n, \t)
    - Loại bỏ header/footer phổ biến
    - Loại bỏ số trang
    - Loại bỏ mục lục
    - Chuẩn hóa khoảng trắng và xuống dòng
    """
    # ============================================================
    # 1. Unicode Normalize
    # ============================================================
    text = unicodedata.normalize("NFC", text)
    # ============================================================
    # 2. Remove control characters (giữ newline và tab)
    # ============================================================
    text = "".join(
        ch
        for ch in text
        if ch == "\n"
        or ch == "\t"
        or unicodedata.category(ch)[0] != "C"
    )

    # ============================================================
    # 3. Remove header/footer
    # ============================================================
    header_patterns = [
        r"AI VIET NAM\s*\(AIO2025\)",
        r"AI VIET NAM\s*–\s*AI COURSE\s*2025",
        r"aivietnam\.edu\.vn",
    ]

    for pattern in header_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # ============================================================
    # 4. Remove page numbers
    # ============================================================
    text = re.sub(r"(?m)^\s*\d+\s*$", "", text)

    # ============================================================
    # 5. Remove Table of Contents
    # ============================================================
    toc_pattern = (
        r"Mục lục.*?(?=\n(?:Bảng Giải Thích Thuật Ngữ|I\.\s*Bối cảnh|II\.))"
    )

    text = re.sub(
        toc_pattern,
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # ============================================================
    # 6. Remove multiple blank lines
    # ============================================================
    text = re.sub(r"\n{3,}", "\n\n", text)

    # ============================================================
    # 7. Remove trailing spaces
    # ============================================================
    text = re.sub(r"[ \t]+\n", "\n", text)

    # ============================================================
    # 8. Compress multiple spaces
    # ============================================================
    text = re.sub(r"[ ]{2,}", " ", text)

    # ============================================================
    # 9. Strip
    # ============================================================
    return text.strip()