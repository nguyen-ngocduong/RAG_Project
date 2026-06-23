import re
import unicodedata
from typing import List

def clean_vn_text(text: str) -> str:
    # Chuẩn hóa Unicode về NFC cho tiếng việt
    text = unicodedata.normalize('NFC', text)
    text = "".join(char for char in text if not unicodedata.category(char).startswith('C') or char in '\n\t')
    # Gộp khoảng trắng thừa và dòng trống
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n', text)
    return text.strip()
