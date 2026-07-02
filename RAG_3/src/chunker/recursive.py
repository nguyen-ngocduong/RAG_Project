from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.Config import CHUNK_SIZE, CHUNK_OVERLAP
from src.loader import load_document

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
    separators = ["\n\n", "\n", " ", ""]
    text_splitter = splitter(separators)
    chunks = text_splitter.split_text(docs)
    return chunks