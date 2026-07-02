def chunk_fixed_size(text, chunk_size, overlap):
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