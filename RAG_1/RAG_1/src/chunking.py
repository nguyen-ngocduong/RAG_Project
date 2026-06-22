from pypdf import PdfReader

def chunk_text(text, chunk_size=1000, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks
def recursive_chunk_text(text, chunk_size):
    # Phương pháp 2: Recursive Character Chunking
    # Các ký tự phân tách ưu tiên: Chương -> Mục -> Xuống dòng -> Khoảng trắng
    separators = ["\n# ", "\n## ", "\n### ", "\n", " ", ""]
    return split_recursive(text, chunk_size, separators)
def split_recursive(input_text, chunk_size, separators):
    if len(input_text) <= chunk_size:
        return [input_text]
    if not separators:
        result = []
        for i in range(0, len(input_text), chunk_size):
            result.append(input_text[i:i + chunk_size])
        return result

    separator = separators[0]
    pieces = input_text.split(separator)
    chunks = []
    current = ""
    for piece in pieces:
        candidate = current + separator + piece
        if len(candidate) < chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = piece
    if current:
        chunks.append(current)
    final_chunks = []
    for chunk in chunks:
        if len(chunk) > chunk_size:
            final_chunks.extend(split_recursive(chunk, chunk_size, separators[1:]))
        else:
            final_chunks.append(chunk)
    return final_chunks
