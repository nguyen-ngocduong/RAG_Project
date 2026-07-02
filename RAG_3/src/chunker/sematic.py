from src.loader import load_document

def chunk_semantic(docs):
    """
    Thay vì cắt văn bản theo 1000 ký tự hoặc 500 token, 
    ta cắt theo ý nghĩa cua cau hoac cat theo tung paragraph
    """
    chunks = [doc.strip() for doc in docs.strip("\n\n") if doc.strip()]
    return chunks

if __name__ == "__main__":
    docs = load_document("data/raw")
    chunks = chunk_semantic(docs)
    print(len(chunks))
    print(chunks[0])