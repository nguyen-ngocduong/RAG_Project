from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.clean_vietnamese_text import clean_vn_text
def chunking(docs, separators):
  # Bước 2: Chunking
  for doc in docs:
    doc.page_content = clean_vn_text(doc.page_content)
  print("Documents cleaned.")
  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=1000, # A common chunk size
      chunk_overlap=200, # A common overlap to maintain context
      add_start_index = True, # Lưu lại vị trí ban đầu mỗi Chunk
      strip_whitespace = True, # loại bỏ khoảng trắng ở đầu, cuối
      separators = separators # Các separator để chia nhỏ tài liệu
  )
  chunks = text_splitter.split_documents(docs)
  print(f"Number of chunks: {len(chunks)}")
  print("First chunk:\n")
  print(chunks[0].page_content)
  return chunks
