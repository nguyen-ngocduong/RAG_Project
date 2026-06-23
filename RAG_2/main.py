from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from src.chunks import chunking
from src.embedding_model import get_embedding
from src.clean_vietnamese_text import clean_vn_text
import json
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List
from langchain_core.documents import Document
from google.colab import userdata

# Đường dẫn tuyệt đối đến dữ liệu trên Drive
raw_dir = "/content/drive/MyDrive/RAG_Project/RAG_2/data/raw"
key_gemini = userdata.get("GEMINI_KEY") #lấy từ file.env

# B1: load document 
loader = DirectoryLoader(
    raw_dir, 
    glob="**/*.pdf", 
    loader_cls=UnstructuredFileLoader,
    show_progress=True,
    use_multithreading=True)
docs = loader.load()

# B2: Chunking
seperators = ["\n\n", "\n", " ", ""]
chunks = chunking(docs, seperators)

# B3 & B4: Embedding & Vector Store
vector_stores_db = []
for chunk in chunks:
    text = chunk.page_content
    vector = get_embedding(text)
    vector_stores_db.append({"text": text, "vector": vector})

# B5: Retriever
def custom_vector_db_retriever_func(question: str, top_k: int = 3) -> List[Document]:
    query_embedding = get_embedding(question)
    similarities = []
    for item in vector_stores_db:
        doc_vector = item['vector']
        score = cosine_similarity(np.array(query_embedding).reshape(1, -1), np.array(doc_vector).reshape(1, -1))[0][0]
        similarities.append((score, item['text']))

    similarities.sort(key=lambda x: x[0], reverse=True)
    return [Document(page_content=text) for score, text in similarities[:top_k]]

retriever = RunnableLambda(custom_vector_db_retriever_func)

# B6: LLM setup
prompt_template = """Bạn là một trợ lý AI chuyên nghiệp.
Context:
{context}

Question: {question}"""

prompt = PromptTemplate.from_template(prompt_template)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=key_gemini
)

# Pipeline
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()} 
    | prompt 
    | llm 
    | StrOutputParser()
)

if __name__ == "__main__":
    print("\n--- Hệ thống đã sẵn sàng ---\n")
    while True:
        user_input = input("Nhập câu hỏi (gõ 'exit' để thoát): ")
        if user_input.lower() == 'exit':
            break
        if not user_input.strip():
            continue
        answer = rag_chain.invoke(user_input)
        print(f"\nTrả lời: {answer}\n")
