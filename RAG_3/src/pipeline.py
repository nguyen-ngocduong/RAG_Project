from src.retrieval import retrieval
from src.generation.generation import generate_answer
from src.generation.prompt import prompt
from config.Config import TOP_K, MODEL_NAME
from src.embedding import load_embedding_model
from src.ingestion import process_and_add_document
import logging

def rag_pipeline(query, embedding_model):
    retrieved_docs = retrieval(
        query=query,
        embedding_model=embedding_model,
        top_k=TOP_K
    )
    # Gộp các chunk kết quả tìm kiếm được thành 1 context duy nhất 
    context = "\n\n".join(
        doc["chunk"] for doc in retrieved_docs
    )
    # Ghép context và câu hỏi (query) vào prompt template
    final_prompt = prompt.format(
        context=context,
        question=query
    )
    #Gửi prompt đã ghép hoàn chỉnh qua LLM để sinh câu trả lời
    answer = generate_answer(final_prompt)

    return answer

if __name__ == "__main__":
    logging.basicConfig(
        filename = "logs/app.log",
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    logging.info("="*20 + "Logging pipeline.py" + "="*20)
    query = "Server phân bổ tài nguyên như thế nào"
    embedding_model = load_embedding_model(MODEL_NAME)
    logging.info(rag_pipeline(query, embedding_model))