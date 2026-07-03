from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template(
    """
    Bạn là một trợ lý AI thông minh.
    Chỉ sử dụng thông tin trong phần ngữ cảnh dưới đây để trả lời.
    Nếu trong ngữ cảnh không có thông tin thì hãy trả lời:
    "Tôi không tìm thấy thông tin trong tài liệu."
    và dựa trên kiến thức đã có hãy trả lời chính xác câu hỏi
    ==========================
    Ngữ cảnh:{context}
    ==========================
    Câu hỏi:{question}
    ==========================
    Trả lời bằng tiếng Việt.
    """
)