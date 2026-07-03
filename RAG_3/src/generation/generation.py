from langchain_google_genai import ChatGoogleGenerativeAI 
from config.Config import GEMINI_KEY, TEMPERATURE, TOP_K 
llm = ChatGoogleGenerativeAI( 
    model="gemini-2.5-flash-lite", 
    google_api_key=GEMINI_KEY, 
    temperature=TEMPERATURE, 
    top_p = 0.5,# Ngưỡng xác suất tích lũy khi chọn từ tiếp theo. 
    top_k = 40 # Giới hạn số lượng từ có xác suất cao nhất được xem xét. 
)
def generate_answer(prompt):
    response = llm.invoke(prompt)
    return response.content