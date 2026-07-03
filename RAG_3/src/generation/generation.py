from langchain_google_genai import ChatGoogleGenerativeAI 
from config.Config import GEMINI_KEY, TEMPERATURE, TOP_K 
llm = ChatGoogleGenerativeAI( 
    model="gemini-2.5-flash-lite", 
    google_api_key=GEMINI_KEY, 
    temperature=TEMPERATURE #tham số dùng để điều khiển tính ngẫu nhiên và sáng tạo 
)
def generate_answer(prompt):
    response = llm.invoke(prompt)
    return response.content