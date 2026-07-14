FROM python:3.10-slim

WORKDIR /app

# Cài đặt các thư viện cần thiết
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ code vào trong container
COPY . .

# Chạy một file mặc định (bạn có thể đổi thành app.py hoặc file ingest tùy nhu cầu)
CMD ["python", "test.py"]
