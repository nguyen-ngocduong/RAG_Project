# Hướng dẫn Hệ thống RAG với Vector Database (Chroma & Qdrant)

Tài liệu này mô tả toàn bộ quá trình tích hợp, cấu hình và sử dụng **Chroma** và **Qdrant** thay thế cho file Pickle (.pkl) tĩnh trong dự án RAG (Retrieval-Augmented Generation).

---

## 1. Tại sao chuyển từ Pickle sang Vector Database?

Trong phiên bản cũ, hệ thống RAG lưu các vector embedding vào một file `embedding.pkl`. Khi người dùng thực hiện truy vấn (query), hệ thống phải tính toán **Cosine Similarity** bằng cách duyệt qua (vòng lặp for) **toàn bộ** các vector đã lưu để tìm ra Top K tài liệu giống nhất.
- **Hạn chế:** Rất chậm khi lượng dữ liệu lớn.
- **Giải pháp:** Sử dụng Vector Database chuyên nghiệp như Chroma hoặc Qdrant. Các cơ sở dữ liệu này được tối ưu hóa cho không gian vector bằng thuật toán phân cụm không gian (như HNSW), giúp lập chỉ mục (index) và tìm kiếm (search) cực nhanh với độ phức tạp cực thấp.

---

## 2. Quá trình hoạt động (Workflow)

### Quá trình nạp tài liệu & Đánh Index (Ingestion)
1. **Đọc dữ liệu (Loader):** Tự động đọc và parse định dạng file PDF/DOCX thành văn bản.
2. **Làm sạch (Clean Text):** Chuẩn hóa khoảng trắng, xóa kí tự lạ.
3. **Semantic Chunking:** Chia nhỏ văn bản dựa trên ý nghĩa ngữ nghĩa (chuyển câu/đoạn).
4. **Embedding:** Chuyển đổi các chunk văn bản thành các vector số (ví dụ: dùng BAAI/bge-m3).
5. **Đánh Index (Vector Store Ingestion):**
   - **Chroma:** Tự động lập chỉ mục nội bộ khi ta đưa tài liệu, embedding và ID vào bằng lệnh `collection.add()`.
   - **Qdrant:** Hệ thống tạo một Collection định trước kích thước vector (ví dụ 1024 chiều), đóng gói payload (metadata, nội dung chunk) bằng `PointStruct`, và lập chỉ mục qua `client.upsert()`.

### Quá trình Truy xuất (Retrieval)
1. Nhận câu hỏi từ người dùng.
2. Embedding câu hỏi thành vector.
3. Thực hiện Vector Search:
   - **Chroma:** Sử dụng hàm `collection.query()`. Hệ thống tự động so khớp cấu trúc HNSW tìm ra K vector gần nhất.
```
            collections
                  │
                  ▼
              segments
                  │
                  ▼
            embeddings
                  │
            ┌─────┴────────┐
            ▼              ▼
          embedding_metadata
          embedding_metadata_array
```
   - **Qdrant:** Sử dụng hàm `client.search()`. Kết quả trả ra ngay lập tức với điểm số tương đồng `score`.
   - docker-compose up -d qdrant
   - http://localhost:6333/dashboard

---

## 3. Hướng dẫn cài đặt và chạy hệ thống

### 3.1. Cài đặt thư viện yêu cầu
Để hệ thống có thể chạy được với Vector Databases, bạn cần cài đặt 2 thư viện sau:
```bash
pip install chromadb qdrant-client
```

### 3.2. Cấu hình chuyển đổi Vector DB
Hệ thống được thiết kế để bạn có thể tuỳ ý thay đổi giữa 2 Database mà không cần đụng vào code. Bạn chỉ cần mở file cấu hình `config/config.yaml` và thay đổi `db_type`:

```yaml
vectorstore:
  # Chọn 'chroma' hoặc 'qdrant'
  db_type: chroma
  # Thư mục lưu dữ liệu database
  persist_directory: data/vector_store
  collection_name: rag_collection
  vector_size: 1024 # Size cho embedding model BAAI/bge-m3
```

### 3.3. Chạy hệ thống
Bạn có thể chạy hệ thống qua 2 cách:

**Cách 1: Giao diện Terminal (CLI)**
Mở terminal và chạy lệnh:
```bash
python main.py
```
> Khi có lời nhắc `Nhập câu hỏi: `, bạn gõ câu hỏi liên quan đến tài liệu. Hệ thống sẽ tự tìm kiếm qua VectorDB và trả về kết quả qua LLM (Gemini).

**Cách 2: Giao diện web Streamlit**
Mở terminal và chạy lệnh:
```bash
streamlit run FE_Streamlit/app.py
```
> Trong giao diện Streamlit:
> 1. Nhấn nút "Tải tài liệu lên" ở Sidebar bên trái.
> 2. Chọn nút "Chạy Ingestion" để hệ thống xử lý, embed và đưa vào VectorDB (Tuỳ vào cấu hình `config.yaml` bạn đang chọn).
> 3. Chat trực tiếp với trợ lý AI ở màn hình giữa.

---

## 4. Cách thay đổi mô hình Embedding & Vector Size
Mặc định hệ thống dùng mô hình `BAAI/bge-m3` có đầu ra vector chiều dài là `1024`.
Nếu bạn đổi sang model khác ở `config.yaml` (Ví dụ `keepitreal/vietnamese-sbert` có `size` là `768`), bạn **BẮT BUỘC** phải đổi thông số `vector_size` trong `config.yaml` tương ứng:
```yaml
model:
  MODEL_NAME: keepitreal/vietnamese-sbert
vectorstore:
  vector_size: 768
```
Nếu không đổi, Qdrant sẽ báo lỗi khi bạn Upsert dữ liệu có số chiều vector không khớp với Collection đã tạo.

*(Lưu ý: Đối với Chroma, thư viện sẽ tự nội suy số chiều mà không cần định nghĩa cứng trước như Qdrant, nên thông số này chỉ áp dụng chủ yếu cho Qdrant).*
