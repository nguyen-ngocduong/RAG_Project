from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid
import numpy as np

class QdrantStore:
    def __init__(self, persist_directory: str = None, collection_name: str = "rag_collection", vector_size: int = 768):
        """
        Khởi tạo Qdrant client kết nối tới Qdrant Server (Docker).
        """
        import os
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.client = QdrantClient(url=qdrant_url)
        self.collection_name = collection_name
        
        # Kiểm tra xem collection đã tồn tại chưa, nếu chưa thì tạo mới với kích thước vector tương ứng
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

    def add_documents(self, chunks: list[str], embeddings: np.ndarray, metadata: list[dict]):
        """
        Thêm tài liệu (points) vào Qdrant DB.
        """
        if not chunks:
            return
            
        points = []
        for chunk, embedding, meta in zip(chunks, embeddings, metadata):
            vector = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
            if hasattr(vector, 'numpy'):
                vector = vector.numpy().tolist()
                
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    # Lưu trữ chunk text và source trong payload
                    payload={"chunk": chunk, **meta}
                )
            )
            
        # Upsert (Cập nhật hoặc Thêm mới) points vào collection
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        """
        Tìm kiếm theo vector similarity.
        """
        vector = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
        if hasattr(vector, 'numpy'):
            vector = vector.numpy().tolist()

        # qdrant-client >= 1.12.0: dùng query_points thay vì search
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=top_k
        )

        formatted_results = []
        for result in search_result.points:
            formatted_results.append({
                "chunk": result.payload.get("chunk", ""),
                "score": result.score,
                "source": result.payload.get("source", "Unknown")
            })
        return formatted_results
