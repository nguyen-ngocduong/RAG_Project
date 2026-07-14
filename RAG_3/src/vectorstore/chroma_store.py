import chromadb
import uuid
import numpy as np

class ChromaStore:
    def __init__(self, persist_directory: str, collection_name: str = "rag_collection"):
        """
        Khởi tạo Chroma client với thư mục lưu trữ cục bộ.
        """
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_documents(self, chunks: list[str], embeddings: np.ndarray, metadata: list[dict]):
        """
        Thêm các chunks, embeddings và metadata vào Chroma DB.
        """
        if not chunks:
            return

        # Tạo IDs duy nhất cho mỗi chunk
        ids = [str(uuid.uuid4()) for _ in range(len(chunks))]
        
        # Chuyển numpy array thành list of lists để Chroma có thể đọc được
        if isinstance(embeddings, np.ndarray):
            embeddings = embeddings.tolist()
        elif hasattr(embeddings, 'numpy'):
            # Trong trường hợp embeddings là PyTorch/TensorFlow tensor
            embeddings = embeddings.numpy().tolist()

        # Chroma tự động lập chỉ mục (index) khi thêm vào
        self.collection.add(
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadata,
            ids=ids
        )

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        """
        Tìm kiếm top_k vector gần nhất với truy vấn.
        """
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()
        elif hasattr(query_embedding, 'numpy'):
            query_embedding = query_embedding.numpy().tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        formatted_results = []
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                # Cấu trúc trả về giống với format của retrieval.py
                formatted_results.append({
                    "chunk": results['documents'][0][i],
                    "score": results['distances'][0][i] if 'distances' in results and results['distances'] else 0.0,
                    "source": results['metadatas'][0][i]['source'] if 'metadatas' in results and results['metadatas'] else "Unknown"
                })
        return formatted_results
