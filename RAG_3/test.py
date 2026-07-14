from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")

points, next_page = client.scroll(
    collection_name="rag_collection",
    limit=10,                 # số lượng point muốn lấy
    with_payload=True,
    with_vectors=True
)
print(client.collection_exists("rag_collection"))