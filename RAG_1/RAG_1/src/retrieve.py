import numpy as np
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
# Tìm top k
def find_top_k(query, vectors, k=3):
    scores = []
    for item in vectors:
        score = cosine_similarity(query, item["vector"])
        scores.append((score, item['text']))
    scores.sort(reverse=True)
    return scores[:k]
