import numpy as np
def cosine_similarity(vector_a, vector_b):
    dot = np.dot(vector_a, vector_b)
    return dot / (np.linalg.norm(vector_a) * np.linalg.norm(vector_b))

#print(cosine_similarity(np.array([1,2,3]), np.array([4,5,6])))