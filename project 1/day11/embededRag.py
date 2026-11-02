import os 

from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer





# cosine_similarity

def cosine_similarity(a, b):
    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )



model = SentenceTransformer("all-MiniLM-L6-v2")

text = "Machine learning is fun."

embedding=model.encode(text)



# print(embedding.shape)
# print(embedding[:18])




t1="There are 24 paid leaves"
t2="cat is a wild animal"



v1=model.encode(t1)
v2=model.encode(t2)
print(cosine_similarity(v1, v2))