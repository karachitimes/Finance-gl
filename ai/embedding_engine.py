
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_text(texts):
    return model.encode(texts, normalize_embeddings=True)
