
import numpy as np

class KnowledgeMemory:
    def __init__(self):
        self.vectors = []
        self.meta = []

    def add(self, vector, metadata):
        self.vectors.append(vector)
        self.meta.append(metadata)

    def search(self, vector, top_k=5):
        sims = [float(np.dot(vector, v)) for v in self.vectors]
        idx = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:top_k]
        return [(self.meta[i], sims[i]) for i in idx]
