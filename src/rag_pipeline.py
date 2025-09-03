import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from src.utils import get_logger

logger = get_logger("RAGPipeline")

class RAGPipeline:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = None
        self.chunks = []

    def build_index(self, chunks):
        logger.info("Building FAISS index")
        self.chunks = chunks
        embeddings = self.model.encode(chunks, convert_to_numpy=True)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

    def query(self, question, top_k=3):
        q_emb = self.model.encode([question], convert_to_numpy=True)
        D, I = self.index.search(q_emb, top_k)
        return [self.chunks[i] for i in I[0]]
