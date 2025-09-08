from sentence_transformers import SentenceTransformer
from transformers import pipeline
from src.utils import get_logger
import numpy as np

logger = get_logger("SummarizerAgent")

class SummarizerAgent:
    def __init__(self):
        # Embedding model
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        self.summarizer = pipeline("summarization",
                                   model="google-t5/t5-large",
                                   device=-1)

    def embed_chunks(self, chunks, normalize=True):
        logger.info("Generating embeddings for chunks")
        return self.embedding_model.encode(
            chunks, convert_to_numpy=True, normalize_embeddings=normalize
        )

    def get_top_k_chunks(self, chunks, embeddings, query_embedding, k=10):
        scores = np.dot(embeddings, query_embedding)
        top_k_idx = np.argsort(scores)[-k:][::-1]
        return [chunks[i] for i in top_k_idx]

    def summarize_chunks(self, chunks, query=None, k=10):
        logger.info("Summarizing retrieved chunks")

        if isinstance(chunks, dict):
            chunks = list(chunks.values())
        if chunks and isinstance(chunks[0], tuple):
            chunks = [c[1] for c in chunks]

        if not chunks:
            return "No chunks available."

        # Rank by query relevance if given
        if query:
            query_emb = self.embed_chunks([query])[0]
            chunk_embs = self.embed_chunks(chunks)
            chunks = self.get_top_k_chunks(chunks, chunk_embs, query_emb, k)
        else:
            chunks = chunks[:k]  # fallback = first k chunks

        summaries = []
        for i, chunk in enumerate(chunks, start=1):
            summary = self.summarizer(
                chunk,
                max_length=80,
                min_length=30,
                do_sample=False
            )[0]['summary_text']
            summaries.append(f"### Paper {i}\n{summary}")

        return "\n\n".join(summaries)
