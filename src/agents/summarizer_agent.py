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

    def summarize_chunks(self, paper_data, query=None, k=10):
        """
        Summarize a paper given its abstract + chunks.
        - Always prioritize the abstract.
        - Optionally augment with top-k body chunks.
        """
        logger.info("Summarizing retrieved chunks")

        # Handle both old (list) and new (dict with abstract+chunks) input
        abstract = None
        chunks = []

        if isinstance(paper_data, dict):
            abstract = paper_data.get("abstract")
            chunks = [c["text"] for c in paper_data.get("chunks", [])]
        elif isinstance(paper_data, list):  # fallback
            chunks = [c if isinstance(c, str) else c[1] for c in paper_data]

        if not chunks and not abstract:
            return "No content available."

        # --- Step 1: Summarize abstract if present ---
        abs_summary = ""
        if abstract:
            abs_summary = self.summarizer(
                abstract,
                max_length=120,
                min_length=50,
                do_sample=False
            )[0]["summary_text"]

        # --- Step 2: Select top-k body chunks ---
        body_summary = ""
        if chunks:
            if query:
                query_emb = self.embed_chunks([query])[0]
                chunk_embs = self.embed_chunks(chunks)
                top_chunks = self.get_top_k_chunks(chunks, chunk_embs, query_emb, k)
            else:
                top_chunks = chunks[:k]

            mini_summaries = []
            for chunk in top_chunks:
                mini = self.summarizer(
                    chunk,
                    max_length=120,
                    min_length=40,
                    do_sample=False
                )[0]["summary_text"]
                mini_summaries.append(mini)

            combined = " ".join(mini_summaries)
            body_summary = self.summarizer(
                combined,
                max_length=200,
                min_length=80,
                do_sample=False
            )[0]["summary_text"]

        # --- Step 3: Merge ---
        if abs_summary and body_summary:
            return f"{abs_summary} {body_summary}"
        return abs_summary or body_summary
