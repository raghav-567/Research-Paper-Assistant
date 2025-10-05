import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from src.utils import get_logger
import numpy as np

logger = get_logger("SummarizerAgent")

class SummarizerAgent:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-pro-latest") 
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_chunks(self, chunks, normalize=True):
        logger.info("Generating embeddings for chunks")
        return self.embedding_model.encode(
            chunks, convert_to_numpy=True, normalize_embeddings=normalize
        )

    def get_top_k_chunks(self, chunks, embeddings, query_embedding, k=10):
        scores = np.dot(embeddings, query_embedding)
        top_k_idx = np.argsort(scores)[-k:][::-1]
        return [chunks[i] for i in top_k_idx]

    def _summarize_text(self, text, max_output_tokens=256):
        try:
            prompt = f"Summarize the following text in a clear and concise way:\n\n{text}"
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.2,
                    "max_output_tokens": max_output_tokens,
                },
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"Gemini summarization error: {e}")
            return "Error generating summary."

    def summarize_chunks(self, paper_data, query=None, k=10):
        logger.info("Summarizing retrieved chunks")

        abstract = None
        chunks = []

        # Handle input format
        if isinstance(paper_data, dict):
            abstract = paper_data.get("abstract")
            chunks = [c["text"] for c in paper_data.get("chunks", [])]
        elif isinstance(paper_data, list):  # fallback
            chunks = [c if isinstance(c, str) else c[1] for c in paper_data]

        if not chunks and not abstract:
            return "No content available."

        # Step 1: Summarize abstract
        abs_summary = ""
        if abstract:
            abs_summary = self._summarize_text(abstract, max_output_tokens=120)

        # Step 2: Summarize body
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
                mini = self._summarize_text(chunk, max_output_tokens=100)
                mini_summaries.append(mini)

            combined = " ".join(mini_summaries)
            body_summary = self._summarize_text(combined, max_output_tokens=200)

        # Step 3: Merge summaries
        if abs_summary and body_summary:
            return f"{abs_summary} {body_summary}"
        return abs_summary or body_summary
