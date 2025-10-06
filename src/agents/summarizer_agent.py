import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from src.utils import get_logger
import numpy as np
import time

logger = get_logger("SummarizerAgent")

class SummarizerAgent:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash-exp") 
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_chunks(self, chunks, normalize=True):
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        if not chunks:
            return np.array([])
        return self.embedding_model.encode(
            chunks, convert_to_numpy=True, normalize_embeddings=normalize
        )

    def get_top_k_chunks(self, chunks, embeddings, query_embedding, k=10):
        if len(chunks) == 0 or embeddings.size == 0:
            return []
        scores = np.dot(embeddings, query_embedding)
        k = min(k, len(chunks))  # Don't try to get more chunks than available
        top_k_idx = np.argsort(scores)[-k:][::-1]
        return [chunks[i] for i in top_k_idx]

    def _summarize_text(self, text, max_output_tokens=256, retry_count=3):
        """Summarize text with retry logic"""
        if not text or not text.strip():
            return "No content available to summarize."
        
        # Truncate if too long
        max_input_length = 5000
        if len(text) > max_input_length:
            text = text[:max_input_length] + "..."
            logger.warning(f"Text truncated to {max_input_length} characters")
        
        for attempt in range(retry_count):
            try:
                prompt = f"Summarize the following academic text concisely, focusing on key findings and methods:\n\n{text}"
                response = self.model.generate_content(
                    prompt,
                    generation_config={
                        "temperature": 0.3,
                        "max_output_tokens": max_output_tokens,
                    },
                )
                
                if response and response.text:
                    summary = response.text.strip()
                    logger.info(f"Successfully generated summary ({len(summary)} chars)")
                    return summary
                else:
                    logger.warning(f"Empty response from Gemini (attempt {attempt + 1})")
                    
            except Exception as e:
                logger.error(f"Gemini API error (attempt {attempt + 1}/{retry_count}): {e}")
                if attempt < retry_count - 1:
                    time.sleep(2 ** attempt) 
                else:
                    return f"Unable to generate summary after {retry_count} attempts."
        
        return "Error generating summary."

    def summarize_chunks(self, paper_data, query=None, k=10):
        logger.info("Starting paper summarization")

        abstract = None
        chunks = []

        # Handle input format
        if isinstance(paper_data, dict):
            abstract = paper_data.get("abstract", "")
            chunk_list = paper_data.get("chunks", [])
            chunks = [c["text"] if isinstance(c, dict) else c for c in chunk_list]
        elif isinstance(paper_data, list):
            chunks = [c if isinstance(c, str) else c.get("text", "") for c in paper_data]

        logger.info(f"Processing abstract: {bool(abstract)}, chunks: {len(chunks)}")

        if not chunks and not abstract:
            logger.warning("No content available for summarization")
            return "No content available."

        # Step 1: Summarize abstract
        abs_summary = ""
        if abstract and abstract.strip():
            logger.info("Summarizing abstract...")
            abs_summary = self._summarize_text(abstract, max_output_tokens=150)

        # Step 2: Summarize body chunks
        body_summary = ""
        if chunks:
            # Filter out empty chunks
            chunks = [c for c in chunks if c and c.strip()]
            
            if not chunks:
                logger.warning("All chunks were empty after filtering")
            else:
                logger.info(f"Processing {len(chunks)} non-empty chunks")
                
                # Select relevant chunks
                if query:
                    logger.info(f"Using query-based chunk selection: {query}")
                    try:
                        query_emb = self.embed_chunks([query])[0]
                        chunk_embs = self.embed_chunks(chunks)
                        top_chunks = self.get_top_k_chunks(chunks, chunk_embs, query_emb, k)
                    except Exception as e:
                        logger.error(f"Error in query-based selection: {e}")
                        top_chunks = chunks[:k]
                else:
                    top_chunks = chunks[:k]

                logger.info(f"Selected {len(top_chunks)} chunks for summarization")

                # Generate mini-summaries for each chunk
                mini_summaries = []
                for i, chunk in enumerate(top_chunks):
                    logger.info(f"Summarizing chunk {i+1}/{len(top_chunks)}")
                    mini = self._summarize_text(chunk, max_output_tokens=100)
                    if mini and not mini.startswith("Error") and not mini.startswith("Unable"):
                        mini_summaries.append(mini)

                # Combine mini-summaries
                if mini_summaries:
                    combined = " ".join(mini_summaries)
                    logger.info("Generating final body summary from mini-summaries")
                    body_summary = self._summarize_text(combined, max_output_tokens=250)
                else:
                    logger.warning("No valid mini-summaries generated")

        # Step 3: Merge summaries
        final_summary = ""
        if abs_summary and body_summary:
            if not abs_summary.startswith("Error") and not body_summary.startswith("Error"):
                final_summary = f"{abs_summary}\n\n{body_summary}"
            elif not abs_summary.startswith("Error"):
                final_summary = abs_summary
            elif not body_summary.startswith("Error"):
                final_summary = body_summary
        elif abs_summary:
            final_summary = abs_summary
        elif body_summary:
            final_summary = body_summary
        
        if not final_summary:
            final_summary = "Unable to generate summary from available content."
            
        logger.info(f"Final summary generated ({len(final_summary)} chars)")
        return final_summary