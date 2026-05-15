try:
    import google.generativeai as genai
    _GENAI_IMPORT_ERROR = None
except Exception as e:
    genai = None
    _GENAI_IMPORT_ERROR = str(e)
from src.utils import get_logger
import numpy as np
import os
import re
import time

try:
    import faiss
    _FAISS_IMPORT_ERROR = None
except Exception as e:
    faiss = None
    _FAISS_IMPORT_ERROR = str(e)

logger = get_logger("SummarizerAgent")

class SummarizerAgent:
    def __init__(self, api_key: str):
        if genai is None:
            raise RuntimeError(
                "Gemini client dependency is missing. Install dependencies with "
                "'pip install -r requirements.txt'. "
                f"Import error: {_GENAI_IMPORT_ERROR}"
            )
        genai.configure(api_key=api_key)
        # Use a current free-tier text model by default.
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
        self.model = genai.GenerativeModel(self.model_name)
        self.embedding_model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self.embedding_model = None
        self.embedding_error = None
        self.quota_exhausted = False
        self.quota_error_message = None

    def _get_embedding_model(self):
        if self.embedding_model is not None:
            return self.embedding_model

        if self.embedding_error is not None:
            return None

        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            return self.embedding_model
        except Exception as e:
            self.embedding_error = str(e)
            logger.warning(
                f"Embedding model '{self.embedding_model_name}' is unavailable. "
                f"RAG indexing will be skipped. Error: {e}"
            )
            return None

    def get_embedding_dimension(self):
        model = self._get_embedding_model()
        if model is None:
            return None
        return model.get_sentence_embedding_dimension()

    def embed_chunks(self, chunks, normalize=True):
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        if not chunks:
            return np.array([])
        model = self._get_embedding_model()
        if model is None:
            return np.array([])
        return self.embedding_model.encode(
            chunks, convert_to_numpy=True, normalize_embeddings=normalize
        )

    def get_top_k_chunks(self, chunks, embeddings, query_embedding, k=10):
        if len(chunks) == 0 or embeddings.size == 0:
            return []
        if faiss is None:
            logger.warning(
                "FAISS is unavailable; using first chunks instead of vector ranking. "
                f"Import error: {_FAISS_IMPORT_ERROR}"
            )
            return chunks[: min(k, len(chunks))]

        # Build FAISS index
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim) 
        index.add(embeddings.astype(np.float32))

        query_embedding = np.array([query_embedding]).astype(np.float32)
        scores, indices = index.search(query_embedding, min(k, len(chunks)))

        top_k_idx = indices[0]
        return [chunks[i] for i in top_k_idx]

    def _extract_response_text(self, response):
        if not response:
            return ""

        text = getattr(response, "text", "") or ""
        if text.strip():
            return text.strip()

        candidates = getattr(response, "candidates", None) or []
        parts = []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            content_parts = getattr(content, "parts", None) or []
            for part in content_parts:
                part_text = getattr(part, "text", "") or ""
                if part_text.strip():
                    parts.append(part_text.strip())

        return "\n".join(parts).strip()

    def _is_quota_error(self, error_message):
        message = (error_message or "").lower()
        return (
            "exceeded your current quota" in message
            or "quota exceeded" in message
            or "generate_content_free_tier" in message
            or "429" in message and "quota" in message
        )

    def _fallback_summary(self, text, max_chars=700):
        cleaned = re.sub(r"\s+", " ", text or "").strip()
        if not cleaned:
            return "No content available to summarize."

        cleaned = re.sub(r"\b(Abstract:|Main Content:)\s*", "", cleaned, flags=re.I)
        sentences = re.split(r"(?<=[.!?])\s+", cleaned)

        selected = []
        current_length = 0
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            projected = current_length + len(sentence) + (1 if selected else 0)
            if projected > max_chars and selected:
                break
            selected.append(sentence)
            current_length = projected
            if len(selected) >= 4:
                break

        if selected:
            return " ".join(selected)

        words = cleaned.split()
        truncated = " ".join(words[: min(len(words), 100)])
        return truncated + ("..." if len(words) > 100 else "")

    def _summarize_text(self, text, max_output_tokens=256, retry_count=3):
        if not text or not text.strip():
            return "No content available to summarize."

        if self.quota_exhausted:
            logger.warning("Gemini quota already exhausted; using local fallback summary.")
            return self._fallback_summary(text)
        
        max_input_length = 4000
        if len(text) > max_input_length:
            text = text[:max_input_length] + "..."
            logger.warning(f"Text truncated to {max_input_length} characters")
        
        last_error = None
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
                
                summary = self._extract_response_text(response)
                if summary:
                    logger.info(f"Successfully generated summary ({len(summary)} chars)")
                    return summary
                else:
                    finish_reason = ""
                    candidates = getattr(response, "candidates", None) or []
                    if candidates:
                        finish_reason = getattr(candidates[0], "finish_reason", "")
                    logger.warning(
                        f"Empty response from Gemini model {self.model_name} "
                        f"(attempt {attempt + 1}/{retry_count}, finish_reason={finish_reason})"
                    )
                    last_error = "Gemini returned an empty response."
                    
            except Exception as e:
                last_error = str(e)
                logger.error(
                    f"Gemini API error with model {self.model_name} "
                    f"(attempt {attempt + 1}/{retry_count}): {e}"
                )
                if self._is_quota_error(last_error):
                    self.quota_exhausted = True
                    self.quota_error_message = last_error
                    logger.warning("Gemini quota exhausted; falling back to extractive summaries for remaining papers.")
                    return self._fallback_summary(text)
                if attempt < retry_count - 1:
                    time.sleep(2 ** attempt) 
                else:
                    return (
                        f"Unable to generate summary after {retry_count} attempts. "
                        f"Last error: {last_error}"
                    )
        
        if last_error:
            return f"Error generating summary. Last error: {last_error}"
        return "Error generating summary."

    def summarize_chunks(self, paper_data, query=None, k=10):
        logger.info("Starting paper summarization")

        abstract = None
        chunks = []

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

        abs_summary = ""
        if abstract and abstract.strip():
            logger.info("Summarizing abstract...")
            abs_summary = self._summarize_text(abstract, max_output_tokens=150)

        body_summary = ""
        if chunks:
            chunks = [c for c in chunks if c and c.strip()]
            if chunks:
                logger.info(f"Processing {len(chunks)} non-empty chunks")
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

                mini_summaries = []
                for i, chunk in enumerate(top_chunks):
                    logger.info(f"Summarizing chunk {i+1}/{len(top_chunks)}")
                    mini = self._summarize_text(chunk, max_output_tokens=100)
                    if mini and not mini.startswith("Error") and not mini.startswith("Unable"):
                        mini_summaries.append(mini)

                if mini_summaries:
                    combined = " ".join(mini_summaries)
                    logger.info("Generating final body summary from mini-summaries")
                    body_summary = self._summarize_text(combined, max_output_tokens=250)
                else:
                    logger.warning("No valid mini-summaries generated")

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
