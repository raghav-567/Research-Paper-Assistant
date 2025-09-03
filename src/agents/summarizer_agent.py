from sentence_transformers import SentenceTransformer
from transformers import pipeline
from src.utils import get_logger

logger = get_logger("SummarizerAgent")

class SummarizerAgent:
    def __init__(self):
        # Embedding model
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Summarizer (T5 or BART)
        self.summarizer = pipeline("summarization", model="t5-small", device=0)  # device=0 for MPS/GPU

    def embed_chunks(self, chunks):
        logger.info("Generating embeddings for chunks")
        return self.embedding_model.encode(chunks, convert_to_numpy=True)

    def summarize_chunks(self, chunks, query=None):
        logger.info("Summarizing retrieved chunks")
        summaries = []

        for i, chunk in enumerate(chunks[:20]): 
            input_len = len(chunk.split()) 

            target_len = min(60, int(input_len * 0.5))

            summary = self.summarizer(
                chunk[:512],
                max_new_tokens=target_len,
                do_sample=False
            )[0]['summary_text']

            summaries.append(f"Chunk {i+1} Summary: {summary}")

        return "\n".join(summaries)
