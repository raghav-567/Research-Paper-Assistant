from sentence_transformers import SentenceTransformer
from src.utils import get_logger

logger = get_logger("SummarizerAgent")

class SummarizerAgent:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_chunks(self, chunks):
        logger.info("Generating embeddings for chunks")
        return self.model.encode(chunks, convert_to_numpy=True)

    def summarize_chunks(self, chunks):
        summaries = [c[:200] + "..." for c in chunks]  
        return " ".join(summaries)
