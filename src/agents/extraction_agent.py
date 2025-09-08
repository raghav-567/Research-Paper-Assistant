import fitz  # PyMuPDF
from src.utils import get_logger

logger = get_logger("ExtractionAgent")

class ExtractionAgent:
    def parse_pdf(self, pdf_path, paper_id=None):
        """Extract full text from a PDF and return as chunks"""
        logger.info(f"Extracting text from PDF: {pdf_path}")
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text("text") + "\n"

        # Return one big text block for chunking
        return self.chunk_text(text, paper_id=paper_id)

    def chunk_text(self, text, paper_id=None, chunk_size=300):
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk_words = words[i:i + chunk_size]
            chunk = " ".join(chunk_words)
            chunks.append({"text": chunk, "metadata": {"paper_id": paper_id}})
        return chunks
