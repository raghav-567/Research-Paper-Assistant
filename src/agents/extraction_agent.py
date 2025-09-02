from pypdf import PdfReader
from src.utils import get_logger

logger = get_logger("ExtractionAgent")

class ExtractionAgent:
    def parse_pdf(self, pdf_path):
        logger.info(f"Extracting text from: {pdf_path}")
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text

    def chunk_text(self, text, chunk_size=512, overlap=50):
        logger.info(f"Chunking text (size={chunk_size})")
        words = text.split()
        chunks = []
        i = 0
        while i < len(words):
            chunk = words[i:i+chunk_size]
            chunks.append(" ".join(chunk))
            i += chunk_size - overlap
        return chunks
