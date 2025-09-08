import fitz  # PyMuPDF
import re
from src.utils import get_logger

logger = get_logger("ExtractionAgent")

class ExtractionAgent:
    def parse_pdf(self, pdf_path, paper_id=None):
        """Extract full text from a PDF, clean it, and return as chunks"""
        logger.info(f"Extracting text from PDF: {pdf_path}")
        doc = fitz.open(pdf_path)

        text = []
        for page in doc:
            raw = page.get_text("text")
            cleaned = self.clean_text(raw)
            if cleaned:  # skip empty/noisy pages
                text.append(cleaned)

        full_text = " ".join(text)

        # Return as chunks
        return self.chunk_text(full_text, paper_id=paper_id)

    @staticmethod
    def clean_text(text: str) -> str:
        """Remove LaTeX, equations, code-like content, citations, references, and noise"""
        # Math and LaTeX
        text = re.sub(r"\$.*?\$", " ", text)                
        text = re.sub(r"\\\[.*?\\\]", " ", text, flags=re.S) 
        
        # Inline code
        text = re.sub(r"`.*?`", " ", text)                  
        
        # Braces, angle brackets
        text = re.sub(r"[{}<>]", " ", text)                 
        
        # URLs
        text = re.sub(r"http\S+", " ", text)                
        
        # Citations like [12], [67], (Smith et al., 2020)
        text = re.sub(r"\[\d+(,\s*\d+)*\]", " ", text)      
        text = re.sub(r"\([A-Z][A-Za-z]+ et al\., \d{4}\)", " ", text)
        
        # References/journals like "RSC Adv. 4 (2014), 1120–1127"
        text = re.sub(r"[A-Z][a-zA-Z]+\s+[A-Z][a-zA-Z]+\.\s*\d+\s*\(\d{4}\).*?\d+", " ", text)

        #after refrences 
        text = re.split(r"(?i)references", text)[0]

        
        # Figure/Table captions
        text = re.sub(r"(Figure|Table)\s*\d+[:\.].*", " ", text)
        
        # Collapse extra spaces
        text = re.sub(r"\s+", " ", text) 

        # Acknowledgment
        text = re.sub(r"(?i)(acknowledg(e)?ments|funding|grants?).*", " ", text)
                   
        return text.strip()


    def chunk_text(self, text, paper_id=None, chunk_size=300):
        """Split cleaned text into word-based chunks"""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk_words = words[i:i + chunk_size]
            chunk = " ".join(chunk_words)
            if len(chunk_words) >= 30:  # skip tiny/noisy chunks
                chunks.append({"text": chunk, "metadata": {"paper_id": paper_id}})
        return chunks
