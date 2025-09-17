import fitz  
import re
from src.utils import get_logger

logger = get_logger("ExtractionAgent")

class ExtractionAgent:
    def parse_pdf(self, pdf_path, paper_id=None):
        """Extract abstract + body from PDF, clean, and return chunks"""
        logger.info(f"Extracting text from PDF: {pdf_path}")
        doc = fitz.open(pdf_path)

        text = []
        for page in doc:
            raw = page.get_text("text")
            cleaned = self.clean_text(raw)
            if cleaned:
                text.append(cleaned)

        full_text = " ".join(text)

        # extracting abstract separately
        abstract = self.extract_abstract(full_text)

        # if no abstract found, use first ~600 words
        if not abstract:
            logger.warning(f"No explicit abstract found in {pdf_path}, using intro fallback.")
            abstract = " ".join(full_text.split()[:600])

        # Chunk the body (excluding abstract text if found)
        body_text = full_text.replace(abstract, "")
        chunks = self.chunk_text(body_text, paper_id=paper_id)

        return {"abstract": abstract, "chunks": chunks}

    @staticmethod
    def clean_text(text: str) -> str:
        text = re.sub(
            r"(?:[A-Z][a-z]+(?:\s[A-Z][a-z]+)*[\*,\d¹²³⁴⁵⁶⁷⁸⁹†]*\s*,\s*){2,}.*",
            " ",
            text,
        )
        text = re.sub(
            r"^\s*[¹²³⁴⁵⁶⁷⁸⁹]\s?.*(University|Institute|Academy|School|Department).*",
            " ",
            text,
            flags=re.M | re.I,
        )
        text = re.sub(r"\$.*?\$", " ", text)
        text = re.sub(r"\\\[.*?\\\]", " ", text, flags=re.S)
        text = re.sub(r"`.*?`", " ", text)
        text = re.sub(r"[{}<>]", " ", text)
        text = re.sub(r"http\S+", " ", text)
        text = re.sub(r"\[\d+(,\s*\d+)*\]", " ", text)
        text = re.sub(r"\([A-Z][A-Za-z]+ et al\., \d{4}\)", " ", text)
        text = re.sub(r"[A-Z][a-zA-Z]+\s+[A-Z][a-zA-Z]+\.\s*\d+\s*\(\d{4}\).*?\d+", " ", text)
        text = re.split(r"(?i)references", text)[0]
        text = re.sub(r"(Figure|Table)\s*\d+[:\.].*", " ", text)
        text = re.sub(r"(?i)(acknowledg(e)?ments|funding|grants?).*", " ", text)

        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def extract_abstract(text: str) -> str:
        # Retrieving abstract section explicitly
        match = re.search(r"(?i)(abstract)(.*?)(introduction|keywords|1\s)", text, re.S)
        if match:
            return match.group(2).strip()
        return None

    def chunk_text(self, text, paper_id=None, chunk_size=2000):
        # Spliting cleaned text into word-based chunks
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk_words = words[i:i + chunk_size]
            chunk = " ".join(chunk_words)
            if len(chunk_words) >= 30:
                chunks.append({"text": chunk, "metadata": {"paper_id": paper_id}})
        return chunks
