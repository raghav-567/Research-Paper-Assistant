import arxiv
import requests
import os
from pathlib import Path
from src.utils import get_logger

logger = get_logger("SearchAgent")

class SearchAgent:
    def __init__(self, pdf_dir="pdfs"):
        self.pdf_dir = Path(pdf_dir)
        self.pdf_dir.mkdir(parents=True, exist_ok=True)

    def search_arxiv(self, query, max_results=10):
        logger.info(f"Searching Arxiv for: {query}")
        results = arxiv.Search(
            query=query, 
            max_results=max_results, 
            sort_by=arxiv.SortCriterion.Relevance
        )
        papers = []
        for r in results.results():
            paper_id = r.entry_id.split("/")[-1]
            pdf_url = r.pdf_url
            pdf_path = self.download_pdf(pdf_url, paper_id) if pdf_url else None
            if not pdf_path:
                logger.warning(f"No PDF for {r.title}, falling back to abstract")

            papers.append({
                "id": paper_id,
                "title": r.title,
                "summary": r.summary,
                "authors": [a.name for a in r.authors],
                "pdf_path": pdf_path,  # path to downloaded PDF, or None
                "url": r.entry_id,
                "published": r.published
            })
        return papers

    def download_pdf(self, pdf_url: str, paper_id: str):
        try:
            response = requests.get(pdf_url, stream=True, timeout=30)
            response.raise_for_status()
            pdf_path = self.pdf_dir / f"{paper_id}.pdf"
            with open(pdf_path, "wb") as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            logger.info(f"Downloaded PDF: {pdf_path}")
            return str(pdf_path)
        except Exception as e:
            logger.warning(f"Failed to download PDF {pdf_url}: {e}")
            return None
