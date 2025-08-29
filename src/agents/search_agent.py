import arxiv
import requests
from src.utils import get_logger

logger = get_logger("SearchAgent")

class SearchAgent:
    def __init__(self):
        self.s2_url = "https://api.semanticscholar.org/graph/v1/paper/search"

    def search_arxiv(self, query, max_results=5):
        logger.info(f"Searching Arxiv for: {query}")
        results = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
        papers = []
        for r in results.results():
            papers.append({
                "title": r.title,
                "summary": r.summary,
                "authors": [a.name for a in r.authors],
                "url": r.entry_id,
                "published": r.published
            })
        return papers

    def search_semantic_scholar(self, query, limit=5):
        logger.info(f"Searching Semantic Scholar for: {query}")
        params = {"query": query, "limit": limit, "fields": "title,abstract,authors,url,year"}
        r = requests.get(self.s2_url, params=params)
        if r.status_code == 200:
            return r.json().get("data", [])
        return []
