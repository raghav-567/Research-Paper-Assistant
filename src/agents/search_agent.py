import arxiv
import requests
import os
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
from src.utils import get_logger

# Load environment variables from .env file
load_dotenv()

logger = get_logger("SearchAgent")

class SearchAgent:
    def __init__(self):
        self.s2_url = "https://api.semanticscholar.org/graph/v1"
        self.s2_api_key = os.getenv('S2_API_KEY')
        self.session = requests.Session()
    
        if self.s2_api_key:
            self.session.headers.update({"x-api-key": self.s2_api_key})
        else:
            logger.warning("S2_API_KEY not found. Semantic Scholar requests will be unauthenticated.")
    
        self.session.timeout = 30

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
    

    def search_semantic_scholar(self, query: str, limit: int = 5):
        logger.info(f"Searching Semantic Scholar for: {query}")
        
        params = {
            "query": query,
            "limit": limit,
            "fields": "title,abstract,authors,url,year,externalIds,paperId,citationCount"
        }
        
        try:
            response = self.session.get(
                f"{self.s2_url}/paper/search",
                params=params,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json().get("data", [])
            for paper in data:
                paper["source"] = "semantic_scholar"
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"S2 Search failed: {e}")
            return []
        except ValueError as e:
            logger.error(f"Failed to parse S2 response JSON: {e}")
            return []

    def get_citations(self, paper_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        logger.info(f"Fetching citations for: {paper_id}")
        
        params = {
            "fields": "title,authors,url,year,abstract",
            "limit": limit
        }
        
        try:
            response = self.session.get(
                f"{self.s2_url}/paper/{paper_id}/citations",
                params=params,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json().get("data", [])
            citations = []
            
            for citation in data:
                if "citingPaper" in citation:
                    citing_paper = citation["citingPaper"]
                    citing_paper["source"] = "semantic_scholar_citation"
                    citations.append(citing_paper)
            
            return citations
            
        except requests.exceptions.RequestException as e:
            logger.error(f"S2 Citations fetch failed: {e}")
            return []
        except ValueError as e:
            logger.error(f"Failed to parse S2 citations JSON: {e}")
            return []