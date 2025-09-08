from src.utils import get_logger

logger = get_logger("SynthesizerAgent")

class SynthesizerAgent:
    def synthesize(self, papers, summaries):
        logger.info("Synthesizing literature review")
        review = "# Literature Review\n\n"
        
        for i, (p, s) in enumerate(zip(papers, summaries), start=1):
            review += f"## {i}. {p['title']}\n"
            review += f"**Authors**: {', '.join(p.get('authors', []))}\n\n"
            review += f"**Summary**: {s}\n\n"
        
        return review
