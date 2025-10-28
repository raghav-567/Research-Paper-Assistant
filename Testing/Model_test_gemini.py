# gemini_summarizer_agent.py
import os
import time
import numpy as np
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
from src.utils import get_logger
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("API_KEY")

logger = get_logger("GeminiSummarizerAgent")


class GeminiSummarizerAgent:
    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.0-flash",
        max_output_tokens: int = 512
    ):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.max_output_tokens = max_output_tokens

    def embed_chunks(self, chunks: List[str], normalize: bool = True) -> np.ndarray:
        if not chunks:
            return np.array([])
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        return self.embedding_model.encode(
            chunks,
            convert_to_numpy=True,
            normalize_embeddings=normalize
        )

    def get_top_k_chunks(
        self,
        chunks: List[str],
        embeddings: np.ndarray,
        query_embedding: np.ndarray,
        k: int = 10
    ) -> List[str]:
        if len(chunks) == 0 or embeddings.size == 0:
            return []
        scores = np.dot(embeddings, query_embedding)
        k = min(k, len(chunks))
        top_k_idx = np.argsort(scores)[-k:][::-1]
        return [chunks[i] for i in top_k_idx]

    def _create_prompt(self, text: str, task: str = "summarize") -> str:
        if task == "summarize":
            return f"""
You are an academic summarization expert specializing in research and technical writing.
Produce a concise, high-quality summary in three short sections:
1. Overview and Core Idea
2. Details or Methodology
3. Significance or Broader Context

Maintain a professional tone, 200–250 words max.
Text to summarize:
{text}
"""
        elif task == "abstract":
            return f"""
Write a formal academic abstract (150–200 words) including:
- Background or motivation
- Core methods or structure
- Key results or findings
- Brief conclusion or implications

Text:
{text}
"""
        elif task == "key_points":
            return f"""
Extract key insights or findings from the text as numbered bullet points.
Focus on the most important information only.

Text:
{text}
"""
        else:
            return f"Summarize concisely:\n{text}"

    def _summarize_text(
        self,
        text: str,
        max_output_tokens: int = 512,
        task: str = "summarize"
    ) -> str:
        if not text or not text.strip():
            return "No content available to summarize."

        max_input_length = 3000
        if len(text) > max_input_length:
            logger.warning(f"Text truncated from {len(text)} to {max_input_length} chars")
            text = text[:max_input_length] + "..."

        prompt = self._create_prompt(text, task)
        logger.info(f"Generating summary with Gemini (max {max_output_tokens} tokens)...")
        start_time = time.time()

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_output_tokens
                )
            )
            elapsed = time.time() - start_time
            logger.info(f"✓ Summary generated in {elapsed:.2f}s")
            summary = response.text.strip()
            return summary if summary else "Unable to generate summary."
        except Exception as e:
            logger.error(f"Gemini generation error: {e}")
            return f"Error: {str(e)}"

    def summarize_chunks(
        self,
        paper_data: Dict,
        query: Optional[str] = None,
        k: int = 5
    ) -> str:
        logger.info("Starting paper summarization with Gemini")

        abstract = None
        chunks = []

        if isinstance(paper_data, dict):
            abstract = paper_data.get("abstract", "")
            chunk_list = paper_data.get("chunks", [])
            chunks = [c["text"] if isinstance(c, dict) else c for c in chunk_list]
        elif isinstance(paper_data, list):
            chunks = [c if isinstance(c, str) else c.get("text", "") for c in paper_data]

        logger.info(f"Processing: Abstract={bool(abstract)}, Chunks={len(chunks)}")

        if not chunks and not abstract:
            logger.warning("No content available for summarization")
            return "No content available."

        abs_summary = ""
        if abstract.strip():
            logger.info("Summarizing abstract...")
            abs_summary = self._summarize_text(abstract, max_output_tokens=150, task="abstract")

        body_summary = ""
        if chunks:
            chunks = [c for c in chunks if c.strip()]
            logger.info(f"Processing {len(chunks)} non-empty chunks")

            k = min(k, 3)
            if query:
                logger.info(f"Using query-based selection: {query}")
                try:
                    query_emb = self.embed_chunks([query])[0]
                    chunk_embs = self.embed_chunks(chunks)
                    top_chunks = self.get_top_k_chunks(chunks, chunk_embs, query_emb, k)
                except Exception as e:
                    logger.error(f"Error in selection: {e}")
                    top_chunks = chunks[:k]
            else:
                top_chunks = chunks[:k]

            combined = " ".join(top_chunks)
            if len(combined) > 4000:
                combined = combined[:4000]

            logger.info("Generating body summary...")
            body_summary = self._summarize_text(
                combined,
                max_output_tokens=512,
                task="summarize"
            )

        final_summary = (abs_summary + "\n\n" + body_summary).strip()
        if not final_summary:
            if abstract:
                final_summary = abstract[:500] + "..."
            elif chunks:
                final_summary = chunks[0][:500] + "..."
            else:
                final_summary = "Unable to generate summary."

        logger.info(f"Final summary length: {len(final_summary)} chars")
        return final_summary

    def __del__(self):
        try:
            logger.info("Cleaning up Gemini Summarizer Agent...")
        except Exception:
            pass


# Example usage
if __name__ == "__main__":
    summarizer = GeminiSummarizerAgent(api_key)

    test_text = """
   Cricket is a bat-and-ball game that is played between two teams of eleven players on a field, at the centre of which is a 22-yard (20-metre; 66-foot) pitch with a wicket at each end, each comprising two bails (small sticks) balanced on three stumps. Two players from the batting team, the striker and nonstriker, stand in front of either wicket holding bats, while one player from the fielding team, the bowler, bowls the ball toward the striker's wicket from the opposite end of the pitch. The striker's goal is to hit the bowled ball with the bat and then switch places with the nonstriker, with the batting team scoring one run for each of these swaps. Runs are also scored when the ball reaches the boundary of the field or when the ball is bowled illegally.

The fielding team aims to prevent runs by dismissing batters (so they are "out"). Dismissal can occur in various ways, including being bowled (when the ball hits the striker's wicket and dislodges the bails), and by the fielding side either catching the ball after it is hit by the bat but before it hits the ground, or hitting a wicket with the ball before a batter can cross the crease line in front of the wicket. When ten batters have been dismissed, the innings (playing phase) ends and the teams swap roles. Forms of cricket range from traditional Test matches played over five days to the newer Twenty20 format (also known as T20), in which each team bats for a single innings of 20 overs (each "over" being a set of 6 fair opportunities for the batting team to score) and the game generally lasts three to four hours.

Traditionally, cricketers play in all-white kit, but in limited overs cricket, they wear club or team colours. In addition to the basic kit, some players wear protective gear to prevent injury caused by the ball, which is a hard, solid spheroid made of compressed leather with a slightly raised sewn seam enclosing a cork core layered with tightly wound string.

The earliest known definite reference to cricket is to it being played in South East England in the mid-16th century. It spread globally with the expansion of the British Empire, with the first international matches in the second half of the 19th century. The game's governing body is the International Cricket Council (ICC), which has over 100 members, twelve of which are full members who play Test matches. The game's rules, the Laws of Cricket, are maintained by Marylebone Cricket Club (MCC) in London. The sport is followed primarily in South Asia, Australia, New Zealand, the United Kingdom, Southern Africa, and the West Indies.[2]

While cricket has traditionally been played largely by men, women's cricket has experienced large growth in the 21st century.[3]

The most successful side playing international cricket is Australia, which has won eight One Day International trophies, including six World Cups, more than any other country, and has been the top-rated Test side more than any other country.[4][5]
    """

    paper = {"chunks": [{"text": test_text}]}
    summary = summarizer.summarize_chunks(paper)
    print("Summary:\n", summary)
