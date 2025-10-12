import mlx.core as mx
from mlx_lm import load, generate
from sentence_transformers import SentenceTransformer
from utils import get_logger
import numpy as np
from typing import List, Dict, Optional
import time

logger = get_logger("MLXSummarizerAgent")

class MLXSummarizerAgent:
    
    def __init__(
        self,
        model_name: str = "mlx-community/Meta-Llama-3-8B-Instruct-4bit",
        max_tokens: int = 256,
        use_lora: bool = False,
        lora_adapter_path: Optional[str] = None
    ):
       
        logger.info(f"Initializing MLX Summarizer with model: {model_name}")
        
        self.model_name = model_name
        self.max_tokens = max_tokens
        
        try:
            logger.info("Loading MLX model... This may take a moment.")
            self.model, self.tokenizer = load(model_name)
            logger.info("✓ Model loaded successfully")
            
          
            if use_lora and lora_adapter_path:
                logger.info(f"Loading LoRA adapter from: {lora_adapter_path}")
                logger.info("✓ LoRA adapter loaded")
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        
        logger.info("Loading embedding model...")
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("✓ Embedding model loaded")
        
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
            prompt = f"""<s>[INST] 
    You are an academic summarization expert specializing in research and technical writing.
    Your goal is to produce a concise, high-quality summary that captures the *core idea*, *methodology or structure*, and *major implications or findings*.

    Write the summary in three short sections:
    1. **Overview and Core Idea**
    2. **Details or Methodology (if applicable)**
    3. **Significance or Broader Context**

    Maintain a professional, coherent tone. Avoid bullet points unless conceptually needed.
    Limit the output to around 200–250 words.
    Use smooth transitions between sections.

    Text to summarize:
    {text}
    [/INST]</s>"""
            
        elif task == "abstract":
            prompt = f"""<s>[INST]
    You are an AI research writing assistant.
    Write a formal academic abstract (150–200 words) for the following text.
    The abstract should include:
    - Background or motivation
    - Core methods or structure
    - Key results or findings
    - Brief conclusion or implications

    Keep it concise, objective, and professional in tone.

    Text:
    {text}
    [/INST]</s>"""
            
        elif task == "key_points":
            prompt = f"""<s>[INST]
    Extract the key insights, findings, or implications from the following text.
    List them clearly using numbered bullet points, focusing only on the most important information.
    Avoid general background or filler sentences.

    Text:
    {text}
    [/INST]</s>"""
        
        else:
            prompt = f"""<s>[INST]
    Summarize the following text concisely and clearly.
    Text:
    {text}
    [/INST]</s>"""
        
        return prompt

    
    def _summarize_text(
        self, 
        text: str, 
        max_output_tokens: int = 256,
        task: str = "summarize"
    ) -> str:
       
        if not text or not text.strip():
            return "No content available to summarize."
        
     
        max_input_length = 3000  
        if len(text) > max_input_length:
            logger.warning(f"Text truncated from {len(text)} to {max_input_length} chars")
            text = text[:max_input_length] + "..."
        
        try:
            
            prompt = self._create_prompt(text, task)
            logger.info(f"Generating summary (max {max_output_tokens} tokens)...")
            start_time = time.time()
            
            response = generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=max_output_tokens,
                
                verbose=False
            )
            
            elapsed = time.time() - start_time
            logger.info(f"✓ Summary generated in {elapsed:.2f}s")
            summary = response.strip()
            if "[/INST]" in summary:
                summary = summary.split("[/INST]")[-1].strip()
            
            return summary if summary else "Unable to generate summary."
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return f"Error: {str(e)}"
    
    def summarize_chunks(
        self, 
        paper_data: Dict, 
        query: Optional[str] = None, 
        k: int = 5
    ) -> str:
       
        logger.info("Starting paper summarization with MLX")
        
        abstract = None
        chunks = []
        
        # Handle input format
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
        if abstract and abstract.strip():
            logger.info("Summarizing abstract...")
            abs_summary = self._summarize_text(
                abstract, 
                max_output_tokens=150,
                task="abstract"
            )
        body_summary = ""
        if chunks:
            chunks = [c for c in chunks if c and c.strip()]
            
            if chunks:
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
                    max_output_tokens=200,
                    task=input()
                )
   
        final_summary = ""
        
        if abs_summary and not abs_summary.startswith("Error"):
            final_summary = abs_summary
        
        if body_summary and not body_summary.startswith("Error"):
            if final_summary:
                final_summary += f"\n\n{body_summary}"
            else:
                final_summary = body_summary
        
        if not final_summary:
          
            if abstract:
                final_summary = abstract[:500] + "..."
            elif chunks:
                final_summary = chunks[0][:500] + "..."
            else:
                final_summary = "Unable to generate summary."
        
        logger.info(f"Final summary: {len(final_summary)} chars")
        return final_summary
    
    def batch_summarize(
        self, 
        texts: List[str], 
        max_output_tokens: int = 256
    ) -> List[str]:
       
        logger.info(f"Batch summarizing {len(texts)} texts...")
        summaries = []
        
        for i, text in enumerate(texts):
            logger.info(f"Processing {i+1}/{len(texts)}")
            summary = self._summarize_text(text, max_output_tokens)
            summaries.append(summary)
        
        return summaries
    
    def extract_key_points(self, text: str) -> List[str]:
  
        response = self._summarize_text(text, max_output_tokens=200, task="key_points")
        lines = response.strip().split('\n')
        key_points = []
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                point = line.lstrip('0123456789.-•').strip()
                if point:
                    key_points.append(point)
        
        return key_points if key_points else [response]
    
    def __del__(self):
        try:
            if hasattr(self, "logger") and self.logger:
                self.logger.info("Cleaning up MLX model...")
        except Exception:
            pass




RECOMMENDED_MODELS = {
    "fast": "mlx-community/Mistral-7B-Instruct-v0.2-4bit",  # 4-bit, fastest
    "balanced": "mlx-community/Mistral-7B-Instruct-v0.3-4bit",  # 8-bit, good quality
    "quality": "mlx-community/Meta-Llama-3-8B-Instruct-4bit",  # Best quality
    "small": "mlx-community/phi-2-4bit",  # Smallest, very fast
}


if __name__ == "__main__":
    summarizer = MLXSummarizerAgent(
        model_name=RECOMMENDED_MODELS["balanced"],
        
    )
    
  
    test_text = """
   Cricket is a bat-and-ball game that is played between two teams of eleven players on a field, at the centre of which is a 22-yard (20-metre; 66-foot) pitch with a wicket at each end, each comprising two bails (small sticks) balanced on three stumps. Two players from the batting team, the striker and nonstriker, stand in front of either wicket holding bats, while one player from the fielding team, the bowler, bowls the ball toward the striker's wicket from the opposite end of the pitch. The striker's goal is to hit the bowled ball with the bat and then switch places with the nonstriker, with the batting team scoring one run for each of these swaps. Runs are also scored when the ball reaches the boundary of the field or when the ball is bowled illegally.

The fielding team aims to prevent runs by dismissing batters (so they are "out"). Dismissal can occur in various ways, including being bowled (when the ball hits the striker's wicket and dislodges the bails), and by the fielding side either catching the ball after it is hit by the bat but before it hits the ground, or hitting a wicket with the ball before a batter can cross the crease line in front of the wicket. When ten batters have been dismissed, the innings (playing phase) ends and the teams swap roles. Forms of cricket range from traditional Test matches played over five days to the newer Twenty20 format (also known as T20), in which each team bats for a single innings of 20 overs (each "over" being a set of 6 fair opportunities for the batting team to score) and the game generally lasts three to four hours.

Traditionally, cricketers play in all-white kit, but in limited overs cricket, they wear club or team colours. In addition to the basic kit, some players wear protective gear to prevent injury caused by the ball, which is a hard, solid spheroid made of compressed leather with a slightly raised sewn seam enclosing a cork core layered with tightly wound string.

The earliest known definite reference to cricket is to it being played in South East England in the mid-16th century. It spread globally with the expansion of the British Empire, with the first international matches in the second half of the 19th century. The game's governing body is the International Cricket Council (ICC), which has over 100 members, twelve of which are full members who play Test matches. The game's rules, the Laws of Cricket, are maintained by Marylebone Cricket Club (MCC) in London. The sport is followed primarily in South Asia, Australia, New Zealand, the United Kingdom, Southern Africa, and the West Indies.[2]

While cricket has traditionally been played largely by men, women's cricket has experienced large growth in the 21st century.[3]

The most successful side playing international cricket is Australia, which has won eight One Day International trophies, including six World Cups, more than any other country, and has been the top-rated Test side more than any other country.[4][5]
    """
    
    summary = summarizer._summarize_text(test_text)
    print("Summary:", summary)