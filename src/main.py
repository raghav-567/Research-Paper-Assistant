from src.agents.search_agent import SearchAgent
from src.agents.extraction_agent import ExtractionAgent
from src.agents.summarizer_agent import SummarizerAgent
from src.agents.synthesizer_agent import SynthesizerAgent
from src.rag_pipeline import RAGPipeline
from src.utils import get_logger
import faiss
import os

logger = get_logger("Main")

def run(query):
    # 1. Init agents
    search = SearchAgent(pdf_dir="pdfs")
    extraction = ExtractionAgent()
    summarizer = SummarizerAgent()
    synthesizer = SynthesizerAgent()

    # 2. Init FAISS + metadata store
    dim = 384  # embedding size of all-MiniLM-L6-v2
    index = faiss.IndexFlatL2(dim)
    id_to_metadata = {}

    rag = RAGPipeline(
        search_agent=search,
        extraction_agent=extraction,
        summarizer_agent=summarizer,
        index=index,
        id_to_metadata=id_to_metadata
    )

    # 3. Retrieve papers
    papers = search.search_arxiv(query, max_results=3)

    # 4. Chunk + embed + add to FAISS
    for p in papers:
        if p.get("pdf_path") and os.path.exists(p["pdf_path"]):
            pages = extraction.parse_pdf(p["pdf_path"])
            text = " ".join([page["text"] for page in pages if "text" in page])
        else:
            logger.warning(f"No PDF for {p['title']}, falling back to abstract")
            text = p["summary"]

        chunks = extraction.chunk_text(text, paper_id=p["id"])
        rag.build_index(chunks, paper_info=p)   # ✅ Add chunks + paper metadata to FAISS

    # 5. Query RAG (retrieve top papers + chunks)
    papers, summaries = rag.query(query, top_k_chunks=100, top_k_papers=3, chunks_per_paper=3)

    # 6. Synthesize into final lit review
    review = synthesizer.synthesize(papers, summaries)

    # 7. Save output
    os.makedirs("outputs", exist_ok=True)
    output_file = "outputs/sample_review_full_5.md"
    with open(output_file, "w") as f:
        f.write(review)
    logger.info(f"Literature review saved in {output_file}")

if __name__ == "__main__":
    run("machine learning is needed")
