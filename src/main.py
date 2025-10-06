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
    # 1. Initialize agents
    search = SearchAgent(pdf_dir="pdfs")
    extraction = ExtractionAgent()
    summarizer = SummarizerAgent(api_key="AIzaSyBeLf_qcAilC-_Ljqk5Pqd9R86cmsesmZ8")
    synthesizer = SynthesizerAgent()

    # 2. Initialize FAISS + metadata store
    dim = summarizer.embedding_model.get_sentence_embedding_dimension()
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
        text = ""
        if p.get("pdf_path") and os.path.exists(p["pdf_path"]):
            parsed = extraction.parse_pdf(p["pdf_path"], paper_id=p["id"])
            # Fallback for failed parsing:
            if parsed and (parsed.get("abstract") or (parsed.get("chunks") and len(parsed["chunks"]) > 0)):
                # Prefer abstract + body if failed parsing whole paper
                text = (parsed.get("abstract") or "") + " " + " ".join([c["text"] for c in parsed.get("chunks", [])])
            else:
                logger.warning(f"PDF parsing empty for {p['title']}, using arXiv summary.")
                text = p.get("summary", "")
        else:
            logger.warning(f"No PDF for {p['title']}, using arXiv summary.")
            text = p.get("summary", "") 

        # Ensure we add at least the abstract/summary as a chunk if the above failed
        if not text.strip():
            logger.warning(f"No extractable text for {p['title']}, skipping.")
            continue
        chunks = extraction.chunk_text(text, paper_id=p["id"])
        if not chunks:
            summary = p.get("summary", "")
            if summary.strip():
                chunks = [{"text": summary, "metadata": {"paper_id": p["id"]}}]
            else:
                logger.warning(f"No chunks or summary for {p['title']}")
                continue
        rag.build_index(chunks, paper_info=p)


    # 5. Query RAG (retrieve top papers + chunks)
    papers, summaries = rag.query(query, top_k_chunks=300, top_k_papers=3, chunks_per_paper=10)

    # 6. Synthesize into final literature review
    review = synthesizer.synthesize(papers, summaries)

    # 7. Save output
    os.makedirs("outputs", exist_ok=True)
    output_file = "outputs/sample_review_full_7.md"
    with open(output_file, "w") as f:
        f.write(review)
    logger.info(f"Literature review saved in {output_file}")

if __name__ == "__main__":
    run("Machine learning")
