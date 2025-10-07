from src.agents.search_agent import SearchAgent
from src.agents.extraction_agent import ExtractionAgent
from src.agents.summarizer_agent import SummarizerAgent
from src.agents.synthesizer_agent import SynthesizerAgent
from src.rag_pipeline import RAGPipeline
from src.utils import get_logger
import faiss
import os
import time
from dotenv import load_dotenv

logger = get_logger("Main")
load_dotenv()

api_key = os.getenv("API_KEY")

def run(query):
    # 1. Initialize agents
    search = SearchAgent(pdf_dir="pdfs")
    extraction = ExtractionAgent()
    summarizer = SummarizerAgent(api_key)
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
    logger.info(f"Found {len(papers)} papers from arXiv")

    # 4. Process each paper with single-pass summarization
    papers_with_summaries = []
    
    for i, p in enumerate(papers):
        paper_id = p["id"]
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing paper {i+1}/{len(papers)}: {p['title']}")
        logger.info(f"{'='*60}")
        
        # Extract content
        content_to_summarize = ""
        chunks_for_index = []
        
        if p.get("pdf_path") and os.path.exists(p["pdf_path"]):
            try:
                parsed = extraction.parse_pdf(p["pdf_path"], paper_id=paper_id)
                if parsed:
                    abstract = parsed.get("abstract", "")
                    chunks = parsed.get("chunks", [])
                    
                    logger.info(f"Extracted: Abstract={bool(abstract)}, Chunks={len(chunks)}")
                    
                    # Prepare content for summarization
                    if abstract:
                        content_to_summarize = f"Abstract: {abstract}\n\n"
                    
                    # Add top 3 chunks to content
                    if chunks:
                        chunks_for_index = chunks  # Save for indexing
                        chunk_texts = [c["text"] for c in chunks[:3]]
                        content_to_summarize += "Main Content: " + " ".join(chunk_texts)
                    
                    # Truncate if too long (keep within token limits)
                    if len(content_to_summarize) > 6000:
                        content_to_summarize = content_to_summarize[:6000]
                        
            except Exception as e:
                logger.error(f"Error parsing PDF: {e}")
        
        # Fallback to arXiv summary if no content extracted
        if not content_to_summarize.strip():
            logger.warning(f"No PDF content, using arXiv summary")
            content_to_summarize = p.get("summary", "")
            # Create a chunk from the summary for indexing
            if content_to_summarize:
                chunks_for_index = [{
                    "text": content_to_summarize,
                    "metadata": {"paper_id": paper_id}
                }]
        
        # Generate summary with single API call
        summary = ""
        if content_to_summarize.strip():
            logger.info(f"Generating summary for paper {i+1}...")
            summary = summarizer._summarize_text(
                content_to_summarize, 
                max_output_tokens=300
            )
            logger.info(f"Summary generated: {len(summary)} chars")
            
            # Add delay between papers to avoid rate limiting
            if i < len(papers) - 1:
                logger.info("Waiting 2s before next paper...")
                time.sleep(2)
        else:
            summary = "No content available to summarize."
            logger.warning(f"No content for {p['title']}")
        
        # Add to FAISS index
        if chunks_for_index:
            try:
                rag.build_index(chunks_for_index, paper_info=p)
                logger.info(f"Added {len(chunks_for_index)} chunks to index")
            except Exception as e:
                logger.error(f"Error building index: {e}")
        
        papers_with_summaries.append({
            "paper": p,
            "summary": summary
        })

    # 5. Synthesize into final literature review
    logger.info("\n" + "="*60)
    logger.info("Generating final literature review...")
    logger.info("="*60)
    
    paper_list = [item["paper"] for item in papers_with_summaries]
    summary_list = [item["summary"] for item in papers_with_summaries]
    
    review = synthesizer.synthesize(paper_list, summary_list)

    # 6. Save output
    os.makedirs("outputs", exist_ok=True)
    output_file = "outputs/sample_review_optimized.md"
    with open(output_file, "w") as f:
        f.write(review)
    logger.info(f"Literature review saved in {output_file}")
    
    # Print summary stats
    logger.info("\n" + "="*60)
    logger.info("Summary Statistics:")
    for i, item in enumerate(papers_with_summaries):
        logger.info(f"{i+1}. {item['paper']['title'][:50]}...")
        logger.info(f"   Summary length: {len(item['summary'])} chars")
        logger.info(f"   Status: {'✓ Success' if len(item['summary']) > 100 else '✗ Failed'}")
    logger.info("="*60)
    
    return review

if __name__ == "__main__":
    run("Machine learning")