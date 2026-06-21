from src.agents.extraction_agent import ExtractionAgent


def test_extract_abstract_returns_text_and_end_index():
    text = "Title Abstract This is the abstract body. Introduction starts here."
    abstract, end = ExtractionAgent.extract_abstract(text)
    assert "abstract body" in abstract.lower()
    # The body marker ("Introduction") should begin at the reported end index.
    assert text[end:].lower().startswith("introduction")


def test_extract_abstract_missing_returns_none_and_zero():
    abstract, end = ExtractionAgent.extract_abstract("No marker section here at all.")
    assert abstract is None
    assert end == 0


def test_chunk_text_respects_size_and_overlap():
    agent = ExtractionAgent()
    words = " ".join(f"w{i}" for i in range(700))
    chunks = agent.chunk_text(words, paper_id="p1", chunk_size=300, overlap=50)

    # Each chunk fits within the embedding token budget (<= 300 words).
    assert all(len(c["text"].split()) <= 300 for c in chunks)
    # More than one chunk for 700 words at step 250.
    assert len(chunks) >= 2
    # Metadata is attached.
    assert all(c["metadata"]["paper_id"] == "p1" for c in chunks)

    # Consecutive chunks overlap (last 50 words of chunk0 == first 50 of chunk1).
    first = chunks[0]["text"].split()
    second = chunks[1]["text"].split()
    assert first[-50:] == second[:50]


def test_chunk_text_drops_tiny_tail():
    agent = ExtractionAgent()
    # 20 words is below the 30-word minimum, so no chunk is produced.
    chunks = agent.chunk_text(" ".join(["x"] * 20))
    assert chunks == []


def test_chunk_text_empty_input():
    assert ExtractionAgent().chunk_text("") == []


def test_clean_text_strips_references_section():
    text = "Real content here. References [1] Some Author, A Paper. 2020."
    cleaned = ExtractionAgent.clean_text(text)
    assert "Real content here" in cleaned
    assert "Some Author" not in cleaned
