from datetime import datetime

from src.agents.synthesizer_agent import SynthesizerAgent


def test_format_year_from_datetime():
    assert SynthesizerAgent._format_year(datetime(2021, 5, 1)) == "2021"


def test_format_year_from_string():
    assert SynthesizerAgent._format_year("2019-03-01") == "2019"


def test_format_year_missing():
    assert SynthesizerAgent._format_year(None) == ""


def test_build_references_formats_entry():
    agent = SynthesizerAgent()
    papers = [{
        "title": "A Great Paper",
        "authors": ["Ada Lovelace", "Alan Turing"],
        "url": "https://arxiv.org/abs/1234.5678",
        "published": datetime(2020, 1, 1),
    }]
    refs = agent._build_references(papers)
    assert "A Great Paper" in refs
    assert "Ada Lovelace" in refs
    assert "(2020)" in refs
    assert "https://arxiv.org/abs/1234.5678" in refs


def test_build_references_truncates_many_authors():
    agent = SynthesizerAgent()
    papers = [{
        "title": "Many Authors",
        "authors": ["A", "B", "C", "D", "E"],
        "url": "",
        "published": None,
    }]
    refs = agent._build_references(papers)
    assert "et al." in refs


def test_synthesize_without_llm_still_lists_papers_and_refs():
    # No summarizer => no LLM overview, but the per-paper listing and the
    # References section must still render.
    agent = SynthesizerAgent(summarizer_agent=None)
    papers = [{
        "title": "Paper One",
        "authors": ["Jane Doe"],
        "url": "https://arxiv.org/abs/1",
        "published": datetime(2022, 1, 1),
    }]
    summaries = ["A concise summary."]
    review = agent.synthesize(papers, summaries, query="test topic")

    assert "# Literature Review" in review
    assert "## Papers" in review
    assert "Paper One" in review
    assert "## References" in review
    assert "## Overview" not in review  # no LLM available
