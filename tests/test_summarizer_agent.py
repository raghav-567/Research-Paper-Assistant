from src.agents.summarizer_agent import SummarizerAgent


def _bare_agent():
    # Bypass __init__ (which needs the genai client) — the methods under test
    # only use regex/string logic and no instance state.
    return object.__new__(SummarizerAgent)


def test_fallback_summary_extracts_sentences():
    agent = _bare_agent()
    text = "Abstract: First sentence. Second sentence. Third sentence. Fourth. Fifth."
    out = agent._fallback_summary(text, max_chars=200)
    assert "First sentence" in out
    # The "Abstract:" label is stripped.
    assert not out.startswith("Abstract:")
    # Capped at 4 sentences.
    assert "Fifth" not in out


def test_fallback_summary_empty():
    agent = _bare_agent()
    assert agent._fallback_summary("") == "No content available to summarize."


def test_is_quota_error_detects_quota_messages():
    agent = _bare_agent()
    assert agent._is_quota_error("You exceeded your current quota")
    assert agent._is_quota_error("429 ... quota ...")
    assert agent._is_quota_error("generate_content_free_tier limit")


def test_is_quota_error_ignores_other_errors():
    agent = _bare_agent()
    assert not agent._is_quota_error("Connection reset by peer")
    assert not agent._is_quota_error("")
    assert not agent._is_quota_error(None)
