import time

from src.utils import mask_api_key, prune_directory


def test_mask_api_key_long():
    assert mask_api_key("AIzaSyABCDEFGHIJKLMNOP") == "AIza**************MNOP"


def test_mask_api_key_short_is_fully_masked():
    assert mask_api_key("short") == "*****"


def test_mask_api_key_empty():
    assert mask_api_key("") == ""
    assert mask_api_key(None) == ""


def test_prune_directory_keeps_newest(tmp_path):
    for i in range(5):
        f = tmp_path / f"review_{i}.md"
        f.write_text("x")
        # Stagger mtimes so ordering is deterministic.
        t = time.time() + i
        import os
        os.utime(f, (t, t))

    prune_directory(str(tmp_path), pattern="review_*.md", keep=2)

    remaining = sorted(p.name for p in tmp_path.glob("review_*.md"))
    assert remaining == ["review_3.md", "review_4.md"]


def test_prune_directory_noop_when_under_limit(tmp_path):
    (tmp_path / "review_0.md").write_text("x")
    prune_directory(str(tmp_path), pattern="review_*.md", keep=10)
    assert len(list(tmp_path.glob("review_*.md"))) == 1


def test_prune_directory_missing_dir_is_safe():
    # Should not raise on a non-existent directory.
    prune_directory("/nonexistent/path/xyz", keep=1)
