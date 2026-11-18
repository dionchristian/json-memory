import pytest
from unittest.mock import patch, MagicMock

from json_memory.smart import SmartMemory
from json_memory.search import AdvancedSearch, create_search, SearchResult

@pytest.fixture
def mem(tmp_path):
    path = tmp_path / "test_search.json"
    mem = SmartMemory(str(path))
    mem.remember("user.name", "Alice")
    mem.remember("user.email", "alice@example.com")
    mem.remember("project.name", "Trading Bot")
    mem.remember("project.description", "A bot for trading cryptocurrencies")
    mem.remember("project.tags", ["crypto", "trading", "bot"])
    mem.remember("settings.theme", "dark")
    mem.remember("settings.notifications", True)
    return mem

@pytest.fixture
def search(mem):
    return AdvancedSearch(mem)

def test_create_search(mem):
    s = create_search(mem)
    assert isinstance(s, AdvancedSearch)
    assert s.memory is mem


def test_regex_search_value(search):
    results = search.regex_search("alice", field="value")
    assert len(results) == 2
    paths = {r.path for r in results}
    assert "user.name" in paths
    assert "user.email" in paths
    result = next(r for r in results if r.path == "user.name")
    assert result.score == 0.5
    assert result.match_type == "regex"
    assert len(result.highlights) == 1
    assert "value:" in result.highlights[0]

def test_regex_search_path(search):
    results = search.regex_search(r"^user\.", field="path")
    assert len(results) == 2
    paths = {r.path for r in results}
    assert "user.name" in paths
    assert "user.email" in paths

def test_regex_search_both(search):
    search.memory.remember("bot.trading.status", "Trading is active")
    results = search.regex_search("trading", field="both")
    assert len(results) >= 1
    result = next(r for r in results if r.path == "bot.trading.status")
    assert result.score == 1.0
    assert len(result.highlights) == 2
    assert any("path:" in h for h in result.highlights)
    assert any("value:" in h for h in result.highlights)

def test_regex_search_case_sensitive(search):
    results = search.regex_search("Alice", case_sensitive=True, field="value")
    assert len(results) == 1
    assert results[0].path == "user.name"
    results_lower = search.regex_search("alice", case_sensitive=True, field="value")
    assert len(results_lower) == 1
    assert results_lower[0].path == "user.email"

def test_regex_search_invalid_regex(search):
    with pytest.raises(ValueError, match="Invalid regex pattern"):
        search.regex_search("[invalid")

def test_regex_search_no_results(search):
    results = search.regex_search("nonexistent_pattern")
    assert len(results) == 0


def test_fuzzy_search(search):
    results = search.fuzzy_search("tradng", threshold=0.4)
    assert len(results) > 0
    paths = {r.path for r in results}
    assert "project.name" in paths or "project.description" in paths or "project.tags" in paths
    results = search.fuzzy_search("projct", field="path", threshold=0.5)
    assert len(results) >= 2
    assert all(r.path.startswith("project.") for r in results)

def test_full_text_search(search):
    results = search.full_text_search("bot trading")
    assert len(results) > 0
    paths = {r.path for r in results}
    assert "project.description" in paths
    assert "project.name" in paths
    assert search.full_text_search("") == []
    assert search.full_text_search("   ") == []

def test_semantic_search_with_mock(search):
    mock_index = MagicMock()
    mock_index.search.return_value = [
        {"path": "project.description", "score": 0.95},
        {"path": "user.name", "score": 0.45}
    ]
    with patch("json_memory.semantic.get_semantic_index", return_value=mock_index, create=True):
        results = search.semantic_search("crypto bot", top_k=5)
        assert len(results) == 2
        assert results[0].path == "project.description"
        assert results[0].score == 0.95
        assert results[0].match_type == "semantic"
        assert results[1].path == "user.name"
        assert results[1].score == 0.45

def test_search_unified(search):
    with patch.object(search, "regex_search") as mock_regex:
        search.search("/^project/")
        mock_regex.assert_called_once_with("^project")
    with patch.object(search, "full_text_search") as mock_full_text:
        search.search("crypto trading bot")
        mock_full_text.assert_called_once_with("crypto trading bot")
    with patch.object(search, "fuzzy_search") as mock_fuzzy:
        search.search("tradng")
        mock_fuzzy.assert_called_once_with("tradng")
    with patch.object(search, "regex_search") as mock_regex:
        search.search("pattern", search_type="regex")
        mock_regex.assert_called_once_with("pattern")
    with patch.object(search, "semantic_search") as mock_semantic:
        search.search("crypto", search_type="semantic")
        mock_semantic.assert_called_once_with("crypto")
    with pytest.raises(ValueError, match="Unknown search type: invalid"):
        search.search("query", search_type="invalid")

def test_suggest(search):
    suggestions = search.suggest("proje")
    assert len(suggestions) == 3
    assert all(s.startswith("project.") for s in suggestions)
    suggestions = search.suggest("PROJE")
    assert len(suggestions) == 3
    suggestions = search.suggest("proje", limit=2)
    assert len(suggestions) == 2
    assert search.suggest("nonexistent") == []

