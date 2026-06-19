"""
Tests for retriever module.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from retriever import (
    CleanerConfig,
    clean_chunk_text,
    _remove_chunk_id_prefix,
    _strip_html,
    _normalize_whitespace,
    _remove_decorative,
    _trim_incomplete_sentence,
)


class TestRemoveChunkIdPrefix:
    """Tests for chunk ID prefix removal."""
    
    def test_with_chunk_id(self) -> None:
        """Test removing chunk ID prefix."""
        assert _remove_chunk_id_prefix("[38644] Корреспондентский счёт") == "Корреспондентский счёт"
        assert _remove_chunk_id_prefix("[0] Текст") == "Текст"
        assert _remove_chunk_id_prefix("[123456] Данные") == "Данные"
    
    def test_without_chunk_id(self) -> None:
        """Test text without prefix unchanged."""
        assert _remove_chunk_id_prefix("Текст без префикса") == "Текст без префикса"
    
    def test_empty_text(self) -> None:
        """Test empty text handling."""
        assert _remove_chunk_id_prefix("") == ""


class TestStripHtml:
    """Tests for HTML stripping."""
    
    def test_html_tags(self) -> None:
        """Test removing HTML tags."""
        assert _strip_html("<b>Счёт</b>") == " Счёт "
        assert _strip_html("<p>Текст</p>") == " Текст "
    
    def test_html_entities(self) -> None:
        """Test decoding HTML entities."""
        assert "текст" in _strip_html("&nbsp;текст&nbsp;")
    
    def test_combined(self) -> None:
        """Test combined HTML and entities."""
        result = _strip_html("<b>Счёт</b>&nbsp;открывается")
        assert "Счёт" in result
        assert "открывается" in result


class TestNormalizeWhitespace:
    """Tests for whitespace normalization."""
    
    def test_nbsp(self) -> None:
        """Test non-breaking space replacement."""
        assert _normalize_whitespace("текст\xa0данные") == "текст данные"
    
    def test_tabs(self) -> None:
        """Test tab replacement."""
        assert _normalize_whitespace("слово\tслово") == "слово слово"
    
    def test_multiple_spaces(self) -> None:
        """Test multiple space collapsing."""
        assert _normalize_whitespace("слово  слово") == "слово слово"
    
    def test_multiple_newlines(self) -> None:
        """Test multiple newline collapsing."""
        assert _normalize_whitespace("текст\n\n\n\nтекст") == "текст\n\nтекст"


class TestRemoveDecorative:
    """Tests for decorative character removal."""
    
    def test_stars(self) -> None:
        """Test removing star separators."""
        assert _remove_decorative("Заголовок\n***\nТекст") == "Заголовок\n\nТекст"
    
    def test_pipes(self) -> None:
        """Test removing pipe separators."""
        assert _remove_decorative("Текст\n||\nДанные") == "Текст\n\nДанные"
    
    def test_underscores(self) -> None:
        """Test removing underscore separators."""
        assert _remove_decorative("Текст\n___\nДанные") == "Текст\n\nДанные"


class TestTrimIncompleteSentence:
    """Tests for incomplete sentence trimming."""
    
    def test_complete_sentence(self) -> None:
        """Test complete sentence unchanged."""
        text = "Позвоните по номеру 8-800."
        assert _trim_incomplete_sentence(text) == text
    
    def test_incomplete_sentence(self) -> None:
        """Test incomplete sentence trimmed."""
        text = "Счёт открывается. Для этого нужно"
        result = _trim_incomplete_sentence(text)
        assert result == "Счёт открывается."
    
    def test_no_sentence_end(self) -> None:
        """Test text without sentence end."""
        text = "Текст без знаков"
        result = _trim_incomplete_sentence(text)
        assert result == text


class TestCleanChunkText:
    """Tests for full chunk cleaning pipeline."""
    
    def test_full_cleaning(self) -> None:
        """Test full cleaning pipeline."""
        text = "[38644] <b>Счёт</b>&nbsp;открывается в банке. Для"
        result = clean_chunk_text(text)
        assert "Счёт" in result
        assert "открывается" in result
        assert "<b>" not in result
        assert "[38644]" not in result
    
    def test_empty_text(self) -> None:
        """Test empty text returns empty."""
        assert clean_chunk_text("") == ""
    
    def test_short_text_filtered(self) -> None:
        """Test short text filtered by min_length."""
        config = CleanerConfig(min_length=100)
        result = clean_chunk_text("Короткий текст", config)
        assert result == ""
    
    def test_custom_config(self) -> None:
        """Test with custom config."""
        config = CleanerConfig(min_length=10, trim_incomplete_sentences=False)
        text = "[1] <b>Текст</b> без обрезки"
        result = clean_chunk_text(text, config)
        assert "Текст" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])