"""
Tests for chunker module.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chunker import (
    split_into_sentences,
    create_chunks,
    chunk_all_websites,
    Chunk,
    clean_text,
    Chunker,
    ChunkerConfig,
)


class TestSplitIntoSentences:
    """Tests for sentence splitting."""
    
    def test_simple_russian_text(self) -> None:
        """Test splitting simple Russian text."""
        text = "Первое предложение. Второе предложение. Третье предложение."
        sentences = split_into_sentences(text)
        assert len(sentences) == 3
        assert sentences[0] == "Первое предложение."
        assert sentences[1] == "Второе предложение."
        assert sentences[2] == "Третье предложение."
    
    def test_empty_text(self) -> None:
        """Test empty text returns empty list."""
        sentences = split_into_sentences("")
        assert sentences == []
    
    def test_whitespace_only(self) -> None:
        """Test whitespace-only text returns empty list."""
        sentences = split_into_sentences("   \n\t  ")
        assert sentences == []
    
    def test_single_sentence(self) -> None:
        """Test single sentence."""
        text = "Это одно предложение."
        sentences = split_into_sentences(text)
        assert len(sentences) == 1
        assert sentences[0] == "Это одно предложение."


class TestCreateChunks:
    """Tests for chunk creation."""
    
    def test_single_short_text(self) -> None:
        """Test single short text creates one chunk."""
        text = "Короткий текст для тестирования."
        chunks = create_chunks(web_id=1, text=text)
        assert len(chunks) == 1
        assert chunks[0].web_id == 1
        assert chunks[0].text == text
    
    def test_long_text_multiple_chunks(self) -> None:
        """Test long text creates multiple chunks."""
        # Create text longer than CHUNK_SIZE (400 chars)
        sentences = ["Это тестовое предложение номер " + str(i) + "." for i in range(20)]
        text = " ".join(sentences)
        chunks = create_chunks(web_id=1, text=text)
        assert len(chunks) > 1
    
    def test_chunk_ids_sequential(self) -> None:
        """Test chunk IDs are sequential."""
        text = "Первое предложение. " * 100
        chunks = create_chunks(web_id=1, text=text, start_chunk_id=5)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_id == 5 + i
    
    def test_no_broken_sentences(self) -> None:
        """Test that sentences are not broken across chunks."""
        text = "Первое предложение. Второе предложение. Третье предложение."
        chunks = create_chunks(web_id=1, text=text)
        
        # All sentences should be present in chunks
        all_text = " ".join(c.text for c in chunks)
        assert "Первое предложение." in all_text
        assert "Второе предложение." in all_text
        assert "Третье предложение." in all_text
    
    def test_empty_text_returns_empty(self) -> None:
        """Test empty text returns empty list."""
        chunks = create_chunks(web_id=1, text="")
        assert chunks == []


class TestChunkAllWebsites:
    """Tests for batch chunking."""
    
    def test_multiple_websites(self) -> None:
        """Test chunking multiple websites."""
        websites = [
            (1, "Текст первого сайта. " * 50),
            (2, "Текст второго сайта. " * 50),
            (3, "Текст третьего сайта. " * 50),
        ]
        chunks = chunk_all_websites(websites)
        
        # All chunks should have unique IDs
        chunk_ids = [c.chunk_id for c in chunks]
        assert len(chunk_ids) == len(set(chunk_ids))
        
        # All chunks should have correct web_id
        for chunk in chunks:
            assert chunk.web_id in [1, 2, 3]


class TestCleanText:
    """Tests for text cleaning before chunking."""
    
    def test_html_tags_removed(self) -> None:
        """Test HTML tags are removed."""
        text = "<b>Счёт</b> открыт."
        result = clean_text(text)
        assert "<b>" not in result
        assert "Счёт" in result
    
    def test_html_entities_decoded(self) -> None:
        """Test HTML entities are decoded."""
        text = "Счёт&nbsp;открыт."
        result = clean_text(text)
        assert "Счёт" in result
        assert "открыт" in result
    
    def test_service_phrases_removed(self) -> None:
        """Test service phrases are removed."""
        text = "Счёт открыт. Обратите внимание: это важно."
        result = clean_text(text)
        assert "Счёт открыт" in result
        assert "Обратите внимание" not in result
    
    def test_empty_text(self) -> None:
        """Test empty text returns empty."""
        assert clean_text("") == ""
    
    def test_whitespace_normalized(self) -> None:
        """Test whitespace is normalized."""
        text = "Слово\xa0\xa0другое"
        result = clean_text(text)
        assert "\xa0" not in result


class TestChunker:
    """Tests for Chunker class."""
    
    def test_chunker_basic(self) -> None:
        """Test basic chunking with Chunker."""
        chunker = Chunker()
        text = "Первое предложение. Второе предложение. Третье предложение."
        chunks = chunker.chunk_text(text)
        assert len(chunks) >= 1
    
    def test_chunker_with_config(self) -> None:
        """Test chunking with custom config."""
        config = ChunkerConfig(chunk_size=100, chunk_overlap=20)
        chunker = Chunker(config)
        text = "Первое предложение. " * 10
        chunks = chunker.chunk_text(text)
        assert len(chunks) >= 1
    
    def test_chunker_empty_text(self) -> None:
        """Test empty text returns empty list."""
        chunker = Chunker()
        chunks = chunker.chunk_text("")
        assert chunks == []
    
    def test_chunker_invalid_overlap(self) -> None:
        """Test invalid overlap raises error."""
        with pytest.raises(ValueError):
            Chunker(ChunkerConfig(chunk_size=100, chunk_overlap=200))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])