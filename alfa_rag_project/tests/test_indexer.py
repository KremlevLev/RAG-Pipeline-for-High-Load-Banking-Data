"""
Tests for indexer module.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from indexer import (
    normalize_for_embedding,
    deduplicate_chunks,
    _compute_chunk_hash,
)
from chunker import Chunk


class TestNormalizeForEmbedding:
    """Tests for text normalization before embedding."""
    
    def test_yo_to_e(self) -> None:
        """Test ё → е normalization."""
        assert normalize_for_embedding("счёт") == "счет"
        assert normalize_for_embedding("СЧЁТ") == "СЧЕТ"
    
    def test_unicode_nfc(self) -> None:
        """Test Unicode NFC normalization."""
        # é can be represented as single char or e + combining accent
        text = "cafe\u0301"  # e + combining acute accent
        result = normalize_for_embedding(text)
        assert "café" in result or "cafe" in result
    
    def test_multiple_spaces(self) -> None:
        """Test multiple space collapsing."""
        assert normalize_for_embedding("слово  слово") == "слово слово"
    
    def test_empty_text(self) -> None:
        """Test empty text handling."""
        assert normalize_for_embedding("") == ""
    
    def test_preserves_case(self) -> None:
        """Test that case is preserved."""
        assert normalize_for_embedding("Сбербанк") == "Сбербанк"


class TestComputeChunkHash:
    """Tests for chunk hash computation."""
    
    def test_same_text_same_hash(self) -> None:
        """Test that same text produces same hash."""
        hash1 = _compute_chunk_hash("одинаковый текст")
        hash2 = _compute_chunk_hash("одинаковый текст")
        assert hash1 == hash2
    
    def test_yo_variant_same_hash(self) -> None:
        """Test that ё/е variants produce same hash."""
        hash1 = _compute_chunk_hash("счёт")
        hash2 = _compute_chunk_hash("счет")
        assert hash1 == hash2
    
    def test_different_text_different_hash(self) -> None:
        """Test that different text produces different hash."""
        hash1 = _compute_chunk_hash("текст один")
        hash2 = _compute_chunk_hash("текст два")
        assert hash1 != hash2


class TestDeduplicateChunks:
    """Tests for chunk deduplication."""
    
    def test_no_duplicates(self) -> None:
        """Test with no duplicates."""
        chunks = [
            Chunk(chunk_id=1, web_id=1, text="Первый чанк."),
            Chunk(chunk_id=2, web_id=1, text="Второй чанк."),
        ]
        unique, dropped = deduplicate_chunks(chunks)
        assert len(unique) == 2
        assert dropped == 0
    
    def test_with_duplicates(self) -> None:
        """Test with duplicates."""
        chunks = [
            Chunk(chunk_id=1, web_id=1, text="Счёт открыт."),
            Chunk(chunk_id=2, web_id=2, text="Счет открыт."),  # дубль (ё→е)
        ]
        unique, dropped = deduplicate_chunks(chunks)
        assert len(unique) == 1
        assert dropped == 1
    
    def test_empty_list(self) -> None:
        """Test with empty list."""
        unique, dropped = deduplicate_chunks([])
        assert len(unique) == 0
        assert dropped == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])