"""
RAG Pipeline for Alfa-Bank MIPT Hackathon.
"""

from .config import *
from .chunker import (
    Chunk,
    chunk_all_websites,
    chunk_website,
    Chunker,
    ChunkerConfig,
    clean_text,
)
from .indexer import (
    Indexer,
    build_and_save_index,
    load_index,
    normalize_for_embedding,
    deduplicate_chunks,
)
from .retriever import (
    Retriever,
    create_retriever,
    CleanerConfig,
    clean_chunk_text,
)
from .generator import (
    Generator,
    create_generator,
    ExtractorConfig,
    extract_answer_from_context,
)
from .kaggle_main import KaggleGenerator

__all__ = [
    "Chunk",
    "chunk_all_websites",
    "chunk_website",
    "Chunker",
    "ChunkerConfig",
    "clean_text",
    "Indexer",
    "build_and_save_index",
    "load_index",
    "normalize_for_embedding",
    "deduplicate_chunks",
    "Retriever",
    "create_retriever",
    "CleanerConfig",
    "clean_chunk_text",
    "Generator",
    "create_generator",
    "ExtractorConfig",
    "extract_answer_from_context",
    "KaggleGenerator",
]