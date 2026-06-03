"""
Retrieval module for RAG pipeline.
Performs vector search, cross-encoder reranking, and context formatting.
"""

import re
import html
import logging
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
from sentence_transformers import CrossEncoder

from config import TOP_K_RETRIEVAL, TOP_K_RERANK, RERANKER_MODEL
from indexer import Indexer

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Конфигурация очистки
# ─────────────────────────────────────────────

@dataclass
class CleanerConfig:
    """Настройки очистки текста чанка."""
    
    # Минимальная длина текста после очистки (символы)
    min_length: int = 20
    
    # Обрезать незавершённые предложения в конце чанка
    trim_incomplete_sentences: bool = True


# ─────────────────────────────────────────────
# Паттерны очистки
# ─────────────────────────────────────────────

# HTML-теги: <b>, </p>, <br/> и т.д.
_HTML_TAG_RE = re.compile(r"<[^>]{1,100}>", re.UNICODE)

# HTML-сущности: &nbsp; & &#160; и т.д.
# html.unescape() покрывает большинство, но на случай битых сущностей:
_BROKEN_HTML_ENTITY_RE = re.compile(r"&[a-zA-Z]{2,8};?|&#\d{1,5};?", re.UNICODE)

# Chunk ID в начале строки: [38644], [0], [123456]
_CHUNK_ID_PREFIX_RE = re.compile(r"^\s*\[\d+\]\s*", re.UNICODE)

# Неразрывные и управляющие пробелы: \xa0, \u200b, \t и подобные
_WHITESPACE_VARIANTS_RE = re.compile(
    r"[\xa0\u00a0\u200b\u200c\u200d\u2060\ufeff\t]+",
    re.UNICODE,
)

# Повторяющиеся пробелы (после замены спецсимволов)
_MULTI_SPACE_RE = re.compile(r" {2,}", re.UNICODE)

# Повторяющиеся переносы строк (больше двух подряд)
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}", re.UNICODE)

# Мусорные символы: вертикальная черта, звёздочки, подчёркивания как разделители
_DECORATIVE_RE = re.compile(r"[|*]{2,}|_{3,}", re.UNICODE)


# ─────────────────────────────────────────────
# Функции очистки
# ─────────────────────────────────────────────

def _remove_chunk_id_prefix(text: str) -> str:
    """
    Удаляет префикс chunk ID из начала строки.
    
    Examples:
        "[38644] Корреспондентский счёт..." → "Корреспондентский счёт..."
        "[0] Текст"                         → "Текст"
    """
    return _CHUNK_ID_PREFIX_RE.sub("", text)


def _strip_html(text: str) -> str:
    """
    Убирает HTML-теги и декодирует HTML-сущности.
    
    Порядок важен: сначала unescape (чтобы <b> стало <b>),
    затем strip тегов.
    
    Examples:
        "<b>Счёт</b>"      → "Счёт"
        "&nbsp;текст&" → " текст&"  → после trim → "текст"
        "&#160;данные"     → "\xa0данные" → заменяется далее
    """
    # 1. Декодируем HTML-сущности (&nbsp; → \xa0, & → &)
    text = html.unescape(text)
    
    # 2. Убираем теги
    text = _HTML_TAG_RE.sub(" ", text)
    
    # 3. На случай битых сущностей которые html.unescape не поймал
    text = _BROKEN_HTML_ENTITY_RE.sub(" ", text)
    
    return text


def _normalize_whitespace(text: str) -> str:
    """
    Нормализует все виды пробельных символов.
    
    Examples:
        "текст\\xa0\\xa0данные" → "текст данные"
        "слово\\t\\tслово"      → "слово слово"
    """
    # Заменяем спецпробелы на обычный
    text = _WHITESPACE_VARIANTS_RE.sub(" ", text)
    
    # Схлопываем многократные пробелы
    text = _MULTI_SPACE_RE.sub(" ", text)
    
    # Схлопываем многократные переносы строк
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    
    return text


def _remove_decorative(text: str) -> str:
    """
    Убирает декоративные символы-разделители.
    
    Examples:
        "Заголовок\n***\nТекст"    → "Заголовок\n\nТекст"
        "Раздел\n---|---\nДанные"  → без изменений (не попадает)
    """
    return _DECORATIVE_RE.sub("", text)


def _trim_incomplete_sentence(text: str) -> str:
    """
    Обрезает незавершённое предложение в конце чанка.
    
    Чанки часто обрываются посередине предложения из-за
    разбивки документа по фиксированному размеру.
    
    Logic:
        Если текст не заканчивается на [.!?»")],
        находим последнее полное предложение и обрезаем до него.
    
    Examples:
        "Счёт открывается. Для этого нужно" → "Счёт открывается."
        "Позвоните по номеру 8-800."         → без изменений (полное)
    """
    stripped = text.rstrip()
    
    # Проверяем: текст заканчивается на знак конца предложения?
    if re.search(r"[.!?»\"\)]\s*$", stripped, re.UNICODE):
        return stripped  # Уже полное
    
    # Ищем последнюю точку/восклицательный/вопросительный знак
    last_end = max(
        stripped.rfind("."),
        stripped.rfind("!"),
        stripped.rfind("?"),
        stripped.rfind("»"),
    )
    
    if last_end == -1:
        # Нет ни одного конца предложения — возвращаем как есть
        # (лучше неполный текст, чем пустота)
        return stripped
    
    return stripped[:last_end + 1]


def clean_chunk_text(
    text: str,
    config: Optional[CleanerConfig] = None,
) -> str:
    """
    Полная очистка текста чанка для подачи в LLM/fallback.
    
    Pipeline:
        1. Убрать chunk ID префикс
        2. Убрать HTML
        3. Нормализовать пробелы
        4. Убрать декоративные символы
        5. Обрезать незавершённые предложения
        6. Финальный trim
    
    Args:
        text: Сырой текст чанка из индекса.
        config: Настройки очистки.
        
    Returns:
        Очищенный текст или пустая строка если текст стал слишком коротким.
        
    Examples:
        >>> clean_chunk_text("[38644] <b>Счёт</b>&nbsp;открывается в банке. Для")
        "Счёт открывается в банке."
    """
    if config is None:
        config = CleanerConfig()
    
    if not text or not text.strip():
        return ""
    
    # Шаг 1: chunk ID
    text = _remove_chunk_id_prefix(text)
    
    # Шаг 2: HTML
    text = _strip_html(text)
    
    # Шаг 3: пробелы
    text = _normalize_whitespace(text)
    
    # Шаг 4: декоративные символы
    text = _remove_decorative(text)
    
    # Шаг 5: незавершённые предложения
    if config.trim_incomplete_sentences:
        text = _trim_incomplete_sentence(text)
    
    # Шаг 6: финальный trim
    text = text.strip()
    
    # Проверка минимальной длины
    if len(text) < config.min_length:
        logger.debug("Chunk too short after cleaning (%d chars), skipping", len(text))
        return ""
    
    return text


# ─────────────────────────────────────────────
# Retriever
# ─────────────────────────────────────────────

class Retriever:
    """
    Handles retrieval with FAISS and reranking.
    """
    
    def __init__(
        self,
        indexer: Indexer,
        reranker_model: str = RERANKER_MODEL,
        cleaner_config: Optional[CleanerConfig] = None,
    ):
        """
        Initialize retriever with indexer and reranker.
        
        Args:
            indexer: Indexer instance with loaded FAISS index.
            reranker_model: Name of the cross-encoder model.
            cleaner_config: Settings for chunk text cleaning.
        """
        self.indexer = indexer
        self.reranker = CrossEncoder(reranker_model)
        self.cleaner_config = cleaner_config or CleanerConfig()
    
    def retrieve(self, query: str) -> List[Tuple[int, str, float]]:
        """
        Retrieve top-k chunks for a query.
        
        Args:
            query: Search query.
            
        Returns:
            List of (chunk_id, text, score) tuples, top-k after reranking.
            Text здесь — сырой, очистка происходит в get_context().
            
        Raises:
            RuntimeError: If indexer is not built or loaded.
        """
        if not self.indexer.is_built():
            raise RuntimeError("Indexer not built or loaded")
        
        query_embedding = self.indexer.model.encode(
            [query],
            normalize_embeddings=True,
        )
        query_embedding = query_embedding.astype(np.float32)
        
        scores, indices = self.indexer.index.search(
            query_embedding,
            TOP_K_RETRIEVAL,
        )
        
        candidate_ids = indices[0].tolist()
        candidates: List[Tuple[int, str]] = []
        
        for cid in candidate_ids:
            chunk_data = self.indexer.get_chunk_by_id(cid)
            if chunk_data is not None:
                candidates.append((cid, chunk_data["text"]))
        
        if not candidates:
            return []
        
        rerank_scores = self.reranker.predict(
            [(query, text) for _, text in candidates]
        )
        
        top_indices = np.argsort(rerank_scores)[::-1][:TOP_K_RERANK]
        
        results = []
        for idx in top_indices:
            chunk_id, text = candidates[idx]
            score = float(rerank_scores[idx])
            results.append((chunk_id, text, score))
        
        return results
    
    def get_context(
        self,
        query: str,
        cleaner_config: Optional[CleanerConfig] = None,
    ) -> str:
        """
        Get formatted context for LLM / fallback generation.
        
        Changes vs original:
            - Chunk ID убран из текста контекста
            - Каждый чанк проходит clean_chunk_text()
            - Пустые чанки после очистки пропускаются
            - Разделитель между чанками — чистый \\n\\n
        
        Args:
            query: Search query.
            cleaner_config: Override cleaner settings for this call.
            
        Returns:
            Clean context string ready for LLM, or "" if nothing retrieved.
        """
        results = self.retrieve(query)
        
        if not results:
            return ""
        
        config = cleaner_config or self.cleaner_config
        
        context_parts = []
        
        for chunk_id, raw_text, score in results:
            cleaned = clean_chunk_text(raw_text, config)
            
            if not cleaned:
                logger.debug(
                    "Chunk %d dropped after cleaning (score=%.3f)",
                    chunk_id,
                    score,
                )
                continue
            
            context_parts.append(cleaned)
        
        return "\n\n".join(context_parts)


# ─────────────────────────────────────────────
# Фабрика
# ─────────────────────────────────────────────

def create_retriever(indexer: Indexer) -> Retriever:
    """
    Convenience function to create retriever.
    
    Args:
        indexer: Indexer instance.
        
    Returns:
        Retriever instance with default settings.
    """
    return Retriever(indexer)