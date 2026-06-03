# Code Review - RAG Pipeline for Alfa-Bank MIPT Hackathon

## Overview

This document provides a comprehensive code review of the RAG pipeline project. The codebase has been enhanced with sophisticated text processing, cleaning, and extraction methods.

---

## Critical Issues (Must Fix)

### 1. Configuration Mismatch - `config.py` vs `ChunkerConfig`
- **File**: `alfa_rag_project/src/config.py:31-32`
- **Issue**: `CHUNK_SIZE = 400` and `CHUNK_OVERLAP = 50` are defined in config, but `ChunkerConfig` defaults to `chunk_size: int = 250`
- **Impact**: Legacy `create_chunks()` uses config values, but `Chunker.chunk_text()` uses its own config with different defaults
- **Recommendation**: Either align defaults or document the intentional difference

### 2. Missing `word_matches` Import in `indexer.py`
- **File**: `alfa_rag_project/src/indexer.py`
- **Issue**: `normalize_for_embedding()` in indexer doesn't use the same `word_matches` function that handles Russian morphology
- **Impact**: Inconsistent text normalization between indexing and extraction
- **Recommendation**: Consider using shared normalization utilities

### 3. Hardcoded API Key in `generator.py`
- **File**: `alfa_rag_project/src/generator.py:465`
- **Issue**: `api_key="ollama"` is hardcoded
- **Impact**: Security concern - should use environment variable
- **Recommendation**: Use `os.environ.get("OLLAMA_API_KEY", "ollama")` or similar

---

## High Priority Issues

### 4. Duplicate `normalize_text` Functions
- **Files**: `alfa_rag_project/src/generator.py:225-235` and `alfa_rag_project/src/indexer.py:52-72`
- **Issue**: Both modules have separate `normalize_text`/`normalize_for_embedding` functions with different implementations
- **Impact**: Code duplication, potential inconsistency
- **Recommendation**: Consolidate into shared `text_utils.py` module

### 5. Unused Import in `chunker.py`
- **File**: `alfa_rag_project/src/chunker.py:9`
- **Issue**: `field` imported from dataclasses but not used
- **Impact**: Minor code cleanliness issue

### 6. Missing Error Handling in `test_submission.py`
- **File**: `alfa_rag_project/src/test_submission.py`
- **Issue**: No try/except around generation, unlike `main.py`
- **Impact**: Single failure will stop entire test run
- **Recommendation**: Add error handling similar to main.py

---

## Medium Priority Issues

### 7. Inconsistent Return Types
- **File**: `alfa_rag_project/src/generator.py:480`
- **Issue**: `generate()` returns "Недостаточно информации" for empty context, but `extract_answer_from_context()` returns empty string
- **Impact**: Inconsistent API behavior
- **Recommendation**: Standardize on one approach (empty string or default message)

### 8. Logging vs Print in `test_submission.py`
- **File**: `alfa_rag_project/src/test_submission.py:17-18`
- **Issue**: Uses `print()` while `main.py` uses `logger.info()`
- **Impact**: Inconsistent logging approach
- **Recommendation**: Use logging consistently

### 9. Missing Type Hints in `main.py`
- **File**: `alfa_rag_project/src/main.py`
- **Issue**: Some functions lack complete type hints (e.g., `AnswerCache.get` return type)
- **Impact**: Reduced type safety
- **Recommendation**: Add complete type hints

---

## Low Priority Issues / Suggestions

### 10. Magic Numbers in `generator.py`
- **File**: `alfa_rag_project/src/generator.py:257-260`
- **Issue**: `0.5` threshold for sentence boundary is hardcoded
- **Recommendation**: Make configurable or add constant

### 11. Test Coverage Gaps
- **Missing tests for**:
  - `AnswerCache` class in `main.py`
  - `validate_answer` function
  - `ChunkerConfig` validation edge cases
  - Integration tests for full pipeline

### 12. Documentation Inconsistencies
- **File**: `alfa_rag_project/src/chunker.py:27-37`
- **Issue**: `ChunkerConfig` docstring mentions 250 chars but config.py has 400
- **Recommendation**: Update documentation to match or explain the difference

---

## Architecture Observations

### Positive Patterns
1. **Dataclass Configuration**: All modules use dataclasses for configuration (`ExtractorConfig`, `CleanerConfig`, `ChunkerConfig`)
2. **Pipeline Pattern**: Each module follows a clear pipeline approach (clean → split → build → filter)
3. **Private Helper Functions**: Good use of `_private` functions for internal logic
4. **Backward Compatibility**: Legacy API maintained in `chunker.py`

### Potential Improvements
1. **Shared Text Utilities**: Consider extracting `normalize_text`, `word_matches` to shared module
2. **Configuration Centralization**: All configs could inherit from a base or use a unified config pattern
3. **Error Handling Strategy**: Consider using custom exceptions for better error propagation

---

## Test Coverage Summary

| Module | Tests | Coverage |
|--------|-------|----------|
| chunker.py | 15 | Good - covers cleaning, chunking, legacy API |
| retriever.py | 20 | Good - covers all cleaning functions |
| generator.py | 10 | Good - covers extraction and truncation |
| indexer.py | 11 | Good - covers normalization and deduplication |
| main.py | 0 | Missing - needs cache and validation tests |

**Total**: 70 tests passing

---

## Security Checklist

- [x] No hardcoded passwords in source code
- [x] API key is hardcoded (see issue #3)
- [x] No SQL injection vectors (no SQL used)
- [x] No path traversal vulnerabilities
- [x] No XSS vectors (no web output)
- [ ] No CSRF protection needed (CLI tool)

---

## Performance Considerations

1. **Model Loading**: `SentenceTransformer` and `CrossEncoder` models loaded on init - consider lazy loading
2. **Batch Processing**: `model.encode()` uses batch_size=32 - good for memory
3. **FAISS Index**: `IndexFlatIP` is appropriate for normalized embeddings
4. **Caching**: `AnswerCache` provides good performance optimization for repeated runs

---

## Recommendations Summary

1. **Fix configuration mismatch** between `config.py` and `ChunkerConfig`
2. **Move text utilities** to shared module
3. **Use environment variables** for API keys
4. **Add tests** for `main.py` components
5. **Standardize return types** across modules
6. **Add error handling** to `test_submission.py`