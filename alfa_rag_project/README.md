# Alfa-Bank RAG Hackathon Pipeline

Pure Python RAG system built for the Alfa-Bank MIPT hackathon. No LangChain, no LlamaIndex — just explicit retrieval, reranking, generation, and post-processing.

## Current Entry Point

**`src/kaggle_main.py`** is the only active pipeline entry point.

All other legacy scripts have been removed from the repository.

## What This Project Does

The pipeline answers Alfa-Bank FAQ questions using a RAG architecture:

1. **Chunk** Russian banking website text into sentence-aware chunks
2. **Index** chunks with BGE-M3 embeddings in FAISS
3. **Retrieve** candidates using hybrid search:
   - FAISS semantic search
   - BM25 lexical search
   - Cross-encoder reranking with `BAAI/bge-reranker-v2-m3`
4. **Generate** answers with a local LLM
5. **Post-process** answers to remove noise and enforce BERTScore-Recall-L length constraints

## Metric: BERTScore-Recall-L

The hackathon metric is **BERTScore-Recall-L**, which includes a relative length penalty:

- **No penalty** if answer length `L <= 1.5 * reference_length`
- **Linear penalty** from `1.5x` to `3x` reference length
- **Zero score** if `L >= 3x` reference length

Reference answers in `data/sample_submission.csv` are typically **250–450+ characters**, often 3–6 sentences or bullet lists.

**Important:** Hard truncation to 150 characters destroys recall on complex list answers. The current pipeline uses adaptive truncation with `MAX_RESPONSE_CHARS=450` and `MAX_SENTENCES=5`.

## Project Structure

```text
alfa_rag_project/
├── README.md
├── requirements.txt
├── evaluate_metric.py
├── hypotheses.md
├── data/
│   ├── websites.csv              # Input: web_id, text
│   ├── questions.csv             # Input: q_id, query
│   ├── sample_submission.csv     # Reference answers / allowed leak
│   ├── submission.csv            # Output predictions
│   ├── faiss_index.bin           # Generated FAISS index
│   └── chunk_mapping.json        # Generated chunk metadata
└── src/
    ├── config.py                 # Centralized configuration
    ├── chunker.py                # Sentence-aware chunking
    ├── indexer.py                # FAISS index + metadata
    ├── retriever.py              # Hybrid retrieval + reranking
    ├── generator.py              # LLM generation + fallback extraction
    ├── kaggle_main.py            # Main pipeline orchestrator
    ├── __init__.py
    └── __main__.py
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Build index and generate submission

```bash
cd alfa_rag_project/src
python kaggle_main.py --build-index --fast-quality --no-validate
```

### Generate only first 100 questions for debugging

```bash
cd alfa_rag_project/src
python kaggle_main.py --limit 100 --fast-quality --no-validate
```

### Use Vikhr-1B instead

```bash
cd alfa_rag_project/src
python kaggle_main.py --build-index --model vikhr-1b --fast-quality --no-validate
```

### Use vLLM explicitly

```bash
cd alfa_rag_project/src
python kaggle_main.py --build-index --model vikhr-1b --vllm --vllm-batch-size 8 --no-validate
```

## CLI Options

| Flag | Description |
|------|-------------|
| `--build-index` | Rebuild FAISS index from scratch |
| `--model MODEL` | LLM model key from `KAGGLE_MODELS` |
| `--vllm` | Use vLLM instead of Hugging Face pipeline |
| `--vllm-batch-size N` | Batch size for vLLM continuous batching |
| `--fastGPU` | Single L4 mode with higher GPU memory utilization |
| `--fast-quality` | Fast quality mode: fewer candidates + stronger model |
| `--limit N` | Generate only first N questions for debugging |
| `--no-validate` | Disable answer validation |
| `--min-overlap N` | Minimum word overlap for validation |
| `--cache-path PATH` | Custom answer cache path |

## Available Models

In `src/config.py`:

| Key | Model | Notes |
|-----|-------|-------|
| `vikhr-1b-finetuned` | `lirex111/vikhrllama1B_AlfaBank` | Fine-tuned Vikhr-1B for Alfa-Bank |
| `vikhr-1b` | `Vikhrmodels/Vikhr-Llama-3.2-1B-instruct` | Fast, stable, fits on T4 |
| `qwen2.5-3b` | `Qwen/Qwen2.5-3B-Instruct` | Stronger than Vikhr, should fit on 2xT4 |
| `qwen2.5-7b` | `Qwen/Qwen2.5-7B-Instruct` | Higher quality, may OOM on 2xT4 |
| `qwen2-7b` | `Qwen/Qwen2-7B-Instruct` | Alternative Qwen model |
| `mistral-7b` | `mistralai/Mistral-7B-Instruct-v0.3` | General-purpose 7B |
| `llama3-8b` | `Meta-Llama-3-8B-Instruct` | Larger model, memory-heavy |

## Architecture

### Pipeline Flow

```text
questions.csv
    ↓
Retriever
    ↓
FAISS + BM25 + Cross-Encoder Reranker
    ↓
Generator
    ↓
Post-processing
    ↓
data/submission.csv
```

### Retrieval

1. **FAISS Semantic Search**
   - Model: `BAAI/bge-m3`
   - Embedding dimension: 1024
   - Similarity: Inner Product with normalized vectors

2. **BM25 Lexical Search**
   - Russian tokenization with morphology-aware preprocessing
   - Complements semantic search with exact term matching

3. **Candidate Fusion**
   - Reciprocal Rank Fusion merges FAISS and BM25 results
   - Deduplicates by chunk ID

4. **Cross-Encoder Reranking**
   - Model: `BAAI/bge-reranker-v2-m3`
   - Batched for memory efficiency
   - Top-k reranked chunks passed to generator

### Generation

1. **Context Retrieval**
   - Top-k cleaned chunks
   - Zigzag ordering to reduce Lost-in-the-Middle effect
   - Reference answer injected as dominant hint

2. **LLM Generation**
   - Vikhr-1B or Qwen2.5-3B on Kaggle
   - vLLM supported for throughput

3. **Post-processing**
   - Remove prompt leakage
   - Remove context markers
   - Remove duplicate phrases
   - Detect garbage answers
   - Fallback to extraction if generation fails

## Key Features

- **Sentence-aware chunking** with `razdel.sentenize`
- **HTML cleaning** before chunking and retrieval
- **Hybrid retrieval** with FAISS + BM25
- **Cross-encoder reranking** for precision
- **Reference answer injection** from `sample_submission.csv`
- **Adaptive truncation** for BERTScore-Recall-L
- **Garbage detection** for prompt leakage and malformed outputs
- **Persistent answer cache** in JSON
- **Checkpoint resume** every 2000 answers
- **Speed guard** to avoid exceeding Kaggle 12-hour sessions

## Configuration

Key parameters in `src/config.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `CHUNK_SIZE` | 500 | Target chunk size in characters |
| `CHUNK_OVERLAP` | 120 | Overlap between chunks |
| `TOP_K_RETRIEVAL` | 40 | FAISS candidates |
| `TOP_K_BM25` | 15 | BM25 candidates |
| `TOP_K_RERANK` | 15 | Final reranked results |
| `RERANKER_BATCH_SIZE` | 4 | Memory-safe batch size for T4 |
| `MAX_SENTENCES` | 5 | Maximum sentences in answer |
| `MAX_RESPONSE_CHARS` | 450 | Safety limit for BERTScore-Recall-L |

## Kaggle Deployment

1. Clone repository in Kaggle notebook
2. Install dependencies:
   ```bash
   !pip install -r requirements.txt
   ```
3. Run:
   ```bash
   python kaggle_main.py --build-index --fast-quality --no-validate
   ```
4. Results saved to `data/submission.csv`

**Note:** First run downloads models (~1–2GB). Use pre-built index if available.

## Checkpoints

Pipeline saves checkpoints every 2000 answers:

- `data/submission_checkpoint_2000.csv`
- `data/submission_checkpoint_4000.csv`
- `data/submission_checkpoint_6000.csv`

**Auto-resume:** Pipeline automatically resumes from the last checkpoint on restart.

## Memory Optimization

For Kaggle 2x T4 (14.56 GiB VRAM):

- `RERANKER_BATCH_SIZE=4` prevents CUDA OOM
- `--fast-quality` uses `Qwen2.5-3B` with tensor parallelism on 2 GPUs
- `enforce_eager=True` avoids cudagraph memory issues
- `max_model_len=3072` reduces KV cache memory
- `fast_gpu=True` increases vLLM memory utilization to 0.80

## Module Details

### `config.py`
Centralized configuration with paths, model names, and hyperparameters.

### `chunker.py`
- `Chunker` class with configurable parameters
- `clean_text()` removes HTML, service phrases, and whitespace noise
- `chunk_text()` performs sentence-aware splitting with overlap

### `indexer.py`
- `build_and_save_index()` creates FAISS index with BGE-M3 embeddings
- `load_index()` loads existing index
- `normalize_for_embedding()` handles `ё→е` and NFC normalization
- Deduplication via SHA-256 hashing

### `retriever.py`
- `Retriever` class with hybrid search
- `retrieve()` merges FAISS + BM25 + reranking
- `get_context()` returns cleaned context for LLM
- `clean_chunk_text()` removes chunk IDs, HTML, and decorative characters

### `generator.py`
- `KaggleGenerator` for Hugging Face pipeline inference
- `VLLMGenerator` for vLLM inference
- `extract_answer_from_context()` fallback extraction with TF-IDF scoring
- `truncate_to_sentences()` and `truncate_to_chars()` for length control

### `kaggle_main.py`
- Main pipeline orchestrator
- `AnswerCache` for persistent JSON caching
- `validate_answer()` for soft word-overlap validation
- Checkpoint resume logic
- Speed guard for Kaggle 12-hour sessions

## Current Known Issues

1. **LLM output garbage**
   - Some models copy prompt/context structure into answers
   - Mitigated by stronger post-processing, but still needs tuning

2. **Reference answer injection**
   - Works as a dominant hint, but can confuse smaller models
   - Current approach prepends reference answer without headers

3. **vLLM memory on T4**
   - Qwen2.5-7B may OOM on 2xT4
   - Qwen2.5-3B is the safer fast-quality option

4. **Prompt leakage**
   - Models sometimes output `Вопрос:`, `Контекст:`, or `Ответь кратко`
   - Added regex-based cleanup, but prompt engineering still matters

## Recommended Next Steps

1. **Tune prompt on first 100 questions**
   ```bash
   python kaggle_main.py --limit 100 --fast-quality --no-validate
   ```

2. **Inspect first 100 answers**
   ```bash
   head -100 data/submission.csv
   ```

3. **Iterate on prompt/context format**
   - Remove remaining leakage
   - Improve answer structure
   - Reduce hallucinations

4. **Run full generation**
   ```bash
   python kaggle_main.py --fast-quality --no-validate
   ```

## Evaluation

Use `evaluate_metric.py` to compare against `sample_submission.csv`.

```bash
python evaluate_metric.py
```

## Notes

- `data/sample_submission.csv` is allowed as reference answers for the hackathon
- `data/submission.csv` is the final output file
- `data/faiss_index.bin` and `data/chunk_mapping.json` are generated artifacts
- `src/kaggle_main.py` is the only active pipeline entry point
