# RAG Pipeline for Alfa-Bank MIPT Hackathon

Pure Python RAG system without heavy frameworks (LangChain, Haystack).

## Project Structure

```
alfa_rag_project/
├── requirements.txt     # Dependencies
├── README.md           # This file
├── data/
│   ├── websites.csv    # Input: web_id, website, text
│   ├── questions.csv   # Input: q_id, query
│   ├── submission.csv  # Output: predictions
│   ├── faiss_index.bin # FAISS index (generated)
│   └── chunk_mapping.json  # Chunk metadata (generated)
└── src/
    ├── config.py       # Configuration and constants
    ├── chunker.py      # Text chunking with razdel
    ├── indexer.py      # FAISS index management
    ├── retriever.py    # Vector search + cross-encoder reranking
    ├── generator.py    # LLM generation with brevity constraints
    ├── main.py         # Pipeline orchestrator (Ollama/local)
    ├── kaggle_main.py  # Kaggle-optimized pipeline (Hugging Face)
    ├── OR_main.py      # OpenRouter API pipeline
    ├── __main__.py     # Entry point for `python -m main`
    └── __init__.py     # Package exports
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Local/Ollama (main.py)
```bash
cd alfa_rag_project/src
python main.py --build-index --model qwen2.5:7b
```

### Kaggle (kaggle_main.py) - Open-source models with 2x T4 GPU
```bash
cd alfa_rag_project/src
python -m main --build-index --model qwen2.5-7b
```

Or directly:
```bash
python kaggle_main.py --build-index --model qwen2.5-7b
```

### Available Kaggle models
- `qwen2.5-7b` - Qwen/Qwen2.5-7B-Instruct
- `qwen2-7b` - Qwen/Qwen2-7B-Instruct
- `mistral-7b` - Mistral-7B-Instruct-v0.3
- `llama3-8b` - Meta-Llama-3-8B-Instruct

### OpenRouter (OR_main.py) - API-based inference
```bash
# Set API key
export OPENROUTER_API_KEY="sk-or-..."

# Run
python OR_main.py --model qwen2.5-7b
```

No model downloads - uses OpenRouter API for open-source models.

## Key Features

- **Chunking**: Sentence-aware splitting using `razdel.sentenize` (no broken sentences)
- **Embeddings**: BGE-M3 (1024-dim) with normalized vectors
- **Reranking**: BGE-reranker-v2-m3 cross-encoder
- **Brevity**: System prompt + post-processing to limit response to ~30 words
- **FAISS**: Inner Product similarity for cosine with normalized vectors
- **Kaggle Support**: Hugging Face models with automatic 2x T4 GPU detection

## Configuration

Key parameters in `config.py`:
- `CHUNK_SIZE=400` - Target chunk size in characters
- `CHUNK_OVERLAP=50` - Overlap between chunks
- `TOP_K_RETRIEVAL=15` - FAISS candidates
- `TOP_K_RERANK=3` - Final results after reranking
- `MAX_RESPONSE_WORDS=30` - Hard limit for LLM output

## Kaggle Deployment

1. Clone repository in Kaggle notebook
2. Install dependencies: `!pip install -r requirements.txt`
3. Run: `python -m main --build-index --model qwen2.5-7b`
4. Results saved to `data/submission.csv`

**Note**: First run will download models (~1-2GB). Use pre-built index if available.

## Checkpoints

Pipeline saves checkpoints every 2000 answers:
- `data/submission_checkpoint_2000.csv`
- `data/submission_checkpoint_4000.csv`
- `data/submission_checkpoint_6000.csv`

**Auto-resume:** Pipeline automatically resumes from the last checkpoint on restart.

## Memory Optimization

For Kaggle 2x T4 (16GB total VRAM):
- Batch size reduced to 8 for embedding generation
- GPU cache cleared after each batch
- Index auto-saved after build to prevent data loss