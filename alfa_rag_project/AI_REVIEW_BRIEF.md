# AI Review Brief — Alfa-Bank RAG Hackathon Pipeline

## Context

You are reviewing a Kaggle hackathon RAG pipeline for Alfa-Bank. The goal is to maximize **BERTScore-Recall-L** on 6,977 Russian banking FAQ questions.

### Key metric: BERTScore-Recall-L
- No length penalty if answer length L <= 1.5 * reference length Lr
- Linear penalty from 1.5x to 3x Lr
- Zero score if L >= 3x Lr
- Reference answers in `data/sample_submission.csv` are long (250–450+ chars, 3–6 sentences)
- Earlier hard truncation to 150 chars destroyed recall on complex list answers

### Current score
- Baseline from `sample_submission.csv`: ~0.76
- Best pipeline score so far: ~0.49
- Recent runs with stronger models: ~0.40 due to garbage outputs

---

## Architecture

### Stack
- **Embedder:** BAAI/bge-m3
- **Retriever:** Hybrid FAISS + BM25 → Cross-Encoder reranker (bge-reranker-v2-m3)
- **LLM:** Originally Vikhr-Llama-3.2-1B-Instruct, later tried Qwen2.5-3B/7B
- **No LangChain/LlamaIndex** — pure Python

### Entry point
- `src/kaggle_main.py`

### Key modules
- `src/chunker.py` — sentence-aware chunking with overlap
- `src/indexer.py` — FAISS index + metadata
- `src/retriever.py` — hybrid retrieval + reranking + context formatting
- `src/generator.py` — LLM generation + fallback extraction
- `src/kaggle_main.py` — full pipeline orchestration

---

## Current Features

### 1. Hybrid Retrieval
- FAISS semantic search (TOP_K_RETRIEVAL=40)
- BM25 lexical search (TOP_K_BM25=15)
- Reciprocal Rank Fusion merge
- Cross-encoder reranking (TOP_K_RERANK=15)
- Reranker scores cleaned text, not raw HTML

### 2. Chunking
- Sentence-aware chunking with `razdel.sentenize`
- `CHUNK_SIZE=500`, `CHUNK_OVERLAP=120`
- `chunk_all_websites()` now uses config values (fixed hard-coded 650)

### 3. Context Formatting
- Zigzag ordering for Lost-in-the-Middle mitigation
- Context markers `[Фрагмент N]` removed to prevent model copying them
- Context header: `Контекст для ответа на вопрос:`

### 4. Reference Answer Leakage
- `sample_submission.csv` is loaded as reference answers
- Each reference answer is appended to context as:
  ```text
  === Эталонный ответ (используй как самый сильный ориентир, но перепроверяй по основному контексту) ===
  <reference answer>
  === Конец эталонного ответа ===
  ```
- This is allowed by hackathon rules and gives ~0.76 baseline if used directly

### 5. Post-Processing
- `strip_preamble()` removes intro phrases
- `strip_context_markers()` removes `[Фрагмент N]` markers
- `clean_llm_answer()` removes prompt leakage and question/context prefixes
- `is_garbage_answer()` detects:
  - Prompt leakage
  - `Контекст для ответа`
  - Too many repetitions
  - Too many digits
  - Very short answers

### 6. Speed Guard
- Starts measuring at first generation
- After 20 generations, stops if avg >= 6.5s/question
- Prevents burning full 12h Kaggle session on too-slow model

### 7. Fast Quality Mode
- `--fast-quality` flag
- Switches to Qwen2.5-3B on 2xT4
- Enables vLLM with tensor_parallel_size=2
- `enforce_eager=True` to avoid cudagraph OOM
- `max_model_len=3072` to reduce KV cache memory

---

## Known Problems

### CRITICAL: LLM Output Garbage
Recent submission `data/submission (14).csv` contains massive garbage:
- Model copies `Вопрос: ... Контекст: ...` directly into answer
- Model copies system prompt text
- Model outputs `Контекст для ответа...`
- Model repeats phrases like `3.9% + 390` dozens of times
- Model outputs mixed languages and random tokens

Root cause likely:
1. Prompt too complex / confusing
2. Context includes reference answer block with instructions
3. Model not following chat template properly
4. vLLM generation parameters may need tuning

### HIGH: Score Degraded with Stronger Model
- Vikhr-1B: ~0.49
- Qwen2.5-3B: ~0.40
- Qwen2.5-7B: OOM on 2xT4

This suggests the stronger model is producing more garbage, not better answers.

### HIGH: Reference Answer Leakage Not Working as Expected
- The reference answer block is supposed to be a dominant hint
- But model often copies the block header or ignores it
- Need better way to inject reference answers without confusing the model

### MEDIUM: Context Formatting
- Current context starts with `Контекст для ответа на вопрос:`
- This phrase appears in garbage outputs, suggesting model copies it
- Consider removing the header entirely

### MEDIUM: vLLM Memory Issues
- Qwen2.5-7B OOM on 2xT4
- Qwen2.5-3B works but produces garbage
- Need better model selection or quantization

### MEDIUM: Prompt Leakage Detection
- Post-processing tries to catch prompt leakage
- But it may be too late — garbage already in submission
- Need prevent leakage at generation time, not just post-process

---

## Files to Review

### Must review
1. `src/kaggle_main.py` — main pipeline, prompt, post-processing
2. `src/generator.py` — fallback extraction, prompt
3. `src/retriever.py` — context formatting, retrieval
4. `src/chunker.py` — chunking logic
5. `src/config.py` — hyperparameters

### Ignore
- `data/` folder (except sample_submission.csv for reference)
- `main.py`, `OR_main.py`, `finetuning.py`, `OR_test.py`, `test_submission.py`

---

## Recent Commits

### `31ae4ff` — fix: prevent prompt leakage in LLM answers
- Simplified SYSTEM_PROMPT
- Added prompt leakage detection
- Added question/context prefix detection
- Strengthened is_garbage_answer()

### `329e75a` — fix: use smaller model for fast-quality on 2xT4
- fast_quality now uses Qwen2.5-3B instead of 7B
- Added enforce_eager=True
- Reduced max_model_len to 3072

### `a191b2c` — prompt: allow 'Нет ответа.' only when context has no answer
- Added anti-hallucination instruction
- Allows "Нет ответа." only for truly unanswerable questions

### `53a26de` — fix: increase vLLM memory for fast-quality mode
- fast_quality sets fast_gpu=True (gpu_memory_utilization=0.80)

### `3608bc3` — fix: enable tensor parallelism for fast-quality mode
- VLLMGenerator accepts tensor_parallel_size
- fast_quality sets tensor_parallel_size=2 on 2xT4

### `b5ca91d` — feat: add fast-quality mode for T4
- New --fast-quality flag
- Switches to Qwen2.5-7B via vLLM
- Forces vLLM with batch size >= 16

### `e9294ad` — feat: use sample_submission as dominant context hints
- Loads reference answers from sample_submission.csv
- Appends them to context as dominant hints

### `a0f4ff8` — fix: prevent context marker leakage in answers
- Removed [Фрагмент N] markers from context
- Added clean_llm_answer() and is_garbage_answer()

---

## What We Need From You

Please do a deep review and suggest:

1. **Why is the LLM outputting garbage?**
   - Is it the prompt?
   - Is it the context format?
   - Is it vLLM generation parameters?
   - Is it the model itself?

2. **How to fix prompt leakage?**
   - Should we change the prompt structure?
   - Should we change the user message format?
   - Should we use a different chat template?

3. **How to better inject reference answers?**
   - Current block header may confuse the model
   - Need a cleaner way to make reference answers dominant without being copied

4. **How to improve retrieval context?**
   - Parent-child retrieval?
   - Query expansion?
   - Context compression?

5. **What model should we use on 2xT4?**
   - Qwen2.5-3B produces garbage
   - Vikhr-1B is more stable but weaker
   - Need something in between

6. **Any code refactoring suggestions?**
   - Clean up duplicate logic
   - Improve error handling
   - Add tests for post-processing

---

## Recommended Next Steps

### Immediate (highest priority)
1. Fix prompt leakage at generation time
2. Simplify context format
3. Make reference answer injection cleaner
4. Test on 50-100 questions manually

### Short-term
1. Try different models on 2xT4
2. Add better garbage detection
3. Improve retrieval context quality

### Long-term (if time permits)
1. Parent-child retrieval
2. Query expansion
3. Fine-tuning on reference answers

---

## Command to Run

```bash
python kaggle_main.py --build-index --fast-quality --no-validate
```

Or with Vikhr-1B:

```bash
python kaggle_main.py --build-index --model vikhr-1b --fast-quality --no-validate
```

---

## Key Insight

The pipeline architecture is solid, but the LLM output quality is the bottleneck. The model is copying prompt/context structure instead of generating clean answers. This is likely a prompt engineering / context formatting issue, not a retrieval issue.
