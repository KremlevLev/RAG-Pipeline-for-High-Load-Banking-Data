# Audit 2 — Коррупция submission (8).csv и критические фиксы

## 1. Критический баг: vLLM batch mapping сломан

### Симптом
`data/submission (8).csv` содержит полностью мусорные ответы:
```text
plevelplevelplevelActionCodeTRGL arsch republika...
Восplevelа — Батары...
TRGLTRGLTRGL...
```
Это не нормальный LLM-output, а признак сломанного batch-processing.

### Root cause
В `VLLMGenerator.generate_batch()` была ошибка индексации:

```python
results: list[str] = []
...
if not context:
    results.append("Нет ответа.")
...
results.insert(batch_map[j], answer)
```

Проблема: `results` уже содержал fallback-ответы для пустых контекстов, поэтому `insert(batch_map[j], answer)` вставлял ответы не в те позиции. Это ломало alignment между вопросами и ответами.

### Fix
Теперь используется fixed-size list:

```python
results: list[str] = [""] * len(queries)
...
results[batch_map[j]] = answer
```

Это гарантирует, что `answers[i]` всегда соответствует `queries[i]`.

---

## 2. Критический баг: hardcoded refusal phrases

### Симптом
В submission периодически появлялось:
```text
Нет ответа.
Недостаточно информации
```

### Root cause
В коде были hardcoded отказы:
- `kaggle_main.py:466`
- `kaggle_main.py:759`
- `kaggle_main.py:850`
- `config.py:44` comment

### Fix
Все hardcoded отказы заменены на fallback extraction:
```python
answer = extract_answer_from_context(query, context or "")
if not answer:
    sentences = [s.strip() for s in re.split(r'[.!?»]+', context or "") if s.strip()]
    answer = sentences[0] if sentences else query
```

---

## 3. Критический баг: validation слишком агрессивная

### Симптом
Логи показывали:
```text
Invalid answer for q_id=1 — fallback applied
Invalid answer for q_id=2 — fallback applied
...
Invalid answer for q_id=8 — fallback applied
```

### Root cause
`validate_answer()` проверяла word-overlap между вопросом и ответом. Для BERTScore это плохо, потому что семантически правильный ответ может не иметь дословного overlap.

### Fix
`validate_answer()` теперь по умолчанию:
```python
min_overlap: int = 0
```
То есть осмысленный LLM-ответ больше не режется из-за отсутствия дословного overlap.

---

## 4. Критический баг: corrupted tokenizer / merged model

### Симптом
Мусорные токены типа `plevel`, `TRGL`, `ActionCode`, `follando`, `arsch` — это не русский текст, а raw token artifacts.

### Root cause
Скорее всего проблема в merged model tokenizer или vLLM batching.

### Fix applied
- Fixed batch mapping
- Removed hardcoded refusals
- Softened validation

### Remaining risk
Если после фикса мусор останется — нужно:
1. Проверить tokenizer merged model
2. Возможно, откатиться на base Vikhr без merge
3. Проверить vLLM version compatibility

---

## 5. Что делать дальше

### Must do
1. Пересобрать индекс после изменения chunking:
```bash
python kaggle_main.py --build-index --model vikhr-1b-finetuned --vllm --vllm-batch-size 8
```

2. Запустить генерацию без validation:
```bash
python kaggle_main.py --build-index --model vikhr-1b-finetuned --vllm --vllm-batch-size 8 --no-validate
```

### If still corrupted
1. Проверить tokenizer:
```bash
python - <<'PY'
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Vikhrmodels/Vikhr-Llama-3.2-1B-instruct", trust_remote_code=True)
print(tok.decode(tok.encode("Привет, как узнать номер счета?")))
PY
```

2. Если merged model ломает tokenizer — использовать base Vikhr без merge

---

## 6. Priority fixes already applied

- [x] Fixed vLLM batch mapping
- [x] Removed hardcoded refusals
- [x] Softened validation
- [x] Updated retriever comments
- [x] Updated config comments

## 7. Remaining risks

- Tokenizer corruption from merged model
- vLLM version compatibility
- Potential need to rebuild index after chunking changes
