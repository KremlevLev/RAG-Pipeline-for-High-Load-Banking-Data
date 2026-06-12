# Audit 2 — Детальный код-ревью и улучшения контекста

## 1. Критические баги, которые уже исправлены

### 1.1 vLLM batch mapping был сломан
**Симптом:** `submission (8).csv` содержал мусор типа `plevel`, `TRGL`, `ActionCode`.

**Root cause:** В `generate_batch()` `results.insert(batch_map[j], answer)` ломал alignment между вопросами и ответами.

**Fix:** `results[batch_map[j]] = answer` с fixed-size list.

### 1.2 Hardcoded refusals
**Симптом:** В submission попадали `Нет ответа.` и `Недостаточно информации`.

**Fix:** Все заменены на fallback extraction.

### 1.3 Validation была слишком агрессивной
**Симптом:** 8/8 ответов в логах были invalid.

**Fix:** `min_overlap=0` по умолчанию.

---

## 2. Найденные проблемы в retrieval/context

### 2.1 CHUNK_SIZE из config НЕ использовался
**Файл:** [`chunker.py`](alfa_rag_project/src/chunker.py:431)

**Проблема:** В `chunk_all_websites()` был hard-coded `chunk_size=650`, хотя в `config.py` стоит `CHUNK_SIZE=500`.

**Fix:** Теперь используется `CHUNK_SIZE` и `CHUNK_OVERLAP` из config.

**Почему важно:** Все изменения в config теперь реально влияют на чанкинг.

### 2.2 Контекст обрезался слишком агрессивно
**Файл:** [`kaggle_main.py`](alfa_rag_project/src/kaggle_main.py:383)

**Проблема:** vLLM prompt truncation был на 2500 символов. При TOP_K_RERANK=15 и CHUNK_SIZE=500 это означало потерю части контекста.

**Fix:** Увеличено до 3200 символов.

**Почему важно:** Больше контекста → выше recall, но всё ещё в пределах 4096 token limit.

### 2.3 Контекст не имел явного header
**Файл:** [`retriever.py`](alfa_rag_project/src/retriever.py:503)

**Проблема:** Модель получала просто `[Фрагмент 1] ... [Фрагмент 2] ...` без явного объяснения, что это контекст.

**Fix:** Добавлен header:
```python
"Контекст для ответа на вопрос:\n\n"
```

**Почему важно:** LLM лучше понимает структуру промпта.

---

## 3. Что насчёт Parent-Child Retrieval?

### Мой verdict: НЕ СЕЙЧАС

**Почему:**
1. Это большая архитектурная перестройка (`chunker.py`, `indexer.py`, `retriever.py`)
2. Нужно пересобирать индекс с нуля
3. Риск сломать то, что уже работает
4. У нас уже есть более простые улучшения, которые могут дать прирост

### Когда стоит пробовать Parent-Child
Только если после всех простых фиксов score всё ещё низкий.

### Альтернатива на сейчас
1. Увеличить `TOP_K_RERANK` до 20
2. Увеличить `MAX_RESPONSE_CHARS` до 600
3. Поиграться с `CHUNK_SIZE` 500→600
4. Добавить query expansion для банковских терминов

---

## 4. Рекомендованный следующий запуск

```bash
python kaggle_main.py --build-index --model vikhr-1b --vllm --vllm-batch-size 8 --no-validate
```

### Почему base Vikhr
- Fine-tuned модель могла быть проблемой
- Base Vikhr стабильнее
- Если score улучшится — значит проблема была в fine-tuned модели

---

## 5. Что я бы улучшил следующим шагом

### 5.1 Query expansion
Добавить синонимы для банковских терминов:
```python
QUERY_SYNONYMS = {
    "счет": ["счёт", "расчётный счёт", "номер счёта"],
    "карта": ["банковская карта", "дебетовая карта"],
    "кредит": ["займ", "кредитование"],
}
```

### 5.2 Context compression
Вместо простой truncation — выбрать top-N chunks по релевантности и склеить только их.

### 5.3 Better prompt
Добавить few-shot examples прямо в SYSTEM_PROMPT для Vikhr.

---

## 6. Итог

### Уже исправлено
- [x] vLLM batch mapping
- [x] Hardcoded refusals
- [x] Validation
- [x] CHUNK_SIZE config usage
- [x] Context truncation limit
- [x] Context header

### Не исправлено (пока)
- [ ] Parent-child retrieval
- [ ] Query expansion
- [ ] Context compression
- [ ] Better prompt few-shot

### Главный риск
Если base Vikhr тоже даст мусор — значит проблема не в fine-tuned модели, а в tokenizer/vLLM. Тогда нужно дебажить tokenizer отдельно.
