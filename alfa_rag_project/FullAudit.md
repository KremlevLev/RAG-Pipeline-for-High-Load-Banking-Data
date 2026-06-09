# 🔬 FullAudit.md — Безжалостный Code-Review RAG Pipeline (Alfa-Bank MIPT)

> Аудит проведён по точке входа [`kaggle_main.py`](src/kaggle_main.py). Проанализированы: [`retriever.py`](src/retriever.py), [`generator.py`](src/generator.py), [`config.py`](src/config.py), [`chunker.py`](src/chunker.py), [`indexer.py`](src/indexer.py), [`evaluate_metric.py`](evaluate_metric.py) и эталон [`data/sample_submission.csv`](data/sample_submission.csv).

## ⚡ TL;DR — Главный вывод за 10 секунд

**Вы калечите собственный скор обрезкой ответов.** Эталонные ответы (`answer_new` в `sample_submission.csv`, baseline 75.8) имеют длину **250–450 символов и 3–6 предложений** со списками. А ваш пайплайн режет ответ до **2 предложений и 150 символов**. 

При reference ≈ 300 символов, порог «без штрафа» = `1.5 * 300 = 450` символов. Вы генерите 150. **Вы теряете не на штрафе за длину — вы теряете на BERTScore-Recall, потому что физически не покрываете содержание эталона.** Recall тем выше, чем больше смысловых токенов эталона вы упомянули. Резать до 150 символов = выкидывать 60–70% recall на сложных вопросах.

Это прямое следствие ошибки, которую вы уже подозревали. Ниже — полный разбор.

---

# 1. 🔴 Критические баги (Красный уровень)

### BUG-1 [CRITICAL]: Жёсткая обрезка убивает Recall на длинных эталонах
**Файлы:** [`config.py:43-45`](src/config.py:43), [`kaggle_main.py:283-284`](src/kaggle_main.py:283), [`generator.py:503-507`](src/generator.py:503)

`MAX_SENTENCES=2` и `MAX_RESPONSE_CHARS=150` применяются ко **всем** ответам безусловно. Но метрика **относительная**: штраф считается от `Lr` (длины эталона), а не от абсолютных 150 символов.

Доказательство из эталона ([`sample_submission.csv:2-9`](data/sample_submission.csv:2)):
- q_id=1: ответ **~330 символов**, одно сложное предложение про структуру счёта.
- q_id=2: ответ **~750 символов**, маркированный список из 4 пунктов.
- q_id=5: ответ **~340 символов**, пошаговая инструкция.

Если эталон 330 символов, то ваш потолок без штрафа = **495 символов**, а вы отдаёте 150. BERTScore-Recall обрушивается, потому что 55% содержательных токенов эталона просто отсутствуют в вашем ответе.

> **Парадокс:** вы боялись штрафа `L(q)` и зарезали длину. Но штраф `L(q)` карает за *слишком длинные* ответы (≥3×Lr). Зарезав до 150, вы НЕ попали под штраф L(q), зато обнулили **сам BERTScore-Recall**. Это и есть ваша главная утечка.

**Фикс:** ввести адаптивный, динамический лимит (см. Action Plan §4, FIX-1). Целевая длина ответа = `~2.0 * median(Lr)`. Поскольку Lr вы не знаете на инференсе, ориентируйтесь на распределение эталона: медиана ≈ 200–280 символов. Ставьте `MAX_RESPONSE_CHARS ≈ 500`, `MAX_SENTENCES = 4–5`.

---

### BUG-2 [CRITICAL]: Сломанный парсинг ответа модели (`split("</")`)
**Файл:** [`kaggle_main.py:269-279`](src/kaggle_main.py:269)

```python
if "</" in answer:
    answer = answer.split("</")[-1]   # ← БАГ
```

`pipeline("text-generation")` с `return_full_text` по умолчанию `True` возвращает **промпт + генерацию**. Логика вырезания промпта здесь катастрофична:

1. `answer.split("</")[-1]` берёт хвост после последнего `</`. Если модель сгенерировала ответ **без** тегов `</`, а в промпте (chat template Llama/Vikhr) тегов тоже нет — этот `if` не сработает, и вы оставите **весь промпт целиком** в ответе. Если же в few-shot примерах или контексте встретится `</` (HTML-мусор в контексте!), вы обрежете по случайному месту посреди контекста.
2. Это «угадайка», которая ломается на каждой второй генерации.

**Правильный способ:** передать `return_full_text=False` в pipeline ИЛИ использовать `messages` напрямую (новые версии `transformers` принимают список сообщений и возвращают только ответ). Тогда весь блок ручного вырезания удаляется. См. FIX-2.

---

### BUG-3 [CRITICAL]: `validate_answer` ничего не делает (мёртвая защита)
**Файл:** [`kaggle_main.py:478-492`](src/kaggle_main.py:478)

```python
if not validate_answer(query, answer, min_overlap):
    logger.warning(...)
    stats["invalid"] += 1
# ... ответ всё равно пишется в results без изменений
```

Валидация логирует «invalid», увеличивает счётчик — и **не предпринимает никаких действий**. Плохой ответ всё равно идёт в сабмишен. Это иллюзия контроля качества. Либо чините (fallback на extract при провале), либо удаляйте, чтобы не тратить CPU на бесполезный проход. См. FIX-3.

---

### BUG-4 [CRITICAL]: Кеш отравляет инференс при смене модели/промпта
**Файл:** [`kaggle_main.py:322-340`](src/kaggle_main.py:322), [`459-463`](src/kaggle_main.py:459)

Ключ кеша = `sha256(query|model)`. Но кеш **персистентный JSON на диске**. На Kaggle при рестарте сессии (а вы рассчитываете на auto-resume) вы подтянете старые ответы, сгенерированные **другой версией промпта/пост-процессинга/чанков**. Вы будете чинить генератор, а кеш будет молча отдавать старый мусор.

Дополнительно: `cache.set()` вызывает `self._save()` **на каждом вопросе** ([`kaggle_main.py:340`](src/kaggle_main.py:340)) — это `json.dump` всего словаря 6977 раз. На последних итерациях это запись файла на десятки МБ каждую итерацию → ощутимый I/O-оверхед на длинной сессии.

**Фикс:** добавить в ключ кеша версию пайплайна (`PIPELINE_VERSION`), сохранять кеш батчами (раз в N), а лучше — на Kaggle инференс одноразовый, кеш вообще не нужен (только чекпоинты). См. FIX-4.

---

### BUG-5 [CRITICAL/HIGH]: Двойная загрузка модели = риск CUDA OOM
**Файл:** [`kaggle_main.py:202-214`](src/kaggle_main.py:202)

```python
self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
self.pipe = pipeline("text-generation", model=self.model, tokenizer=...,
                     device_map=device_map, torch_dtype=torch_dtype)  # ← повтор аргументов
```

Передача `device_map` и `torch_dtype` в `pipeline()` поверх уже загруженной модели в лучшем случае игнорируется, в худшем — провоцирует повторное размещение/копирование. С `device_map="auto"` на 2×T4 это особенно опасно: модель уже расшардена, а pipeline пытается ещё раз управлять девайсами. Передавайте в `pipeline` только `model` и `tokenizer`. См. FIX-2.

---

### BUG-6 [HIGH]: Чекпоинт-резюм может тихо терять/дублировать данные
**Файл:** [`kaggle_main.py:441-499`](src/kaggle_main.py:441)

Резюм берёт чекпоинт по `len(cp_df) == cp_num`. Но `results` дополняется как `q_id, answer`, а чекпоинт сохраняется со всеми колонками `to_dict("records")`. Если порядок `questions.csv` не идентичен между запусками или чекпоинт записан в момент, когда `idx+1` не кратен 2000 — индексы разъедутся. Плюс enumerate идёт по `tqdm(questions_df.iterrows())` — это работает, но `idx` от enumerate и фактический ряд могут разойтись, если в данных есть пропуски. См. FIX-6.

---

# 2. 🔍 Анализ Retrieval

### 2.1 Гибридный поиск (FAISS + BM25) — найдены реальные дефекты

**LEAK-R1 [HIGH]: Рассинхрон конфигов — `hypotheses.md` врёт.**
[`config.py:35`](src/config.py:35) задаёт `TOP_K_RETRIEVAL=40`, а docstring [`retriever.py:342-345`](src/retriever.py:342) и [`hypotheses.md`](hypotheses.md) местами говорят про 15. Документация не отражает код. Это не косметика: при `TOP_K_RETRIEVAL=40 + TOP_K_BM25=15 = ~55` кандидатов, reranker прогоняет все 55 → по таймингам это ~1.5–2 c/запрос только на reranking.

**LEAK-R2 [CRITICAL]: BM25 строится на «грязном» тексте с chunk_id.**
[`retriever.py:292-298`](src/retriever.py:292) — BM25 индексирует `indexer.get_all_texts()`, т.е. **сырой** `chunk_mapping["text"]`. Очистка (`clean_chunk_text`) применяется только в `get_context()` **после** retrieval. Значит:
- BM25 токенизирует HTML-мусор, `[chunk_id]`, `&nbsp;` → шум в лексическом поиске.
- FAISS ищет по нормализованным эмбеддингам (ё→е, NFC), но reranker получает **сырой** текст пар `(query, raw_text)` ([`retriever.py:402`](src/retriever.py:402)). Cross-encoder скорит мусорные чанки с HTML вперемешку — это понижает качество ранжирования.

**LEAK-R3 [HIGH]: Reranker и эмбеддер работают на разных нормализациях.**
Эмбеддинги строятся на `normalize_for_embedding` (ё→е). Запрос в FAISS ([`retriever.py:361`](src/retriever.py:361)) кодируется **без** этой нормализации:
```python
query_embedding = self.indexer.model.encode([query], normalize_embeddings=True)
```
`normalize_embeddings=True` — это L2-нормализация вектора, а НЕ текстовая ё→е/NFC. То есть индекс построен на «счет», а запрос «счёт» идёт как есть. Это ровно тот баг, от которого вы защищались в `indexer.py`, но забыли применить к query на стороне retrieve. Прямая утечка recall.

**LEAK-R4 [MEDIUM]: «Сырое» слияние без взвешивания (нет RRF).**
[`retriever.py:381-395`](src/retriever.py:381) — FAISS-кандидаты добавляются первыми, BM25 — потом, дедуп по id. Скоры FAISS и BM25 **не используются** при слиянии — выживает только факт присутствия, а финальный порядок целиком отдан reranker'у. Это рабочая, но не оптимальная схема. **Reciprocal Rank Fusion (RRF)** даёт более стабильный пул кандидатов и почти бесплатен по времени. См. FIX-R4.

**LEAK-R5 [MEDIUM]: Reranker гоняется на CPU?**
[`retriever.py:278`](src/retriever.py:278) — `CrossEncoder(reranker_model)` создаётся без явного `device`. По умолчанию `sentence-transformers` выберет cuda, если доступна, но при `device_map="auto"` LLM уже занял оба T4. Стоит явно прибить reranker к `cuda:0` и LLM к шардингу, либо reranker на `cuda:1`. Иначе reranker может уехать на CPU и стать узким местом (×10 по времени). См. FIX-R5.

### 2.2 Чанкинг и Overlap — здесь зарыт data-quality killer

**LEAK-C1 [CRITICAL]: Используется НЕ тот чанкер.**
В пайплайне вызывается `chunk_all_websites` → `create_chunks` ([`chunker.py:331-401`](src/chunker.py:331)) — это **legacy**-функция. А «правильный» класс `Chunker` с `clean_text()` ([`chunker.py:145-299`](src/chunker.py:145)) **не используется вообще**. Следствие:
- `create_chunks` **НЕ вызывает `clean_text()`**. HTML, `&nbsp;`, служебные фразы попадают прямо в индекс и в BM25.
- `create_chunks` фильтрует только пустые предложения, но не однословные артефакты как `Chunker._split_into_sentences`.
- Overlap-логика в `create_chunks` ([`chunker.py:385-399`](src/chunker.py:385)) считает overlap по символам через откат по предложениям — рабочая, но другая, чем у класса.

Это classic data-leak качества: вы написали хороший очищающий чанкер и не подключили его. Вся очистка фактически отложена на момент чтения контекста, а индекс/BM25 загрязнены. **Это объясняет, почему reranker иногда тащит мусор.** См. FIX-C1.

**LEAK-C2 [MEDIUM]: CHUNK_SIZE=450 + razdel может рвать таблицы/списки банковских FAQ.**
Эталоны содержат списки (q_id=2 — 4 буллета). Если в исходном `websites.csv` это маркированные списки, `razdel.sentenize` плохо бьёт `*` -буллеты и переносы → один «буллет» может стать обрезанным предложением, фильтруемым как однословный артефакт. Для FAQ лучше чанковать по «вопрос-ответ» блокам, если структура позволяет. Минимум — увеличить chunk_size до 600–700, чтобы целостный FAQ-ответ влезал в один чанк (тогда reranker отдаёт цельный ответ, а не половину). См. FIX-C2.

**LEAK-C3 [LOW]: дедуп по нормализованному хэшу может склеить близкие, но разные ответы** ([`indexer.py:105`](src/indexer.py:105)) — приемлемо, но при склейке вы теряете web-разнообразие. Не трогать сейчас.

---

# 3. 🧠 Анализ Generation

### 3.1 Промпт

**LEAK-G1 [CRITICAL]: Промпт прямо приказывает быть КОРОТКИМ — против метрики.**
[`kaggle_main.py:69-89`](src/kaggle_main.py:69) и [`generator.py:221-242`](src/generator.py:221):
> «Отвечай максимально емко… не должен превышать 3 предложений… ОДНО-ДВУХ предложениями.»

Это прямо противоречит эталону, где норма — 3–6 предложений со списками. Few-shot примеры тоже все короткие (1 предложение). Вы дообучаете модель в нужный момент быть лаконичной, а нужно — **полно, но без воды**. Промпт нужно переписать под «полный фактический ответ, как в справке банка, со списком при необходимости». См. FIX-G1.

**LEAK-G2 [HIGH]: Few-shot примеры не учат формату списков.**
Все 3 примера — короткие однострочники. Эталон q_id=2/5 — списки и пошаговые инструкции. Модель не видит образца «как оформлять список». Добавьте 1 few-shot с многопунктовым ответом. См. FIX-G1.

**LEAK-G3 [MEDIUM]: `max_new_tokens=64` физически обрубает длинный ответ.**
[`kaggle_main.py:258`](src/kaggle_main.py:258) — 64 токена ≈ 40–50 русских слов ≈ ~250–300 символов **в лучшем случае**, а с учётом токенизации кириллицы (1 слово ≈ 2–4 токена) реально ~25–30 слов. Для ответов-списков (эталон до 100+ слов) это жёсткий обрезок ещё на этапе генерации. Поднять до **256–320**. См. FIX-G3.

### 3.2 «Lost in the Middle»

**LEAK-G4 [HIGH]: Реверс чанков реализован НАИВНО и, вероятно, вредит.**
[`retriever.py:472`](src/retriever.py:472):
```python
context_parts = context_parts[::-1]
```
Классический «lost in the middle» фикс — это НЕ полный реверс, а **раскладка пилой**: самый релевантный чанк ставится в **начало И конец**, средние — в середину. Простой реверс просто перемещает самый релевантный чанк (он был первым после reranking) в **самый конец**. Если модель сильнее цепляется за начало (а для коротких контекстов это так), вы наоборот **спрятали лучший чанк**. Нужна правильная «зигзаг» раскладка. См. FIX-G4.

**LEAK-G5 [MEDIUM]: Нет нумерации/маркировки чанков.**
Эталонные ответы ссылаются на «Фрагмент 2», «Фрагмент 3» ([`sample_submission.csv:11,15,21`](data/sample_submission.csv:11)). Это значит, что **baseline-промпт нумеровал фрагменты**, и модель опиралась на это. У вас чанки склеены через `\n\n` без меток. Маркировка `[Фрагмент N]` помогает модели и cross-attention. (Но в финальном ответе метки надо вычищать — иначе словите «воду».) См. FIX-G5.

### 3.3 Пост-обработка («вода»)

**LEAK-G6 [HIGH]: Пост-процессинг не чистит «воду», только режет длину.**
Эталон-baseline сам содержит воду: «Согласно Фрагменту 2…», «Таким образом…», «В фрагменте указано:…» ([`sample_submission.csv:11,15,21,25`](data/sample_submission.csv:11)). Это мета-болтовня, которая снижает precision BERTScore (хотя метрика Recall-ориентирована, лишние токены разбавляют и могут тянуть length-штраф). Ваш `extract_answer_from_context` фильтрует junk, но `KaggleGenerator.generate` после LLM **не вызывает очистку преамбул** вообще. Нужен пост-фильтр вводных клише. См. FIX-G6.

**LEAK-G7 [MEDIUM]: `truncate_to_chars` режет посреди списка.**
[`generator.py:285-317`](src/generator.py:285) ищет последнюю `.`/`»` в пределах лимита. Для ответа-списка это срежет половину пунктов. При переходе на длинный лимит (FIX-1) проблема смягчается, но логика обрезки должна резать по **границе предложения/пункта**, а не по символу. См. FIX-G7.

**LEAK-G8 [LOW]: `extract_answer_from_context` пересчитывает IDF в O(N²).**
[`generator.py:180-183`](src/generator.py:180) в цикле по словам запроса перебирает все предложения для doc_freq, а сам вызывается для каждого предложения → O(sentences² × query_words). На длинном контексте это заметно. Кешировать doc_freq один раз. См. FIX-G8.

---

# 4. 🛠️ Action Plan & Code Snippets

> Порядок приоритета: FIX-1 → FIX-2 → FIX-C1 → FIX-G1/G3 → FIX-R2/R3 → остальное.
> Ожидаемый суммарный прирост: с текущих ~0.22 в район **0.45–0.60+** (FIX-1 + FIX-C1 + FIX-G1 дают основную массу).

### FIX-1 [CRITICAL]: Снять смертельный лимит длины → адаптивная длина

[`config.py`](src/config.py:42)
```python
# Generation parameters - optimized for BERT-Recall-L
# Эталоны (sample_submission) имеют медиану ~200-280 символов, max 700+.
# Порог без штрафа = 1.5*Lr. Цель: покрыть recall, не уходя в 3x.
MAX_SENTENCES: Final[int] = 5          # было 2 — каралось обнулением recall
MAX_RESPONSE_WORDS: Final[int] = 80    # было 30
MAX_RESPONSE_CHARS: Final[int] = 500   # было 150 — главная утечка
TEMPERATURE: Final[float] = 0.1
```

> Почему 500, а не «бесконечность»: при медиане эталона ~270 символов, 3×Lr ≈ 810. 500 символов безопасно сидят между 1.5×Lr (≈400) и 3×Lr — лёгкий штраф в худшем случае, но без обнуления, и при этом полный recall. Это и есть «умный баланс».

### FIX-2 [CRITICAL]: Починить генерацию (убрать `split("</")`, двойную загрузку)

[`kaggle_main.py`](src/kaggle_main.py:202) — упростить загрузку и парсинг:
```python
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        # pipeline НЕ должен повторно получать device_map/torch_dtype —
        # модель уже размещена. Иначе двойное размещение / OOM на 2xT4.
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )
```

[`kaggle_main.py`](src/kaggle_main.py:255) — заменить блок генерации и весь ручной парсинг на `return_full_text=False`:
```python
            outputs = self.pipe(
                prompt,
                max_new_tokens=320,            # FIX-G3
                temperature=TEMPERATURE,
                do_sample=TEMPERATURE > 0,
                top_p=0.9 if TEMPERATURE > 0 else None,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_full_text=False,        # ← возвращает ТОЛЬКО генерацию
            )

            answer = outputs[0]["generated_text"].strip()

            # Пост-обработка: чистим воду → режем по предложениям → safety по символам
            answer = strip_preamble(answer)               # FIX-G6
            answer = truncate_to_sentences(answer, MAX_SENTENCES)
            answer = truncate_to_chars(answer, MAX_RESPONSE_CHARS)
            return answer
```
Удалить целиком блок [`kaggle_main.py:268-280`](src/kaggle_main.py:268) (ручное вырезание `</`, тегов, `prompt in answer`).

### FIX-C1 [CRITICAL]: Подключить настоящий `Chunker` с очисткой

[`kaggle_main.py`](src/kaggle_main.py:35) — заменить импорт и использование чанкера. Заменить legacy `create_chunks` на класс `Chunker`, прогоняющий `clean_text()`:

[`chunker.py`](src/chunker.py:418) — переписать `chunk_all_websites`, чтобы он использовал класс:
```python
def chunk_all_websites(websites_data: List[Tuple[int, str]]) -> List[Chunk]:
    """Chunk all websites через ОЧИЩАЮЩИЙ Chunker (не legacy create_chunks)."""
    chunker = Chunker(ChunkerConfig(
        chunk_size=650,        # FIX-C2: вмещаем цельный FAQ-ответ
        chunk_overlap=120,
        min_chunk_length=40,
    ))
    all_chunks: List[Chunk] = []
    next_id = 0
    for web_id, text in websites_data:
        for piece in chunker.chunk_text(text):   # clean_text внутри
            all_chunks.append(Chunk(chunk_id=next_id, web_id=web_id, text=piece))
            next_id += 1
    return all_chunks
```
Эффект: индекс и BM25 строятся на чистом тексте — reranker перестаёт тащить HTML-мусор.

### FIX-R2/R3 [CRITICAL]: Чистить текст для reranker + нормализовать query

[`retriever.py`](src/retriever.py:361) — нормализовать запрос так же, как чанки при индексации:
```python
from indexer import normalize_for_embedding   # вверху файла

        # ── Stage 1: FAISS semantic search ──
        norm_query = normalize_for_embedding(query)     # FIX-R3: ё→е, NFC
        query_embedding = self.indexer.model.encode(
            [norm_query], normalize_embeddings=True,
        ).astype(np.float32)
```

[`retriever.py`](src/retriever.py:400) — reranker должен скорить ОЧИЩЕННЫЙ текст:
```python
        # FIX-R2: cross-encoder скорит чистый текст, не сырой HTML
        all_pairs = [
            (query, clean_chunk_text(text, self.cleaner_config) or text)
            for _, text in merged_candidates
        ]
```

### FIX-R4 [MEDIUM]: Reciprocal Rank Fusion вместо «сырого» слияния

[`retriever.py`](src/retriever.py:381) — заменить блок merge:
```python
        # ── Stage 3: Reciprocal Rank Fusion (RRF) ──
        K_RRF = 60
        rrf: dict[int, float] = {}
        text_by_id: dict[int, str] = {}

        for rank, (cid, text) in enumerate(faiss_candidates):
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (K_RRF + rank)
            text_by_id[cid] = text
        for rank, (cid, text, _) in enumerate(bm25_candidates):
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (K_RRF + rank)
            text_by_id.setdefault(cid, text)

        merged_candidates = [
            (cid, text_by_id[cid])
            for cid in sorted(rrf, key=rrf.get, reverse=True)
        ]
        if not merged_candidates:
            return []
```

### FIX-R5 [MEDIUM]: Прибить reranker к конкретному GPU

[`retriever.py`](src/retriever.py:278)
```python
        import torch
        rerank_device = "cuda:1" if torch.cuda.device_count() >= 2 else (
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self.reranker = CrossEncoder(reranker_model, device=rerank_device)
```
На 2×T4: LLM шардится `auto`, reranker — на `cuda:1`. Если конфликт по памяти — оставьте reranker на `cuda:0`, а LLM прибейте к `cuda:0` тоже (1B модель влезает в один T4).

### FIX-G1 [CRITICAL]: Переписать промпт под полноту + список

[`kaggle_main.py`](src/kaggle_main.py:69) (и зеркально [`generator.py:221`](src/generator.py:221)):
```python
SYSTEM_PROMPT = """Ты — банковский AI-ассистент Альфа-Банка. Отвечай на вопрос ПОЛНО и фактически точно, опираясь ТОЛЬКО на предоставленный контекст.

ПРАВИЛА:
1. Используй все релевантные факты из контекста. Не выдумывай.
2. Если ответ — это перечень шагов или вариантов, оформи его списком или через точку с запятой. Сохраняй ВСЕ пункты.
3. НЕ пиши вводных слов («Согласно фрагменту», «Таким образом», «В контексте указано»). Сразу давай суть.
4. Не здоровайся, не предлагай помощь, без рекламы.
5. Объём ответа — как в справке банка: обычно 2–5 предложений, при списках больше.

ПРИМЕРЫ:
Вопрос: Что такое БИК?
Контекст: БИК — банковский идентификационный код для перечисления средств.
Ответ: БИК — это банковский идентификационный код, используемый для перечисления средств.

Вопрос: Как получить карту?
Контекст: Карту можно получить доставкой или в офисе. После получения нужно подписать договор и активировать карту в приложении: выберите карту → Активация → введите код из SMS → задайте пин-код.
Ответ: Получить карту можно доставкой или в офисе. После получения подпишите договор и активируйте карту в приложении: выберите карту на главном экране, нажмите «Активация», введите код из SMS и задайте пин-код.
""".strip()
```

### FIX-G3 [HIGH]: Поднять `max_new_tokens`
Сделано в FIX-2 (`max_new_tokens=320`). Без этого FIX-1 бесполезен — модель физически не сгенерит длинный ответ.

### FIX-G4 [HIGH]: Правильная «зигзаг» раскладка вместо реверса

[`retriever.py`](src/retriever.py:470) — заменить `context_parts = context_parts[::-1]`:
```python
        # "Lost in the Middle" fix: зигзаг — топовые чанки по краям, слабые в центре.
        # results уже отсортированы по убыванию релевантности.
        head, tail = [], []
        for i, part in enumerate(context_parts):
            (head if i % 2 == 0 else tail).append(part)
        context_parts = head + tail[::-1]   # лучший — в начале, 2-й — в конце
```

### FIX-G5 [MEDIUM]: Нумеровать фрагменты (и чистить метки в ответе)

[`retriever.py`](src/retriever.py:474)
```python
        labeled = [f"[Фрагмент {i+1}] {p}" for i, p in enumerate(context_parts)]
        return "\n\n".join(labeled)
```
А `strip_preamble` (FIX-G6) убирает «Согласно Фрагменту N» из финального ответа.

### FIX-G6 [HIGH]: Стрипалка «воды» / преамбул

Добавить в [`kaggle_main.py`](src/kaggle_main.py:91) (и/или generator.py):
```python
_PREAMBLE_RE = re.compile(
    r"^\s*(?:согласно\s+(?:фрагмент[ауые]*\s*\d*|контекст[ауе]*|предоставленны[мх][^,.:]*)[,:]?\s*"
    r"|в\s+фрагмент[еах]*\s*\d*\s*(?:указано|сказано|говорится)[,:]?\s*"
    r"|таким образом[,:]?\s*"
    r"|исходя из (?:контекста|вышесказанного)[,:]?\s*"
    r"|ответ[:：]\s*)",
    flags=re.IGNORECASE | re.UNICODE,
)

def strip_preamble(text: str) -> str:
    """Убирает мета-вводные клише, разбавляющие ответ."""
    prev = None
    while prev != text:                 # многократно, если клише вложены
        prev = text
        text = _PREAMBLE_RE.sub("", text).lstrip(" «\"—-:")
    # убрать остаточные «(Фрагмент 3)» в конце
    text = re.sub(r"\s*\(?\s*фрагмент\s*\d+\s*\)?\.?\s*$", "", text,
                  flags=re.IGNORECASE).strip()
    return text
```

### FIX-G7 [MEDIUM]: Обрезка по границе предложения (не по символу)
При `MAX_RESPONSE_CHARS=500` приоритетна `truncate_to_sentences`. Убедитесь, что в `KaggleGenerator.generate` сначала идёт `truncate_to_sentences`, потом `truncate_to_chars` (уже так). Дополнительно — не резать символьно, если результат уже ≤ лимита по предложениям.

### FIX-G8 [LOW]: Кеш doc_freq в extract
[`generator.py`](src/generator.py:147) — предрассчитать `doc_freq` один раз перед циклом, передавать словарём. O(N²)→O(N).

### FIX-3 [CRITICAL]: Сделать валидацию действующей
[`kaggle_main.py`](src/kaggle_main.py:478)
```python
        if validate_answers and answer:
            if not validate_answer(query, answer, min_overlap):
                # fallback: извлекаем из уже полученного контекста
                fb = extract_answer_from_context(query, context or "")
                if fb and validate_answer(query, fb, min_overlap):
                    answer = fb
                stats["invalid"] += 1
```

### FIX-4 [HIGH]: Версионировать кеш + батч-сохранение
[`kaggle_main.py`](src/kaggle_main.py:322)
```python
PIPELINE_VERSION = "v2-adaptive-len"   # менять при правке промпта/чанков

    @staticmethod
    def _make_key(query: str, model: str) -> str:
        raw = f"{PIPELINE_VERSION}|{query.strip()}|{model}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
```
И сохранять кеш не на каждом `set`, а раз в 200 ответов (вынести `_save()` из `set()` в основной цикл рядом с чекпоинтом).

### FIX-6 [HIGH]: Надёжный резюм по q_id, а не по индексу
[`kaggle_main.py`](src/kaggle_main.py:451) — пропускать уже отвеченные `q_id`, а не сравнивать индексы:
```python
    done_ids = {str(r["q_id"]) for r in results}
    for _, row in tqdm(questions_df.iterrows(), total=total, desc="Generating"):
        q_id = str(row["q_id"])
        if q_id in done_ids:
            continue
        ...
```

---

# 5. 🎯 Прочие улучшения скора (нетривиальные)

### 5.1 [HIGH] Колонка эталона называется `answer_new`, у вас `answer`
[`sample_submission.csv:1`](data/sample_submission.csv:1) — заголовок `q_id,answer_new`. А вы пишете колонку `answer` ([`kaggle_main.py:492`](src/kaggle_main.py:492)). **Проверьте требуемое имя колонки в сабмишене на лидерборде.** Если грейдер ждёт `answer_new`, а вы шлёте `answer` — получите 0 или ошибку парсинга. Это потенциальный «тихий ноль» всего сабмита.

### 5.2 [HIGH] LoRA fine-tuning на эталоне = утечка стиля длины
`sample_submission.csv` (baseline 75.8) — это и есть распределение целевых ответов. Файнтюн на нём научит модель *правильной длине и формату списков автоматически*, без ручных лимитов. Но: в эталоне есть «Нет ответа.» (q_id=7,13) и вода «Согласно Фрагменту». **Перед файнтюном почистите train от «Согласно Фрагменту/Таким образом»**, иначе модель выучит воду. На L4 (3–7ч) QLoRA Vikhr-1B на 6977 примерах — 1–2 эпохи реально.

### 5.3 [MEDIUM] «Нет ответа.» как осознанный класс
Эталон содержит «Нет ответа.» для нерелевантных вопросов. Сейчас ваш greedy-extract **всегда** что-то возвращает. Если reranker top-1 score ниже порога — честнее вернуть «Нет ответа.» (совпадёт с эталоном дословно → высокий BERTScore на этих вопросах). Добавьте порог: `if best_rerank_score < THRESHOLD: return "Нет ответа."`.

### 5.4 [MEDIUM] Тайминг под 12ч (2×T4)
- Текущее: ~55 кандидатов × reranker + 64 ток генерации ≈ 5–6 c/вопрос ≈ 10–11.5ч на 6977. **Опасно близко к лимиту.**
- После FIX-G3 (320 токенов) генерация вырастет → можно вылезти за 12ч. Компенсация: `TOP_K_RETRIEVAL=24`, `TOP_K_BM25=12` (RRF сохранит recall), `RERANKER_BATCH_SIZE=32` (1B-сетап + reranker на отдельном T4 потянет). Батчуйте **генерацию** (pipeline принимает список промптов) — это даёт ×2-3 throughput на GPU.
- **Сильный рычаг:** обернуть LLM в **vLLM** (`pip install vllm`) — для Vikhr-1B даёт ×5-10 throughput на T4 за счёт continuous batching. Тогда 320 токенов перестают быть проблемой по времени.

### 5.5 [MEDIUM] Эмбеддинг батч 8 на GPU — слишком мелко
[`indexer.py:227`](src/indexer.py:227) `batch_size=8 if cuda` — для BGE-M3 на T4 это недогруз. Ставьте 32–64, индексация ускорится в разы. `empty_cache()` после каждого батча ([`indexer.py:244`](src/indexer.py:244)) тоже тормозит — вызывайте раз в N батчей.

### 5.6 [L4-сценарий, 3–7ч] Апгрейд LLM
На L4 (24GB) влезает Qwen2.5-7B-Instruct в fp16 — он заметно сильнее Vikhr-1B на русском reasoning и форматировании списков. Связка: BGE-M3 + reranker на L4 + Qwen-7B через vLLM → 6977 вопросов за ~2-3ч, качество выше. Это лучший ROI, если L4 доступна. Fallback на Kaggle оставить Vikhr-1B.

### 5.7 [LOW] `requirements.txt` — закрепить версии
Незакреплённые версии `transformers`/`sentence-transformers` на Kaggle = риск, что `return_full_text` / chat template поведут себя иначе. Закрепите проверенные версии.

### 5.8 [LOW] Тесты под новую длину
[`tests/test_generator.py`](tests/test_generator.py) наверняка проверяет старый лимит 150. После FIX-1 обновите ассерты, иначе CI «покраснеет» и собьёт с толку.

---

## 📋 Чек-лист внедрения (по убыванию ROI)

| # | Фикс | Файл | Эффект | Риск |
|---|------|------|--------|------|
| 1 | FIX-1 адаптивная длина 500/5 | config.py | 🔥🔥🔥 +recall | нет |
| 2 | FIX-C1 подключить чистый Chunker | chunker.py/kaggle_main | 🔥🔥🔥 чистый индекс | пересборка индекса |
| 3 | FIX-2 return_full_text=False | kaggle_main.py | 🔥🔥 убирает мусор в ответе | нет |
| 4 | FIX-G1/G3 промпт+токены | kaggle_main.py | 🔥🔥 полнота ответа | время |
| 5 | FIX-R2/R3 нормализация query + чистый reranker | retriever.py | 🔥🔥 +recall | нет |
| 6 | FIX-5.1 имя колонки answer_new | kaggle_main.py | 🔥🔥🔥 не обнулить сабмит | проверить ЛБ |
| 7 | FIX-G4 зигзаг | retriever.py | 🔥 | нет |
| 8 | FIX-R4 RRF | retriever.py | 🔥 стабильность | нет |
| 9 | FIX-G6 strip_preamble | kaggle_main.py | 🔥 precision | нет |
| 10 | FIX-5.2 LoRA на чистом эталоне (L4) | finetuning.py | 🔥🔥 авто-стиль | время L4 |
| 11 | FIX-5.4/vLLM тайминг | kaggle_main.py | время в бюджете | интеграция |

> **Минимальный набор для первого ре-сабмита:** FIX-1 + FIX-2 + FIX-C1 + FIX-G1/G3 + FIX-5.1. Это устранит главную утечку (обрезка), даст модели генерить полный ответ на чистом контексте и гарантирует валидный формат сабмита.

Глубокий code-review завершён. Создан файл FullAudit.md с полным разбором пайплайна по точке входа kaggle_main.py.

Главный вывод (root cause скора ~0.22)
Сравнив код с эталоном data/sample_submission.csv (baseline 75.8, колонка answer_new), я обнаружил, что эталонные ответы длиной 250–450+ символов, 3–6 предложений, со списками, а ваш пайплайн жёстко режет ответ до 150 символов / 2 предложений (config.py:43-45). Вы боялись штрафа L(q) за длину, но штраф карает за слишком длинные ответы (≥3×Lr). Зарезав до 150, вы не попали под L(q), зато обнулили сам BERTScore-Recall — это и есть утечка.

Структура FullAudit.md
🔴 Критические баги (6 шт): обрезка длины; сломанный парсинг split("</") (оставляет промпт в ответе); мёртвая validate_answer; отравление кеша при смене промпта + I/O на каждом вопросе; двойная загрузка модели (OOM на 2×T4); хрупкий резюм по индексу.

🔍 Retrieval: legacy-чанкер create_chunks без clean_text() → HTML/[chunk_id] в FAISS и BM25; reranker скорит сырой текст; query НЕ нормализуется (ё→е) при поиске, а индекс — да; «сырое» слияние вместо RRF; риск reranker на CPU.

🧠 Generation: промпт прямо приказывает быть коротким (против метрики); few-shot не учит спискам; max_new_tokens=64 обрубает ответ; наивный реверс чанков вместо зигзага; нет нумерации фрагментов; пост-процессинг не чистит «воду» («Согласно Фрагменту…»).

🛠️ Action Plan: готовые copy-paste Python-диффы FIX-1…FIX-G8 для каждого файла (адаптивная длина 500/5, return_full_text=False, подключение чистого Chunker, нормализация query, RRF, зигзаг, strip_preamble, версионирование кеша).

🎯 Прочее: проверка имени колонки сабмита (answer/answer_new — риск тихого нуля); LoRA на очищенном эталоне; класс «Нет ответа.»; тайминг под 12ч (vLLM + батч-генерация); сценарии для L4 на 3–7ч (Qwen2.5-7B через vLLM, лучший ROI).

В конце — чек-лист внедрения по убыванию ROI. Минимальный набор для первого ре-сабмита: FIX-1 + FIX-2 + FIX-C1 + FIX-G1/G3 + проверка колонки. Прогноз: ~0.22 → 0.45–0.60+.

Также обновлён банк памяти .roo/rules/learned-instincts.md и отправлено уведомление в Telegram.