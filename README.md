# booksummary: RAG система для книг и документов

## 1. Описание

Скрипт `text_blocker_rag.py` позволяет работать с документами через систему RAG, используя локальную LLM, и имеет 3 режима:

- `index` — загружает один или несколько UTF-8 `.txt` файлов, делит текст на фрагменты, строит embeddings и сохраняет их в Qdrant.
- `ask` — превращает вопрос в embedding, ищет наиболее близкие фрагменты в Qdrant и передаёт их вместе с вопросом чат-модели Ollama.
- `summarize` — запускает процесс краткого пересказа содержимого текста.  

По умолчанию Qdrant работает в локальном persistent-режиме: сервер Qdrant/Docker не нужен, база сохраняется в `./qdrant_db`.

## 2. Установка

Рекомендуется Python 3.10+ и отдельное виртуальное окружение.

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Установите библиотеки RAG:

```bash
pip install -r requirements-rag.txt
```

Установите Ollama с https://ollama.com/download и загрузите модель генерации:

```bash
ollama pull qwen2.5:3b
```

Если Ollama не запустилась автоматически, запустите её отдельно:

```bash
ollama serve
```

Embedding-модель Hugging Face скачивается автоматически при первом запуске `index` или `ask`.

## 3. Индексация документов

```bash
python text_blocker_rag.py index ./docs/manual.txt ./docs/faq.txt
```

Можно передать один файл или несколько. Файлы должны быть в UTF-8 и пока поддерживается только обычный текст `.txt`.

Параметры по умолчанию:

- Qdrant-каталог: `./qdrant_db`;
- коллекция: `text_knowledge`;
- embedding-модель: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`;
- размер фрагмента: 1100 символов;
- overlap: 180 символов.

Изменить параметры можно так:

```bash
python text_blocker_rag.py index ./docs/manual.txt \
  --chunk-size 900 \
  --chunk-overlap 150 \
  --qdrant-path ./storage/qdrant \
  --collection manuals
```

Повторная индексация использует стабильные ID фрагментов, чтобы не добавлять дубликаты для тех же позиций документа. Если нужно полностью перестроить коллекцию:

```bash
python text_blocker_rag.py index ./docs/manual.txt --reset
```

**Внимание:** `--reset` удаляет всю текущую коллекцию, а не только один указанный файл. Передай все документы, которые должны остаться в коллекции.

## 4. Вопросы к документам

```bash
python text_blocker_rag.py ask "Как оформить возврат товара?"
```

Можно изменить число найденных фрагментов и модель:

```bash
python text_blocker_rag.py ask "Какие сроки возврата?" --k 7 --llm-model qwen2.5:3b
```

Скрипт печатает ответ и пути к документам/номера фрагментов, которые были переданы модели.

## 5. Саммаризация

Поддерживается два режима:

```bash
python text_blocker_rag.py summarize ./docs/manual.txt --format markdown --output summary.md
```

И вызов без слова `summarize`:

```bash
python text_blocker_rag.py ./docs/manual.txt --format markdown --output summary.md
```
При саммаризации вычисляются ключевые для блока слова. Можно указать словарь стоп-слов разместив в папке со скриптом файл `stopwords-ru.txt`.

## 6. Как проходит RAG-запрос

1. **Загрузка:** `index` читает текстовый файл и сохраняет имя файла и номер фрагмента в metadata.
2. **Chunking:** `RecursiveCharacterTextSplitter` делит большой текст на небольшие перекрывающиеся куски. Перекрытие помогает не потерять смысл на границе двух кусков.
3. **Embeddings:** `HuggingFaceEmbeddings` превращает каждый кусок в числовой вектор. Векторы описывают семантическое содержание текста.
4. **Vector Store:** LangChain записывает векторы, исходные тексты и metadata в коллекцию Qdrant.
5. **Retrieval:** при `ask` embedding вопроса сравнивается с векторами в Qdrant. Извлекаются `k` наиболее близких фрагментов.
6. **Generation:** LangChain `ChatOllama` отправляет вопрос и найденный контекст модели Qwen. В системной инструкции модели сказано отвечать только по переданным документам и сообщать, когда информации недостаточно.
7. **Sources:** скрипт показывает источники извлечённых фрагментов, чтобы можно было проверить, на какие документы опирался ответ.


## 7. Перспектива

Текущая версия рассчитана на `.txt`. Для PDF/DOCX планируется добавить соответствующие LangChain loaders; для большой нагрузки — подключить Qdrant Server/Cloud вместо локального persistent-режима; для повышения точности планируетс использовать reranker или hybrid search.
