
"""
text_blocker_neural.py
-----------------------
Делит текст (произведение) на блоки примерно
равного размера (по числу слов) и генерирует для каждого блока 
"человечное" описание содержания — не набор цитат, а
пересказ своими словами. Для этого используется локальная
нейросеть-суммаризатор (по умолчанию — rut5_base_sum_gazeta)
Модель скачивается один раз (~1 ГБ) и кэшируется локально
(~/.cache/huggingface), дальше работает офлайн.

Установка (один раз, нужен интернет):
    pip install transformers torch sentencepiece --break-system-packages

Использование:
    python3 text_blocker_neural.py input.txt \
        --words-per-block 500 \
        --format markdown \
        --output result.md

"""

import argparse
import json
import re
import os
import sys
import spacy
import torchao
from torchao.quantization import quantize_, Int8WeightOnlyConfig
from nltk.tokenize import sent_tokenize
nlp = spacy.load('ru_core_news_sm')
RU_TAG = "[text_blocker_neural]"
with open('stopwords-ru.txt', 'r') as sw:
        stop = set(sw.read().split('\n'))

def log(msg):
    print(f"{RU_TAG} {msg}", file=sys.stderr)


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def split_sentences(text: str):
    return sent_tokenize(text, language='russian')
    
    #nlp splitting
    # doc = nlp(text)
    # return [sent.text for sent in doc.sents if sent.text.strip()]
    
    # regex splitting
    # text = re.sub(r"\s+", " ", text.strip())
    # sentence_endings = re.compile(r'(?<=[.!?…])\s+(?=[A-ZА-ЯЁ«"\(])')
    # raw = sentence_endings.split(text)
    # return [s.strip() for s in raw if s.strip()]
    


def words_of(sentence: str):
    return re.findall(r"[а-яёА-ЯЁa-zA-Z]+", sentence.lower())


def build_blocks(sentences, words_per_block):
    blocks = []
    current = []
    current_count = 0

    for sent in sentences:
        w = len(words_of(sent))
        if not current:
            current.append(sent)
            current_count += w
            continue

        if current_count + w >= words_per_block:
            diff_without = abs(current_count - words_per_block)
            diff_with = abs(current_count + w - words_per_block)
            if diff_with >= diff_without:
                blocks.append((current, current_count))
                current = [sent]
                current_count = w
                continue

        current.append(sent)
        current_count += w

    if current:
        blocks.append((current, current_count))

    return blocks


class NeuralSummarizer:


    def __init__(self, model_name="IlyaGusev/rut5_base_sum_gazeta",
                 device=None, chunk_token_limit=580, num_beams=4, num_threads=None):
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        except ImportError as e:
            raise RuntimeError() from e

        self.torch = torch
        log(f"Загружаю модель {model_name} (при первом запуске — скачивание "
            f"~1 ГБ, дальше берётся из локального кэша)...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        quantize_(self.model, Int8WeightOnlyConfig())        
        self.chunk_token_limit = chunk_token_limit
        self.num_beams = num_beams
        print(num_beams)
        if self.device == "cpu":
            threads = num_threads or os.cpu_count()
            torch.set_num_threads(threads)
            log(f"CPU-потоков для инференса: {threads} "
                f"(torch реально использует: {torch.get_num_threads()})")

        log(f"Модель загружена, работаю на устройстве: {self.device}")
    
    def _generate(self, text, max_length=150, min_length=10):
        inputs = self.tokenizer(
            [text],
            max_length=600,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with self.torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_length=max_length,
                min_length=min_length,
                no_repeat_ngram_size=3,
                num_beams=self.num_beams,
            )[0]

        return self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    def _split_into_token_chunks(self, sentences):
        chunks = []
        current_sents = []
        current_len = 0

        for sent in sentences:
            n_tokens = len(self.tokenizer(sent)["input_ids"])
            if current_sents and current_len + n_tokens > self.chunk_token_limit:
                chunks.append(" ".join(current_sents))
                current_sents = [sent]
                current_len = n_tokens
            else:
                current_sents.append(sent)
                current_len += n_tokens

        if current_sents:
            chunks.append(" ".join(current_sents))

        return chunks

    def summarize(self, block_sentences, max_length=150, min_length=40):
        full_text = " ".join(block_sentences)
        total_tokens = len(self.tokenizer(full_text)["input_ids"])

        if total_tokens <= self.chunk_token_limit:
            return self._generate(full_text, max_length, min_length)

        log(f"Блок большой ({total_tokens} токенов) — делю на подкуски "
            f"для иерархической суммаризации...")
        chunks = self._split_into_token_chunks(block_sentences)
        chunk_summaries = [
            self._generate(c, max_length=100, min_length=25) for c in chunks
        ]
        combined = " ".join(chunk_summaries)
        combined_tokens = len(self.tokenizer(combined)["input_ids"])

        if combined_tokens <= self.chunk_token_limit:
            return self._generate(combined, max_length, min_length)
        else:
            return self.summarize(
                [combined], max_length=max_length, min_length=min_length
            )


def extract_keywords(sentences, top_n=6):
    from collections import Counter
    freq = Counter()
    for sent in sentences:
        for w in words_of(sent):
            if w not in stop and len(w) > 2:
                freq[w] += 1
    return [w for w, _ in freq.most_common(top_n)]


def process(text, words_per_block, summarizer, max_length, min_length, keywords_n):
    sentences = split_sentences(text)
    raw_blocks = build_blocks(sentences, words_per_block)

    results = []
    for i, (block_sentences, word_count) in enumerate(raw_blocks, start=1):
        log(f"Обрабатываю блок {i}/{len(raw_blocks)} ({word_count} слов)...")
        summary = summarizer.summarize(block_sentences, max_length, min_length)
        keywords = extract_keywords(block_sentences, keywords_n)
        results.append({
            "block": i,
            "word_count": word_count,
            "sentence_count": len(block_sentences),
            "summary": summary,
            "keywords": keywords,
        })
    return results


def format_markdown(results):
    lines = []
    for r in results:
        lines.append(f"## Блок {r['block']} "
                     f"(~{r['word_count']} слов, {r['sentence_count']} предл.)")
        lines.append("")
        lines.append(r["summary"])
        lines.append("")
        lines.append(f"**Ключевые слова:** {', '.join(r['keywords'])}")
        lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines)


def format_text(results):
    lines = []
    for r in results:
        lines.append(f"Блок {r['block']} "
                     f"(~{r['word_count']} слов, {r['sentence_count']} предл.)")
        lines.append(r["summary"])
        lines.append("Ключевые слова: " + ", ".join(r["keywords"]))
        lines.append("-" * 60)
    return "\n".join(lines)


def format_json(results):
    return json.dumps(results, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Делит текст на блоки заданного размера (в словах) и "
                    "генерирует человечное, связное описание каждого блока "
                    "с помощью локальной нейросети-суммаризатора."
    )
    parser.add_argument("input", help="Путь к текстовому файлу (utf-8)")
    parser.add_argument("--words-per-block", type=int, default=500,
                        help="Целевой размер блока в словах (по умолчанию 500)")
    parser.add_argument("--model", default="IlyaGusev/rut5_base_sum_gazeta",
                        help="Название модели на HuggingFace Hub")
    parser.add_argument("--max-summary-length", type=int, default=150,
                        help="Максимальная длина описания блока в токенах")
    parser.add_argument("--min-summary-length", type=int, default=40,
                        help="Минимальная длина описания блока в токенах")
    parser.add_argument("--keywords", type=int, default=6,
                        help="Число ключевых слов на блок (по умолчанию 6)")
    parser.add_argument("--device", default=None,
                        help="cpu или cuda (по умолчанию — автоопределение)")
    parser.add_argument("--format", choices=["markdown", "text", "json"],
                        default="text", help="Формат вывода")
    parser.add_argument("--output", default=None,
                        help="Путь для сохранения результата (по умолчанию - stdout)")
    parser.add_argument("--num-beams", type=int, default=4,
                        help="Число лучей в beam search (по умолчанию 4). "
                             "Уменьшите до 2 или 1 (greedy) для заметного "
                             "ускорения на CPU ценой небольшой потери качества.")
    parser.add_argument("--num-threads", type=int, default=None,
                        help="Число потоков CPU для torch (по умолчанию — "
                             "все ядра, os.cpu_count())")

    args = parser.parse_args()

    text = read_text(args.input)

    summarizer = NeuralSummarizer(model_name=args.model, device=args.device, num_beams=args.num_beams, num_threads=args.num_threads)

    results = process(
        text, args.words_per_block, summarizer,
        args.max_summary_length, args.min_summary_length, args.keywords
    )

    if args.format == "markdown":
        out = format_markdown(results)
    elif args.format == "text":
        out = format_text(results)
    else:
        out = format_json(results)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(out)
        log(f"Готово: {len(results)} блок(ов) сохранено в {args.output}")
    else:
        print(out)


if __name__ == "__main__":
    main()