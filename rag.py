"""Ядро RAG: чанкинг, эмбеддинги, ChromaDB, генерация через Gemini."""
from __future__ import annotations

import os
import re
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# кириллица в консоли Windows (cp1251/cp866) без UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------- конфигурация
ROOT = Path(__file__).parent
DOCS_DIR = ROOT / "docs"
CHROMA_DIR = ROOT / "chroma_db"
COLLECTION = "hr_policies"

EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small")
# По умолчанию — gemini-3.1-flash-lite: на ней получены результаты в eval/results.md.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
# Минимальная пауза между запросами к Gemini (бесплатный тариф ограничен по RPM).
GEMINI_MIN_INTERVAL = float(os.getenv("GEMINI_MIN_INTERVAL", "6"))
GEMINI_MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "6"))
# Таймаут одного HTTP-запроса: без него зависший запрос блокирует прогон навсегда.
GEMINI_TIMEOUT = float(os.getenv("GEMINI_TIMEOUT", "120"))

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 5

NO_ANSWER = "В документах нет информации"


# ---------------------------------------------------------------- документы и чанки
@dataclass
class Chunk:
    id: str
    text: str
    source: str  # имя файла
    title: str   # заголовок документа


def doc_title(text: str, fallback: str) -> str:
    m = re.search(r"^#\s+(.+)$", text, flags=re.M)
    return m.group(1).strip() if m else fallback


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Скользящее окно ~size символов с перекрытием overlap.

    Граница чанка сдвигается к ближайшему абзацу / концу строки / предложению / пробелу,
    чтобы не резать слова и по возможности не резать мысли.
    """
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    chunks: list[str] = []
    start, n = 0, len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            floor = start + int(size * 0.5)
            for sep in ("\n\n", "\n", ". ", " "):
                cut = text.rfind(sep, floor, end)
                if cut != -1:
                    end = cut + len(sep)
                    break
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        nxt = max(end - overlap, start + 1)  # гарантируем движение вперёд
        # начинаем следующий чанк с начала слова
        sp = text.find(" ", nxt, end)
        start = sp + 1 if sp != -1 else nxt
    return chunks


def load_chunks(docs_dir: Path = DOCS_DIR) -> list[Chunk]:
    out: list[Chunk] = []
    for path in sorted(docs_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        title = doc_title(text, path.stem)
        for i, piece in enumerate(chunk_text(text)):
            out.append(Chunk(id=f"{path.name}#{i}", text=piece, source=path.name, title=title))
    return out


# ---------------------------------------------------------------- эмбеддинги и база
@lru_cache(maxsize=1)
def get_embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBED_MODEL)


def embed(texts: list[str], kind: str) -> list[list[float]]:
    # e5 обучена с префиксами "query: " / "passage: "
    prefixed = [f"{kind}: {t}" for t in texts]
    vecs = get_embedder().encode(prefixed, normalize_embeddings=True, batch_size=32)
    return vecs.tolist()


def get_collection(reset: bool = False):
    import chromadb

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if reset:
        try:
            client.delete_collection(COLLECTION)
        except Exception:
            pass
    return client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})


@dataclass
class Hit:
    text: str
    source: str
    title: str
    score: float  # косинусное сходство


def retrieve(question: str, k: int = TOP_K) -> list[Hit]:
    col = get_collection()
    if col.count() == 0:
        raise RuntimeError("Индекс пуст. Сначала запустите: python ingest.py")
    res = col.query(query_embeddings=embed([question], "query"), n_results=k)
    return [
        Hit(text=doc, source=meta["source"], title=meta["title"], score=1 - dist)
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]


# ---------------------------------------------------------------- Gemini
class _RateLimiter:
    def __init__(self, interval: float):
        self.interval = interval
        self.last = 0.0

    def wait(self):
        delta = time.monotonic() - self.last
        if delta < self.interval:
            time.sleep(self.interval - delta)
        self.last = time.monotonic()


_limiter = _RateLimiter(GEMINI_MIN_INTERVAL)


@lru_cache(maxsize=1)
def get_client():
    from google import genai
    from google.genai import types

    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError(
            "Не задан GEMINI_API_KEY. Получите ключ в Google AI Studio и задайте переменную "
            "окружения или строку GEMINI_API_KEY=... в файле .env"
        )
    return genai.Client(
        api_key=os.environ["GEMINI_API_KEY"],
        http_options=types.HttpOptions(timeout=int(GEMINI_TIMEOUT * 1000)),  # мс
    )


def _retry_delay(err: Exception, attempt: int) -> float:
    """Берём retryDelay из ответа API, если он есть, иначе экспоненциальный backoff."""
    m = re.search(r"retry(?:Delay)?['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s", str(err), flags=re.I)
    if m:
        return float(m.group(1)) + 1
    return min(5 * 2**attempt, 120)


def gemini_generate(prompt: str, *, system: str | None = None, schema=None,
                    model: str | None = None, temperature: float = 0.0):
    """Вызов Gemini с паузами между запросами и повтором при 429 / 5xx."""
    from google.genai import errors, types

    cfg = types.GenerateContentConfig(temperature=temperature, system_instruction=system)
    if schema is not None:
        cfg.response_mime_type = "application/json"
        cfg.response_schema = schema

    for attempt in range(GEMINI_MAX_RETRIES + 1):
        _limiter.wait()
        try:
            resp = get_client().models.generate_content(
                model=model or GEMINI_MODEL, contents=prompt, config=cfg
            )
            return resp.parsed if schema is not None else resp.text
        except errors.APIError as e:
            retriable = e.code == 429 or (e.code is not None and e.code >= 500)
            if not retriable or attempt == GEMINI_MAX_RETRIES:
                raise
            delay = _retry_delay(e, attempt)
            print(f"  [gemini] {e.code}: повтор через {delay:.0f} с (попытка {attempt + 1})", flush=True)
            time.sleep(delay)
        except Exception as e:  # таймаут / обрыв соединения на уровне HTTP-клиента
            if "timeout" not in type(e).__name__.lower() or attempt == GEMINI_MAX_RETRIES:
                raise
            delay = min(5 * 2**attempt, 120)
            print(f"  [gemini] таймаут: повтор через {delay:.0f} с (попытка {attempt + 1})", flush=True)
            time.sleep(delay)


# ---------------------------------------------------------------- генерация ответа
SYSTEM_PROMPT = f"""Ты — HR-ассистент компании DemoTech. Отвечай на вопросы сотрудников СТРОГО
по предоставленным фрагментам документов HR-политики.

Правила:
1. Используй только факты из фрагментов. Не добавляй знаний извне, не додумывай цифры.
2. Если во фрагментах нет ответа на вопрос, верни answer = "{NO_ANSWER}." , found = false, sources = [].
3. Отвечай кратко и по делу, на русском языке, сохраняя точные цифры, сроки и суммы.
4. В sources перечисли имена файлов документов (поле "source"), из которых взят ответ.
"""


class LLMAnswer(BaseModel):
    answer: str
    found: bool
    sources: list[str]


@dataclass
class Answer:
    text: str
    found: bool
    sources: list[str]  # имена файлов
    hits: list[Hit]


def build_context(hits: list[Hit]) -> str:
    return "\n\n".join(
        f"[Фрагмент {i}] source: {h.source} | документ: {h.title}\n{h.text}"
        for i, h in enumerate(hits, 1)
    )


def answer(question: str, k: int = TOP_K) -> Answer:
    hits = retrieve(question, k)
    prompt = f"Фрагменты документов:\n\n{build_context(hits)}\n\nВопрос сотрудника: {question}"
    parsed: LLMAnswer | None = gemini_generate(prompt, system=SYSTEM_PROMPT, schema=LLMAnswer)
    if parsed is None:
        return Answer(text=f"{NO_ANSWER}.", found=False, sources=[], hits=hits)

    known = {h.source for h in hits}
    sources = [s for s in dict.fromkeys(parsed.sources) if s in known]  # без галлюцинаций
    found = parsed.found and NO_ANSWER.lower() not in parsed.answer.lower()
    text = parsed.answer.strip()
    if found and sources:
        titles = {h.source: h.title for h in hits}
        text += "\n\nИсточник: " + "; ".join(f"{titles[s]} ({s})" for s in sources)
    if not found:
        sources = []
    return Answer(text=text, found=found, sources=sources, hits=hits)
