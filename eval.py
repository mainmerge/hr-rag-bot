"""Оценка качества RAG.

Метрики (по категориям вопросов из eval/questions.json):
  * retrieval hit@3 / hit@1 / MRR@5 — есть ли документ-источник среди top-k фрагментов;
  * all@5 — для вопросов на два документа: оба документа есть в top-5 (в контексте LLM);
  * точность ответов — LLM-as-judge (Gemini) сравнивает ответ с эталоном;
  * верные источники — бот назвал все ожидаемые документы;
  * ложные отказы — «нет информации», хотя ответ есть;
  * правильные отказы — на вопросах без ответа и вопросах-ловушках.

Запуск:
  python eval.py                                # полная оценка
  python eval.py --model gemini-3.1-flash-lite  # другая модель генерации
  python eval.py --retrieval-only               # без обращений к Gemini
  python eval.py --no-cache                     # заново получить ответы и вердикты

Ответы модели и вердикты судьи кэшируются в eval/cache/, поэтому при исчерпании
дневной квоты бесплатного тарифа можно просто перезапустить скрипт позже.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

import rag

EVAL_DIR = Path(__file__).parent / "eval"
QUESTIONS = EVAL_DIR / "questions.json"
RESULTS_MD = EVAL_DIR / "results.md"
CACHE_DIR = EVAL_DIR / "cache"
JUDGE_BATCH = 10  # ответов на один запрос судьи — экономим квоту

ANSWERABLE = ["easy", "paraphrase", "multi_doc", "numeric"]
REFUSAL = ["unanswerable", "trap"]
CAT_RU = {
    "easy": "Простые",
    "paraphrase": "Перефразированные",
    "multi_doc": "Два документа",
    "numeric": "Подвох в цифрах",
    "unanswerable": "Нет в документах",
    "trap": "Ловушки (похожие)",
}


# ---------------------------------------------------------------- retrieval
def eval_retrieval(items: list[dict]) -> dict[int, dict]:
    out = {}
    for it in items:
        srcs = [h.source for h in rag.retrieve(it["question"], k=5)]
        ranks = [srcs.index(s) + 1 for s in it["sources"] if s in srcs]
        best = min(ranks) if ranks else None
        out[it["id"]] = {
            "hit1": best == 1,
            "hit3": best is not None and best <= 3,
            "all5": all(s in srcs for s in it["sources"]),
            "rr": 1 / best if best else 0.0,
            "top5": srcs,
        }
    return out


# ---------------------------------------------------------------- кэш
class Cache:
    def __init__(self, name: str, enabled: bool):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.path = CACHE_DIR / f"{name}.json"
        self.data = json.loads(self.path.read_text("utf-8")) if enabled and self.path.exists() else {}

    def get(self, key):
        return self.data.get(key)

    def set(self, key, value):
        self.data[key] = value
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=1), "utf-8")


def safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in s)


# ---------------------------------------------------------------- генерация
def get_answers(items: list[dict], cache: Cache) -> dict[int, dict]:
    out = {}
    for i, it in enumerate(items, 1):
        key = it["question"]
        cached = cache.get(key)
        if cached is None:
            print(f"  [{i}/{len(items)}] {it['question']}", flush=True)
            a = rag.answer(it["question"])
            cached = {"text": a.text, "found": a.found, "sources": a.sources}
            cache.set(key, cached)
        out[it["id"]] = cached
    return out


# ---------------------------------------------------------------- LLM-as-judge
JUDGE_SYSTEM = """Ты — строгий и беспристрастный эксперт, оценивающий ответы HR-бота.
Для каждого пункта сравни ОТВЕТ БОТА с ЭТАЛОННЫМ ОТВЕТОМ.
correct = true, если ответ бота по существу совпадает с эталоном: все ключевые цифры, сроки,
суммы и выводы (да/нет) верны и нет фактов, противоречащих эталону. Дополнительные верные
детали и строка «Источник: ...» допустимы.
correct = false, если:
- главный факт или итоговая цифра неверны (например, бот не учёл лимит или неверно посчитал);
- бот согласился с неверной посылкой вопроса;
- в эталоне две части ответа, а бот ответил только на одну;
- бот отказался отвечать.
Верни вердикт для каждого id с кратким обоснованием."""


class Verdict(BaseModel):
    id: int
    correct: bool
    reason: str


class Verdicts(BaseModel):
    verdicts: list[Verdict]


def judge(items: list[dict], answers: dict[int, dict], cache: Cache, model: str) -> dict[int, dict | None]:
    def key(it):
        return f"{it['question']}||{answers[it['id']]['text']}"

    todo = [it for it in items if cache.get(key(it)) is None]
    for start in range(0, len(todo), JUDGE_BATCH):
        batch = todo[start:start + JUDGE_BATCH]
        print(f"  судья: пункты {start + 1}–{start + len(batch)} из {len(todo)}", flush=True)
        prompt = "\n\n".join(
            f"id: {it['id']}\nВОПРОС: {it['question']}\nЭТАЛОННЫЙ ОТВЕТ: {it['expected']}\n"
            f"ОТВЕТ БОТА: {answers[it['id']]['text']}"
            for it in batch
        )
        res: Verdicts = rag.gemini_generate(prompt, system=JUDGE_SYSTEM, schema=Verdicts, model=model)
        by_id = {v.id: v for v in (res.verdicts if res else [])}
        for it in batch:
            v = by_id.get(it["id"])
            if v is not None:
                cache.set(key(it), {"correct": v.correct, "reason": v.reason})
    return {it["id"]: cache.get(key(it)) for it in items}


# ---------------------------------------------------------------- отчёт
def frac(xs: list[bool]) -> str:
    return f"{sum(xs) / len(xs) * 100:.0f}% ({sum(xs)}/{len(xs)})" if xs else "—"


def md_table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c).replace("|", "\\|").replace("\n", " ") for c in r) + " |" for r in rows]
    return "\n".join(lines)


def text_table(header: list[str], rows: list[list]) -> str:
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
    fmt = lambda r: "  ".join(str(c).ljust(w) for c, w in zip(r, widths))  # noqa: E731
    return "\n".join([fmt(header), "  ".join("-" * w for w in widths), *map(fmt, rows)])


def short(s: str, n: int = 70) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1] + "…"


def mark(x) -> str:
    return "—" if x is None else ("✅" if x else "❌")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="модель генерации (по умолчанию GEMINI_MODEL)")
    ap.add_argument("--judge-model", help="модель-судья (по умолчанию GEMINI_JUDGE_MODEL или модель генерации)")
    ap.add_argument("--retrieval-only", action="store_true", help="не вызывать Gemini")
    ap.add_argument("--no-cache", action="store_true", help="игнорировать кэш ответов и вердиктов")
    args = ap.parse_args()

    if args.model:
        rag.GEMINI_MODEL = args.model
    judge_model = args.judge_model or os.getenv("GEMINI_JUDGE_MODEL") or rag.GEMINI_MODEL

    qs = json.loads(QUESTIONS.read_text("utf-8"))["questions"]
    ans_items = [q for q in qs if q["category"] in ANSWERABLE]
    ref_items = [q for q in qs if q["category"] in REFUSAL]
    by_cat = {c: [q for q in qs if q["category"] == c] for c in ANSWERABLE + REFUSAL}

    print(f"Retrieval: {len(ans_items)} вопросов…")
    ret = eval_retrieval(ans_items)

    use_llm = not args.retrieval_only
    if use_llm and not os.getenv("GEMINI_API_KEY"):
        print("GEMINI_API_KEY не задан — генерация и LLM-as-judge пропущены (только retrieval).")
        use_llm = False

    answers, verdicts = {}, {}
    if use_llm:
        print(f"Генерация ответов ({rag.GEMINI_MODEL}), пауза {rag.GEMINI_MIN_INTERVAL:.0f} с между запросами…")
        answers = get_answers(ans_items + ref_items, Cache(f"answers_{safe_name(rag.GEMINI_MODEL)}", not args.no_cache))
        print(f"LLM-as-judge ({judge_model})…")
        verdicts = judge(ans_items, answers, Cache(f"judge_{safe_name(judge_model)}", not args.no_cache), judge_model)

    def correct(q):
        v = verdicts.get(q["id"])
        return None if v is None else v["correct"]

    def src_ok(q):
        return set(q["sources"]) <= set(answers[q["id"]]["sources"])

    # ---- сводка по категориям
    cat_header = ["Категория", "N", "hit@3", "all@5", "Точность ответов", "Источники верны", "Ложные отказы"]
    cat_rows = []
    for c in ANSWERABLE + [None]:
        items = ans_items if c is None else by_cat[c]
        row = [
            "**Все с ответом**" if c is None else CAT_RU[c], len(items),
            frac([ret[q["id"]]["hit3"] for q in items]),
            frac([ret[q["id"]]["all5"] for q in items]),
        ]
        if use_llm:
            row += [
                frac([correct(q) for q in items if correct(q) is not None]),
                frac([src_ok(q) for q in items]),
                frac([not answers[q["id"]]["found"] for q in items]),
            ]
        cat_rows.append(row)
    if not use_llm:
        cat_header = cat_header[:4]

    ref_header = ["Категория", "N", "Правильные отказы"]
    ref_rows = []
    if use_llm:
        for c in REFUSAL + [None]:
            items = ref_items if c is None else by_cat[c]
            ref_rows.append(["**Все без ответа**" if c is None else CAT_RU[c], len(items),
                             frac([not answers[q["id"]]["found"] for q in items])])

    n = len(ans_items)
    key_metrics = [
        ["Retrieval hit@1", frac([ret[q["id"]]["hit1"] for q in ans_items])],
        ["Retrieval hit@3", frac([ret[q["id"]]["hit3"] for q in ans_items])],
        ["Retrieval MRR@5", f"{sum(ret[q['id']]['rr'] for q in ans_items) / n:.3f}"],
    ]
    if use_llm:
        key_metrics += [
            ["Точность ответов (LLM-judge)", frac([correct(q) for q in ans_items if correct(q) is not None])],
            ["Правильные отказы", frac([not answers[q["id"]]["found"] for q in ref_items])],
        ]

    # ---- подробности
    det_header = ["#", "Категория", "Вопрос", "Ожид. документы", "top-3", "hit@3"]
    det_rows = []
    for q in ans_items:
        r = ret[q["id"]]
        row = [q["id"], CAT_RU[q["category"]], short(q["question"], 70), ", ".join(q["sources"]),
               ", ".join(r["top5"][:3]), mark(r["hit3"])]
        if use_llm:
            a = answers[q["id"]]
            row += [mark(correct(q)), mark(src_ok(q)), short(a["text"].split("\n\nИсточник:")[0], 110)]
        det_rows.append(row)
    if use_llm:
        det_header += ["Ответ верен", "Источники", "Ответ бота"]

    ref_det = [[q["id"], CAT_RU[q["category"]], q["question"], mark(not answers[q["id"]]["found"]),
                short(answers[q["id"]]["text"], 110)] for q in ref_items] if use_llm else []

    # ---- печать
    print("\n" + text_table(["Метрика", "Значение"], key_metrics))
    print("\n" + text_table(cat_header, [[str(c).replace("**", "") for c in r] for r in cat_rows]))
    if ref_rows:
        print("\n" + text_table(ref_header, [[str(c).replace("**", "") for c in r] for r in ref_rows]))
    print("\n" + text_table(["#", "Категория", "hit@3"] + (["Ответ", "Источн."] if use_llm else []),
                            [[r[0], r[1], r[5]] + (r[6:8] if use_llm else []) for r in det_rows]))
    if ref_det:
        print("\n" + text_table(["#", "Категория", "Отказ"], [[r[0], r[1], r[3]] for r in ref_det]))

    # ---- results.md
    md = [
        "# Результаты оценки RAG-бота",
        "",
        f"- Дата: {datetime.now():%Y-%m-%d %H:%M}",
        f"- Эмбеддинги: `{rag.EMBED_MODEL}`, чанки ≈{rag.CHUNK_SIZE} симв., перекрытие {rag.CHUNK_OVERLAP}, top-k = {rag.TOP_K}",
        f"- Генерация: `{rag.GEMINI_MODEL if use_llm else '— (не запускалась)'}`; судья: `{judge_model if use_llm else '—'}`",
        f"- Вопросов: {len(ans_items)} с ответом ("
        + ", ".join(f"{CAT_RU[c].lower()} — {len(by_cat[c])}" for c in ANSWERABLE)
        + f") + {len(ref_items)} без ответа ("
        + ", ".join(f"{CAT_RU[c].lower()} — {len(by_cat[c])}" for c in REFUSAL) + ")",
        "",
        "## Ключевые метрики",
        "",
        md_table(["Метрика", "Значение"], key_metrics),
        "",
        "## По категориям: вопросы с ответом",
        "",
        md_table(cat_header, cat_rows),
        "",
        "`all@5` — все ожидаемые документы попали в top-5 (контекст LLM); важно для вопросов на два документа.",
    ]
    if ref_rows:
        md += ["", "## По категориям: вопросы без ответа", "", md_table(ref_header, ref_rows)]
    md += ["", "## Вопросы с ответом — подробно", "", md_table(det_header, det_rows)]
    if ref_det:
        md += ["", "## Вопросы без ответа — подробно", "",
               md_table(["#", "Категория", "Вопрос", "Отказ", "Ответ бота"], ref_det)]
    if use_llm:
        wrong = [q for q in ans_items if correct(q) is False]
        if wrong:
            md += ["", "## Разбор ошибок (комментарий судьи)", ""]
            md += [f"- **#{q['id']}** ({CAT_RU[q['category']]}) {q['question']}  \n"
                   f"  Бот: {short(answers[q['id']]['text'], 300)}  \n"
                   f"  Судья: {verdicts[q['id']]['reason']}" for q in wrong]
        missed = [q for q in ref_items if answers[q["id"]]["found"]]
        if missed:
            md += ["", "## Пропущенные отказы", ""]
            md += [f"- **#{q['id']}** ({CAT_RU[q['category']]}) {q['question']}  \n"
                   f"  Бот: {short(answers[q['id']]['text'], 300)}" for q in missed]
    RESULTS_MD.write_text("\n".join(md) + "\n", "utf-8")
    print(f"\nСохранено: {RESULTS_MD}")


if __name__ == "__main__":
    main()
