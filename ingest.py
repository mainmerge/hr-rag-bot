"""Индексация документов: docs/*.md -> чанки -> эмбеддинги e5 -> ChromaDB."""
from collections import Counter

from rag import CHROMA_DIR, CHUNK_OVERLAP, CHUNK_SIZE, EMBED_MODEL, embed, get_collection, load_chunks


def main():
    chunks = load_chunks()
    if not chunks:
        raise SystemExit("В папке docs/ нет .md документов")

    print(f"Чанкинг: size≈{CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")
    for src, n in sorted(Counter(c.source for c in chunks).items()):
        print(f"  {src:<28} {n:>3} чанков")
    print(f"Всего: {len(chunks)} чанков, средняя длина {sum(len(c.text) for c in chunks) // len(chunks)} симв.")

    print(f"Эмбеддинги: {EMBED_MODEL} ...")
    vectors = embed([f"{c.title}\n{c.text}" for c in chunks], "passage")

    col = get_collection(reset=True)
    col.add(
        ids=[c.id for c in chunks],
        documents=[c.text for c in chunks],
        embeddings=vectors,
        metadatas=[{"source": c.source, "title": c.title} for c in chunks],
    )
    print(f"Готово: {col.count()} векторов в {CHROMA_DIR}")


if __name__ == "__main__":
    main()
