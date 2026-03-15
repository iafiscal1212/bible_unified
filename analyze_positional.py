#!/usr/bin/env python3
"""
analyze_positional.py — Patrones posicionales: primera/última palabra de versículo,
distribución de POS por posición, patrones de apertura/cierre de capítulo.
"""
import json, logging
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

BASE = Path(__file__).parent
CORPUS = BASE / "bible_unified.json"
OUT = BASE / "results" / "positional"
OUT.mkdir(parents=True, exist_ok=True)
LOG = BASE / "logs"
LOG.mkdir(parents=True, exist_ok=True)

logging.basicConfig(filename=LOG / "positional.log", level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pos")


def load_corpus():
    with open(CORPUS, encoding="utf-8") as f:
        return json.load(f)


def analyze_book(book_words):
    if not book_words:
        return None
    name = book_words[0]["book"]
    corpus = book_words[0]["corpus"]

    # Agrupar por versículo
    verses = defaultdict(list)
    for w in book_words:
        verses[(w["chapter"], w["verse"])].append(w)

    # Primera y última palabra de cada versículo
    first_words = Counter()
    last_words = Counter()
    first_lemmas = Counter()
    last_lemmas = Counter()
    first_pos = Counter()
    last_pos = Counter()

    for (ch, vs), ws in verses.items():
        ws_sorted = sorted(ws, key=lambda x: x["word_pos"])
        if ws_sorted:
            first_words[ws_sorted[0]["text"]] += 1
            last_words[ws_sorted[-1]["text"]] += 1
            first_lemmas[ws_sorted[0]["lemma"]] += 1
            last_lemmas[ws_sorted[-1]["lemma"]] += 1
            first_pos[ws_sorted[0]["pos"]] += 1
            last_pos[ws_sorted[-1]["pos"]] += 1

    # POS por posición (1-based, para posiciones 1-10)
    pos_by_position = defaultdict(Counter)
    for ws in verses.values():
        for w in ws:
            if w["word_pos"] <= 10:
                pos_by_position[w["word_pos"]][w["pos"]] += 1

    # Primera/última palabra de cada capítulo
    chapters = defaultdict(list)
    for w in book_words:
        chapters[w["chapter"]].append(w)

    first_word_chapter = Counter()
    last_word_chapter = Counter()
    for ch, ws in chapters.items():
        ws_sorted = sorted(ws, key=lambda x: (x["verse"], x["word_pos"]))
        if ws_sorted:
            first_word_chapter[ws_sorted[0]["lemma"]] += 1
            last_word_chapter[ws_sorted[-1]["lemma"]] += 1

    return {
        "book": name,
        "corpus": corpus,
        "n_verses": len(verses),
        "first_word_verse_top10": first_words.most_common(10),
        "last_word_verse_top10": last_words.most_common(10),
        "first_lemma_verse_top10": first_lemmas.most_common(10),
        "last_lemma_verse_top10": last_lemmas.most_common(10),
        "first_pos_verse": dict(first_pos),
        "last_pos_verse": dict(last_pos),
        "pos_by_position": {str(p): dict(cnt) for p, cnt in sorted(pos_by_position.items())},
        "first_lemma_chapter_top5": first_word_chapter.most_common(5),
        "last_lemma_chapter_top5": last_word_chapter.most_common(5),
    }


def main():
    log.info("Cargando corpus...")
    words = load_corpus()
    log.info(f"Corpus: {len(words)} palabras")

    books_data = {}
    for w in words:
        books_data.setdefault(w["book"], []).append(w)

    log.info(f"Analizando {len(books_data)} libros...")
    with Pool() as pool:
        book_results = pool.map(analyze_book, list(books_data.values()))
    book_results = [r for r in book_results if r]

    # Global aggregation
    # Agrupar por versículo global
    verses_global = defaultdict(list)
    for w in words:
        verses_global[(w["book"], w["chapter"], w["verse"])].append(w)

    g_first_lemma = Counter()
    g_last_lemma = Counter()
    g_first_pos = Counter()
    g_last_pos = Counter()
    ot_first_pos = Counter()
    ot_last_pos = Counter()
    nt_first_pos = Counter()
    nt_last_pos = Counter()

    for key, ws in verses_global.items():
        ws_sorted = sorted(ws, key=lambda x: x["word_pos"])
        if ws_sorted:
            g_first_lemma[ws_sorted[0]["lemma"]] += 1
            g_last_lemma[ws_sorted[-1]["lemma"]] += 1
            g_first_pos[ws_sorted[0]["pos"]] += 1
            g_last_pos[ws_sorted[-1]["pos"]] += 1
            if ws_sorted[0]["corpus"] == "OT":
                ot_first_pos[ws_sorted[0]["pos"]] += 1
                ot_last_pos[ws_sorted[-1]["pos"]] += 1
            else:
                nt_first_pos[ws_sorted[0]["pos"]] += 1
                nt_last_pos[ws_sorted[-1]["pos"]] += 1

    # POS by position (global, positions 1-10)
    g_pos_by_position = defaultdict(Counter)
    for w in words:
        if w["word_pos"] <= 10:
            g_pos_by_position[w["word_pos"]][w["pos"]] += 1

    summary = {
        "global_first_lemma_top30": g_first_lemma.most_common(30),
        "global_last_lemma_top30": g_last_lemma.most_common(30),
        "global_first_pos": dict(g_first_pos),
        "global_last_pos": dict(g_last_pos),
        "ot_first_pos": dict(ot_first_pos),
        "ot_last_pos": dict(ot_last_pos),
        "nt_first_pos": dict(nt_first_pos),
        "nt_last_pos": dict(nt_last_pos),
        "global_pos_by_position": {
            str(p): dict(cnt) for p, cnt in sorted(g_pos_by_position.items())
        },
        "per_book": book_results,
    }

    out_file = OUT / "positional_analysis.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log.info(f"Resultados en {out_file}")
    print(f"[positional] DONE — {out_file}")


if __name__ == "__main__":
    main()
