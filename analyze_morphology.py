#!/usr/bin/env python3
"""
analyze_morphology.py — Distribuciones morfológicas (POS) por libro, capítulo, posición.
Incluye: ratios verbo/nombre, bigramas POS, diversidad morfológica.
"""
import json, logging, os
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

BASE = Path(__file__).parent
CORPUS = BASE / "bible_unified.json"
OUT = BASE / "results" / "morphology"
OUT.mkdir(parents=True, exist_ok=True)
LOG = BASE / "logs"
LOG.mkdir(parents=True, exist_ok=True)

logging.basicConfig(filename=LOG / "morphology.log", level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("morph")


def load_corpus():
    with open(CORPUS, encoding="utf-8") as f:
        return json.load(f)


def analyze_book(book_words):
    if not book_words:
        return None
    name = book_words[0]["book"]
    corpus = book_words[0]["corpus"]

    pos_counts = Counter(w["pos"] for w in book_words)
    morph_codes = Counter(w["morph"] for w in book_words)
    n = len(book_words)

    # Verb/noun ratio
    verbs = pos_counts.get("verb", 0)
    nouns = pos_counts.get("noun", 0)
    vn_ratio = verbs / nouns if nouns > 0 else None

    # POS bigrams
    pos_bigrams = Counter()
    for i in range(len(book_words) - 1):
        # Only within same verse
        if (book_words[i]["chapter"] == book_words[i+1]["chapter"] and
            book_words[i]["verse"] == book_words[i+1]["verse"]):
            bg = (book_words[i]["pos"], book_words[i+1]["pos"])
            pos_bigrams[bg] += 1

    # POS by chapter
    chapters = {}
    for w in book_words:
        ch = w["chapter"]
        chapters.setdefault(ch, Counter())
        chapters[ch][w["pos"]] += 1

    # Morph diversity = unique morph codes / total words
    morph_diversity = len(morph_codes) / n if n else 0

    return {
        "book": name,
        "corpus": corpus,
        "n_words": n,
        "pos_distribution": dict(pos_counts),
        "verb_noun_ratio": round(vn_ratio, 4) if vn_ratio is not None else None,
        "morph_diversity": round(morph_diversity, 6),
        "n_unique_morph_codes": len(morph_codes),
        "top20_bigrams": [list(x) for x in pos_bigrams.most_common(20)],
        "pos_by_chapter": {str(ch): dict(cnt) for ch, cnt in sorted(chapters.items())},
    }


def main():
    log.info("Cargando corpus...")
    words = load_corpus()
    log.info(f"Corpus: {len(words)} palabras")

    books = {}
    for w in words:
        books.setdefault(w["book"], []).append(w)

    log.info(f"Analizando {len(books)} libros...")
    with Pool() as pool:
        book_results = pool.map(analyze_book, list(books.values()))
    book_results = [r for r in book_results if r]

    # Global POS distribution
    global_pos = Counter(w["pos"] for w in words)
    ot_pos = Counter(w["pos"] for w in words if w["corpus"] == "OT")
    nt_pos = Counter(w["pos"] for w in words if w["corpus"] == "NT")

    # Global bigrams
    global_bigrams = Counter()
    for i in range(len(words) - 1):
        if (words[i]["book"] == words[i+1]["book"] and
            words[i]["chapter"] == words[i+1]["chapter"] and
            words[i]["verse"] == words[i+1]["verse"]):
            global_bigrams[(words[i]["pos"], words[i+1]["pos"])] += 1

    summary = {
        "global_pos_distribution": dict(global_pos),
        "ot_pos_distribution": dict(ot_pos),
        "nt_pos_distribution": dict(nt_pos),
        "global_verb_noun_ratio": round(global_pos.get("verb", 0) / max(global_pos.get("noun", 0), 1), 4),
        "ot_verb_noun_ratio": round(ot_pos.get("verb", 0) / max(ot_pos.get("noun", 0), 1), 4),
        "nt_verb_noun_ratio": round(nt_pos.get("verb", 0) / max(nt_pos.get("noun", 0), 1), 4),
        "top30_bigrams_global": [[list(k), v] for k, v in global_bigrams.most_common(30)],
        "per_book": book_results,
    }

    out_file = OUT / "morphology_analysis.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log.info(f"Resultados en {out_file}")
    print(f"[morphology] DONE — {out_file}")


if __name__ == "__main__":
    main()
