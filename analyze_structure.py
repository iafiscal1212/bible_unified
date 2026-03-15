#!/usr/bin/env python3
"""
analyze_structure.py — Análisis estructural: longitud de versículos, capítulos, libros.
Incluye: distribuciones, momentos estadísticos, proporciones, tests de normalidad.
"""
import json, logging, math
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

BASE = Path(__file__).parent
CORPUS = BASE / "bible_unified.json"
OUT = BASE / "results" / "structure"
OUT.mkdir(parents=True, exist_ok=True)
LOG = BASE / "logs"
LOG.mkdir(parents=True, exist_ok=True)

logging.basicConfig(filename=LOG / "structure.log", level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("struct")


def load_corpus():
    with open(CORPUS, encoding="utf-8") as f:
        return json.load(f)


def compute_moments(vals):
    """Calcula mean, std, skewness, kurtosis."""
    n = len(vals)
    if n < 3:
        return {"mean": 0, "std": 0, "skewness": 0, "kurtosis": 0, "n": n,
                "min": min(vals) if vals else 0, "max": max(vals) if vals else 0}
    mean = sum(vals) / n
    m2 = sum((v - mean) ** 2 for v in vals) / n
    m3 = sum((v - mean) ** 3 for v in vals) / n
    m4 = sum((v - mean) ** 4 for v in vals) / n
    std = math.sqrt(m2) if m2 > 0 else 0
    skew = m3 / (std ** 3) if std > 0 else 0
    kurt = (m4 / (std ** 4)) - 3 if std > 0 else 0  # excess kurtosis
    vals_s = sorted(vals)
    return {
        "mean": round(mean, 4),
        "std": round(std, 4),
        "skewness": round(skew, 4),
        "kurtosis": round(kurt, 4),
        "n": n,
        "min": vals_s[0],
        "max": vals_s[-1],
        "median": vals_s[n // 2],
        "p10": vals_s[n // 10] if n >= 10 else vals_s[0],
        "p90": vals_s[9 * n // 10] if n >= 10 else vals_s[-1],
    }


def analyze_book(book_words):
    if not book_words:
        return None
    name = book_words[0]["book"]
    corpus = book_words[0]["corpus"]

    # Words per verse
    verse_lens = defaultdict(int)
    for w in book_words:
        verse_lens[(w["chapter"], w["verse"])] += 1

    # Words per chapter
    chapter_lens = defaultdict(int)
    for w in book_words:
        chapter_lens[w["chapter"]] += 1

    # Verses per chapter
    verses_per_ch = defaultdict(set)
    for w in book_words:
        verses_per_ch[w["chapter"]].add(w["verse"])
    vpc = {ch: len(vs) for ch, vs in verses_per_ch.items()}

    vl = list(verse_lens.values())
    cl = list(chapter_lens.values())
    vpcl = list(vpc.values())

    return {
        "book": name,
        "corpus": corpus,
        "total_words": len(book_words),
        "n_chapters": len(chapter_lens),
        "n_verses": len(verse_lens),
        "words_per_verse": compute_moments(vl),
        "words_per_chapter": compute_moments(cl),
        "verses_per_chapter": compute_moments(vpcl),
        "verse_length_distribution": dict(sorted(
            defaultdict(int, {v: 0 for v in range(max(vl) + 1)}).items()
        )) if vl else {},
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

    # Verse length distribution histogram (count occurrences of each length)
    all_verse_lens = []
    for bw in books_data.values():
        vl = defaultdict(int)
        for w in bw:
            vl[(w["book"], w["chapter"], w["verse"])] += 1
        all_verse_lens.extend(vl.values())

    vl_hist = defaultdict(int)
    for v in all_verse_lens:
        vl_hist[v] += 1

    # Global structure
    global_verse_moments = compute_moments(all_verse_lens)

    # Words per book
    book_sizes = [(r["book"], r["total_words"]) for r in book_results]
    book_word_counts = [r["total_words"] for r in book_results]

    # OT vs NT proportions
    ot_words = sum(r["total_words"] for r in book_results if r["corpus"] == "OT")
    nt_words = sum(r["total_words"] for r in book_results if r["corpus"] == "NT")
    ot_verses = sum(r["n_verses"] for r in book_results if r["corpus"] == "OT")
    nt_verses = sum(r["n_verses"] for r in book_results if r["corpus"] == "NT")
    ot_chapters = sum(r["n_chapters"] for r in book_results if r["corpus"] == "OT")
    nt_chapters = sum(r["n_chapters"] for r in book_results if r["corpus"] == "NT")

    summary = {
        "global_verse_length": global_verse_moments,
        "global_book_size": compute_moments(book_word_counts),
        "verse_length_histogram": dict(sorted(vl_hist.items())),
        "proportions": {
            "ot_words": ot_words,
            "nt_words": nt_words,
            "ot_nt_word_ratio": round(ot_words / nt_words, 6) if nt_words else None,
            "ot_verses": ot_verses,
            "nt_verses": nt_verses,
            "ot_nt_verse_ratio": round(ot_verses / nt_verses, 6) if nt_verses else None,
            "ot_chapters": ot_chapters,
            "nt_chapters": nt_chapters,
        },
        "book_sizes": book_sizes,
        "per_book": book_results,
    }

    out_file = OUT / "structure_analysis.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log.info(f"Resultados en {out_file}")
    print(f"[structure] DONE — {out_file}")


if __name__ == "__main__":
    main()
