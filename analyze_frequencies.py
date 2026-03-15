#!/usr/bin/env python3
"""
analyze_frequencies.py — Frecuencias absolutas/relativas de palabras y lemas.
Incluye: Zipf fit, hapax legomena, TTR por libro, comparación OT/NT.
"""
import json, logging, os, sys, math
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

BASE = Path(__file__).parent
CORPUS = BASE / "bible_unified.json"
OUT = BASE / "results" / "frequencies"
OUT.mkdir(parents=True, exist_ok=True)
LOG = BASE / "logs"
LOG.mkdir(parents=True, exist_ok=True)

logging.basicConfig(filename=LOG / "frequencies.log", level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("freq")


def load_corpus():
    with open(CORPUS, encoding="utf-8") as f:
        return json.load(f)


def analyze_book(book_words):
    """Analiza un libro: retorna dict con métricas."""
    if not book_words:
        return None
    name = book_words[0]["book"]
    corpus = book_words[0]["corpus"]
    texts = [w["text"] for w in book_words]
    lemmas = [w["lemma"] for w in book_words]

    word_freq = Counter(texts)
    lemma_freq = Counter(lemmas)
    n_tokens = len(texts)
    n_types_word = len(word_freq)
    n_types_lemma = len(lemma_freq)
    hapax_words = sum(1 for c in word_freq.values() if c == 1)
    hapax_lemmas = sum(1 for c in lemma_freq.values() if c == 1)
    ttr_word = n_types_word / n_tokens if n_tokens else 0
    ttr_lemma = n_types_lemma / n_tokens if n_tokens else 0

    return {
        "book": name,
        "corpus": corpus,
        "n_tokens": n_tokens,
        "n_types_word": n_types_word,
        "n_types_lemma": n_types_lemma,
        "hapax_words": hapax_words,
        "hapax_lemmas": hapax_lemmas,
        "ttr_word": round(ttr_word, 6),
        "ttr_lemma": round(ttr_lemma, 6),
        "top20_words": word_freq.most_common(20),
        "top20_lemmas": lemma_freq.most_common(20),
    }


def zipf_fit(freq_counter):
    """Fit Zipf's law: log(freq) = a - s*log(rank). Returns (s, R²)."""
    freqs = sorted(freq_counter.values(), reverse=True)
    n = len(freqs)
    if n < 3:
        return None, None
    log_ranks = [math.log(i + 1) for i in range(n)]
    log_freqs = [math.log(f) for f in freqs]
    # Simple linear regression
    mean_x = sum(log_ranks) / n
    mean_y = sum(log_freqs) / n
    ss_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(log_ranks, log_freqs))
    ss_xx = sum((x - mean_x) ** 2 for x in log_ranks)
    ss_yy = sum((y - mean_y) ** 2 for y in log_freqs)
    if ss_xx == 0 or ss_yy == 0:
        return None, None
    slope = ss_xy / ss_xx
    r_sq = (ss_xy ** 2) / (ss_xx * ss_yy)
    return round(-slope, 4), round(r_sq, 6)


def main():
    log.info("Cargando corpus...")
    words = load_corpus()
    log.info(f"Corpus cargado: {len(words)} palabras")

    # Agrupar por libro
    books = {}
    for w in words:
        books.setdefault(w["book"], []).append(w)

    # Paralelo por libro
    log.info(f"Analizando {len(books)} libros en paralelo...")
    with Pool() as pool:
        book_results = pool.map(analyze_book, list(books.values()))
    book_results = [r for r in book_results if r]

    # Global frequencies
    all_texts = [w["text"] for w in words]
    all_lemmas = [w["lemma"] for w in words]
    global_word_freq = Counter(all_texts)
    global_lemma_freq = Counter(all_lemmas)

    # OT vs NT
    ot_words = [w["text"] for w in words if w["corpus"] == "OT"]
    nt_words = [w["text"] for w in words if w["corpus"] == "NT"]
    ot_lemmas = [w["lemma"] for w in words if w["corpus"] == "OT"]
    nt_lemmas = [w["lemma"] for w in words if w["corpus"] == "NT"]

    # Zipf fits
    zipf_global_s, zipf_global_r2 = zipf_fit(global_word_freq)
    zipf_ot_s, zipf_ot_r2 = zipf_fit(Counter(ot_words))
    zipf_nt_s, zipf_nt_r2 = zipf_fit(Counter(nt_words))

    summary = {
        "total_tokens": len(words),
        "total_word_types": len(global_word_freq),
        "total_lemma_types": len(global_lemma_freq),
        "global_hapax_words": sum(1 for c in global_word_freq.values() if c == 1),
        "global_hapax_lemmas": sum(1 for c in global_lemma_freq.values() if c == 1),
        "ot_tokens": len(ot_words),
        "nt_tokens": len(nt_words),
        "ot_word_types": len(set(ot_words)),
        "nt_word_types": len(set(nt_words)),
        "ot_lemma_types": len(set(ot_lemmas)),
        "nt_lemma_types": len(set(nt_lemmas)),
        "zipf": {
            "global": {"exponent": zipf_global_s, "r_squared": zipf_global_r2},
            "OT": {"exponent": zipf_ot_s, "r_squared": zipf_ot_r2},
            "NT": {"exponent": zipf_nt_s, "r_squared": zipf_nt_r2},
        },
        "top50_words_global": global_word_freq.most_common(50),
        "top50_lemmas_global": global_lemma_freq.most_common(50),
        "per_book": book_results,
    }

    out_file = OUT / "frequency_analysis.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log.info(f"Resultados guardados en {out_file}")
    print(f"[frequencies] DONE — {out_file}")


if __name__ == "__main__":
    main()
