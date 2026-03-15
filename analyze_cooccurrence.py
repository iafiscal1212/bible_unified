#!/usr/bin/env python3
"""
analyze_cooccurrence.py — Coocurrencia de lemas en ventanas de versículo.
Incluye: PMI (Pointwise Mutual Information), top pares coocurrentes.
"""
import json, logging, math
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

BASE = Path(__file__).parent
CORPUS = BASE / "bible_unified.json"
OUT = BASE / "results" / "cooccurrence"
OUT.mkdir(parents=True, exist_ok=True)
LOG = BASE / "logs"
LOG.mkdir(parents=True, exist_ok=True)

logging.basicConfig(filename=LOG / "cooccurrence.log", level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("coocc")

MIN_FREQ = 5  # Umbral mínimo de frecuencia de lema para considerar


def load_corpus():
    with open(CORPUS, encoding="utf-8") as f:
        return json.load(f)


def analyze_book(book_words):
    """Calcula coocurrencia de lemas dentro de cada versículo."""
    if not book_words:
        return None
    name = book_words[0]["book"]
    corpus = book_words[0]["corpus"]

    # Agrupar por versículo
    verses = defaultdict(list)
    for w in book_words:
        verses[(w["chapter"], w["verse"])].append(w["lemma"])

    # Frecuencias de lema en el libro
    lemma_freq = Counter()
    for lemmas in verses.values():
        lemma_freq.update(lemmas)

    # Coocurrencia: pares de lemas que aparecen en el mismo versículo
    pair_counts = Counter()
    lemma_verse_count = Counter()  # en cuántos versículos aparece cada lema

    for verse_lemmas in verses.values():
        unique = set(verse_lemmas)
        for lem in unique:
            lemma_verse_count[lem] += 1
        unique_list = sorted(unique)
        for i in range(len(unique_list)):
            for j in range(i + 1, len(unique_list)):
                pair_counts[(unique_list[i], unique_list[j])] += 1

    n_verses = len(verses)

    return {
        "book": name,
        "corpus": corpus,
        "n_verses": n_verses,
        "n_unique_lemmas": len(lemma_freq),
        "n_pairs_found": len(pair_counts),
        "top20_pairs": [
            [list(pair), count] for pair, count in pair_counts.most_common(20)
        ],
        # Pass raw data for global aggregation
        "_pair_counts": dict(pair_counts),
        "_lemma_verse_count": dict(lemma_verse_count),
        "_n_verses": n_verses,
    }


def compute_pmi(pair_counts, lemma_verse_count, n_verses, min_count=3, top_n=100):
    """Calcula PMI para pares de lemas.
    PMI(a,b) = log2(P(a,b) / (P(a) * P(b)))
    donde P(x) = nº versículos con x / total versículos
    """
    pmi_scores = []
    for (a, b), count in pair_counts.items():
        if count < min_count:
            continue
        pa = lemma_verse_count.get(a, 0) / n_verses
        pb = lemma_verse_count.get(b, 0) / n_verses
        pab = count / n_verses
        if pa > 0 and pb > 0 and pab > 0:
            pmi = math.log2(pab / (pa * pb))
            pmi_scores.append({
                "pair": [a, b],
                "pmi": round(pmi, 4),
                "co_count": count,
                "freq_a": lemma_verse_count[a],
                "freq_b": lemma_verse_count[b],
            })
    pmi_scores.sort(key=lambda x: -x["pmi"])
    return pmi_scores[:top_n]


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
    global_pairs = Counter()
    global_lemma_vc = Counter()
    global_n_verses = 0
    ot_pairs = Counter()
    ot_lemma_vc = Counter()
    ot_n_verses = 0
    nt_pairs = Counter()
    nt_lemma_vc = Counter()
    nt_n_verses = 0

    for r in book_results:
        pc = {tuple(k.split("|||")) if isinstance(k, str) else k: v
              for k, v in r["_pair_counts"].items()}
        # pair keys are tuples stored as strings in dict; need to handle
        for pair_key, count in r["_pair_counts"].items():
            if isinstance(pair_key, str):
                pair = tuple(pair_key.split("|||"))
            else:
                pair = pair_key
            global_pairs[pair] += count
            if r["corpus"] == "OT":
                ot_pairs[pair] += count
            else:
                nt_pairs[pair] += count

        for lem, vc in r["_lemma_verse_count"].items():
            global_lemma_vc[lem] += vc
            if r["corpus"] == "OT":
                ot_lemma_vc[lem] += vc
            else:
                nt_lemma_vc[lem] += vc

        global_n_verses += r["_n_verses"]
        if r["corpus"] == "OT":
            ot_n_verses += r["_n_verses"]
        else:
            nt_n_verses += r["_n_verses"]

    # PMI
    log.info("Calculando PMI global...")
    global_pmi = compute_pmi(global_pairs, global_lemma_vc, global_n_verses)
    ot_pmi = compute_pmi(ot_pairs, ot_lemma_vc, ot_n_verses)
    nt_pmi = compute_pmi(nt_pairs, nt_lemma_vc, nt_n_verses)

    # Clean book results (remove raw data)
    clean_results = []
    for r in book_results:
        cr = {k: v for k, v in r.items() if not k.startswith("_")}
        clean_results.append(cr)

    summary = {
        "global_n_verses": global_n_verses,
        "global_unique_pairs": len(global_pairs),
        "top50_cooccurring_pairs": [
            [list(pair), count] for pair, count in global_pairs.most_common(50)
        ],
        "global_top_pmi": global_pmi[:50],
        "ot_top_pmi": ot_pmi[:30],
        "nt_top_pmi": nt_pmi[:30],
        "per_book": clean_results,
    }

    out_file = OUT / "cooccurrence_analysis.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log.info(f"Resultados en {out_file}")
    print(f"[cooccurrence] DONE — {out_file}")


if __name__ == "__main__":
    main()
