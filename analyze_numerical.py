#!/usr/bin/env python3
"""
analyze_numerical.py — Valores numéricos: gematría hebrea + isopsefia griega.
Calcula valores por palabra, versículo, capítulo y libro.
"""
import json, logging, math
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

BASE = Path(__file__).parent
CORPUS = BASE / "bible_unified.json"
OUT = BASE / "results" / "numerical"
OUT.mkdir(parents=True, exist_ok=True)
LOG = BASE / "logs"
LOG.mkdir(parents=True, exist_ok=True)

logging.basicConfig(filename=LOG / "numerical.log", level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("num")

# ── Gematría hebrea estándar ────────────────────────────────────────────
HEBREW_GEMATRIA = {
    'א': 1, 'ב': 2, 'ג': 3, 'ד': 4, 'ה': 5, 'ו': 6, 'ז': 7, 'ח': 8, 'ט': 9,
    'י': 10, 'כ': 20, 'ל': 30, 'מ': 40, 'נ': 50, 'ס': 60, 'ע': 70, 'פ': 80, 'צ': 90,
    'ק': 100, 'ר': 200, 'ש': 300, 'ת': 400,
    # Formas finales (sofit) — mismo valor
    'ך': 20, 'ם': 40, 'ן': 50, 'ף': 80, 'ץ': 90,
}

# ── Isopsefia griega ────────────────────────────────────────────────────
GREEK_ISOPSEPHY = {
    'α': 1, 'β': 2, 'γ': 3, 'δ': 4, 'ε': 5, 'ϛ': 6, 'ζ': 7, 'η': 8, 'θ': 9,
    'ι': 10, 'κ': 20, 'λ': 30, 'μ': 40, 'ν': 50, 'ξ': 60, 'ο': 70, 'π': 80, 'ϟ': 90,
    'ρ': 100, 'σ': 200, 'τ': 300, 'υ': 400, 'φ': 500, 'χ': 600, 'ψ': 700, 'ω': 800,
    'ς': 200,  # sigma final = sigma
    # Variantes con diacríticos: extraer letra base
}


def word_value(text, lang):
    """Calcula el valor numérico de una palabra."""
    val = 0
    if lang == "heb":
        for ch in text:
            val += HEBREW_GEMATRIA.get(ch, 0)
    elif lang == "grc":
        for ch in text.lower():
            val += GREEK_ISOPSEPHY.get(ch, 0)
    return val


def strip_accents_greek(text):
    """Intenta mapear caracteres con diacríticos a base para isopsefia."""
    import unicodedata
    result = []
    for ch in text:
        # NFD decomposition separates base char from combining marks
        decomp = unicodedata.normalize('NFD', ch)
        base = decomp[0] if decomp else ch
        result.append(base.lower())
    return ''.join(result)


def word_value_robust(text, lang):
    """Versión robusta que maneja diacríticos."""
    if lang == "heb":
        return sum(HEBREW_GEMATRIA.get(ch, 0) for ch in text)
    elif lang == "grc":
        clean = strip_accents_greek(text)
        return sum(GREEK_ISOPSEPHY.get(ch, 0) for ch in clean)
    return 0


def analyze_book(book_words):
    if not book_words:
        return None
    name = book_words[0]["book"]
    corpus = book_words[0]["corpus"]
    lang = book_words[0]["lang"]

    word_vals = []
    verse_sums = defaultdict(int)
    chapter_sums = defaultdict(int)
    book_total = 0

    for w in book_words:
        v = word_value_robust(w["text"], lang)
        word_vals.append(v)
        verse_key = (w["chapter"], w["verse"])
        verse_sums[verse_key] += v
        chapter_sums[w["chapter"]] += v
        book_total += v

    n = len(word_vals)
    nonzero_vals = [v for v in word_vals if v > 0]
    verse_vals = list(verse_sums.values())
    chapter_vals = list(chapter_sums.values())

    def stats(vals):
        if not vals:
            return {"mean": 0, "median": 0, "std": 0, "min": 0, "max": 0, "n": 0}
        vals_s = sorted(vals)
        n = len(vals_s)
        mean = sum(vals_s) / n
        median = vals_s[n // 2] if n % 2 else (vals_s[n // 2 - 1] + vals_s[n // 2]) / 2
        var = sum((v - mean) ** 2 for v in vals_s) / n if n > 1 else 0
        return {
            "mean": round(mean, 2),
            "median": median,
            "std": round(math.sqrt(var), 2),
            "min": vals_s[0],
            "max": vals_s[-1],
            "n": n,
        }

    # Distribución de valores de palabras (bins)
    val_dist = Counter(word_vals)

    return {
        "book": name,
        "corpus": corpus,
        "lang": lang,
        "book_total_value": book_total,
        "word_value_stats": stats(nonzero_vals),
        "verse_sum_stats": stats(verse_vals),
        "chapter_sum_stats": stats(chapter_vals),
        "n_words_zero_value": sum(1 for v in word_vals if v == 0),
        "n_verses": len(verse_sums),
        "n_chapters": len(chapter_sums),
        "top10_verse_sums": sorted(
            [(f"{k[0]}:{k[1]}", v) for k, v in verse_sums.items()],
            key=lambda x: -x[1]
        )[:10],
    }


def load_corpus():
    with open(CORPUS, encoding="utf-8") as f:
        return json.load(f)


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

    # Global stats
    ot_total = sum(r["book_total_value"] for r in book_results if r["corpus"] == "OT")
    nt_total = sum(r["book_total_value"] for r in book_results if r["corpus"] == "NT")

    # Verse value distribution globally
    all_verse_sums = []
    for w_group in books.values():
        verse_sums = defaultdict(int)
        lang = w_group[0]["lang"]
        for w in w_group:
            v = word_value_robust(w["text"], lang)
            verse_sums[(w["book"], w["chapter"], w["verse"])] += v
        all_verse_sums.extend(verse_sums.values())

    vs_sorted = sorted(all_verse_sums)
    n_vs = len(vs_sorted)

    summary = {
        "ot_total_value": ot_total,
        "nt_total_value": nt_total,
        "grand_total": ot_total + nt_total,
        "ot_nt_ratio": round(ot_total / nt_total, 6) if nt_total else None,
        "verse_value_global_stats": {
            "n": n_vs,
            "mean": round(sum(vs_sorted) / n_vs, 2) if n_vs else 0,
            "median": vs_sorted[n_vs // 2] if n_vs else 0,
            "min": vs_sorted[0] if n_vs else 0,
            "max": vs_sorted[-1] if n_vs else 0,
            "p25": vs_sorted[n_vs // 4] if n_vs else 0,
            "p75": vs_sorted[3 * n_vs // 4] if n_vs else 0,
        },
        "per_book": book_results,
    }

    out_file = OUT / "numerical_analysis.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log.info(f"Resultados en {out_file}")
    print(f"[numerical] DONE — {out_file}")


if __name__ == "__main__":
    main()
