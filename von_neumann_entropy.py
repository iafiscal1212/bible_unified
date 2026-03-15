#!/usr/bin/env python3
"""
von_neumann_entropy.py — Fase 4, Investigación 2
Entropía de Von Neumann vs Shannon por libro y género.
Todo numpy/scipy — cero frameworks cuánticos.
"""
import json, logging, math, time
from collections import Counter
from pathlib import Path
import numpy as np
from scipy import stats as sp_stats

BASE = Path(__file__).parent
CORPUS = BASE / "bible_unified.json"
OUT = BASE / "results" / "von_neumann"
OUT.mkdir(parents=True, exist_ok=True)
LOG = BASE / "logs"
LOG.mkdir(parents=True, exist_ok=True)

logging.basicConfig(filename=LOG / "von_neumann_entropy.log", level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("von_neumann_entropy")

POS_CATEGORIES = ["noun", "verb", "pronoun", "adjective", "adverb",
                   "preposition", "conjunction", "particle", "other"]
N_POS = len(POS_CATEGORIES)

GENRE_MAP = {
    "poetic": ["Psalms", "Proverbs", "Song of Songs", "Ecclesiastes", "Lamentations"],
    "legal": ["Leviticus", "Deuteronomy"],
    "epistolar": ["Romans", "1 Corinthians", "2 Corinthians", "Galatians", "Ephesians",
                  "Philippians", "Colossians", "1 Thessalonians", "2 Thessalonians",
                  "1 Timothy", "2 Timothy", "Titus", "Philemon", "Hebrews",
                  "James", "1 Peter", "2 Peter", "1 John", "2 John", "3 John", "Jude"],
}


def get_genre(book):
    for genre, books in GENRE_MAP.items():
        if book in books:
            return genre
    return "narrative"


def pos_to_index(pos):
    try:
        return POS_CATEGORIES.index(pos)
    except ValueError:
        return POS_CATEGORIES.index("other")


def build_density_matrix(word_list):
    """Build density matrix ρ from POS frequency vectors of each verse."""
    # Group by verse
    verses = {}
    for w in word_list:
        vk = (w["book_num"], w["chapter"], w["verse"])
        verses.setdefault(vk, []).append(w)

    n_verses = len(verses)
    if n_verses == 0:
        return np.zeros((N_POS, N_POS)), 0

    rho = np.zeros((N_POS, N_POS))
    for vk in sorted(verses.keys()):
        vw = verses[vk]
        # POS frequency vector
        vec = np.zeros(N_POS)
        for w in vw:
            idx = pos_to_index(w["pos"])
            vec[idx] += 1
        # Normalize to unit vector
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        # Outer product: |v><v|
        rho += np.outer(vec, vec)

    # Normalize: Tr(ρ) = 1
    tr = np.trace(rho)
    if tr > 0:
        rho = rho / tr

    return rho, n_verses


def von_neumann_entropy(rho):
    """S_vN = -Tr(ρ log₂ ρ) = -Σ λᵢ log₂ λᵢ"""
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = eigenvalues[eigenvalues > 1e-15]  # Remove near-zero
    return float(-np.sum(eigenvalues * np.log2(eigenvalues)))


def shannon_entropy_pos(word_list):
    """Shannon entropy of POS distribution (bits)."""
    pos_counts = Counter(w["pos"] for w in word_list)
    total = sum(pos_counts.values())
    if total == 0:
        return 0.0
    probs = np.array([pos_counts.get(p, 0) / total for p in POS_CATEGORIES])
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def main():
    log.info("=== VON NEUMANN ENTROPY — INICIO ===")
    t0 = time.time()

    with open(CORPUS, encoding="utf-8") as f:
        words = json.load(f)
    log.info(f"Corpus: {len(words)} palabras")

    # Group by book
    books = {}
    for w in words:
        books.setdefault(w["book"], []).append(w)

    # Load Hurst and Zipf data from previous phases
    hurst_data = {}
    zipf_data = {}
    vn_ratio_data = {}
    try:
        with open(BASE / "results" / "deep_fractal" / "fractal_by_genre.json") as f:
            fractal_genre = json.load(f)
    except Exception:
        fractal_genre = {}
    try:
        with open(BASE / "results" / "deep_zipf_semantic" / "zipf_by_at_book.json") as f:
            for entry in json.load(f):
                zipf_data[entry["book"]] = entry.get("zipf_s_lemma")
    except Exception:
        pass
    try:
        with open(BASE / "results" / "deep_vn_ratio" / "vn_by_book.json") as f:
            for entry in json.load(f):
                vn_ratio_data[entry["book"]] = entry.get("vn_ratio")
    except Exception:
        pass

    # === Per-book analysis ===
    log.info("Calculando entropías por libro...")
    book_results = []
    density_summaries = []

    for book_name in sorted(books.keys(), key=lambda b: books[b][0]["book_num"]):
        bw = books[book_name]
        corpus = bw[0]["corpus"]
        genre = get_genre(book_name)

        rho, n_v = build_density_matrix(bw)
        s_vn = von_neumann_entropy(rho)
        s_sh = shannon_entropy_pos(bw)
        delta_s = round(s_vn - s_sh, 6)

        eigenvalues = np.linalg.eigvalsh(rho)
        eigenvalues = np.sort(eigenvalues)[::-1]

        book_results.append({
            "book": book_name,
            "corpus": corpus,
            "genre": genre,
            "n_verses": n_v,
            "n_words": len(bw),
            "s_von_neumann": round(s_vn, 6),
            "s_shannon": round(s_sh, 6),
            "delta_s": delta_s,
            "top3_eigenvalues": [round(float(e), 6) for e in eigenvalues[:3]],
            "zipf_s": zipf_data.get(book_name),
            "vn_ratio": vn_ratio_data.get(book_name),
        })

        density_summaries.append({
            "book": book_name,
            "trace": round(float(np.trace(rho)), 8),
            "rank": int(np.sum(eigenvalues > 1e-10)),
            "purity": round(float(np.trace(rho @ rho)), 6),
        })

        log.info(f"  {book_name}: S_vN={s_vn:.4f}, S_Sh={s_sh:.4f}, ΔS={delta_s:.4f}")

    # === Corpus-level comparison ===
    log.info("Comparación AT vs NT...")
    ot_delta = [r["delta_s"] for r in book_results if r["corpus"] == "OT"]
    nt_delta = [r["delta_s"] for r in book_results if r["corpus"] == "NT"]

    comparison = {}
    if ot_delta and nt_delta:
        u_stat, u_pval = sp_stats.mannwhitneyu(ot_delta, nt_delta, alternative='two-sided')
        comparison["ot_vs_nt"] = {
            "ot_mean_delta_s": round(float(np.mean(ot_delta)), 6),
            "ot_std_delta_s": round(float(np.std(ot_delta)), 6),
            "nt_mean_delta_s": round(float(np.mean(nt_delta)), 6),
            "nt_std_delta_s": round(float(np.std(nt_delta)), 6),
            "mannwhitney_U": round(float(u_stat), 2),
            "mannwhitney_p": round(float(u_pval), 8),
            "significant": bool(u_pval < 0.05),
        }

    # === Genre comparison ===
    log.info("Comparación por género...")
    genre_groups = {}
    for r in book_results:
        genre_groups.setdefault(r["genre"], []).append(r["delta_s"])

    genre_stats = {}
    for g, vals in genre_groups.items():
        genre_stats[g] = {
            "n_books": len(vals),
            "mean_delta_s": round(float(np.mean(vals)), 6),
            "std_delta_s": round(float(np.std(vals)), 6),
        }
    comparison["by_genre"] = genre_stats

    # === Correlations ===
    log.info("Correlaciones ΔS con otras métricas...")
    correlations = {}
    delta_arr = np.array([r["delta_s"] for r in book_results])

    for metric_name in ["zipf_s", "vn_ratio"]:
        vals = []
        ds = []
        for r in book_results:
            v = r.get(metric_name)
            if v is not None:
                vals.append(v)
                ds.append(r["delta_s"])
        if len(vals) >= 5:
            r_val, p_val = sp_stats.pearsonr(ds, vals)
            correlations[f"delta_s_vs_{metric_name}"] = {
                "pearson_r": round(float(r_val), 4),
                "p_value": round(float(p_val), 6),
                "significant": bool(p_val < 0.05),
                "n": len(vals),
            }

    # === Save ===
    log.info("Guardando resultados...")
    with open(OUT / "density_matrices_summary.json", "w", encoding="utf-8") as f:
        json.dump(density_summaries, f, ensure_ascii=False, indent=2)
    with open(OUT / "entropy_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    with open(OUT / "delta_s_by_book.json", "w", encoding="utf-8") as f:
        json.dump(book_results, f, ensure_ascii=False, indent=2)
    with open(OUT / "correlations.json", "w", encoding="utf-8") as f:
        json.dump(correlations, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t0
    log.info(f"=== DONE en {elapsed:.1f}s ===")
    print(f"[von_neumann_entropy] DONE — {elapsed:.1f}s")


if __name__ == "__main__":
    main()
