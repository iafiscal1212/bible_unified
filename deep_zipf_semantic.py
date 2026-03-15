#!/usr/bin/env python3
"""
deep_zipf_semantic.py — Fase 3, Investigación 4
El AT tiene Zipf anómalo en lemas (s=0.715).
¿Qué libros concentran la anomalía? ¿Verbos o nombres? ¿Cabeza o cola?
"""
import json, logging, math, time
from collections import Counter
from pathlib import Path
import numpy as np
from scipy import stats as sp_stats
from scipy.optimize import curve_fit

BASE = Path(__file__).parent
CORPUS = BASE / "bible_unified.json"
OUT = BASE / "results" / "deep_zipf_semantic"
OUT.mkdir(parents=True, exist_ok=True)
LOG = BASE / "logs"
LOG.mkdir(parents=True, exist_ok=True)

logging.basicConfig(filename=LOG / "deep_zipf_semantic.log", level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("deep_zipf_semantic")


def zipf_func(r, C, s):
    return C * np.power(r, -s)


def fit_zipf(freq_counter):
    """Fit Zipf via curve_fit. Returns (s, R², n_types)."""
    freqs = np.array(sorted(freq_counter.values(), reverse=True), dtype=float)
    n = len(freqs)
    if n < 5:
        return None, None, n
    ranks = np.arange(1, n + 1, dtype=float)
    try:
        popt, _ = curve_fit(zipf_func, ranks, freqs, p0=[freqs[0], 1.0], maxfev=5000)
        C_fit, s_fit = popt
        predicted = zipf_func(ranks, C_fit, s_fit)
        ss_res = np.sum((freqs - predicted) ** 2)
        ss_tot = np.sum((freqs - np.mean(freqs)) ** 2)
        r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        return round(float(s_fit), 6), round(float(r_sq), 6), n
    except Exception:
        log_r = np.log(ranks)
        log_f = np.log(freqs)
        slope, intercept, r_val, _, _ = sp_stats.linregress(log_r, log_f)
        return round(float(-slope), 6), round(float(r_val**2), 6), n


def shannon_entropy(freq_counter):
    """Shannon entropy (bits) of a frequency distribution."""
    total = sum(freq_counter.values())
    if total == 0:
        return 0.0
    probs = np.array(list(freq_counter.values()), dtype=float) / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def main():
    log.info("=== DEEP ZIPF SEMANTIC — INICIO ===")
    t0 = time.time()

    with open(CORPUS, encoding="utf-8") as f:
        words = json.load(f)
    log.info(f"Corpus: {len(words)} palabras")

    # Only AT words
    ot_words = [w for w in words if w["corpus"] == "OT"]
    log.info(f"AT: {len(ot_words)} palabras")

    # Group by book
    books = {}
    for w in ot_words:
        books.setdefault(w["book"], []).append(w)

    # === 1. Zipf en lemas para cada libro del AT ===
    log.info("Zipf en lemas para cada libro del AT...")
    zipf_by_book = []
    for book_name in sorted(books.keys(), key=lambda b: books[b][0]["book_num"]):
        bw = books[book_name]
        lemma_freq = Counter(w["lemma"] for w in bw)
        s, r2, n_types = fit_zipf(lemma_freq)
        entropy = shannon_entropy(lemma_freq)

        zipf_by_book.append({
            "book": book_name,
            "book_num": bw[0]["book_num"],
            "n_tokens": len(bw),
            "n_lemma_types": n_types,
            "zipf_s_lemma": s,
            "zipf_r2": r2,
            "entropy_bits": round(entropy, 4),
        })
        log.info(f"  {book_name}: s={s}, H={entropy:.4f}")

    # Sort by s (most anomalous first)
    zipf_by_book.sort(key=lambda x: x["zipf_s_lemma"] if x["zipf_s_lemma"] is not None else 999)

    # === 2. Top 5 most anomalous: head vs tail analysis ===
    log.info("Top 5 más anómalos: análisis cabeza vs cola...")
    top5 = zipf_by_book[:5]
    top5_analysis = []

    for entry in top5:
        book_name = entry["book"]
        bw = books[book_name]
        lemma_freq = Counter(w["lemma"] for w in bw)
        total_tokens = sum(lemma_freq.values())

        # Sort by frequency
        sorted_lemmas = lemma_freq.most_common()

        # Head: top 50 lemmas
        head_50 = sorted_lemmas[:50]
        head_tokens = sum(f for _, f in head_50)
        head_pct = round(head_tokens / total_tokens * 100, 2)

        # Tail: hapax legomena (freq=1)
        hapax = [l for l, f in sorted_lemmas if f == 1]
        hapax_pct = round(len(hapax) / len(sorted_lemmas) * 100, 2)

        # Mid zone: rank 51 to median rank
        mid = sorted_lemmas[50:len(sorted_lemmas)//2]
        mid_tokens = sum(f for _, f in mid)
        mid_pct = round(mid_tokens / total_tokens * 100, 2)

        # Where is the anomaly? Compare to Zipf(s=1) prediction
        n = len(sorted_lemmas)
        ranks = np.arange(1, n + 1, dtype=float)
        actual_freqs = np.array([f for _, f in sorted_lemmas], dtype=float)
        zipf_predicted = actual_freqs[0] * np.power(ranks, -1.0)  # s=1.0

        # Excess ratio: actual/predicted (>1 means more than Zipf predicts)
        excess = actual_freqs / np.maximum(zipf_predicted, 1e-10)

        # Where is excess maximum?
        # Head (rank 1-10), middle (rank 11-100), tail (rank 100+)
        head_excess = float(np.mean(excess[:10]))
        mid_excess = float(np.mean(excess[10:min(100, n)]))
        tail_excess = float(np.mean(excess[min(100, n):]))

        anomaly_location = "head" if head_excess > mid_excess and head_excess > tail_excess else \
                          "middle" if mid_excess > tail_excess else "tail"

        top5_analysis.append({
            "book": book_name,
            "zipf_s": entry["zipf_s_lemma"],
            "n_lemmas": n,
            "head_50_pct_tokens": head_pct,
            "hapax_pct_types": hapax_pct,
            "mid_zone_pct_tokens": mid_pct,
            "excess_over_zipf1": {
                "head_1_10": round(head_excess, 4),
                "mid_11_100": round(mid_excess, 4),
                "tail_100_plus": round(tail_excess, 4),
            },
            "anomaly_location": anomaly_location,
        })
        log.info(f"  {book_name}: anomaly in {anomaly_location}, "
                 f"excess head={head_excess:.2f} mid={mid_excess:.2f} tail={tail_excess:.2f}")

    # === 3. Entropy vs Zipf correlation ===
    log.info("Correlación entropía vs Zipf...")
    s_values = [x["zipf_s_lemma"] for x in zipf_by_book if x["zipf_s_lemma"] is not None]
    h_values = [x["entropy_bits"] for x in zipf_by_book if x["zipf_s_lemma"] is not None]

    if len(s_values) >= 5:
        pearson_r, pearson_p = sp_stats.pearsonr(s_values, h_values)
        spearman_r, spearman_p = sp_stats.spearmanr(s_values, h_values)
    else:
        pearson_r, pearson_p = None, None
        spearman_r, spearman_p = None, None

    entropy_correlation = {
        "n_books": len(s_values),
        "pearson": {"r": round(float(pearson_r), 4) if pearson_r is not None else None,
                    "p": round(float(pearson_p), 6) if pearson_p is not None else None},
        "spearman": {"r": round(float(spearman_r), 4) if spearman_r is not None else None,
                     "p": round(float(spearman_p), 6) if spearman_p is not None else None},
        "interpretation": (
            "Low s (anomalous Zipf) correlates with HIGH entropy → more uniform distribution"
            if pearson_r is not None and pearson_r < -0.2 else
            "Low s correlates with LOW entropy → more concentrated distribution"
            if pearson_r is not None and pearson_r > 0.2 else
            "No significant correlation between Zipf exponent and entropy"
        ),
    }
    log.info(f"  Pearson r={pearson_r}, p={pearson_p}")

    # === 4. Zipf by POS (verbs vs nouns) ===
    log.info("Zipf por POS (verbos vs nombres) en cada libro...")
    zipf_by_pos = []

    for book_name in sorted(books.keys(), key=lambda b: books[b][0]["book_num"]):
        bw = books[book_name]

        verb_lemmas = Counter(w["lemma"] for w in bw if w["pos"] == "verb")
        noun_lemmas = Counter(w["lemma"] for w in bw if w["pos"] == "noun")

        s_verb, r2_verb, n_verb = fit_zipf(verb_lemmas)
        s_noun, r2_noun, n_noun = fit_zipf(noun_lemmas)

        zipf_by_pos.append({
            "book": book_name,
            "book_num": bw[0]["book_num"],
            "verb": {"s": s_verb, "r2": r2_verb, "n_types": n_verb, "n_tokens": sum(verb_lemmas.values())},
            "noun": {"s": s_noun, "r2": r2_noun, "n_types": n_noun, "n_tokens": sum(noun_lemmas.values())},
            "delta_verb_noun": round(s_verb - s_noun, 6) if s_verb and s_noun else None,
        })

    # Which POS drives the anomaly?
    verb_s_list = [x["verb"]["s"] for x in zipf_by_pos if x["verb"]["s"] is not None]
    noun_s_list = [x["noun"]["s"] for x in zipf_by_pos if x["noun"]["s"] is not None]

    if verb_s_list and noun_s_list:
        verb_mean_s = float(np.mean(verb_s_list))
        noun_mean_s = float(np.mean(noun_s_list))
        u_stat, u_pval = sp_stats.mannwhitneyu(verb_s_list, noun_s_list, alternative='two-sided')
    else:
        verb_mean_s, noun_mean_s = None, None
        u_stat, u_pval = None, None

    pos_summary = {
        "verb_mean_s": round(verb_mean_s, 4) if verb_mean_s else None,
        "noun_mean_s": round(noun_mean_s, 4) if noun_mean_s else None,
        "more_anomalous": "verbs" if verb_mean_s and noun_mean_s and verb_mean_s < noun_mean_s else "nouns",
        "mannwhitney_U": round(float(u_stat), 2) if u_stat else None,
        "mannwhitney_p": round(float(u_pval), 6) if u_pval else None,
    }

    # === 5. Anomaly characteristics: correlate with Fase 1/2 data ===
    log.info("Características de los libros anómalos...")
    # Load Fase 2 V/N data if available
    vn_data_path = BASE / "results" / "deep_vn_ratio" / "vn_by_book.json"
    vn_by_book = {}
    if vn_data_path.exists():
        with open(vn_data_path) as f:
            vn_raw = json.load(f)
            for entry in vn_raw:
                vn_by_book[entry["book"]] = entry.get("vn_ratio")

    # Build characteristics table
    anomaly_chars = []
    for entry in zipf_by_book:
        book_name = entry["book"]
        bw = books[book_name]
        n_tokens = len(bw)

        # POS distribution
        pos_dist = Counter(w["pos"] for w in bw)
        total_pos = sum(pos_dist.values())
        verb_pct = round(pos_dist.get("verb", 0) / total_pos * 100, 2) if total_pos > 0 else 0
        noun_pct = round(pos_dist.get("noun", 0) / total_pos * 100, 2) if total_pos > 0 else 0

        # Mean verse length
        verses = {}
        for w in bw:
            vk = (w["chapter"], w["verse"])
            verses[vk] = verses.get(vk, 0) + 1
        mean_verse_len = round(float(np.mean(list(verses.values()))), 2) if verses else None

        anomaly_chars.append({
            "book": book_name,
            "zipf_s_lemma": entry["zipf_s_lemma"],
            "entropy": entry["entropy_bits"],
            "n_tokens": n_tokens,
            "verb_pct": verb_pct,
            "noun_pct": noun_pct,
            "vn_ratio": vn_by_book.get(book_name),
            "mean_verse_length": mean_verse_len,
        })

    # Correlations between zipf_s and characteristics
    valid = [x for x in anomaly_chars if x["zipf_s_lemma"] is not None]
    s_arr = np.array([x["zipf_s_lemma"] for x in valid])

    correlations = {}
    for metric in ["verb_pct", "noun_pct", "n_tokens", "mean_verse_length"]:
        vals = np.array([x[metric] for x in valid if x[metric] is not None])
        s_sub = np.array([x["zipf_s_lemma"] for x in valid if x[metric] is not None])
        if len(vals) >= 5:
            r, p = sp_stats.pearsonr(s_sub, vals)
            correlations[f"zipf_s_vs_{metric}"] = {
                "pearson_r": round(float(r), 4),
                "p_value": round(float(p), 6),
                "significant": float(p) < 0.05,
            }

    # === Save all results ===
    log.info("Guardando resultados...")
    with open(OUT / "zipf_by_at_book.json", "w", encoding="utf-8") as f:
        json.dump(zipf_by_book, f, ensure_ascii=False, indent=2)
    with open(OUT / "top5_anomalous.json", "w", encoding="utf-8") as f:
        json.dump(top5_analysis, f, ensure_ascii=False, indent=2)
    with open(OUT / "entropy_correlation.json", "w", encoding="utf-8") as f:
        json.dump(entropy_correlation, f, ensure_ascii=False, indent=2)
    with open(OUT / "zipf_by_pos.json", "w", encoding="utf-8") as f:
        json.dump({"by_book": zipf_by_pos, "summary": pos_summary}, f, ensure_ascii=False, indent=2)
    with open(OUT / "anomaly_characteristics.json", "w", encoding="utf-8") as f:
        json.dump({"by_book": anomaly_chars, "correlations": correlations},
                  f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t0
    log.info(f"=== DONE en {elapsed:.1f}s ===")
    print(f"[deep_zipf_semantic] DONE — {elapsed:.1f}s")


if __name__ == "__main__":
    main()
