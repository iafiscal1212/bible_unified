#!/usr/bin/env python3
"""
deep_vn_ratio.py — Investigación 4: ¿La transición V/N de 0.55 (AT) a 0.99 (NT)
es gradual o tiene quiebres abruptos?
- V/N por libro
- Change point detection (sliding window + Welch's t-test)
- Autocorrelación
- Correlaciones con otras métricas
"""
import json, logging, time, math
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from scipy import stats as sp_stats

BASE = Path(__file__).parent
CORPUS = BASE / "bible_unified.json"
OUT = BASE / "results" / "deep_vn_ratio"
OUT.mkdir(parents=True, exist_ok=True)
LOG = BASE / "logs"
LOG.mkdir(parents=True, exist_ok=True)

logging.basicConfig(filename=LOG / "deep_vn_ratio.log", level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("deep_vn")


def compute_autocorr(series, max_lag=10):
    """Autocorrelación de la serie."""
    x = np.array(series)
    n = len(x)
    if n < max_lag + 2:
        max_lag = n - 2
    mean = np.mean(x)
    var = np.var(x)
    if var == 0:
        return {}
    result = {}
    for lag in range(1, max_lag + 1):
        cov = np.mean((x[:-lag] - mean) * (x[lag:] - mean))
        result[lag] = round(float(cov / var), 6)
    return result


def detect_changepoints(series, min_segment=3):
    """Detección de quiebres usando sliding window + Welch's t-test."""
    n = len(series)
    results = []
    for i in range(min_segment, n - min_segment):
        left = series[:i]
        right = series[i:]
        if len(left) < 3 or len(right) < 3:
            continue
        t_stat, p_val = sp_stats.ttest_ind(left, right, equal_var=False)
        results.append({
            "position": i,
            "t_stat": round(float(t_stat), 4),
            "p_value": float(f"{p_val:.2e}"),
            "left_mean": round(float(np.mean(left)), 4),
            "right_mean": round(float(np.mean(right)), 4),
        })

    # Find most significant changepoints (local minima of p-value)
    if not results:
        return []

    # Sort by p-value and take top changepoints
    results.sort(key=lambda x: x["p_value"])
    significant = [r for r in results if r["p_value"] < 0.01]

    # Remove redundant (within 3 books of each other)
    final = []
    for r in significant:
        if not final or all(abs(r["position"] - f["position"]) >= 3 for f in final):
            final.append(r)
        if len(final) >= 5:
            break
    return final


def main():
    log.info("Cargando corpus...")
    t0 = time.time()
    with open(CORPUS, encoding="utf-8") as f:
        words = json.load(f)
    log.info(f"Corpus: {len(words)} palabras")

    # Agrupar por libro
    books = {}
    for w in words:
        books.setdefault(w["book"], []).append(w)

    # === 1. V/N ratio por libro ===
    log.info("Calculando V/N por libro...")
    vn_by_book = []
    vn_series = []
    book_order = []

    for book_name in sorted(books.keys(), key=lambda b: books[b][0]["book_num"]):
        bw = books[book_name]
        pos_counts = Counter(w["pos"] for w in bw)
        verbs = pos_counts.get("verb", 0)
        nouns = pos_counts.get("noun", 0)
        vn = verbs / nouns if nouns > 0 else 0

        # Hapax rate
        word_freq = Counter(w["text"] for w in bw)
        hapax_pct = sum(1 for c in word_freq.values() if c == 1) / len(word_freq) if word_freq else 0

        # Mean verse length
        verses = defaultdict(int)
        for w in bw:
            verses[(w["chapter"], w["verse"])] += 1
        mean_vl = np.mean(list(verses.values())) if verses else 0

        # Zipf exponent (quick fit via log-log regression)
        freqs = np.array(sorted(word_freq.values(), reverse=True), dtype=float)
        if len(freqs) >= 5:
            log_r = np.log(np.arange(1, len(freqs) + 1))
            log_f = np.log(freqs)
            slope, _, r_val, _, _ = sp_stats.linregress(log_r, log_f)
            zipf_s = round(float(-slope), 4)
        else:
            zipf_s = None

        entry = {
            "book": book_name,
            "book_num": bw[0]["book_num"],
            "corpus": bw[0]["corpus"],
            "vn_ratio": round(vn, 4),
            "verbs": verbs,
            "nouns": nouns,
            "n_words": len(bw),
            "hapax_pct": round(hapax_pct, 4),
            "mean_verse_length": round(float(mean_vl), 2),
            "zipf_s": zipf_s,
        }
        vn_by_book.append(entry)
        vn_series.append(vn)
        book_order.append(book_name)

    # === 2. Change point detection ===
    log.info("Detectando quiebres...")
    vn_arr = np.array(vn_series)
    changepoints = detect_changepoints(vn_arr)

    # Map positions to book names
    for cp in changepoints:
        pos = cp["position"]
        cp["book_before"] = book_order[pos - 1] if pos > 0 else None
        cp["book_after"] = book_order[pos] if pos < len(book_order) else None

    log.info(f"  Quiebres significativos: {len(changepoints)}")
    for cp in changepoints:
        log.info(f"    pos={cp['position']}: {cp['book_before']} → {cp['book_after']} "
                 f"(p={cp['p_value']})")

    # === 3. Autocorrelación ===
    log.info("Calculando autocorrelación...")
    autocorr = compute_autocorr(vn_series, max_lag=min(15, len(vn_series) - 2))

    # === 4. Correlaciones ===
    log.info("Calculando correlaciones...")
    mean_vls = [b["mean_verse_length"] for b in vn_by_book]
    hapax_pcts = [b["hapax_pct"] for b in vn_by_book]
    zipf_ss = [b["zipf_s"] for b in vn_by_book if b["zipf_s"] is not None]
    vn_for_zipf = [b["vn_ratio"] for b in vn_by_book if b["zipf_s"] is not None]

    correlations = {}

    # V/N vs mean verse length
    r, p = sp_stats.pearsonr(vn_series, mean_vls)
    correlations["vn_vs_mean_verse_length"] = {
        "pearson_r": round(float(r), 4),
        "p_value": float(f"{p:.2e}"),
        "significant": bool(p < 0.05),
    }

    # V/N vs hapax rate
    r, p = sp_stats.pearsonr(vn_series, hapax_pcts)
    correlations["vn_vs_hapax_rate"] = {
        "pearson_r": round(float(r), 4),
        "p_value": float(f"{p:.2e}"),
        "significant": bool(p < 0.05),
    }

    # V/N vs Zipf exponent
    if len(zipf_ss) >= 5:
        r, p = sp_stats.pearsonr(vn_for_zipf, zipf_ss)
        correlations["vn_vs_zipf_s"] = {
            "pearson_r": round(float(r), 4),
            "p_value": float(f"{p:.2e}"),
            "significant": bool(p < 0.05),
        }

    # V/N vs book size
    book_sizes = [b["n_words"] for b in vn_by_book]
    r, p = sp_stats.pearsonr(vn_series, book_sizes)
    correlations["vn_vs_book_size"] = {
        "pearson_r": round(float(r), 4),
        "p_value": float(f"{p:.2e}"),
        "significant": bool(p < 0.05),
    }

    # Save
    with open(OUT / "vn_by_book.json", "w") as f:
        json.dump(vn_by_book, f, indent=2, ensure_ascii=False)
    with open(OUT / "changepoints.json", "w") as f:
        json.dump({"changepoints": changepoints, "n_books": len(book_order),
                    "book_order": book_order}, f, indent=2, ensure_ascii=False)
    with open(OUT / "autocorrelation.json", "w") as f:
        json.dump(autocorr, f, indent=2)
    with open(OUT / "correlations.json", "w") as f:
        json.dump(correlations, f, indent=2)

    elapsed = time.time() - t0
    log.info(f"DONE en {elapsed:.1f}s")
    print(f"[deep_vn_ratio] DONE — {elapsed:.1f}s")


if __name__ == "__main__":
    main()
