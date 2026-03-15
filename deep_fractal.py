#!/usr/bin/env python3
"""
deep_fractal.py — Fase 3, Investigación 3
Dimensión fractal real del corpus: box-counting, Hurst exponent, DFA.
¿Consistente entre AT y NT? ¿Entre géneros?
"""
import json, logging, math, time
from collections import Counter
from pathlib import Path
import numpy as np
from scipy import stats as sp_stats

BASE = Path(__file__).parent
CORPUS = BASE / "bible_unified.json"
OUT = BASE / "results" / "deep_fractal"
OUT.mkdir(parents=True, exist_ok=True)
LOG = BASE / "logs"
LOG.mkdir(parents=True, exist_ok=True)

logging.basicConfig(filename=LOG / "deep_fractal.log", level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("deep_fractal")

# Genre classification from Fase 2
GENRE_MAP = {
    "poetic": ["Psalms", "Proverbs", "Song of Solomon", "Ecclesiastes", "Lamentations"],
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


def build_verse_length_series(word_list):
    """Build ordered series of verse lengths (words per verse)."""
    verses = {}
    for w in word_list:
        key = (w["book_num"], w["chapter"], w["verse"])
        verses[key] = verses.get(key, 0) + 1
    # Sort by canonical order
    ordered = [v for _, v in sorted(verses.items())]
    return np.array(ordered, dtype=float)


def box_counting_dimension(series, scales=None):
    """
    Box-counting dimension of a 1D signal.
    The signal is binarized: cells occupied by the signal trajectory.
    """
    if len(series) < 10:
        return None, None, None

    n = len(series)
    if scales is None:
        scales = [2**i for i in range(0, int(math.log2(n))) if 2**i < n]
        if not scales:
            scales = [1, 2, 4, 8]

    # Normalize series to [0, 1]
    smin, smax = series.min(), series.max()
    if smax == smin:
        return None, None, None
    normalized = (series - smin) / (smax - smin)

    log_eps = []
    log_n = []
    details = []

    for eps in scales:
        if eps >= n:
            continue
        # Number of boxes in time axis
        n_time_boxes = int(math.ceil(n / eps))
        # Number of boxes in value axis
        n_val_boxes = max(1, int(math.ceil(1.0 / (eps / n))))

        # Count occupied boxes
        occupied = set()
        for i, val in enumerate(normalized):
            t_box = i // eps
            v_box = min(int(val * n_val_boxes), n_val_boxes - 1)
            occupied.add((t_box, v_box))

        count = len(occupied)
        if count > 0:
            log_eps.append(math.log(1.0 / eps))
            log_n.append(math.log(count))
            details.append({"epsilon": eps, "n_boxes": count})

    if len(log_eps) < 3:
        return None, None, details

    # Linear regression: log(N) vs log(1/eps) = D_f
    slope, intercept, r_val, p_val, stderr = sp_stats.linregress(log_eps, log_n)
    return round(float(slope), 4), round(float(r_val**2), 4), details


def hurst_exponent_rs(series):
    """
    Hurst exponent via Rescaled Range (R/S) analysis.
    H > 0.5: persistent (long memory)
    H < 0.5: anti-persistent
    H = 0.5: random walk
    """
    n = len(series)
    if n < 20:
        return None, None

    # Divide into subseries of increasing length
    min_len = 10
    max_divs = min(50, n // min_len)

    log_ns = []
    log_rs = []

    for div_count in range(2, max_divs + 1):
        sub_len = n // div_count
        if sub_len < min_len:
            break

        rs_values = []
        for i in range(div_count):
            sub = series[i * sub_len:(i + 1) * sub_len]
            mean = np.mean(sub)
            deviations = sub - mean
            cumulative = np.cumsum(deviations)
            R = np.max(cumulative) - np.min(cumulative)
            S = np.std(sub, ddof=1)
            if S > 0:
                rs_values.append(R / S)

        if rs_values:
            mean_rs = np.mean(rs_values)
            if mean_rs > 0:
                log_ns.append(math.log(sub_len))
                log_rs.append(math.log(mean_rs))

    if len(log_ns) < 3:
        return None, None

    slope, intercept, r_val, p_val, stderr = sp_stats.linregress(log_ns, log_rs)
    return round(float(slope), 4), round(float(r_val**2), 4)


def dfa_exponent(series, min_box=4, max_box=None):
    """
    Detrended Fluctuation Analysis (DFA).
    α ≈ 0.5: uncorrelated
    α > 0.5: long-range correlations
    α < 0.5: anti-correlated
    """
    n = len(series)
    if n < 20:
        return None, None

    if max_box is None:
        max_box = n // 4

    # Integrate the series (cumulative sum of deviations from mean)
    mean = np.mean(series)
    y = np.cumsum(series - mean)

    # Box sizes
    box_sizes = []
    s = min_box
    while s <= max_box:
        box_sizes.append(int(s))
        s *= 1.5
    box_sizes = sorted(set(box_sizes))

    if len(box_sizes) < 3:
        return None, None

    log_s = []
    log_f = []

    for s in box_sizes:
        n_boxes = n // s
        if n_boxes < 1:
            continue

        fluctuations = []
        for i in range(n_boxes):
            segment = y[i * s:(i + 1) * s]
            # Linear detrend
            x_range = np.arange(len(segment), dtype=float)
            if len(segment) < 2:
                continue
            slope, intercept = np.polyfit(x_range, segment, 1)
            trend = slope * x_range + intercept
            residual = segment - trend
            rms = np.sqrt(np.mean(residual ** 2))
            fluctuations.append(rms)

        if fluctuations:
            mean_f = np.mean(fluctuations)
            if mean_f > 0:
                log_s.append(math.log(s))
                log_f.append(math.log(mean_f))

    if len(log_s) < 3:
        return None, None

    slope, intercept, r_val, p_val, stderr = sp_stats.linregress(log_s, log_f)
    return round(float(slope), 4), round(float(r_val**2), 4)


def analyze_series(series, label):
    """Run all three fractal analyses on a series."""
    log.info(f"  Analizando {label} (n={len(series)})...")
    D_f, D_r2, D_details = box_counting_dimension(series)
    H, H_r2 = hurst_exponent_rs(series)
    alpha, alpha_r2 = dfa_exponent(series)

    result = {
        "n_verses": len(series),
        "mean_length": round(float(np.mean(series)), 4) if len(series) > 0 else None,
        "box_counting": {"D_f": D_f, "R2": D_r2},
        "hurst": {"H": H, "R2": H_r2},
        "dfa": {"alpha": alpha, "R2": alpha_r2},
    }

    log.info(f"    D_f={D_f}, H={H}, α={alpha}")
    return result


def main():
    log.info("=== DEEP FRACTAL — INICIO ===")
    t0 = time.time()

    with open(CORPUS, encoding="utf-8") as f:
        words = json.load(f)
    log.info(f"Corpus: {len(words)} palabras")

    # === 1. Global analysis ===
    log.info("Análisis fractal global...")
    global_series = build_verse_length_series(words)
    global_result = analyze_series(global_series, "global")

    # === 2. By corpus (AT vs NT) ===
    log.info("Análisis por corpus...")
    ot_words = [w for w in words if w["corpus"] == "OT"]
    nt_words = [w for w in words if w["corpus"] == "NT"]

    ot_series = build_verse_length_series(ot_words)
    nt_series = build_verse_length_series(nt_words)

    ot_result = analyze_series(ot_series, "OT")
    nt_result = analyze_series(nt_series, "NT")

    # === 3. By genre ===
    log.info("Análisis por género...")
    genre_words = {}
    for w in words:
        genre = get_genre(w["book"])
        genre_words.setdefault(genre, []).append(w)

    genre_results = {}
    for genre, gw in sorted(genre_words.items()):
        series = build_verse_length_series(gw)
        if len(series) >= 20:
            genre_results[genre] = analyze_series(series, genre)

    # === 4. Statistical comparison AT vs NT ===
    log.info("Comparación estadística AT vs NT...")
    # Bootstrap comparison of Hurst exponents
    np.random.seed(42)
    n_boot = 1000

    ot_H_boots = []
    nt_H_boots = []

    for i in range(n_boot):
        # Resample verse lengths with replacement
        ot_sample = np.random.choice(ot_series, size=len(ot_series), replace=True)
        nt_sample = np.random.choice(nt_series, size=len(nt_series), replace=True)
        h_ot, _ = hurst_exponent_rs(ot_sample)
        h_nt, _ = hurst_exponent_rs(nt_sample)
        if h_ot is not None:
            ot_H_boots.append(h_ot)
        if h_nt is not None:
            nt_H_boots.append(h_nt)

        if (i + 1) % 200 == 0:
            log.info(f"  Bootstrap {i+1}/{n_boot}")

    comparison = {}
    if ot_H_boots and nt_H_boots:
        ot_arr = np.array(ot_H_boots)
        nt_arr = np.array(nt_H_boots)
        # Welch's t-test on bootstrap distributions
        t_stat, p_val = sp_stats.ttest_ind(ot_arr, nt_arr, equal_var=False)
        comparison["hurst_comparison"] = {
            "ot_mean": round(float(np.mean(ot_arr)), 4),
            "ot_std": round(float(np.std(ot_arr)), 4),
            "nt_mean": round(float(np.mean(nt_arr)), 4),
            "nt_std": round(float(np.std(nt_arr)), 4),
            "t_stat": round(float(t_stat), 4),
            "p_value": round(float(p_val), 6),
            "significant": float(p_val) < 0.05,
        }
        log.info(f"  Hurst comparison: OT={np.mean(ot_arr):.4f}±{np.std(ot_arr):.4f}, "
                 f"NT={np.mean(nt_arr):.4f}±{np.std(nt_arr):.4f}, p={p_val:.6f}")

    # === 5. Compile results ===
    box_counting = {
        "global": global_result["box_counting"],
        "OT": ot_result["box_counting"],
        "NT": nt_result["box_counting"],
    }

    hurst = {
        "global": global_result["hurst"],
        "OT": ot_result["hurst"],
        "NT": nt_result["hurst"],
    }

    dfa_results = {
        "global": global_result["dfa"],
        "OT": ot_result["dfa"],
        "NT": nt_result["dfa"],
    }

    fractal_by_corpus = {
        "global": global_result,
        "OT": ot_result,
        "NT": nt_result,
        "comparison": comparison,
    }

    fractal_by_genre = genre_results

    # === Save results ===
    log.info("Guardando resultados...")
    with open(OUT / "box_counting.json", "w", encoding="utf-8") as f:
        json.dump(box_counting, f, ensure_ascii=False, indent=2)
    with open(OUT / "hurst_exponent.json", "w", encoding="utf-8") as f:
        json.dump(hurst, f, ensure_ascii=False, indent=2)
    with open(OUT / "dfa_results.json", "w", encoding="utf-8") as f:
        json.dump(dfa_results, f, ensure_ascii=False, indent=2)
    with open(OUT / "fractal_by_corpus.json", "w", encoding="utf-8") as f:
        json.dump(fractal_by_corpus, f, ensure_ascii=False, indent=2)
    with open(OUT / "fractal_by_genre.json", "w", encoding="utf-8") as f:
        json.dump(fractal_by_genre, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t0
    log.info(f"=== DONE en {elapsed:.1f}s ===")
    print(f"[deep_fractal] DONE — {elapsed:.1f}s")


if __name__ == "__main__":
    main()
