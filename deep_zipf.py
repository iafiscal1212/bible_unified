#!/usr/bin/env python3
"""
deep_zipf.py — Investigación 1: ¿La anomalía Zipf del AT es morfológica o semántica?
- Zipf sobre formas de superficie vs lemas, por libro
- KS test contra Zipf teórico
- Delta-s análisis
"""
import json, logging, math, time
from collections import Counter
from pathlib import Path
import numpy as np
from scipy import stats as sp_stats
from scipy.optimize import curve_fit

BASE = Path(__file__).parent
CORPUS = BASE / "bible_unified.json"
OUT = BASE / "results" / "deep_zipf"
OUT.mkdir(parents=True, exist_ok=True)
LOG = BASE / "logs"
LOG.mkdir(parents=True, exist_ok=True)

logging.basicConfig(filename=LOG / "deep_zipf.log", level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("deep_zipf")


def zipf_func(r, C, s):
    """f(r) = C * r^(-s)"""
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
        # Fallback: linear regression on log-log
        log_r = np.log(ranks)
        log_f = np.log(freqs)
        slope, intercept, r_val, _, _ = sp_stats.linregress(log_r, log_f)
        return round(float(-slope), 6), round(float(r_val**2), 6), n


def ks_test_zipf(freq_counter, s_theoretical=1.0):
    """KS test: observed frequencies vs Zipf(s=s_theoretical)."""
    freqs = np.array(sorted(freq_counter.values(), reverse=True), dtype=float)
    n = len(freqs)
    if n < 5:
        return None, None
    # Normalize to get empirical PMF
    total = freqs.sum()
    empirical = freqs / total
    # Theoretical Zipf PMF
    ranks = np.arange(1, n + 1, dtype=float)
    theoretical = np.power(ranks, -s_theoretical)
    theoretical = theoretical / theoretical.sum()
    # KS on CDFs
    emp_cdf = np.cumsum(empirical)
    theo_cdf = np.cumsum(theoretical)
    ks_stat = float(np.max(np.abs(emp_cdf - theo_cdf)))
    # Approximate p-value using scipy on the raw sorted data
    # Create samples from empirical distribution for KS
    return round(ks_stat, 6), n


def main():
    log.info("Cargando corpus...")
    t0 = time.time()
    with open(CORPUS, encoding="utf-8") as f:
        words = json.load(f)
    log.info(f"Corpus: {len(words)} palabras, cargado en {time.time()-t0:.1f}s")

    # Agrupar por libro
    books = {}
    for w in words:
        books.setdefault(w["book"], []).append(w)

    # === 1. Zipf por libro: superficie vs lema ===
    log.info("Calculando Zipf por libro (superficie vs lema)...")
    book_results = []
    for book_name in sorted(books.keys(), key=lambda b: books[b][0]["book_num"]):
        bw = books[book_name]
        corpus = bw[0]["corpus"]
        book_num = bw[0]["book_num"]

        surf_freq = Counter(w["text"] for w in bw)
        lemma_freq = Counter(w["lemma"] for w in bw)

        s_surf, r2_surf, n_surf = fit_zipf(surf_freq)
        s_lemma, r2_lemma, n_lemma = fit_zipf(lemma_freq)

        delta_s = round(s_surf - s_lemma, 6) if s_surf and s_lemma else None

        ks_surf, _ = ks_test_zipf(surf_freq, 1.0)
        ks_lemma, _ = ks_test_zipf(lemma_freq, 1.0)

        result = {
            "book": book_name,
            "book_num": book_num,
            "corpus": corpus,
            "n_tokens": len(bw),
            "surface": {"s": s_surf, "r2": r2_surf, "n_types": n_surf},
            "lemma": {"s": s_lemma, "r2": r2_lemma, "n_types": n_lemma},
            "delta_s": delta_s,
            "ks_vs_zipf1_surface": ks_surf,
            "ks_vs_zipf1_lemma": ks_lemma,
        }
        book_results.append(result)
        log.info(f"  {book_name}: s_surf={s_surf}, s_lemma={s_lemma}, Δs={delta_s}")

    # === 2. Global: superficie vs lema por corpus ===
    log.info("Zipf global superficie vs lema...")
    for corpus_name in ["OT", "NT"]:
        cw = [w for w in words if w["corpus"] == corpus_name]
        surf = Counter(w["text"] for w in cw)
        lemma = Counter(w["lemma"] for w in cw)
        s_s, r2_s, n_s = fit_zipf(surf)
        s_l, r2_l, n_l = fit_zipf(lemma)
        log.info(f"  {corpus_name} surface: s={s_s}, R²={r2_s}")
        log.info(f"  {corpus_name} lemma:   s={s_l}, R²={r2_l}")

    surface_vs_lemma = {}
    for corpus_name in ["OT", "NT", "global"]:
        if corpus_name == "global":
            cw = words
        else:
            cw = [w for w in words if w["corpus"] == corpus_name]
        surf = Counter(w["text"] for w in cw)
        lemma = Counter(w["lemma"] for w in cw)
        s_s, r2_s, n_s = fit_zipf(surf)
        s_l, r2_l, n_l = fit_zipf(lemma)
        ks_s, _ = ks_test_zipf(surf, 1.0)
        ks_l, _ = ks_test_zipf(lemma, 1.0)
        surface_vs_lemma[corpus_name] = {
            "surface": {"s": s_s, "r2": r2_s, "n_types": n_s, "ks_stat": ks_s},
            "lemma": {"s": s_l, "r2": r2_l, "n_types": n_l, "ks_stat": ks_l},
            "delta_s": round(s_s - s_l, 6) if s_s and s_l else None,
        }

    # === 3. Análisis de Delta-s ===
    log.info("Análisis de Δs...")
    ot_deltas = [r["delta_s"] for r in book_results if r["corpus"] == "OT" and r["delta_s"] is not None]
    nt_deltas = [r["delta_s"] for r in book_results if r["corpus"] == "NT" and r["delta_s"] is not None]

    delta_analysis = {
        "ot_delta_s_mean": round(float(np.mean(ot_deltas)), 6) if ot_deltas else None,
        "ot_delta_s_std": round(float(np.std(ot_deltas)), 6) if ot_deltas else None,
        "nt_delta_s_mean": round(float(np.mean(nt_deltas)), 6) if nt_deltas else None,
        "nt_delta_s_std": round(float(np.std(nt_deltas)), 6) if nt_deltas else None,
        "interpretation": (
            "If Δs is constant across books, the Zipf anomaly is purely morphological. "
            "If Δs varies significantly, there is semantic structure contributing."
        ),
    }

    # CV of delta_s
    if ot_deltas and np.mean(ot_deltas) != 0:
        delta_analysis["ot_delta_s_cv"] = round(float(np.std(ot_deltas) / abs(np.mean(ot_deltas))), 4)
    if nt_deltas and np.mean(nt_deltas) != 0:
        delta_analysis["nt_delta_s_cv"] = round(float(np.std(nt_deltas) / abs(np.mean(nt_deltas))), 4)

    # Mann-Whitney on delta_s: AT vs NT
    if len(ot_deltas) >= 3 and len(nt_deltas) >= 3:
        u_stat, u_pval = sp_stats.mannwhitneyu(ot_deltas, nt_deltas, alternative='two-sided')
        delta_analysis["mannwhitney_ot_vs_nt"] = {
            "U": round(float(u_stat), 2),
            "p_value": round(float(u_pval), 8),
        }

    # === 4. KS test results ===
    log.info("Compilando KS test results...")
    ks_results = {
        "test": "Kolmogorov-Smirnov: observed frequency distribution vs Zipf(s=1.0)",
        "by_corpus": {},
    }
    for corpus_name in ["OT", "NT"]:
        cw = [w for w in words if w["corpus"] == corpus_name]
        surf = Counter(w["text"] for w in cw)
        lemma = Counter(w["lemma"] for w in cw)
        ks_s, _ = ks_test_zipf(surf, 1.0)
        ks_l, _ = ks_test_zipf(lemma, 1.0)
        ks_results["by_corpus"][corpus_name] = {
            "surface_ks_stat": ks_s,
            "lemma_ks_stat": ks_l,
        }
    ks_results["by_book"] = [
        {"book": r["book"], "corpus": r["corpus"],
         "surface_ks": r["ks_vs_zipf1_surface"],
         "lemma_ks": r["ks_vs_zipf1_lemma"]}
        for r in book_results
    ]

    # Save results
    with open(OUT / "zipf_by_book.json", "w") as f:
        json.dump(book_results, f, indent=2, ensure_ascii=False)
    with open(OUT / "zipf_surface_vs_lemma.json", "w") as f:
        json.dump({"by_corpus": surface_vs_lemma, "delta_analysis": delta_analysis}, f, indent=2)
    with open(OUT / "ks_test_results.json", "w") as f:
        json.dump(ks_results, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    log.info(f"DONE en {elapsed:.1f}s")
    print(f"[deep_zipf] DONE — {elapsed:.1f}s")


if __name__ == "__main__":
    main()
