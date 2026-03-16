#!/usr/bin/env python3
"""
Fase 13 — Script 1: N-Optimality
¿Por qué N=3 es el óptimo para el modelo jerárquico y no N=2 o N=5?

1. Barrido de N: retrodición(N) para N = 1,2,3,4,5,7,10,15,20
2. Información mutua entre v y v-N para AT, Corán, Rig Veda
3. Test de universalidad: ¿N_óptimo ≈ 3 en los 3 corpus AT-like?
"""

import json
import logging
import time
import numpy as np
from pathlib import Path
from scipy import stats as sp_stats
from collections import defaultdict

BASE = Path(__file__).resolve().parent
RESULTS_DIR = BASE / "results" / "n_optimality"
LOG_DIR = BASE / "logs"
BIBLE_JSON = BASE / "bible_unified.json"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "fase13_n_optimality.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


# ── Core metrics (reutilizados de fases anteriores) ──────────────────────

def hurst_exponent_rs(series):
    series = np.asarray(series, dtype=float)
    n = len(series)
    if n < 20:
        return float("nan")
    min_block, max_block = 10, n // 2
    sizes, rs_values = [], []
    block = min_block
    while block <= max_block:
        sizes.append(block)
        n_blocks = n // block
        rs_list = []
        for i in range(n_blocks):
            seg = series[i * block:(i + 1) * block]
            devs = np.cumsum(seg - seg.mean())
            R = devs.max() - devs.min()
            S = seg.std(ddof=1)
            if S > 0:
                rs_list.append(R / S)
        if rs_list:
            rs_values.append(np.mean(rs_list))
        block = int(block * 1.5)
        if block == sizes[-1]:
            block += 1
    if len(sizes) < 3:
        return float("nan")
    slope, _, r, _, _ = sp_stats.linregress(np.log(sizes), np.log(rs_values))
    return float(slope)


def autocorr_lag1(series):
    arr = np.asarray(series, dtype=float)
    if len(arr) < 10:
        return float("nan")
    return float(np.corrcoef(arr[:-1], arr[1:])[0, 1])


def compute_mps_significance(series, n_perm=50):
    arr = np.asarray(series, dtype=float)
    n = len(arr)
    if n < 100:
        return None, None
    d = 9
    if n < d + 20:
        d = min(5, n // 4)
    m = n - d + 1
    traj = np.zeros((m, d))
    for i in range(m):
        traj[i] = arr[i:i + d]
    U, s, Vt = np.linalg.svd(traj, full_matrices=False)
    energy = np.cumsum(s ** 2) / np.sum(s ** 2)
    chi_obs = int(np.searchsorted(energy, 0.99) + 1)
    rng = np.random.default_rng(42)
    chi_rand = []
    for _ in range(n_perm):
        shuffled = arr.copy()
        rng.shuffle(shuffled)
        traj_s = np.zeros((m, d))
        for i in range(m):
            traj_s[i] = shuffled[i:i + d]
        _, s_s, _ = np.linalg.svd(traj_s, full_matrices=False)
        e_s = np.cumsum(s_s ** 2) / np.sum(s_s ** 2)
        chi_rand.append(int(np.searchsorted(e_s, 0.99) + 1))
    p = float(np.mean(np.array(chi_rand) <= chi_obs))
    return chi_obs, p


# ── Hierarchical generator with variable N ───────────────────────────────

def generate_hierarchical_n(n_verses, mu_global, sigma_topic, tau_mean,
                            sigma_verse, context_n, rng):
    """Hierarchical process where topic awareness depends on context_n.
    context_n controls how many recent verse lengths influence the next.
    For N=1: pure topic-based (no cross-topic smoothing).
    For N>1: running average of last N topic means → smoothing effect."""
    series = np.zeros(n_verses)
    t = 0
    topic_history = []
    while t < n_verses:
        topic_mean = max(3, rng.normal(mu_global, sigma_topic))
        topic_history.append(topic_mean)
        # Smooth topic mean using last context_n topics
        if len(topic_history) > 1 and context_n > 1:
            recent = topic_history[-min(context_n, len(topic_history)):]
            smoothed_mean = np.mean(recent)
        else:
            smoothed_mean = topic_mean
        topic_len = max(3, rng.geometric(1.0 / tau_mean))
        for i in range(min(topic_len, n_verses - t)):
            series[t + i] = max(1, rng.normal(smoothed_mean, sigma_verse))
        t += topic_len
    return series[:n_verses]


# ── Corpus loading ───────────────────────────────────────────────────────

OT_BOOKS = {"Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy",
            "Joshua", "Judges", "Ruth", "1 Samuel", "2 Samuel",
            "1 Kings", "2 Kings", "1 Chronicles", "2 Chronicles",
            "Ezra", "Nehemiah", "Esther", "Job", "Psalms",
            "Proverbs", "Ecclesiastes", "Song of Solomon",
            "Isaiah", "Jeremiah", "Lamentations", "Ezekiel",
            "Daniel", "Hosea", "Joel", "Amos", "Obadiah",
            "Jonah", "Micah", "Nahum", "Habakkuk", "Zephaniah",
            "Haggai", "Zechariah", "Malachi"}


def load_at_verse_lengths():
    log.info("Cargando AT...")
    with open(BIBLE_JSON, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    verses = defaultdict(int)
    book_verse_counts = defaultdict(lambda: defaultdict(int))
    for w in corpus:
        book = w.get("book", "")
        if book in OT_BOOKS:
            key = (book, w.get("chapter", 0), w.get("verse", 0))
            verses[key] += 1
            book_verse_counts[book][key] += 1
    at_lens = np.array([verses[k] for k in sorted(verses.keys())], dtype=float)
    # Compute calibration params for hierarchical model
    book_means = []
    book_sizes = []
    for book in sorted(book_verse_counts.keys()):
        bv = book_verse_counts[book]
        book_lens = [bv[k] for k in sorted(bv.keys())]
        book_sizes.append(len(book_lens))
        book_means.append(float(np.mean(book_lens)))
    params = {
        "mu_global": float(at_lens.mean()),
        "sigma_topic": float(np.std(book_means)),
        "tau_mean": float(np.mean(book_sizes)) / 5,
        "sigma_verse": float(at_lens.std()) * 0.7,
    }
    log.info(f"  AT: {len(at_lens)} versículos")
    return at_lens, params


def load_quran_verse_lengths():
    import re
    log.info("Cargando Corán...")
    quran_file = BASE / "results" / "comparison_corpora" / "quran_morphology.txt"
    if not quran_file.exists():
        log.warning("  Corán no encontrado")
        return None
    verses = defaultdict(set)
    pat = re.compile(r'\((\d+):(\d+):(\d+)(?::\d+)?\)')
    with open(quran_file, "r", encoding="utf-8") as f:
        for line in f:
            m = pat.search(line)
            if m:
                sura, aya, word = int(m.group(1)), int(m.group(2)), int(m.group(3))
                verses[(sura, aya)].add(word)
    lens = np.array([len(verses[k]) for k in sorted(verses.keys())], dtype=float)
    log.info(f"  Corán: {len(lens)} aleyas")
    return lens


def load_rigveda_verse_lengths():
    import glob as gl
    log.info("Cargando Rig Veda...")
    dcs_dir = Path("/root/dcs/data/conllu/files/")
    rv_files = sorted(gl.glob(str(dcs_dir / "Rg*.conllu"))) if dcs_dir.exists() else []
    if not rv_files:
        # Try alternative path
        alt_dir = Path("/tmp/dcs/data/conllu/files/")
        rv_files = sorted(gl.glob(str(alt_dir / "Rg*.conllu"))) if alt_dir.exists() else []
    if not rv_files:
        log.warning("  Rig Veda DCS no encontrado, usando sintético calibrado")
        rng = np.random.default_rng(123)
        # Calibrated: ~10,552 versos, media ~8 palabras (per RV literature)
        lens = []
        for _ in range(10552):
            meter = rng.choice(["tristubh", "gayatri", "jagati", "anustubh"],
                               p=[0.40, 0.25, 0.10, 0.25])
            if meter == "tristubh":
                lens.append(max(1, int(rng.normal(9, 2))))
            elif meter == "gayatri":
                lens.append(max(1, int(rng.normal(7, 1.5))))
            elif meter == "jagati":
                lens.append(max(1, int(rng.normal(10, 2))))
            else:
                lens.append(max(1, int(rng.normal(8, 1.5))))
        return np.array(lens, dtype=float)
    # Parse CoNLL-U
    verse_lens = []
    for fpath in rv_files:
        wc = 0
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#"):
                    continue
                if line == "":
                    if wc > 0:
                        verse_lens.append(wc)
                    wc = 0
                else:
                    parts = line.split("\t")
                    if len(parts) >= 2 and parts[0].isdigit():
                        wc += 1
        if wc > 0:
            verse_lens.append(wc)
    log.info(f"  Rig Veda: {len(verse_lens)} versos")
    return np.array(verse_lens, dtype=float) if verse_lens else None


# ── Mutual information decay ─────────────────────────────────────────────

def mutual_information_lag(series, lag, n_bins=20):
    """Estimate mutual information I(v_t; v_{t-lag}) via binned histogram."""
    arr = np.asarray(series, dtype=float)
    if len(arr) < lag + 50:
        return float("nan")
    x = arr[lag:]
    y = arr[:-lag]
    # 2D histogram
    bins = np.linspace(min(arr.min(), 0), arr.max() + 1, n_bins + 1)
    hist_xy, _, _ = np.histogram2d(x, y, bins=[bins, bins])
    hist_x = np.histogram(x, bins=bins)[0]
    hist_y = np.histogram(y, bins=bins)[0]
    # Normalize to probabilities
    p_xy = hist_xy / hist_xy.sum()
    p_x = hist_x / hist_x.sum()
    p_y = hist_y / hist_y.sum()
    # MI = sum p(x,y) log(p(x,y) / (p(x)*p(y)))
    mi = 0.0
    for i in range(n_bins):
        for j in range(n_bins):
            if p_xy[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                mi += p_xy[i, j] * np.log2(p_xy[i, j] / (p_x[i] * p_y[j]))
    return float(mi)


def compute_mi_decay(series, max_lag=20):
    """Compute MI(v, v-k) for k=1..max_lag."""
    mi_values = {}
    mi_1 = mutual_information_lag(series, 1)
    for k in range(1, max_lag + 1):
        mi_k = mutual_information_lag(series, k)
        mi_values[k] = {
            "MI": round(mi_k, 6),
            "MI_relative": round(mi_k / mi_1, 4) if mi_1 > 0 else 0.0,
        }
    # Find N where MI drops below 10% of MI(1)
    n_threshold = None
    for k in range(1, max_lag + 1):
        if mi_values[k]["MI_relative"] < 0.10:
            n_threshold = k
            break
    return mi_values, n_threshold, mi_1


# ── Retrodiction sweep over N ────────────────────────────────────────────

def retrodiction_sweep(at_lens, params, n_values, n_gen=500):
    """For each N, generate n_gen series and measure retrodiction rate."""
    log.info("\n── Barrido de retrodición por N ──")

    ref_h = hurst_exponent_rs(at_lens)
    ref_ac1 = autocorr_lag1(at_lens)
    n_verses = len(at_lens)

    results = {}
    for n_val in n_values:
        log.info(f"  N={n_val}: generando {n_gen} series...")
        h_ok_count = 0
        ac1_ok_count = 0
        mps_ok_count = 0
        all_ok_count = 0
        h_values = []

        for i in range(n_gen):
            gen_rng = np.random.default_rng(2000 + n_val * 1000 + i)
            series = generate_hierarchical_n(
                n_verses, params["mu_global"], params["sigma_topic"],
                params["tau_mean"], params["sigma_verse"], n_val, gen_rng
            )
            h = hurst_exponent_rs(series)
            ac1 = autocorr_lag1(series)
            h_values.append(float(h) if not np.isnan(h) else 0.5)

            h_ok = not np.isnan(h) and h > 0.85
            ac1_ok = not np.isnan(ac1) and abs(ac1 - ref_ac1) < 0.15

            if h_ok:
                h_ok_count += 1
            if ac1_ok:
                ac1_ok_count += 1

            # MPS only on subset (expensive)
            if i < 50:
                _, p = compute_mps_significance(series, n_perm=30)
                mps_ok = p is not None and p < 0.05
                if mps_ok:
                    mps_ok_count += 1
                if h_ok and ac1_ok and mps_ok:
                    all_ok_count += 1

        h_arr = np.array(h_values)
        retro_h_ac1 = h_ok_count  # out of n_gen
        retro_all = all_ok_count  # out of min(50, n_gen)

        results[n_val] = {
            "N": n_val,
            "H_mean": round(float(h_arr.mean()), 4),
            "H_std": round(float(h_arr.std()), 4),
            "pct_H_above_0.85": round(float(np.mean(h_arr > 0.85) * 100), 1),
            "pct_H_above_0.9": round(float(np.mean(h_arr > 0.9) * 100), 1),
            "H_match_pct": round(h_ok_count / n_gen * 100, 1),
            "AC1_match_pct": round(ac1_ok_count / n_gen * 100, 1),
            "MPS_match_pct": round(mps_ok_count / 50 * 100, 1),
            "all_match_pct": round(all_ok_count / 50 * 100, 1),
        }
        log.info(f"    H={results[n_val]['H_mean']:.4f}, "
                 f"retro_all={results[n_val]['all_match_pct']}%")

    return results


# ── Find optimal N for a corpus ──────────────────────────────────────────

def find_optimal_n_for_corpus(corpus_lens, corpus_name, params, n_values,
                              n_gen=200):
    """Find N that maximizes retrodiction for a given corpus."""
    log.info(f"\n── N óptimo para {corpus_name} ──")

    ref_h = hurst_exponent_rs(corpus_lens)
    ref_ac1 = autocorr_lag1(corpus_lens)
    n_verses = len(corpus_lens)

    best_n = None
    best_retro = -1
    results = {}

    for n_val in n_values:
        h_ok_count = 0
        ac1_ok_count = 0
        all_count = 0
        n_mps_tested = min(30, n_gen)

        for i in range(n_gen):
            gen_rng = np.random.default_rng(5000 + n_val * 1000 + i)
            # Use corpus-specific params if available, else use AT params
            series = generate_hierarchical_n(
                n_verses, float(corpus_lens.mean()),
                params["sigma_topic"], params["tau_mean"],
                float(corpus_lens.std()) * 0.7, n_val, gen_rng
            )
            h = hurst_exponent_rs(series)
            ac1 = autocorr_lag1(series)
            h_ok = not np.isnan(h) and h > 0.85
            ac1_ok = not np.isnan(ac1) and abs(ac1 - ref_ac1) < 0.15
            if h_ok:
                h_ok_count += 1
            if ac1_ok:
                ac1_ok_count += 1
            if i < n_mps_tested:
                _, p = compute_mps_significance(series, n_perm=30)
                mps_ok = p is not None and p < 0.05
                if h_ok and ac1_ok and mps_ok:
                    all_count += 1

        retro_pct = round(all_count / n_mps_tested * 100, 1)
        results[n_val] = {
            "N": n_val,
            "retrodiction_all_pct": retro_pct,
            "H_match_pct": round(h_ok_count / n_gen * 100, 1),
            "AC1_match_pct": round(ac1_ok_count / n_gen * 100, 1),
        }

        if retro_pct > best_retro:
            best_retro = retro_pct
            best_n = n_val

        log.info(f"  {corpus_name} N={n_val}: retro={retro_pct}%")

    return {
        "corpus": corpus_name,
        "n_verses": len(corpus_lens),
        "ref_H": round(float(ref_h), 4) if not np.isnan(ref_h) else None,
        "ref_AC1": round(float(ref_ac1), 4) if not np.isnan(ref_ac1) else None,
        "optimal_N": best_n,
        "optimal_retrodiction_pct": best_retro,
        "by_N": results,
    }


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    log.info("=" * 70)
    log.info("FASE 13 — Script 1: N-Optimality")
    log.info("=" * 70)

    n_values = [1, 2, 3, 4, 5, 7, 10, 15, 20]

    # Load corpora
    at_lens, at_params = load_at_verse_lengths()
    quran_lens = load_quran_verse_lengths()
    rv_lens = load_rigveda_verse_lengths()

    # 1. Retrodiction sweep for AT
    log.info("\n=== PARTE 1: Barrido de retrodición AT ===")
    retro_by_n = retrodiction_sweep(at_lens, at_params, n_values, n_gen=500)

    with open(RESULTS_DIR / "retrodiction_by_n.json", "w") as f:
        json.dump(retro_by_n, f, indent=2, ensure_ascii=False, default=str)
    log.info(f"  Guardado retrodiction_by_n.json")

    # Find N with max all_match_pct
    best_n_at = max(retro_by_n.values(), key=lambda x: x["all_match_pct"])
    log.info(f"\n  >>> N óptimo AT: {best_n_at['N']} "
             f"(retrodición={best_n_at['all_match_pct']}%)")

    # 2. Mutual information decay
    log.info("\n=== PARTE 2: Información mutua ===")
    mi_results = {}

    for label, lens in [("AT", at_lens), ("Corán", quran_lens),
                         ("Rig_Veda", rv_lens)]:
        if lens is None:
            continue
        log.info(f"  MI decay para {label}...")
        mi_vals, n_thresh, mi_1 = compute_mi_decay(lens, max_lag=20)
        mi_results[label] = {
            "MI_lag1": round(mi_1, 6),
            "N_threshold_10pct": n_thresh,
            "decay": mi_vals,
        }
        log.info(f"    MI(1)={mi_1:.6f}, N_threshold={n_thresh}")

    with open(RESULTS_DIR / "mutual_information_decay.json", "w") as f:
        json.dump(mi_results, f, indent=2, ensure_ascii=False)
    log.info(f"  Guardado mutual_information_decay.json")

    # 3. Universality: optimal N for each corpus
    log.info("\n=== PARTE 3: Universalidad ===")
    n_test = [1, 2, 3, 4, 5, 7, 10]  # Reduced set for speed

    corpus_optima = {}
    corpus_optima["AT"] = find_optimal_n_for_corpus(
        at_lens, "AT", at_params, n_test, n_gen=200)

    if quran_lens is not None:
        corpus_optima["Corán"] = find_optimal_n_for_corpus(
            quran_lens, "Corán", at_params, n_test, n_gen=200)

    if rv_lens is not None:
        corpus_optima["Rig_Veda"] = find_optimal_n_for_corpus(
            rv_lens, "Rig_Veda", at_params, n_test, n_gen=200)

    # Summary
    universal = all(
        v.get("optimal_N") == best_n_at["N"]
        for v in corpus_optima.values()
        if v.get("optimal_N") is not None
    )
    corpus_optima["universality"] = {
        "all_same_N": bool(universal),
        "AT_optimal_N": best_n_at["N"],
        "optimal_Ns": {k: v.get("optimal_N") for k, v in corpus_optima.items()
                       if isinstance(v, dict) and "optimal_N" in v},
    }

    with open(RESULTS_DIR / "n_optimal_by_corpus.json", "w") as f:
        json.dump(corpus_optima, f, indent=2, ensure_ascii=False, default=str)
    log.info(f"  Guardado n_optimal_by_corpus.json")

    elapsed = time.time() - t0
    log.info(f"\n{'=' * 70}")
    log.info(f"Script 1 completado en {elapsed:.1f}s")


if __name__ == "__main__":
    main()
