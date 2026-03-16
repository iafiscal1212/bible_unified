#!/usr/bin/env python3
"""
Fase 13 — Script 2: Anomalous Books
¿Los libros AT anómalos (Nahúm z=-2.62, Abdías z=-2.16, 1Crónicas z=+2.44)
tienen explicación composicional?

1. Feature profiles de libros anómalos vs control
2. Modelo jerárquico N=3 por libro: ¿reproduce H?
3. Test de efecto de tamaño finito
4. Veredicto: genuinamente anómalos vs artefactos estadísticos
"""

import json
import logging
import time
import numpy as np
from pathlib import Path
from scipy import stats as sp_stats
from collections import defaultdict

BASE = Path(__file__).resolve().parent
RESULTS_DIR = BASE / "results" / "anomalous"
LOG_DIR = BASE / "logs"
BIBLE_JSON = BASE / "bible_unified.json"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "fase13_anomalous.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

OT_BOOKS = {"Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy",
            "Joshua", "Judges", "Ruth", "1 Samuel", "2 Samuel",
            "1 Kings", "2 Kings", "1 Chronicles", "2 Chronicles",
            "Ezra", "Nehemiah", "Esther", "Job", "Psalms",
            "Proverbs", "Ecclesiastes", "Song of Solomon",
            "Isaiah", "Jeremiah", "Lamentations", "Ezekiel",
            "Daniel", "Hosea", "Joel", "Amos", "Obadiah",
            "Jonah", "Micah", "Nahum", "Habakkuk", "Zephaniah",
            "Haggai", "Zechariah", "Malachi"}

ANOMALOUS = ["Nahum", "Obadiah", "1 Chronicles"]
CONTROL = ["Genesis", "Exodus", "Isaiah", "Psalms", "Jeremiah"]


# ── Core metrics ─────────────────────────────────────────────────────────

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
    if n < 50:
        return None, None
    d = min(9, n // 5)
    if d < 3:
        return None, None
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


# ── Compositional features ───────────────────────────────────────────────

def compute_features(verse_lens, word_data=None):
    """Compute compositional features for a book."""
    arr = np.asarray(verse_lens, dtype=float)
    n = len(arr)
    features = {
        "n_verses": n,
        "mean_verse_len": round(float(arr.mean()), 2),
        "std_verse_len": round(float(arr.std()), 2),
        "cv_verse_len": round(float(arr.std() / arr.mean()), 4) if arr.mean() > 0 else 0,
        "skewness": round(float(sp_stats.skew(arr)), 4) if n > 3 else 0,
        "autocorr_lag1": round(autocorr_lag1(arr), 4),
    }
    h = hurst_exponent_rs(arr)
    features["H"] = round(float(h), 4) if not np.isnan(h) else None

    chi, p = compute_mps_significance(arr, n_perm=50)
    features["MPS_chi"] = chi
    features["MPS_p"] = round(float(p), 4) if p is not None else None

    # POS features if word_data available
    if word_data:
        pos_counts = defaultdict(int)
        total_words = 0
        for w in word_data:
            pos = w.get("pos", "unknown")
            pos_counts[pos] += 1
            total_words += 1
        if total_words > 0:
            features["verb_density"] = round(
                pos_counts.get("verb", 0) / total_words, 4)
            features["noun_density"] = round(
                pos_counts.get("noun", 0) / total_words, 4)
            features["proper_ratio"] = round(
                pos_counts.get("proper_noun", 0) / total_words, 4)
            # POS entropy
            probs = np.array([c / total_words for c in pos_counts.values()])
            probs = probs[probs > 0]
            features["pos_entropy"] = round(
                float(-np.sum(probs * np.log2(probs))), 4)

    return features


# ── Hierarchical generator ───────────────────────────────────────────────

def generate_hierarchical(n, mu_global, sigma_topic, tau_mean,
                          sigma_verse, rng):
    series = np.zeros(n)
    t = 0
    while t < n:
        topic_mean = max(3, rng.normal(mu_global, sigma_topic))
        topic_len = max(3, rng.geometric(1.0 / max(3, tau_mean)))
        for i in range(min(topic_len, n - t)):
            series[t + i] = max(1, rng.normal(topic_mean, sigma_verse))
        t += topic_len
    return series[:n]


def model_fit_for_book(book_lens, book_name, n_gen=200):
    """Fit hierarchical model to a book and measure retrodiction."""
    arr = np.asarray(book_lens, dtype=float)
    n = len(arr)
    if n < 20:
        return {"book": book_name, "n_verses": n, "too_short": True}

    ref_h = hurst_exponent_rs(arr)
    ref_ac1 = autocorr_lag1(arr)

    # Calibrate from book data
    mu = float(arr.mean())
    std_val = float(arr.std())

    results = {}
    for context_n in [1, 2, 3, 4, 5, 7, 10]:
        h_values = []
        h_match = 0
        ac1_match = 0

        for i in range(n_gen):
            gen_rng = np.random.default_rng(3000 + context_n * 1000 + i)
            # Use book-specific calibration
            sigma_topic = std_val * 0.4
            tau_mean = max(3, n / 5)
            sigma_verse = std_val * 0.7
            series = generate_hierarchical(
                n, mu, sigma_topic, tau_mean, sigma_verse, gen_rng
            )
            h = hurst_exponent_rs(series)
            ac1 = autocorr_lag1(series)
            h_values.append(float(h) if not np.isnan(h) else 0.5)
            if not np.isnan(h) and not np.isnan(ref_h):
                if abs(h - ref_h) < 0.15:
                    h_match += 1
            if not np.isnan(ac1) and not np.isnan(ref_ac1):
                if abs(ac1 - ref_ac1) < 0.15:
                    ac1_match += 1

        h_arr = np.array(h_values)
        results[context_n] = {
            "N": context_n,
            "H_mean_synth": round(float(h_arr.mean()), 4),
            "H_match_pct": round(h_match / n_gen * 100, 1),
            "AC1_match_pct": round(ac1_match / n_gen * 100, 1),
        }

    best_n = max(results.values(), key=lambda x: x["H_match_pct"])

    return {
        "book": book_name,
        "n_verses": n,
        "ref_H": round(float(ref_h), 4) if not np.isnan(ref_h) else None,
        "ref_AC1": round(float(ref_ac1), 4) if not np.isnan(ref_ac1) else None,
        "best_N": best_n["N"],
        "best_H_match_pct": best_n["H_match_pct"],
        "by_N": results,
    }


# ── Size effect test ─────────────────────────────────────────────────────

def size_effect_test(all_books_lens, anomalous_sizes):
    """Subsample large books to size of anomalous books and compute H variance."""
    log.info("\n── Test de efecto de tamaño finito ──")
    results = {}

    for target_size_name, target_n in anomalous_sizes.items():
        log.info(f"  Target: {target_size_name} (n={target_n})")
        # Subsample large books
        h_subsampled = []
        rng = np.random.default_rng(42)
        for book_name, book_lens in all_books_lens.items():
            if len(book_lens) < target_n * 2:
                continue  # Need at least 2x to subsample
            for _ in range(100):
                start = rng.integers(0, len(book_lens) - target_n)
                subsample = book_lens[start:start + target_n]
                h = hurst_exponent_rs(subsample)
                if not np.isnan(h):
                    h_subsampled.append(float(h))

        if not h_subsampled:
            results[target_size_name] = {"n_subsamples": 0}
            continue

        h_arr = np.array(h_subsampled)
        results[target_size_name] = {
            "target_n": target_n,
            "n_subsamples": len(h_arr),
            "H_mean": round(float(h_arr.mean()), 4),
            "H_std": round(float(h_arr.std()), 4),
            "H_q05": round(float(np.percentile(h_arr, 5)), 4),
            "H_q95": round(float(np.percentile(h_arr, 95)), 4),
            "pct_below_0.7": round(float(np.mean(h_arr < 0.7) * 100), 1),
            "pct_above_1.0": round(float(np.mean(h_arr > 1.0) * 100), 1),
        }
        log.info(f"    H_mean={h_arr.mean():.4f}±{h_arr.std():.4f}, "
                 f"q5-q95=[{np.percentile(h_arr, 5):.3f}, {np.percentile(h_arr, 95):.3f}]")

    return results


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    log.info("=" * 70)
    log.info("FASE 13 — Script 2: Anomalous Books")
    log.info("=" * 70)

    # Load all AT books
    log.info("Cargando AT por libro...")
    with open(BIBLE_JSON, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    # Group by book → verse → word count
    book_verses = defaultdict(lambda: defaultdict(int))
    book_words = defaultdict(list)
    for w in corpus:
        book = w.get("book", "")
        if book in OT_BOOKS:
            key = (w.get("chapter", 0), w.get("verse", 0))
            book_verses[book][key] += 1
            book_words[book].append(w)

    # Compute features for all books
    all_books_lens = {}
    all_features = {}
    for book in sorted(book_verses.keys()):
        bv = book_verses[book]
        lens = np.array([bv[k] for k in sorted(bv.keys())], dtype=float)
        all_books_lens[book] = lens
        all_features[book] = compute_features(lens, book_words.get(book))
        all_features[book]["book"] = book

    log.info(f"  {len(all_features)} libros cargados")

    # 1. Feature profiles: anomalous vs control
    log.info("\n=== PARTE 1: Feature profiles ===")
    profiles = {"anomalous": {}, "control": {}}
    for book in ANOMALOUS:
        if book in all_features:
            profiles["anomalous"][book] = all_features[book]
            log.info(f"  ANÓMALO {book}: n={all_features[book]['n_verses']}, "
                     f"H={all_features[book].get('H')}, "
                     f"AC1={all_features[book].get('autocorr_lag1')}")
    for book in CONTROL:
        if book in all_features:
            profiles["control"][book] = all_features[book]
            log.info(f"  CONTROL {book}: n={all_features[book]['n_verses']}, "
                     f"H={all_features[book].get('H')}, "
                     f"AC1={all_features[book].get('autocorr_lag1')}")

    # Compute z-scores for each feature
    feature_keys = ["mean_verse_len", "std_verse_len", "cv_verse_len",
                    "skewness", "autocorr_lag1", "H"]
    z_scores = {}
    for book in ANOMALOUS:
        if book not in all_features:
            continue
        z_scores[book] = {}
        for feat in feature_keys:
            vals = [all_features[b].get(feat) for b in all_features
                    if all_features[b].get(feat) is not None]
            book_val = all_features[book].get(feat)
            if vals and book_val is not None:
                mean_f = np.mean(vals)
                std_f = np.std(vals)
                z = (book_val - mean_f) / std_f if std_f > 0 else 0
                z_scores[book][feat] = {
                    "value": book_val,
                    "z_score": round(float(z), 3),
                    "anomalous": bool(abs(z) > 2),
                }

    profiles["z_scores"] = z_scores

    with open(RESULTS_DIR / "feature_profiles.json", "w") as f:
        json.dump(profiles, f, indent=2, ensure_ascii=False)
    log.info(f"  Guardado feature_profiles.json")

    # 2. Model fit per book
    log.info("\n=== PARTE 2: Model fit por libro ===")
    model_fits = {}
    test_books = ANOMALOUS + CONTROL
    for book in test_books:
        if book in all_books_lens:
            log.info(f"  Ajustando {book}...")
            model_fits[book] = model_fit_for_book(
                all_books_lens[book], book, n_gen=200)

    with open(RESULTS_DIR / "model_fit_per_book.json", "w") as f:
        json.dump(model_fits, f, indent=2, ensure_ascii=False)
    log.info(f"  Guardado model_fit_per_book.json")

    # 3. Size effect test
    log.info("\n=== PARTE 3: Efecto de tamaño finito ===")
    anomalous_sizes = {}
    for book in ANOMALOUS:
        if book in all_features:
            anomalous_sizes[book] = all_features[book]["n_verses"]

    size_results = size_effect_test(all_books_lens, anomalous_sizes)

    with open(RESULTS_DIR / "size_effect_test.json", "w") as f:
        json.dump(size_results, f, indent=2, ensure_ascii=False)
    log.info(f"  Guardado size_effect_test.json")

    # 4. Specific hypotheses for 1 Chronicles
    log.info("\n=== PARTE 4: Hipótesis específicas ===")
    specific = {}

    # 1 Chronicles vs 2 Chronicles, Ezra
    chronicle_books = ["1 Chronicles", "2 Chronicles", "Ezra"]
    for book in chronicle_books:
        if book in all_features:
            specific[book] = {
                "n_verses": all_features[book]["n_verses"],
                "H": all_features[book].get("H"),
                "AC1": all_features[book].get("autocorr_lag1"),
                "cv": all_features[book].get("cv_verse_len"),
                "mean_verse_len": all_features[book].get("mean_verse_len"),
            }

    # Is 1Chr AC(1) higher than similar books?
    if "1 Chronicles" in all_features and "2 Chronicles" in all_features:
        ac1_1chr = all_features["1 Chronicles"].get("autocorr_lag1", 0)
        ac1_2chr = all_features["2 Chronicles"].get("autocorr_lag1", 0)
        specific["1chr_vs_2chr_ac1"] = {
            "1Chronicles_AC1": ac1_1chr,
            "2Chronicles_AC1": ac1_2chr,
            "ratio": round(ac1_1chr / ac1_2chr, 3) if ac1_2chr != 0 else None,
            "1chr_higher": bool(ac1_1chr > ac1_2chr),
        }

    with open(RESULTS_DIR / "specific_hypotheses.json", "w") as f:
        json.dump(specific, f, indent=2, ensure_ascii=False)

    # 5. Verdict
    log.info("\n=== PARTE 5: Veredicto ===")
    verdict = {}
    for book in ANOMALOUS:
        if book not in all_features:
            continue
        n = all_features[book]["n_verses"]
        h = all_features[book].get("H")

        # Check if anomaly is explained by size
        size_data = size_results.get(book, {})
        q05 = size_data.get("H_q05")
        q95 = size_data.get("H_q95")

        if h is not None and q05 is not None and q95 is not None:
            within_size_range = q05 <= h <= q95
        else:
            within_size_range = None

        # Check if model reproduces H
        model_data = model_fits.get(book, {})
        best_h_match = model_data.get("best_H_match_pct", 0)

        if within_size_range:
            explanation = "SIZE_EFFECT"
            genuine = False
        elif best_h_match > 30:
            explanation = "MODEL_REPRODUCED"
            genuine = False
        else:
            explanation = "GENUINELY_ANOMALOUS"
            genuine = True

        verdict[book] = {
            "n_verses": n,
            "H": h,
            "within_size_effect_range": within_size_range,
            "model_H_match_pct": best_h_match,
            "explanation": explanation,
            "genuinely_anomalous": genuine,
        }
        log.info(f"  {book}: {explanation} (H={h}, size_range={within_size_range}, "
                 f"model_match={best_h_match}%)")

    n_genuine = sum(1 for v in verdict.values() if v["genuinely_anomalous"])
    verdict["summary"] = {
        "n_anomalous_tested": len(ANOMALOUS),
        "n_genuinely_anomalous": n_genuine,
        "n_size_effect": sum(1 for v in verdict.values()
                             if isinstance(v, dict) and v.get("explanation") == "SIZE_EFFECT"),
        "n_model_reproduced": sum(1 for v in verdict.values()
                                   if isinstance(v, dict) and v.get("explanation") == "MODEL_REPRODUCED"),
    }

    with open(RESULTS_DIR / "verdict.json", "w") as f:
        json.dump(verdict, f, indent=2, ensure_ascii=False)
    log.info(f"  Guardado verdict.json")

    elapsed = time.time() - t0
    log.info(f"\n{'=' * 70}")
    log.info(f"Script 2 completado en {elapsed:.1f}s")


if __name__ == "__main__":
    main()
