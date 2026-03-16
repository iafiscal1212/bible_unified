#!/usr/bin/env python3
"""
Fase 12 — Script 3: Compositional Rule
¿Se puede reconstruir la "regla composicional" que produce H>0.9?

4 modelos generativos en competencia:
A. AR(1) — 2 parámetros (μ, φ)
B. HMM-2 — 5 parámetros (μ1, μ2, σ1, σ2, p_transition)
C. ARFIMA — 3 parámetros (d, φ, σ)
D. Jerárquico — 4 parámetros (μ_topic, σ_topic, τ_topic, σ_verse)

Criterios: H, AC(1), MPS significativo
Test de plausibilidad cognitiva: la regla debe ser local (N ≤ 10)
Retrodicción: generar 100 textos y medir qué fracción reproduce simultáneamente
H, AC(1) y MPS del AT.
"""

import json
import logging
import time
import numpy as np
from pathlib import Path
from scipy import stats as sp_stats

BASE = Path(__file__).resolve().parent
RESULTS_DIR = BASE / "results" / "compositional_rule"
LOG_DIR = BASE / "logs"
BIBLE_JSON = BASE / "bible_unified.json"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "fase12_compositional.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


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


def compute_mps_significance(series, n_perm=200):
    """Simplified MPS significance test.
    Bond dimension via SVD of trajectory matrix."""
    arr = np.asarray(series, dtype=float)
    n = len(arr)
    if n < 100:
        return None, None

    # Trajectory matrix (embedding dimension = 9, like previous phases)
    d = 9
    if n < d + 20:
        d = min(5, n // 4)
    m = n - d + 1
    traj = np.zeros((m, d))
    for i in range(m):
        traj[i] = arr[i:i + d]

    # Bond dimension = rank at 99% energy
    U, s, Vt = np.linalg.svd(traj, full_matrices=False)
    energy = np.cumsum(s ** 2) / np.sum(s ** 2)
    chi_obs = int(np.searchsorted(energy, 0.99) + 1)

    # Permutation test
    rng = np.random.default_rng(42)
    chi_rand = []
    for _ in range(n_perm):
        shuffled = arr.copy()
        rng.shuffle(shuffled)
        traj_s = np.zeros((m, d))
        for i in range(m):
            traj_s[i] = shuffled[i:i + d]
        U_s, s_s, _ = np.linalg.svd(traj_s, full_matrices=False)
        e_s = np.cumsum(s_s ** 2) / np.sum(s_s ** 2)
        chi_rand.append(int(np.searchsorted(e_s, 0.99) + 1))

    chi_rand = np.array(chi_rand)
    p = float(np.mean(chi_rand <= chi_obs))

    return chi_obs, p


# ── Load AT reference data ───────────────────────────────────────────────

def load_at_reference():
    """Load AT verse lengths and compute reference metrics."""
    log.info("Cargando AT como referencia...")
    from collections import defaultdict

    OT_BOOKS = {"Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy",
                "Joshua", "Judges", "Ruth", "1 Samuel", "2 Samuel",
                "1 Kings", "2 Kings", "1 Chronicles", "2 Chronicles",
                "Ezra", "Nehemiah", "Esther", "Job", "Psalms",
                "Proverbs", "Ecclesiastes", "Song of Solomon",
                "Isaiah", "Jeremiah", "Lamentations", "Ezekiel",
                "Daniel", "Hosea", "Joel", "Amos", "Obadiah",
                "Jonah", "Micah", "Nahum", "Habakkuk", "Zephaniah",
                "Haggai", "Zechariah", "Malachi"}

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

    # Reference metrics
    h_ref = hurst_exponent_rs(at_lens)
    ac1_ref = autocorr_lag1(at_lens)
    chi_ref, p_ref = compute_mps_significance(at_lens, n_perm=100)

    # Book-level statistics for hierarchical model calibration
    book_sizes = []
    book_means = []
    for book in sorted(book_verse_counts.keys()):
        bv = book_verse_counts[book]
        book_lens = [bv[k] for k in sorted(bv.keys())]
        book_sizes.append(len(book_lens))
        book_means.append(float(np.mean(book_lens)))

    ref = {
        "n": len(at_lens),
        "mean": float(at_lens.mean()),
        "std": float(at_lens.std()),
        "cv": float(at_lens.std() / at_lens.mean()),
        "H": round(float(h_ref), 4) if not np.isnan(h_ref) else None,
        "AC1": round(float(ac1_ref), 4),
        "MPS_chi": chi_ref,
        "MPS_p": round(float(p_ref), 4) if p_ref is not None else None,
        "MPS_significant": bool(p_ref < 0.05) if p_ref is not None else None,
        "n_books": len(book_sizes),
        "book_size_mean": round(float(np.mean(book_sizes)), 1),
        "book_size_std": round(float(np.std(book_sizes)), 1),
        "book_mean_verse_len_mean": round(float(np.mean(book_means)), 2),
        "book_mean_verse_len_std": round(float(np.std(book_means)), 2),
    }

    log.info(f"  AT: n={ref['n']}, H={ref['H']}, AC(1)={ref['AC1']}, "
             f"MPS_χ={ref['MPS_chi']}, MPS_p={ref['MPS_p']}")

    return at_lens, ref


# ── Generative Models ────────────────────────────────────────────────────

def generate_ar1(n, mu, phi, sigma, rng):
    """AR(1): x_t = mu*(1-phi) + phi*x_{t-1} + noise"""
    series = np.zeros(n)
    series[0] = mu + rng.normal(0, sigma)
    for t in range(1, n):
        series[t] = mu * (1 - phi) + phi * series[t-1] + rng.normal(0, sigma)
    return np.maximum(series, 1)


def generate_hmm2(n, mu1, mu2, sigma1, sigma2, p_stay, rng):
    """HMM with 2 Gaussian states."""
    A = np.array([[p_stay, 1 - p_stay], [1 - p_stay, p_stay]])
    mus = [mu1, mu2]
    sigmas = [sigma1, sigma2]
    series = np.zeros(n)
    state = rng.choice(2)
    for t in range(n):
        series[t] = max(1, rng.normal(mus[state], sigmas[state]))
        state = rng.choice(2, p=A[state])
    return series


def generate_arfima(n, d, phi, sigma, mu, rng):
    """ARFIMA(1,d,0): fractional differencing + AR(1).
    Uses truncated MA(∞) representation."""
    # Generate fractional Gaussian noise via MA(∞) coefficients
    max_k = min(500, n)
    psi = np.zeros(max_k)
    psi[0] = 1.0
    for k in range(1, max_k):
        psi[k] = psi[k-1] * (k - 1 + d) / k

    # Generate white noise
    eps = rng.normal(0, sigma, n + max_k)

    # Convolve to get fractional noise
    frac_noise = np.zeros(n)
    for t in range(n):
        frac_noise[t] = sum(psi[k] * eps[t + max_k - k] for k in range(min(t + 1, max_k)))

    # Apply AR(1)
    series = np.zeros(n)
    series[0] = mu + frac_noise[0]
    for t in range(1, n):
        series[t] = mu * (1 - phi) + phi * series[t-1] + frac_noise[t]

    return np.maximum(series, 1)


def generate_hierarchical(n, mu_global, sigma_topic, tau_mean, sigma_verse, rng):
    """Hierarchical process: topics of variable duration with different means.
    Topic durations follow geometric(1/tau) distribution.
    Each topic has a mean drawn from N(mu_global, sigma_topic).
    Within topic, verses are N(topic_mean, sigma_verse)."""
    series = np.zeros(n)
    t = 0
    while t < n:
        topic_mean = max(3, rng.normal(mu_global, sigma_topic))
        # Topic duration: geometric distribution
        topic_len = max(3, rng.geometric(1.0 / tau_mean))
        for i in range(min(topic_len, n - t)):
            series[t + i] = max(1, rng.normal(topic_mean, sigma_verse))
        t += topic_len
    return series[:n]


# ── Model Fitting and Comparison ─────────────────────────────────────────

def evaluate_model(series, label, n_gen=100, ref_metrics=None):
    """Evaluate a generative model by computing H, AC(1), MPS on generated series."""
    arr = np.asarray(series, dtype=float)
    h = hurst_exponent_rs(arr)
    ac1 = autocorr_lag1(arr)

    result = {
        "H": round(float(h), 4) if not np.isnan(h) else None,
        "AC1": round(float(ac1), 4) if not np.isnan(ac1) else None,
        "mean": round(float(arr.mean()), 2),
        "std": round(float(arr.std()), 2),
    }

    return result


def run_model_comparison(at_lens, ref):
    """Compare 4 generative models."""
    log.info("\n── Comparación de modelos generativos ──")

    n = ref["n"]
    mu = ref["mean"]
    std_val = ref["std"]
    ac1_target = ref["AC1"]
    rng = np.random.default_rng(42)

    # Calibrate parameters from data
    # AR(1): phi = AC(1), sigma = std * sqrt(1 - phi^2)
    phi_ar1 = max(0.01, min(0.99, ac1_target))
    sigma_ar1 = std_val * np.sqrt(max(0.01, 1 - phi_ar1 ** 2))

    # HMM-2: estimate from data quartiles
    sorted_lens = np.sort(at_lens)
    q25 = sorted_lens[n // 4]
    q75 = sorted_lens[3 * n // 4]
    mu1_hmm = float(sorted_lens[:n//2].mean())
    mu2_hmm = float(sorted_lens[n//2:].mean())
    sigma1_hmm = float(sorted_lens[:n//2].std())
    sigma2_hmm = float(sorted_lens[n//2:].std())
    # p_stay calibrated to give mean regime length matching book sizes
    p_stay = 0.95  # ~20 verses per regime

    # ARFIMA: d estimated from H ≈ d + 0.5
    h_target = ref["H"] if ref["H"] else 0.88
    d_arfima = max(0.01, min(0.49, h_target - 0.5))
    phi_arfima = max(0.01, min(0.5, ac1_target * 0.5))
    sigma_arfima = std_val * 0.5

    # Hierarchical: calibrate from book-level stats
    sigma_topic = ref.get("book_mean_verse_len_std", 3.0)
    tau_mean = ref.get("book_size_mean", 50.0) / 5  # sub-book topic scale
    sigma_verse = std_val * 0.7

    models = {
        "AR1": {
            "params": {"mu": mu, "phi": phi_ar1, "sigma": sigma_ar1},
            "n_params": 2,
            "generator": lambda rng: generate_ar1(n, mu, phi_ar1, sigma_ar1, rng),
            "description": "Cada versículo depende linealmente del anterior",
            "cognitive_local_N": 1,
        },
        "HMM2": {
            "params": {"mu1": mu1_hmm, "mu2": mu2_hmm, "sigma1": sigma1_hmm,
                        "sigma2": sigma2_hmm, "p_stay": p_stay},
            "n_params": 5,
            "generator": lambda rng: generate_hmm2(n, mu1_hmm, mu2_hmm,
                                                    sigma1_hmm, sigma2_hmm, p_stay, rng),
            "description": "Alterna entre 2 regímenes (corto/largo)",
            "cognitive_local_N": 1,  # Only depends on current state
        },
        "ARFIMA": {
            "params": {"d": d_arfima, "phi": phi_arfima, "sigma": sigma_arfima, "mu": mu},
            "n_params": 3,
            "generator": lambda rng: generate_arfima(n, d_arfima, phi_arfima,
                                                      sigma_arfima, mu, rng),
            "description": "Memoria fraccional de largo alcance + AR(1)",
            "cognitive_local_N": float("inf"),  # Depends on full history
        },
        "Hierarchical": {
            "params": {"mu_global": mu, "sigma_topic": sigma_topic,
                        "tau_mean": tau_mean, "sigma_verse": sigma_verse},
            "n_params": 4,
            "generator": lambda rng: generate_hierarchical(n, mu, sigma_topic,
                                                            tau_mean, sigma_verse, rng),
            "description": "Temas de duración variable con medias distintas",
            "cognitive_local_N": 3,  # Depends on topic awareness (~3 verse context)
        },
    }

    # Generate and evaluate
    n_gen = 100
    results = {}

    for model_name, model in models.items():
        log.info(f"\n  Modelo {model_name} ({model['n_params']} params)...")

        h_values = []
        ac1_values = []
        mps_count = 0

        for i in range(n_gen):
            gen_rng = np.random.default_rng(42 + i)
            series = model["generator"](gen_rng)
            h = hurst_exponent_rs(series)
            ac1 = autocorr_lag1(series)

            if not np.isnan(h):
                h_values.append(float(h))
            if not np.isnan(ac1):
                ac1_values.append(float(ac1))

            # MPS test only on first 20 to save time
            if i < 20:
                _, p = compute_mps_significance(series, n_perm=50)
                if p is not None and p < 0.05:
                    mps_count += 1

        h_arr = np.array(h_values) if h_values else np.array([0.5])
        ac1_arr = np.array(ac1_values) if ac1_values else np.array([0.0])

        # Count how many reproduce all 3 criteria simultaneously
        h_ok = h_arr > 0.85
        ac1_ok = np.abs(ac1_arr[:len(h_ok)] - ref["AC1"]) < 0.15
        both_ok = h_ok & ac1_ok[:len(h_ok)]

        results[model_name] = {
            "n_params": model["n_params"],
            "description": model["description"],
            "cognitive_local_N": model["cognitive_local_N"],
            "params": {k: round(float(v), 4) if isinstance(v, (int, float, np.floating))
                       else v for k, v in model["params"].items()},
            "H_mean": round(float(h_arr.mean()), 4),
            "H_std": round(float(h_arr.std()), 4),
            "H_median": round(float(np.median(h_arr)), 4),
            "pct_H_above_0.85": round(float(np.mean(h_arr > 0.85) * 100), 1),
            "pct_H_above_0.9": round(float(np.mean(h_arr > 0.9) * 100), 1),
            "AC1_mean": round(float(ac1_arr.mean()), 4),
            "AC1_std": round(float(ac1_arr.std()), 4),
            "MPS_significant_pct": round(float(mps_count / 20 * 100), 1),
            "retrodiction_pct": round(float(np.mean(both_ok) * 100), 1),
        }

        log.info(f"    H={results[model_name]['H_mean']:.4f}±{results[model_name]['H_std']:.4f}, "
                 f"AC1={results[model_name]['AC1_mean']:.4f}, "
                 f"H>0.85: {results[model_name]['pct_H_above_0.85']}%, "
                 f"MPS sig: {results[model_name]['MPS_significant_pct']}%")

    return results, models


def minimum_complexity_analysis(model_results):
    """Determine minimum model complexity to reproduce AT metrics."""
    log.info("\n── Complejidad mínima ──")

    # Rank by retrodiction success
    ranked = sorted(model_results.items(),
                    key=lambda x: -x[1]["retrodiction_pct"])

    # Find minimum params that reproduce
    threshold = 10.0  # At least 10% retrodiction
    min_model = None
    for name, res in sorted(model_results.items(), key=lambda x: x[1]["n_params"]):
        if res["retrodiction_pct"] >= threshold:
            min_model = name
            break

    result = {
        "ranking_by_retrodiction": [
            {"model": name, "retrodiction_pct": res["retrodiction_pct"],
             "n_params": res["n_params"], "H_mean": res["H_mean"]}
            for name, res in ranked
        ],
        "minimum_model": min_model,
        "minimum_params": model_results[min_model]["n_params"] if min_model else None,
        "threshold_used": threshold,
    }

    if min_model:
        log.info(f"  Modelo mínimo: {min_model} ({model_results[min_model]['n_params']} params)")
    else:
        log.info("  Ningún modelo alcanza el umbral de retrodicción")

    return result


def cognitive_plausibility(model_results):
    """Test if the winning model is cognitively plausible (N ≤ 10)."""
    log.info("\n── Plausibilidad cognitiva ──")

    result = {}
    for name, res in model_results.items():
        local_n = res["cognitive_local_N"]
        plausible = local_n <= 10
        result[name] = {
            "local_N": local_n if local_n != float("inf") else "infinity",
            "within_working_memory": bool(plausible),
            "description": res["description"],
            "retrodiction_pct": res["retrodiction_pct"],
        }

        if plausible:
            log.info(f"  {name}: N={local_n} ≤ 10 → PLAUSIBLE "
                     f"(retrodicción: {res['retrodiction_pct']}%)")
        else:
            log.info(f"  {name}: N={local_n} > 10 → NO PLAUSIBLE "
                     f"(retrodicción: {res['retrodiction_pct']}%)")

    # Find best plausible model
    plausible_models = {k: v for k, v in result.items()
                        if v["within_working_memory"]}
    if plausible_models:
        best = max(plausible_models.items(),
                   key=lambda x: x[1]["retrodiction_pct"])
        result["best_plausible_model"] = best[0]
        result["best_plausible_retrodiction"] = best[1]["retrodiction_pct"]
    else:
        result["best_plausible_model"] = None

    return result


def retrodiction_test(at_lens, ref, best_model_name, models):
    """Full retrodiction: generate 100 texts and measure all 3 metrics."""
    log.info(f"\n── Test de retrodicción: {best_model_name} ──")

    if best_model_name not in models:
        return {"error": f"Model {best_model_name} not found"}

    model = models[best_model_name]
    n = ref["n"]

    results = []
    h_match = 0
    ac1_match = 0
    mps_match = 0
    all_match = 0

    for i in range(100):
        gen_rng = np.random.default_rng(1000 + i)
        series = model["generator"](gen_rng)

        h = hurst_exponent_rs(series)
        ac1 = autocorr_lag1(series)
        chi, p = compute_mps_significance(series, n_perm=50)

        h_ok = not np.isnan(h) and h > 0.85
        ac1_ok = not np.isnan(ac1) and abs(ac1 - ref["AC1"]) < 0.15
        mps_ok = p is not None and p < 0.05

        if h_ok:
            h_match += 1
        if ac1_ok:
            ac1_match += 1
        if mps_ok:
            mps_match += 1
        if h_ok and ac1_ok and mps_ok:
            all_match += 1

        results.append({
            "H": round(float(h), 4) if not np.isnan(h) else None,
            "AC1": round(float(ac1), 4) if not np.isnan(ac1) else None,
            "MPS_chi": chi,
            "MPS_p": round(float(p), 4) if p is not None else None,
            "H_match": bool(h_ok),
            "AC1_match": bool(ac1_ok),
            "MPS_match": bool(mps_ok),
            "all_match": bool(h_ok and ac1_ok and mps_ok),
        })

        if (i + 1) % 25 == 0:
            log.info(f"    {i+1}/100: H_match={h_match}, AC1_match={ac1_match}, "
                     f"MPS_match={mps_match}, ALL={all_match}")

    return {
        "model": best_model_name,
        "n_tests": 100,
        "H_match_pct": h_match,
        "AC1_match_pct": ac1_match,
        "MPS_match_pct": mps_match,
        "all_match_pct": all_match,
        "compositional_plausibility": f"{all_match}%",
        "reference": {
            "H": ref["H"],
            "AC1": ref["AC1"],
            "MPS_significant": ref["MPS_significant"],
        },
        "individual_results": results[:10],  # Save first 10 for inspection
    }


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    log.info("=" * 70)
    log.info("FASE 12 — Script 3: Compositional Rule")
    log.info("=" * 70)

    # Load reference
    at_lens, ref = load_at_reference()

    with open(RESULTS_DIR / "at_reference.json", "w") as f:
        json.dump(ref, f, indent=2, ensure_ascii=False)

    # Model comparison
    model_results, models = run_model_comparison(at_lens, ref)
    with open(RESULTS_DIR / "model_comparison.json", "w") as f:
        json.dump(model_results, f, indent=2, ensure_ascii=False)

    # Minimum complexity
    min_complexity = minimum_complexity_analysis(model_results)
    with open(RESULTS_DIR / "minimum_complexity.json", "w") as f:
        json.dump(min_complexity, f, indent=2, ensure_ascii=False)

    # Cognitive plausibility
    cog_result = cognitive_plausibility(model_results)
    with open(RESULTS_DIR / "cognitive_plausibility.json", "w") as f:
        json.dump(cog_result, f, indent=2, ensure_ascii=False)

    # Retrodiction with best model overall (not just plausible)
    best_overall = max(model_results.items(),
                       key=lambda x: x[1]["retrodiction_pct"])[0]
    best_plausible = cog_result.get("best_plausible_model", best_overall)
    test_model = best_plausible if best_plausible else best_overall

    retro = retrodiction_test(at_lens, ref, test_model, models)
    with open(RESULTS_DIR / "retrodiction_test.json", "w") as f:
        json.dump(retro, f, indent=2, ensure_ascii=False)

    # Also test best overall if different
    if test_model != best_overall:
        retro2 = retrodiction_test(at_lens, ref, best_overall, models)
        retro2["note"] = "Best overall model (may not be cognitively plausible)"
        with open(RESULTS_DIR / "retrodiction_best_overall.json", "w") as f:
            json.dump(retro2, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    log.info(f"\n{'=' * 70}")
    log.info(f"Script 3 completado en {elapsed:.1f}s")


if __name__ == "__main__":
    main()
