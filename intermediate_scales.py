#!/usr/bin/env python3
"""
Fase 12 — Script 2: Intermediate Scales
¿Qué produce correlaciones a escalas intermedias que AR(1) no captura?

1. ACF profiles (lag 1-200) for 6 corpora
2. Power law fit: ACF(lag) ~ lag^(-β)
3. Block scale analysis: variance of H_local(w) vs w
4. Simple 2-state Gaussian HMM from scratch
5. Regime correspondence with known text divisions
"""

import json
import logging
import re
import time
import numpy as np
from pathlib import Path
from scipy import stats
from collections import defaultdict

BASE = Path(__file__).resolve().parent
RESULTS_DIR = BASE / "results" / "intermediate"
LOG_DIR = BASE / "logs"
BIBLE_JSON = BASE / "bible_unified.json"
QURAN_FILE = BASE / "results" / "comparison_corpora" / "quran_morphology.txt"
HOMER_FILES = [
    BASE / "results" / "comparison_corpora" / "homer_iliad.xml",
    BASE / "results" / "comparison_corpora" / "homer_odyssey.xml",
]
HERODOTUS_FILE = BASE / "results" / "comparison_corpora" / "herodotus.xml"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "fase12_intermediate.log"),
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
    slope, _, r, _, _ = stats.linregress(np.log(sizes), np.log(rs_values))
    return float(slope)


# ── Corpus loading ───────────────────────────────────────────────────────

def load_all_corpora():
    """Load verse-length series for all corpora."""
    log.info("Cargando todos los corpus...")

    # AT and NT from bible_unified.json
    with open(BIBLE_JSON, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    at_verses = defaultdict(int)
    nt_verses = defaultdict(int)
    at_books = defaultdict(lambda: defaultdict(int))

    for w in corpus:
        book = w.get("book", "")
        key = (book, w.get("chapter", 0), w.get("verse", 0))
        if book in OT_BOOKS:
            at_verses[key] += 1
            at_books[book][key] += 1
        else:
            nt_verses[key] += 1

    at_lens = [at_verses[k] for k in sorted(at_verses.keys())]
    nt_lens = [nt_verses[k] for k in sorted(nt_verses.keys())]

    # Book boundaries for AT (for section analysis)
    at_book_boundaries = {}
    idx = 0
    for key in sorted(at_verses.keys()):
        book = key[0]
        if book not in at_book_boundaries:
            at_book_boundaries[book] = {"start": idx}
        at_book_boundaries[book]["end"] = idx
        idx += 1

    log.info(f"  AT: {len(at_lens)} versículos, NT: {len(nt_lens)}")

    # Quran
    quran_lens = []
    if QURAN_FILE.exists():
        aya_words = defaultdict(set)
        with open(QURAN_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("LOCATION"):
                    continue
                match = re.match(r'\((\d+):(\d+):(\d+)(?::\d+)?\)', line)
                if match:
                    sura, aya, word = int(match.group(1)), int(match.group(2)), int(match.group(3))
                    aya_words[(sura, aya)].add(word)
        quran_lens = [len(aya_words[k]) for k in sorted(aya_words.keys())]
        log.info(f"  Corán: {len(quran_lens)} aleyas")

    # Homer
    import xml.etree.ElementTree as ET
    homer_lens = []
    for hf in HOMER_FILES:
        if not hf.exists():
            continue
        with open(hf, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        try:
            root = ET.fromstring(content)
            for sent in root.iter("sentence"):
                n_w = sum(1 for w in sent.iter("word") if w.get("form", "").strip())
                if n_w > 0:
                    homer_lens.append(n_w)
        except ET.ParseError:
            pass
    log.info(f"  Homero: {len(homer_lens)} sentencias")

    # Herodotus
    herodotus_lens = []
    if HERODOTUS_FILE.exists():
        with open(HERODOTUS_FILE, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        sentences = re.findall(r'<sentence[^>]*>(.*?)</sentence>', content, re.DOTALL)
        for sent in sentences:
            words = re.findall(r'<word[^>]*form="([^"]*)"', sent)
            if words:
                herodotus_lens.append(len(words))
    log.info(f"  Heródoto: {len(herodotus_lens)} sentencias")

    corpora = {
        "AT": at_lens,
        "NT": nt_lens,
        "Corán": quran_lens,
        "Homero": homer_lens,
        "Heródoto": herodotus_lens,
    }

    return corpora, at_book_boundaries


# ── ACF Profiles ─────────────────────────────────────────────────────────

def compute_acf_profile(series, max_lag=200):
    """Compute autocorrelation function for lags 1 to max_lag."""
    arr = np.asarray(series, dtype=float)
    n = len(arr)
    if n < max_lag + 10:
        max_lag = n // 2
    mean = arr.mean()
    var = arr.var()
    if var < 1e-10:
        return {}

    acf = {}
    for lag in range(1, max_lag + 1):
        if lag >= n - 5:
            break
        c = np.mean((arr[:-lag] - mean) * (arr[lag:] - mean))
        acf[lag] = round(float(c / var), 6)

    return acf


def fit_power_law_acf(acf_dict):
    """Fit ACF(lag) ~ lag^(-beta) to the positive portion of ACF."""
    lags = []
    values = []
    for lag, val in sorted(acf_dict.items()):
        if val > 0.01 and lag >= 2:  # Only positive values, skip lag 1
            lags.append(lag)
            values.append(val)

    if len(lags) < 5:
        return {"beta": None, "r2": None, "n_points": len(lags)}

    log_lags = np.log(np.array(lags))
    log_vals = np.log(np.array(values))
    slope, intercept, r, p, se = stats.linregress(log_lags, log_vals)

    return {
        "beta": round(float(-slope), 4),  # ACF ~ lag^(-beta), slope is negative
        "r2": round(float(r ** 2), 4),
        "n_points": len(lags),
        "max_lag_positive": max(lags) if lags else 0,
        "intercept": round(float(np.exp(intercept)), 4),
    }


def acf_analysis(corpora):
    """Compute ACF profiles and power law fits for all corpora."""
    log.info("\n── Perfiles de autocorrelación (lag 1-200) ──")

    profiles = {}
    power_law_fits = {}

    for label, lens in corpora.items():
        if not lens or len(lens) < 50:
            continue
        acf = compute_acf_profile(lens, max_lag=200)
        profiles[label] = acf

        # Find lag where ACF drops below 0.05
        drop_lag = None
        for lag in sorted(acf.keys()):
            if acf[lag] < 0.05:
                drop_lag = lag
                break

        pl = fit_power_law_acf(acf)
        pl["drop_below_0.05"] = drop_lag
        power_law_fits[label] = pl

        log.info(f"  {label}: β={pl['beta']}, R²={pl['r2']}, "
                 f"drop<0.05 at lag={drop_lag}, "
                 f"max_lag_positive={pl['max_lag_positive']}")

    return profiles, power_law_fits


# ── Block Scale Analysis ─────────────────────────────────────────────────

def block_scale_analysis(corpora):
    """Find the scale w* where variance of H_local is maximum."""
    log.info("\n── Análisis de escala de bloque ──")

    window_sizes = [5, 10, 15, 20, 30, 50, 75, 100, 150, 200]

    results = {}
    for label, lens in corpora.items():
        if not lens or len(lens) < 200:
            continue

        arr = np.array(lens, dtype=float)
        n = len(arr)

        scale_results = []
        for w in window_sizes:
            if w > n // 3:
                continue
            stride = max(1, w // 4)
            h_locals = []
            for start in range(0, n - w + 1, stride):
                seg = arr[start:start + w]
                h = hurst_exponent_rs(seg)
                if not np.isnan(h):
                    h_locals.append(h)

            if len(h_locals) >= 5:
                scale_results.append({
                    "window_size": w,
                    "n_windows": len(h_locals),
                    "H_local_mean": round(float(np.mean(h_locals)), 4),
                    "H_local_std": round(float(np.std(h_locals)), 4),
                    "H_local_var": round(float(np.var(h_locals)), 6),
                    "H_local_cv": round(float(np.std(h_locals) /
                                   (np.mean(h_locals) + 1e-10)), 4),
                })

        if scale_results:
            # Find w* (maximum variance)
            max_var_idx = np.argmax([s["H_local_var"] for s in scale_results])
            w_star = scale_results[max_var_idx]["window_size"]
            results[label] = {
                "scales": scale_results,
                "w_star": w_star,
                "w_star_var": scale_results[max_var_idx]["H_local_var"],
            }
            log.info(f"  {label}: w*={w_star} (max var={scale_results[max_var_idx]['H_local_var']:.6f})")

    return results


# ── Known Section Sizes ──────────────────────────────────────────────────

def known_section_sizes(at_book_boundaries):
    """Get sizes of known text divisions to compare with w*."""
    log.info("\n── Tamaños de secciones conocidas ──")

    # AT: Torah parashas average ~70-90 verses, prophetic readings ~50-70
    # But we can compute book sizes from boundaries
    book_sizes = {}
    for book, bounds in at_book_boundaries.items():
        size = bounds["end"] - bounds["start"] + 1
        book_sizes[book] = size

    sizes = list(book_sizes.values())
    torah_books = ["Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy"]
    torah_sizes = [book_sizes.get(b, 0) for b in torah_books if b in book_sizes]

    result = {
        "at_book_sizes": {
            "mean": round(float(np.mean(sizes)), 1),
            "median": round(float(np.median(sizes)), 1),
            "min": int(min(sizes)),
            "max": int(max(sizes)),
        },
        "torah_book_sizes": {
            "sizes": {b: book_sizes.get(b, 0) for b in torah_books},
            "mean": round(float(np.mean(torah_sizes)), 1) if torah_sizes else None,
        },
        "known_divisions": {
            "parasha_typical_verses": "54 parashot, ~55 verses each (Torah)",
            "haftarah_typical_verses": "~30-50 verses each",
            "quran_sura_mean_ayas": "~49 ayas (6236/114)",
        }
    }

    log.info(f"  AT book mean size: {result['at_book_sizes']['mean']} verses")
    return result


# ── 2-State Gaussian HMM (from scratch) ─────────────────────────────────

def gaussian_pdf(x, mu, sigma):
    """Gaussian probability density."""
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def hmm_2state_fit(series, n_iter=50, n_restarts=5):
    """Fit a 2-state Gaussian HMM using Baum-Welch (EM).
    Returns best fit across n_restarts random initializations."""
    arr = np.asarray(series, dtype=float)
    n = len(arr)
    if n < 50:
        return None

    best_ll = -np.inf
    best_params = None
    rng = np.random.default_rng(42)

    for restart in range(n_restarts):
        # Initialize
        sorted_arr = np.sort(arr)
        split = n // 2
        if restart == 0:
            mu = np.array([sorted_arr[:split].mean(), sorted_arr[split:].mean()])
            sigma = np.array([sorted_arr[:split].std() + 1e-6,
                              sorted_arr[split:].std() + 1e-6])
        else:
            mu = rng.choice(arr, 2)
            mu.sort()
            sigma = np.array([arr.std() * rng.uniform(0.3, 1.0),
                              arr.std() * rng.uniform(0.3, 1.0)])

        pi = np.array([0.5, 0.5])
        A = np.array([[0.9, 0.1], [0.1, 0.9]])

        for iteration in range(n_iter):
            # E-step: Forward
            alpha = np.zeros((n, 2))
            for k in range(2):
                alpha[0, k] = pi[k] * gaussian_pdf(arr[0], mu[k], sigma[k])
            s = alpha[0].sum()
            if s > 0:
                alpha[0] /= s

            for t in range(1, n):
                for k in range(2):
                    alpha[t, k] = (alpha[t-1] @ A[:, k]) * gaussian_pdf(arr[t], mu[k], sigma[k])
                s = alpha[t].sum()
                if s > 0:
                    alpha[t] /= s

            # E-step: Backward
            beta = np.zeros((n, 2))
            beta[-1] = 1.0
            for t in range(n - 2, -1, -1):
                for k in range(2):
                    beta[t, k] = sum(A[k, j] * gaussian_pdf(arr[t+1], mu[j], sigma[j]) *
                                     beta[t+1, j] for j in range(2))
                s = beta[t].sum()
                if s > 0:
                    beta[t] /= s

            # Gamma and xi
            gamma = alpha * beta
            gamma_sum = gamma.sum(axis=1, keepdims=True)
            gamma_sum[gamma_sum < 1e-300] = 1e-300
            gamma /= gamma_sum

            # M-step
            for k in range(2):
                nk = gamma[:, k].sum()
                if nk > 1e-10:
                    mu[k] = (gamma[:, k] * arr).sum() / nk
                    diff = arr - mu[k]
                    sigma[k] = max(1e-6, np.sqrt((gamma[:, k] * diff**2).sum() / nk))

            pi = gamma[0] / gamma[0].sum()

            # Update transition matrix
            for i in range(2):
                for j in range(2):
                    num = 0.0
                    den = 0.0
                    for t in range(n - 1):
                        xi_ij = (alpha[t, i] * A[i, j] *
                                 gaussian_pdf(arr[t+1], mu[j], sigma[j]) *
                                 beta[t+1, j])
                        num += xi_ij
                        den += gamma[t, i]
                    A[i, j] = num / (den + 1e-300)
                A[i] /= A[i].sum()

        # Log-likelihood approximation
        ll = np.sum(np.log(gamma.sum(axis=1) + 1e-300))

        if ll > best_ll:
            best_ll = ll
            best_params = {
                "mu": mu.copy(),
                "sigma": sigma.copy(),
                "A": A.copy(),
                "pi": pi.copy(),
                "gamma": gamma.copy(),
            }

    return best_params


def hmm_analysis(corpora):
    """Fit 2-state HMM to each corpus and analyze regimes."""
    log.info("\n── HMM 2-estados ──")

    results = {}
    for label, lens in corpora.items():
        if not lens or len(lens) < 100:
            continue

        log.info(f"  Ajustando HMM para {label}...")
        arr = np.array(lens, dtype=float)

        params = hmm_2state_fit(arr, n_iter=30, n_restarts=3)
        if params is None:
            results[label] = {"error": "insufficient data"}
            continue

        mu = params["mu"]
        sigma = params["sigma"]
        A = params["A"]
        gamma = params["gamma"]

        # Most likely state sequence
        states = gamma.argmax(axis=1)
        # Make state 0 the "short" state (lower mean)
        if mu[0] > mu[1]:
            mu = mu[::-1]
            sigma = sigma[::-1]
            states = 1 - states

        # Regime statistics
        regime_lens = [[], []]
        for s, val in zip(states, arr):
            regime_lens[s].append(val)

        # Transition statistics
        n_transitions = sum(1 for i in range(len(states)-1) if states[i] != states[i+1])
        mean_regime_len = len(states) / max(1, n_transitions)

        # H per regime
        h_regimes = []
        for r in range(2):
            r_vals = [arr[i] for i in range(len(arr)) if states[i] == r]
            if len(r_vals) >= 30:
                h = hurst_exponent_rs(r_vals)
                h_regimes.append(round(float(h), 4) if not np.isnan(h) else None)
            else:
                h_regimes.append(None)

        # Generate from HMM and measure H
        rng = np.random.default_rng(42)
        hmm_h_values = []
        for _ in range(100):
            synth = np.zeros(len(arr))
            state = rng.choice(2, p=params["pi"])
            for t in range(len(arr)):
                synth[t] = max(1, rng.normal(mu[state], sigma[state]))
                state = rng.choice(2, p=A[state])
            h = hurst_exponent_rs(synth)
            if not np.isnan(h):
                hmm_h_values.append(float(h))

        results[label] = {
            "mu": [round(float(mu[0]), 2), round(float(mu[1]), 2)],
            "sigma": [round(float(sigma[0]), 2), round(float(sigma[1]), 2)],
            "transition_matrix": [[round(float(A[i, j]), 4) for j in range(2)]
                                  for i in range(2)],
            "n_transitions": n_transitions,
            "mean_regime_length": round(mean_regime_len, 1),
            "regime_fraction": [round(float((states == 0).sum() / len(states)), 3),
                                round(float((states == 1).sum() / len(states)), 3)],
            "H_per_regime": h_regimes,
            "H_global": round(float(hurst_exponent_rs(arr)), 4),
            "H_from_hmm_synthetic": {
                "mean": round(float(np.mean(hmm_h_values)), 4) if hmm_h_values else None,
                "std": round(float(np.std(hmm_h_values)), 4) if hmm_h_values else None,
                "pct_above_0.9": round(float(np.mean(np.array(hmm_h_values) > 0.9) * 100), 1)
                    if hmm_h_values else None,
            },
        }

        log.info(f"    μ=[{mu[0]:.1f}, {mu[1]:.1f}], σ=[{sigma[0]:.1f}, {sigma[1]:.1f}]")
        log.info(f"    Transitions: {n_transitions}, mean regime len: {mean_regime_len:.1f}")
        log.info(f"    H_global={results[label]['H_global']}, "
                 f"H_hmm_synth={results[label]['H_from_hmm_synthetic']['mean']}")

    return results


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    log.info("=" * 70)
    log.info("FASE 12 — Script 2: Intermediate Scales")
    log.info("=" * 70)

    corpora, at_book_boundaries = load_all_corpora()

    # 1. ACF profiles
    profiles, power_law_fits = acf_analysis(corpora)
    with open(RESULTS_DIR / "acf_profiles.json", "w") as f:
        json.dump(profiles, f, indent=2, ensure_ascii=False)
    with open(RESULTS_DIR / "power_law_fits.json", "w") as f:
        json.dump(power_law_fits, f, indent=2, ensure_ascii=False)

    # 2. Block scale analysis
    block_results = block_scale_analysis(corpora)
    section_info = known_section_sizes(at_book_boundaries)
    block_results["known_sections"] = section_info
    with open(RESULTS_DIR / "block_scale_analysis.json", "w") as f:
        json.dump(block_results, f, indent=2, ensure_ascii=False)

    # 3. HMM analysis
    hmm_results = hmm_analysis(corpora)
    with open(RESULTS_DIR / "hmm_fit.json", "w") as f:
        json.dump(hmm_results, f, indent=2, ensure_ascii=False)

    # 4. Regime correspondence
    log.info("\n── Correspondencia de regímenes ──")
    regime_results = {}
    for label in corpora:
        if label in hmm_results and label in block_results:
            hmm = hmm_results[label]
            block = block_results[label]
            regime_results[label] = {
                "w_star": block["w_star"],
                "mean_regime_length_hmm": hmm.get("mean_regime_length"),
                "correspondence": (
                    "CLOSE" if hmm.get("mean_regime_length") and
                    abs(block["w_star"] - hmm["mean_regime_length"]) <
                    0.5 * block["w_star"]
                    else "DISTANT"
                ),
                "hmm_reproduces_H": (
                    hmm.get("H_from_hmm_synthetic", {}).get("pct_above_0.9", 0) > 10
                    if hmm.get("H_from_hmm_synthetic") else False
                ),
            }
            log.info(f"  {label}: w*={block['w_star']}, "
                     f"regime_len={hmm.get('mean_regime_length')}, "
                     f"HMM→H>0.9: {regime_results[label]['hmm_reproduces_H']}")

    with open(RESULTS_DIR / "regime_correspondence.json", "w") as f:
        json.dump(regime_results, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    log.info(f"\n{'=' * 70}")
    log.info(f"Script 2 completado en {elapsed:.1f}s")


if __name__ == "__main__":
    main()
