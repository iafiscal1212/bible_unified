#!/usr/bin/env python3
"""
mps_compression_ratio.py — Fase 4, Investigación 5
Compresión MPS vs clásica + permutation test.
Todo numpy/scipy — cero frameworks cuánticos.
"""
import json, logging, time
from collections import Counter
from pathlib import Path
import numpy as np
from scipy import linalg as la

BASE = Path(__file__).parent
CORPUS = BASE / "bible_unified.json"
OUT = BASE / "results" / "mps_compression"
OUT.mkdir(parents=True, exist_ok=True)
LOG = BASE / "logs"
LOG.mkdir(parents=True, exist_ok=True)

logging.basicConfig(filename=LOG / "mps_compression_ratio.log", level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("mps_compression")


def build_verse_lengths(words, corpus_filter=None):
    verses = {}
    for w in words:
        if corpus_filter and w["corpus"] != corpus_filter:
            continue
        key = (w["book_num"], w["chapter"], w["verse"])
        verses[key] = verses.get(key, 0) + 1
    return np.array([v for _, v in sorted(verses.items())], dtype=float)


def shannon_entropy_bits(series):
    """Shannon entropy in bits."""
    counts = Counter(series.astype(int))
    total = len(series)
    probs = np.array(list(counts.values()), dtype=float) / total
    return float(-np.sum(probs * np.log2(probs)))


def compute_bond_dimension(series, max_lag=256, threshold=0.99):
    """Compute bond dimension χ from autocorrelation matrix SVD."""
    n = min(max_lag, len(series) // 4)
    if n < 2:
        return 1, np.array([1.0])

    mean = np.mean(series)
    centered = series - mean

    # Autocorrelation
    acf = np.zeros(n)
    for lag in range(n):
        if lag >= len(centered):
            break
        acf[lag] = np.mean(centered[:len(centered)-lag] * centered[lag:])

    # Toeplitz matrix
    T = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            lag = abs(i - j)
            if lag < n:
                T[i, j] = acf[lag]

    _, sigma, _ = la.svd(T, full_matrices=False)

    total = np.sum(sigma ** 2)
    if total == 0:
        return 1, sigma

    cumulative = np.cumsum(sigma ** 2) / total
    chi = int(np.searchsorted(cumulative, threshold) + 1)
    return chi, sigma


def main():
    log.info("=== MPS COMPRESSION RATIO — INICIO ===")
    t0 = time.time()

    with open(CORPUS, encoding="utf-8") as f:
        words = json.load(f)
    log.info(f"Corpus: {len(words)} palabras")

    results = {}
    perm_results = {}

    for corpus_name in ["OT", "NT", "global"]:
        log.info(f"=== {corpus_name} ===")
        filt = None if corpus_name == "global" else corpus_name
        series = build_verse_lengths(words, filt)
        n = len(series)
        max_val = int(np.max(series))

        # 1. Direct representation: N * log2(max_L) bits
        bits_direct = n * np.log2(max_val) if max_val > 1 else n

        # 2. Classical compression: Shannon entropy * N bits
        h_shannon = shannon_entropy_bits(series)
        bits_classical = h_shannon * n

        # 3. MPS compression: χ² × d × N parameters
        chi, sigma = compute_bond_dimension(series, max_lag=256, threshold=0.99)
        d = max_val  # local dimension = number of distinct values
        d_unique = len(set(series.astype(int)))
        # MPS parameters: χ² * d_local * N_sites
        # But we use the effective local dimension (unique values)
        params_mps = chi * chi * d_unique
        # Each parameter needs ~64 bits (float64), but for fair comparison
        # we count in bits per site
        bits_mps_per_site = np.log2(params_mps) if params_mps > 1 else 1
        bits_mps = bits_mps_per_site * n

        # Compression ratios
        ratio_classical = bits_direct / bits_classical if bits_classical > 0 else 1
        ratio_mps = bits_direct / bits_mps if bits_mps > 0 else 1

        results[corpus_name] = {
            "n_verses": n,
            "max_verse_length": max_val,
            "unique_lengths": d_unique,
            "bond_dimension_chi": chi,
            "shannon_entropy_bits": round(h_shannon, 4),
            "bits_direct": round(float(bits_direct), 2),
            "bits_classical": round(float(bits_classical), 2),
            "bits_mps": round(float(bits_mps), 2),
            "ratio_direct_vs_classical": round(float(ratio_classical), 4),
            "ratio_direct_vs_mps": round(float(ratio_mps), 4),
            "ratio_classical_vs_mps": round(float(bits_classical / bits_mps), 4) if bits_mps > 0 else None,
            "mps_more_compressible": bool(bits_mps < bits_classical),
        }
        log.info(f"  χ={chi}, H={h_shannon:.4f}, "
                 f"bits: direct={bits_direct:.0f}, classical={bits_classical:.0f}, mps={bits_mps:.0f}")

        # 4. Permutation test for bond dimension
        log.info(f"  Permutation test (n=10000) for χ...")
        np.random.seed(42)
        n_perm = 10000
        chi_permutations = []

        last_log = time.time()
        for i in range(n_perm):
            shuffled = np.random.permutation(series)
            chi_shuf, _ = compute_bond_dimension(shuffled, max_lag=64, threshold=0.99)
            chi_permutations.append(chi_shuf)

            now = time.time()
            if now - last_log >= 30:
                log.info(f"    Permutation {i+1}/{n_perm}")
                last_log = now

        chi_perm_arr = np.array(chi_permutations)
        # P-value: fraction of permutations with χ ≤ observed χ
        # Lower χ = more compressible = more structured
        # So we want: is real χ significantly LOWER than random?
        chi_real, _ = compute_bond_dimension(series, max_lag=64, threshold=0.99)
        p_value = float(np.mean(chi_perm_arr <= chi_real))

        perm_results[corpus_name] = {
            "chi_observed": chi_real,
            "chi_permutation_mean": round(float(np.mean(chi_perm_arr)), 2),
            "chi_permutation_std": round(float(np.std(chi_perm_arr)), 2),
            "chi_permutation_median": round(float(np.median(chi_perm_arr)), 2),
            "chi_permutation_ci_95": [
                round(float(np.percentile(chi_perm_arr, 2.5)), 2),
                round(float(np.percentile(chi_perm_arr, 97.5)), 2),
            ],
            "p_value": round(p_value, 6),
            "significant": bool(p_value < 0.05),
            "interpretation": (
                f"Real χ={chi_real} is {'significantly lower' if p_value < 0.05 else 'not significantly different'} "
                f"than random (mean χ={np.mean(chi_perm_arr):.1f}, p={p_value:.4f}). "
                f"{'The corpus has more MPS-compressible structure than random.' if p_value < 0.05 else 'No evidence of extra MPS compressibility.'}"
            ),
        }
        log.info(f"  Permutation: χ_obs={chi_real}, χ_rand={np.mean(chi_perm_arr):.1f}±{np.std(chi_perm_arr):.1f}, p={p_value:.4f}")

    # Bits comparison summary
    bits_comparison = {
        corpus: {
            "direct": results[corpus]["bits_direct"],
            "classical_shannon": results[corpus]["bits_classical"],
            "mps": results[corpus]["bits_mps"],
            "best": "mps" if results[corpus]["mps_more_compressible"] else "classical",
        }
        for corpus in results
    }

    # Save
    log.info("Guardando resultados...")
    with open(OUT / "compression_ratios.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(OUT / "permutation_test_chi.json", "w", encoding="utf-8") as f:
        json.dump(perm_results, f, ensure_ascii=False, indent=2)
    with open(OUT / "bits_comparison.json", "w", encoding="utf-8") as f:
        json.dump(bits_comparison, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t0
    log.info(f"=== DONE en {elapsed:.1f}s ===")
    print(f"[mps_compression_ratio] DONE — {elapsed:.1f}s")


if __name__ == "__main__":
    main()
