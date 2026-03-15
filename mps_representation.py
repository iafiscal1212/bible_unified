#!/usr/bin/env python3
"""
mps_representation.py — Fase 4, Investigación 1
Matrix Product State: bond dimension, correlaciones, reconstrucción.
Todo numpy/scipy — cero frameworks cuánticos.
"""
import json, logging, time
from pathlib import Path
import numpy as np
from scipy import linalg as la

BASE = Path(__file__).parent
CORPUS = BASE / "bible_unified.json"
OUT = BASE / "results" / "mps"
OUT.mkdir(parents=True, exist_ok=True)
LOG = BASE / "logs"
LOG.mkdir(parents=True, exist_ok=True)

logging.basicConfig(filename=LOG / "mps_representation.log", level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("mps_representation")


def build_verse_lengths(words, corpus_filter=None):
    """Build ordered series of verse lengths."""
    verses = {}
    for w in words:
        if corpus_filter and w["corpus"] != corpus_filter:
            continue
        key = (w["book_num"], w["chapter"], w["verse"])
        verses[key] = verses.get(key, 0) + 1
    return np.array([v for _, v in sorted(verses.items())], dtype=float)


def correlation_function(series, distances):
    """C(d) = <L[i]*L[i+d]> - <L>^2"""
    mean = np.mean(series)
    mean_sq = mean ** 2
    n = len(series)
    results = {}
    for d in distances:
        if d >= n:
            results[d] = None
            continue
        corr = np.mean(series[:n-d] * series[d:]) - mean_sq
        results[d] = round(float(corr), 6)
    return results


def build_transfer_matrix(series, max_lag=256):
    """Build correlation matrix (Toeplitz-like) from autocorrelation."""
    n = min(max_lag, len(series) // 2)
    mean = np.mean(series)
    var = np.var(series)
    if var == 0:
        return np.eye(n)

    centered = series - mean
    # Autocorrelation for lags 0..n-1
    acf = np.zeros(n)
    for lag in range(n):
        if lag >= len(centered):
            break
        acf[lag] = np.mean(centered[:len(centered)-lag] * centered[lag:])

    # Build Toeplitz correlation matrix
    T = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            lag = abs(i - j)
            if lag < n:
                T[i, j] = acf[lag]

    return T


def compute_bond_dimension(sigma, threshold=0.99):
    """Find minimal bond dimension chi for given variance threshold."""
    total = np.sum(sigma ** 2)
    if total == 0:
        return 1, 1.0
    cumulative = np.cumsum(sigma ** 2) / total
    chi = int(np.searchsorted(cumulative, threshold) + 1)
    return chi, float(cumulative[min(chi-1, len(cumulative)-1)])


def main():
    log.info("=== MPS REPRESENTATION — INICIO ===")
    t0 = time.time()

    with open(CORPUS, encoding="utf-8") as f:
        words = json.load(f)
    log.info(f"Corpus: {len(words)} palabras")

    distances = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    results_corr = {}
    results_svd = {}
    results_bond = {}
    results_recon = {}

    for corpus_name in ["OT", "NT", "global"]:
        log.info(f"Procesando {corpus_name}...")
        filt = None if corpus_name == "global" else corpus_name
        series = build_verse_lengths(words, filt)
        n = len(series)
        log.info(f"  {corpus_name}: {n} versículos, μ={np.mean(series):.2f}")

        # 1. Correlation function
        corr = correlation_function(series, distances)
        results_corr[corpus_name] = {
            "n_verses": n,
            "mean_length": round(float(np.mean(series)), 4),
            "variance": round(float(np.var(series)), 4),
            "correlations": {str(d): v for d, v in corr.items()},
        }

        # 2. Transfer matrix + SVD
        max_lag = min(256, n // 4)
        T = build_transfer_matrix(series, max_lag)
        U, sigma, Vt = la.svd(T, full_matrices=False)
        log.info(f"  SVD: {len(sigma)} valores singulares, max={sigma[0]:.4f}")

        results_svd[corpus_name] = {
            "matrix_size": max_lag,
            "top20_singular_values": [round(float(s), 6) for s in sigma[:20]],
            "total_variance": round(float(np.sum(sigma**2)), 4),
            "sigma_ratio_top1": round(float(sigma[0]**2 / np.sum(sigma**2)), 6) if np.sum(sigma**2) > 0 else 0,
        }

        # 3. Bond dimension at various thresholds
        for thresh in [0.90, 0.95, 0.99, 0.999]:
            chi, captured = compute_bond_dimension(sigma, thresh)
            if corpus_name not in results_bond:
                results_bond[corpus_name] = {}
            results_bond[corpus_name][f"chi_{int(thresh*1000)}"] = {
                "threshold": thresh,
                "bond_dimension": chi,
                "variance_captured": round(captured, 6),
            }
            log.info(f"  χ({thresh}) = {chi}")

        # 4. Reconstruction error
        chi_99 = results_bond[corpus_name]["chi_990"]["bond_dimension"]
        # Reconstruct T with rank chi_99
        T_reconstructed = U[:, :chi_99] @ np.diag(sigma[:chi_99]) @ Vt[:chi_99, :]
        recon_error = float(la.norm(T - T_reconstructed, 'fro') / la.norm(T, 'fro'))
        results_recon[corpus_name] = {
            "bond_dimension_used": chi_99,
            "relative_frobenius_error": round(recon_error, 8),
            "original_frobenius_norm": round(float(la.norm(T, 'fro')), 4),
        }
        log.info(f"  Reconstruction error (χ={chi_99}): {recon_error:.6f}")

    # Save
    log.info("Guardando resultados...")
    with open(OUT / "correlation_matrix.json", "w", encoding="utf-8") as f:
        json.dump(results_corr, f, ensure_ascii=False, indent=2)
    with open(OUT / "svd_spectrum.json", "w", encoding="utf-8") as f:
        json.dump(results_svd, f, ensure_ascii=False, indent=2)
    with open(OUT / "bond_dimension.json", "w", encoding="utf-8") as f:
        json.dump(results_bond, f, ensure_ascii=False, indent=2)
    with open(OUT / "reconstruction_error.json", "w", encoding="utf-8") as f:
        json.dump(results_recon, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t0
    log.info(f"=== DONE en {elapsed:.1f}s ===")
    print(f"[mps_representation] DONE — {elapsed:.1f}s")


if __name__ == "__main__":
    main()
