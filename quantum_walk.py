#!/usr/bin/env python3
"""
quantum_walk.py — Fase 4, Investigación 4
Quantum walk vs random walk sobre grafo de coocurrencia de lemas.
Todo numpy/scipy — cero frameworks cuánticos.
"""
import json, logging, time
from collections import Counter
from pathlib import Path
import numpy as np
from scipy import linalg as la

BASE = Path(__file__).parent
CORPUS = BASE / "bible_unified.json"
COOCCURRENCE = BASE / "results" / "cooccurrence" / "cooccurrence_analysis.json"
OUT = BASE / "results" / "qwalk"
OUT.mkdir(parents=True, exist_ok=True)
LOG = BASE / "logs"
LOG.mkdir(parents=True, exist_ok=True)

logging.basicConfig(filename=LOG / "quantum_walk.log", level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("qwalk")

TOP_N_LEMAS = 1000


def build_cooccurrence_graph(words, top_n=TOP_N_LEMAS):
    """Build adjacency matrix for top-n lemmas by cooccurrence."""
    # Get top-n lemmas by frequency
    lemma_freq = Counter(w["lemma"] for w in words)
    top_lemmas = [l for l, _ in lemma_freq.most_common(top_n)]
    lemma_to_idx = {l: i for i, l in enumerate(top_lemmas)}
    n = len(top_lemmas)

    # Group words by verse
    verses = {}
    for w in words:
        vk = (w["book_num"], w["chapter"], w["verse"])
        verses.setdefault(vk, set()).add(w["lemma"])

    # Count cooccurrences
    adj = np.zeros((n, n))
    for vk, lemmas in verses.items():
        present = [lemma_to_idx[l] for l in lemmas if l in lemma_to_idx]
        for i in range(len(present)):
            for j in range(i+1, len(present)):
                adj[present[i], present[j]] += 1
                adj[present[j], present[i]] += 1

    return adj, top_lemmas


def classical_stationary(adj):
    """Classical random walk stationary distribution: π[v] ∝ degree(v)."""
    degrees = np.sum(adj, axis=1)
    total = np.sum(degrees)
    if total == 0:
        return np.ones(len(adj)) / len(adj)
    return degrees / total


def quantum_walk_ctqw(adj, start_node, times, max_size=500):
    """Continuous-time quantum walk on graph.
    U(t) = exp(-iLt), p[v,t] = |<v|U(t)|start>|²
    For large graphs, use truncated eigendecomposition.
    """
    n = adj.shape[0]
    degrees = np.sum(adj, axis=1)
    D = np.diag(degrees)
    L = D - adj  # Graph Laplacian

    # Eigendecomposition of L (symmetric, real)
    log.info(f"  Eigendecomposition of {n}x{n} Laplacian...")
    if n <= max_size:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
    else:
        # For very large graphs, use only top eigenvalues
        from scipy.sparse.linalg import eigsh
        k = min(max_size, n - 1)
        eigenvalues, eigenvectors = eigsh(L, k=k, which='SM')

    # Initial state: |start>
    psi_0 = np.zeros(n)
    psi_0[start_node] = 1.0

    # Project onto eigenbasis
    coeffs = eigenvectors.T @ psi_0

    distributions = {}
    for t in times:
        # U(t)|psi_0> = Σ_k exp(-i λ_k t) c_k |φ_k>
        evolved_coeffs = coeffs * np.exp(-1j * eigenvalues * t)
        psi_t = eigenvectors @ evolved_coeffs
        prob_t = np.abs(psi_t) ** 2
        # Normalize (should be ~1 but numerical errors)
        prob_t = prob_t / np.sum(prob_t)
        distributions[t] = prob_t

    return distributions


def main():
    log.info("=== QUANTUM WALK — INICIO ===")
    t0 = time.time()

    with open(CORPUS, encoding="utf-8") as f:
        words = json.load(f)
    log.info(f"Corpus: {len(words)} palabras")

    times = [1, 2, 5, 10, 20, 50]

    for corpus_name in ["OT", "NT"]:
        log.info(f"=== {corpus_name} ===")
        cw = [w for w in words if w["corpus"] == corpus_name]

        # Build graph
        log.info(f"Building cooccurrence graph (top {TOP_N_LEMAS} lemmas)...")
        adj, lemma_list = build_cooccurrence_graph(cw, TOP_N_LEMAS)
        n = len(lemma_list)
        n_edges = int(np.sum(adj > 0) // 2)
        log.info(f"  Graph: {n} nodes, {n_edges} edges")

        # Classical stationary
        log.info("Classical stationary distribution...")
        pi_classical = classical_stationary(adj)

        # Quantum walk from most frequent lemma (node 0)
        log.info("Quantum walk (CTQW)...")
        start_node = 0  # Most frequent lemma
        qw_distributions = quantum_walk_ctqw(adj, start_node, times)

        # Time-averaged distribution
        all_probs = np.array([qw_distributions[t] for t in times])
        p_avg = np.mean(all_probs, axis=0)
        p_avg = p_avg / np.sum(p_avg)  # Normalize

        # Delta p
        delta_p = p_avg - pi_classical

        # Top positive and negative Δp
        sorted_idx = np.argsort(delta_p)
        top_positive = sorted_idx[-20:][::-1]
        top_negative = sorted_idx[:20]

        # Results
        classical_result = [
            {"lemma": lemma_list[i], "rank": i, "p_classical": round(float(pi_classical[i]), 8)}
            for i in range(min(50, n))
        ]

        quantum_result = {
            "start_node": lemma_list[start_node],
            "times": times,
            "time_averaged": [
                {"lemma": lemma_list[i], "rank": i,
                 "p_quantum": round(float(p_avg[i]), 8),
                 "p_classical": round(float(pi_classical[i]), 8),
                 "delta_p": round(float(delta_p[i]), 8)}
                for i in range(min(50, n))
            ],
        }

        delta_result = {
            "top20_positive": [
                {"lemma": lemma_list[i], "rank": int(i),
                 "delta_p": round(float(delta_p[i]), 8),
                 "p_quantum": round(float(p_avg[i]), 8),
                 "p_classical": round(float(pi_classical[i]), 8)}
                for i in top_positive
            ],
            "top20_negative": [
                {"lemma": lemma_list[i], "rank": int(i),
                 "delta_p": round(float(delta_p[i]), 8),
                 "p_quantum": round(float(p_avg[i]), 8),
                 "p_classical": round(float(pi_classical[i]), 8)}
                for i in top_negative
            ],
            "stats": {
                "mean_abs_delta": round(float(np.mean(np.abs(delta_p))), 8),
                "max_delta": round(float(np.max(delta_p)), 8),
                "min_delta": round(float(np.min(delta_p)), 8),
                "std_delta": round(float(np.std(delta_p)), 8),
            },
        }

        # Save per-corpus
        suffix = corpus_name.lower()
        with open(OUT / f"classical_stationary_{suffix}.json", "w", encoding="utf-8") as f:
            json.dump(classical_result, f, ensure_ascii=False, indent=2)
        with open(OUT / f"quantum_walk_distribution_{suffix}.json", "w", encoding="utf-8") as f:
            json.dump(quantum_result, f, ensure_ascii=False, indent=2)
        with open(OUT / f"delta_p_{suffix}.json", "w", encoding="utf-8") as f:
            json.dump(delta_result, f, ensure_ascii=False, indent=2)
        with open(OUT / f"top_quantum_lemas_{corpus_name}.json", "w", encoding="utf-8") as f:
            json.dump({
                "corpus": corpus_name,
                "top10_quantum_preferred": delta_result["top20_positive"][:10],
                "top10_quantum_avoided": delta_result["top20_negative"][:10],
            }, f, ensure_ascii=False, indent=2)

        log.info(f"  {corpus_name} done. Max Δp={np.max(delta_p):.6f}, "
                 f"top lemma: {lemma_list[top_positive[0]]}")

    elapsed = time.time() - t0
    log.info(f"=== DONE en {elapsed:.1f}s ===")
    print(f"[quantum_walk] DONE — {elapsed:.1f}s")


if __name__ == "__main__":
    main()
