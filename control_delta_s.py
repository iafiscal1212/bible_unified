#!/usr/bin/env python3
"""
control_delta_s.py — Control crítico: ¿ΔS Von Neumann es artefacto o real?
4 tests independientes para separar efecto de dimensión de estructura real.
"""
import json, logging, time
from pathlib import Path
import numpy as np
from scipy import stats as sp_stats

BASE = Path(__file__).parent
CORPUS = BASE / "bible_unified.json"
OUT = BASE / "results" / "control_delta_s"
OUT.mkdir(parents=True, exist_ok=True)
LOG = BASE / "logs"
LOG.mkdir(parents=True, exist_ok=True)

logging.basicConfig(filename=LOG / "control_delta_s.log", level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("control_delta_s")

POS_CATEGORIES = ["noun", "verb", "pronoun", "adjective", "adverb",
                   "preposition", "conjunction", "particle", "other"]
N_POS = len(POS_CATEGORIES)


def pos_to_index(pos):
    try:
        return POS_CATEGORIES.index(pos)
    except ValueError:
        return POS_CATEGORIES.index("other")


def build_density_matrix(verse_vectors):
    """Build density matrix ρ from a list of POS frequency vectors."""
    d = verse_vectors.shape[1]
    rho = np.zeros((d, d))
    for vec in verse_vectors:
        norm = np.linalg.norm(vec)
        if norm > 0:
            v = vec / norm
            rho += np.outer(v, v)
    tr = np.trace(rho)
    if tr > 0:
        rho = rho / tr
    return rho


def von_neumann_entropy(rho):
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = eigenvalues[eigenvalues > 1e-15]
    if len(eigenvalues) == 0:
        return 0.0
    return float(-np.sum(eigenvalues * np.log2(eigenvalues)))


def shannon_entropy(probs):
    probs = probs[probs > 0]
    if len(probs) == 0:
        return 0.0
    return float(-np.sum(probs * np.log2(probs)))


def get_verse_pos_vectors(word_list, categories, cat_fn):
    """Build per-verse vectors for a given set of categories."""
    d = len(categories)
    verses = {}
    for w in word_list:
        vk = (w.get("book_num", 0), w["chapter"], w["verse"])
        verses.setdefault(vk, []).append(w)

    vectors = []
    for vk in sorted(verses.keys()):
        vec = np.zeros(d)
        for w in verses[vk]:
            idx = cat_fn(w)
            if 0 <= idx < d:
                vec[idx] += 1
        vectors.append(vec)
    return np.array(vectors)


def get_pos_marginal(word_list):
    """Get POS marginal distribution for a book."""
    counts = np.zeros(N_POS)
    for w in word_list:
        counts[pos_to_index(w["pos"])] += 1
    total = np.sum(counts)
    if total > 0:
        return counts / total
    return np.ones(N_POS) / N_POS


def main():
    log.info("=== CONTROL DELTA_S — INICIO ===")
    t0 = time.time()

    with open(CORPUS, encoding="utf-8") as f:
        words = json.load(f)
    log.info(f"Corpus: {len(words)} palabras")

    # Group by book
    books = {}
    book_order = []
    for w in words:
        if w["book"] not in books:
            books[w["book"]] = {"words": [], "corpus": w["corpus"], "book_num": w["book_num"]}
            book_order.append(w["book"])
        books[w["book"]]["words"].append(w)
    book_order.sort(key=lambda b: books[b]["book_num"])
    n_books = len(book_order)
    log.info(f"{n_books} libros")

    # Get top 50 lemmas globally for Test 4
    from collections import Counter
    lemma_freq = Counter(w["lemma"] for w in words)
    top50_lemmas = [l for l, _ in lemma_freq.most_common(50)]
    lemma_to_idx50 = {l: i for i, l in enumerate(top50_lemmas)}

    # ═══════════════════════════════════════════════════════════════
    # TEST 1 — Cota teórica
    # ═══════════════════════════════════════════════════════════════
    log.info("=== TEST 1: Cota teórica ===")
    theoretical_results = []

    for bk in book_order:
        bw = books[bk]["words"]
        corpus = books[bk]["corpus"]

        # Get verse vectors (d=9)
        vectors = get_verse_pos_vectors(bw, POS_CATEGORIES, lambda w: pos_to_index(w["pos"]))
        n_verses = len(vectors)

        # S_vN max = log2(min(n_verses, d))
        s_vn_max = np.log2(min(n_verses, N_POS))

        # S_Shannon from marginal POS
        marginal = get_pos_marginal(bw)
        s_shannon = shannon_entropy(marginal)

        # Actual S_vN
        rho = build_density_matrix(vectors)
        s_vn = von_neumann_entropy(rho)

        # Is S_Shannon > S_vN_max?
        ceiling_forces_negative = bool(s_shannon > s_vn_max)

        theoretical_results.append({
            "book": bk,
            "corpus": corpus,
            "n_verses": n_verses,
            "d": N_POS,
            "s_vn_max_theoretical": round(s_vn_max, 6),
            "s_shannon": round(s_shannon, 6),
            "s_vn_observed": round(s_vn, 6),
            "delta_s": round(s_vn - s_shannon, 6),
            "ceiling_forces_negative": ceiling_forces_negative,
        })

    n_forced = sum(1 for r in theoretical_results if r["ceiling_forces_negative"])
    log.info(f"  {n_forced}/{n_books} libros tienen S_Shannon > log2(d) → ΔS negativo inevitable")

    with open(OUT / "theoretical_ceiling.json", "w", encoding="utf-8") as f:
        json.dump({
            "results": theoretical_results,
            "summary": {
                "n_books": n_books,
                "n_ceiling_forced": n_forced,
                "fraction_forced": round(n_forced / n_books, 4),
                "log2_d": round(np.log2(N_POS), 6),
            }
        }, f, ensure_ascii=False, indent=2)

    # ═══════════════════════════════════════════════════════════════
    # TEST 2 — Control con texto aleatorio
    # ═══════════════════════════════════════════════════════════════
    log.info("=== TEST 2: Control aleatorio (1000 simulaciones/libro) ===")
    np.random.seed(42)
    N_SIM = 1000
    random_results = []
    last_log = time.time()

    for bi, bk in enumerate(book_order):
        bw = books[bk]["words"]
        corpus = books[bk]["corpus"]

        # Real ΔS
        vectors_real = get_verse_pos_vectors(bw, POS_CATEGORIES, lambda w: pos_to_index(w["pos"]))
        n_verses = len(vectors_real)
        rho_real = build_density_matrix(vectors_real)
        s_vn_real = von_neumann_entropy(rho_real)
        marginal = get_pos_marginal(bw)
        s_shannon_real = shannon_entropy(marginal)
        delta_s_real = s_vn_real - s_shannon_real

        # Verse sizes
        verse_sizes = {}
        for w in bw:
            vk = (w.get("book_num", 0), w["chapter"], w["verse"])
            verse_sizes[vk] = verse_sizes.get(vk, 0) + 1
        sizes = [verse_sizes[vk] for vk in sorted(verse_sizes.keys())]

        # Simulate
        delta_s_sims = []
        for sim in range(N_SIM):
            # Generate synthetic verses with same sizes and marginal POS distribution
            syn_vectors = np.zeros((n_verses, N_POS))
            for vi, sz in enumerate(sizes):
                # Draw sz POS tags from marginal distribution
                pos_draws = np.random.choice(N_POS, size=sz, p=marginal)
                for p in pos_draws:
                    syn_vectors[vi, p] += 1

            rho_syn = build_density_matrix(syn_vectors)
            s_vn_syn = von_neumann_entropy(rho_syn)

            # Shannon of the synthetic text (should be ≈ s_shannon_real since same marginal)
            syn_total = np.zeros(N_POS)
            for vi, sz in enumerate(sizes):
                syn_total += syn_vectors[vi]
            syn_marginal = syn_total / np.sum(syn_total) if np.sum(syn_total) > 0 else marginal
            s_shannon_syn = shannon_entropy(syn_marginal)

            delta_s_sims.append(s_vn_syn - s_shannon_syn)

        delta_s_sim_arr = np.array(delta_s_sims)
        # t-test: is delta_s_real significantly different from delta_s_sims?
        t_stat, t_pval = sp_stats.ttest_1samp(delta_s_sim_arr, delta_s_real)

        random_results.append({
            "book": bk,
            "corpus": corpus,
            "delta_s_observed": round(delta_s_real, 6),
            "delta_s_random_mean": round(float(np.mean(delta_s_sim_arr)), 6),
            "delta_s_random_std": round(float(np.std(delta_s_sim_arr)), 6),
            "delta_s_random_ci95": [
                round(float(np.percentile(delta_s_sim_arr, 2.5)), 6),
                round(float(np.percentile(delta_s_sim_arr, 97.5)), 6),
            ],
            "t_stat": round(float(t_stat), 4),
            "p_value_raw": round(float(t_pval), 8),
            "p_value_bonferroni": round(min(float(t_pval) * n_books, 1.0), 8),
            "significant_bonferroni": bool(t_pval * n_books < 0.05),
            "obs_more_negative": bool(delta_s_real < np.mean(delta_s_sim_arr)),
        })

        now = time.time()
        if now - last_log >= 15:
            log.info(f"  [{bi+1}/{n_books}] {bk}: ΔS_obs={delta_s_real:.4f}, "
                     f"ΔS_rand={np.mean(delta_s_sim_arr):.4f}, p_bonf={min(t_pval*n_books,1):.4f}")
            last_log = now

    n_sig = sum(1 for r in random_results if r["significant_bonferroni"])
    n_more_neg = sum(1 for r in random_results if r["obs_more_negative"])
    log.info(f"  {n_sig}/{n_books} significativos tras Bonferroni")
    log.info(f"  {n_more_neg}/{n_books} observados más negativos que aleatorio")

    with open(OUT / "random_control.json", "w", encoding="utf-8") as f:
        json.dump({
            "results": random_results,
            "summary": {
                "n_books": n_books,
                "n_simulations": N_SIM,
                "n_significant_bonferroni": n_sig,
                "n_obs_more_negative": n_more_neg,
                "fraction_significant": round(n_sig / n_books, 4),
            }
        }, f, ensure_ascii=False, indent=2)

    # ═══════════════════════════════════════════════════════════════
    # TEST 3 — Sensibilidad a la dimensión
    # ═══════════════════════════════════════════════════════════════
    log.info("=== TEST 3: Sensibilidad a la dimensión ===")

    # Define dimension mappings
    dim_configs = {
        2: {"name": "verb_vs_rest", "fn": lambda w: 0 if w["pos"] == "verb" else 1},
        3: {"name": "noun_verb_other", "fn": lambda w: 0 if w["pos"] == "noun" else (1 if w["pos"] == "verb" else 2)},
        5: {"name": "5cat", "fn": lambda w: (
            0 if w["pos"] == "noun" else
            1 if w["pos"] == "verb" else
            2 if w["pos"] == "pronoun" else
            3 if w["pos"] == "adjective" else 4
        )},
        7: {"name": "7cat", "fn": lambda w: (
            0 if w["pos"] == "noun" else
            1 if w["pos"] == "verb" else
            2 if w["pos"] == "pronoun" else
            3 if w["pos"] == "adjective" else
            4 if w["pos"] == "adverb" else
            5 if w["pos"] == "preposition" else 6
        )},
        9: {"name": "full_9cat", "fn": lambda w: pos_to_index(w["pos"])},
    }

    dim_results = {d: [] for d in dim_configs}

    for bk in book_order:
        bw = books[bk]["words"]
        corpus = books[bk]["corpus"]

        for d, config in dim_configs.items():
            categories = list(range(d))
            cat_fn = config["fn"]
            vectors = get_verse_pos_vectors(bw, categories, lambda w, cf=cat_fn: cf(w))

            rho = build_density_matrix(vectors)
            s_vn = von_neumann_entropy(rho)

            # Shannon for this dimensionality
            counts = np.zeros(d)
            for w in bw:
                idx = cat_fn(w)
                if 0 <= idx < d:
                    counts[idx] += 1
            total = np.sum(counts)
            probs = counts / total if total > 0 else np.ones(d) / d
            s_sh = shannon_entropy(probs)

            dim_results[d].append({
                "book": bk,
                "corpus": corpus,
                "d": d,
                "s_vn": round(s_vn, 6),
                "s_shannon": round(s_sh, 6),
                "delta_s": round(s_vn - s_sh, 6),
            })

    # Compute mean ΔS per dimension
    dim_summary = {}
    for d in dim_configs:
        deltas = [r["delta_s"] for r in dim_results[d]]
        dim_summary[str(d)] = {
            "name": dim_configs[d]["name"],
            "mean_delta_s": round(float(np.mean(deltas)), 6),
            "std_delta_s": round(float(np.std(deltas)), 6),
            "min_delta_s": round(float(np.min(deltas)), 6),
            "max_delta_s": round(float(np.max(deltas)), 6),
            "log2_d": round(np.log2(d), 4),
        }
        log.info(f"  d={d}: mean ΔS = {np.mean(deltas):.4f} ± {np.std(deltas):.4f}")

    with open(OUT / "dimension_sensitivity.json", "w", encoding="utf-8") as f:
        json.dump({
            "per_book": {str(d): dim_results[d] for d in dim_configs},
            "summary": dim_summary,
        }, f, ensure_ascii=False, indent=2)

    # ═══════════════════════════════════════════════════════════════
    # TEST 4 — Dimensión expandida (d=50 lemas)
    # ═══════════════════════════════════════════════════════════════
    log.info("=== TEST 4: Dimensión expandida (d=50 lemas) ===")
    D_HIGH = 50
    high_dim_results = []

    for bk in book_order:
        bw = books[bk]["words"]
        corpus = books[bk]["corpus"]

        # Verse vectors with top-50 lemmas
        verses = {}
        for w in bw:
            vk = (w.get("book_num", 0), w["chapter"], w["verse"])
            verses.setdefault(vk, []).append(w)

        vectors = np.zeros((len(verses), D_HIGH))
        for vi, vk in enumerate(sorted(verses.keys())):
            for w in verses[vk]:
                idx = lemma_to_idx50.get(w["lemma"])
                if idx is not None:
                    vectors[vi, idx] += 1

        rho = build_density_matrix(vectors)
        s_vn = von_neumann_entropy(rho)

        # Shannon: marginal distribution over 50 lemmas
        counts = np.zeros(D_HIGH)
        for w in bw:
            idx = lemma_to_idx50.get(w["lemma"])
            if idx is not None:
                counts[idx] += 1
        total = np.sum(counts)
        probs = counts / total if total > 0 else np.ones(D_HIGH) / D_HIGH
        s_sh = shannon_entropy(probs)

        delta_s = s_vn - s_sh

        high_dim_results.append({
            "book": bk,
            "corpus": corpus,
            "d": D_HIGH,
            "s_vn": round(s_vn, 6),
            "s_shannon": round(s_sh, 6),
            "delta_s": round(delta_s, 6),
        })

    ot_deltas_high = [r["delta_s"] for r in high_dim_results if r["corpus"] == "OT"]
    nt_deltas_high = [r["delta_s"] for r in high_dim_results if r["corpus"] == "NT"]

    if ot_deltas_high and nt_deltas_high:
        u_stat, u_pval = sp_stats.mannwhitneyu(ot_deltas_high, nt_deltas_high, alternative='two-sided')
    else:
        u_stat, u_pval = 0, 1

    high_dim_summary = {
        "d": D_HIGH,
        "ot_mean_delta_s": round(float(np.mean(ot_deltas_high)), 6),
        "nt_mean_delta_s": round(float(np.mean(nt_deltas_high)), 6),
        "all_negative": bool(all(r["delta_s"] < 0 for r in high_dim_results)),
        "any_positive": bool(any(r["delta_s"] >= 0 for r in high_dim_results)),
        "mannwhitney_U": round(float(u_stat), 2),
        "mannwhitney_p": round(float(u_pval), 8),
        "ot_nt_significant": bool(u_pval < 0.05),
    }
    log.info(f"  d=50: OT mean ΔS = {np.mean(ot_deltas_high):.4f}, "
             f"NT mean ΔS = {np.mean(nt_deltas_high):.4f}, p = {u_pval:.6f}")

    with open(OUT / "high_dim_delta_s.json", "w", encoding="utf-8") as f:
        json.dump({
            "results": high_dim_results,
            "summary": high_dim_summary,
        }, f, ensure_ascii=False, indent=2)

    # ═══════════════════════════════════════════════════════════════
    # VERDICT — Síntesis
    # ═══════════════════════════════════════════════════════════════
    log.info("=== VEREDICTO ===")
    verdicts = []

    for bi, bk in enumerate(book_order):
        # Gather evidence
        ceiling = theoretical_results[bi]["ceiling_forces_negative"]
        random_sig = random_results[bi]["significant_bonferroni"]
        random_more_neg = random_results[bi]["obs_more_negative"]
        delta_obs = random_results[bi]["delta_s_observed"]
        delta_rand = random_results[bi]["delta_s_random_mean"]

        if not ceiling and random_sig and random_more_neg:
            verdict = "real"
        elif ceiling and not random_sig:
            verdict = "artefacto"
        elif ceiling and random_sig and random_more_neg:
            verdict = "real_beyond_ceiling"
        else:
            verdict = "indeterminado"

        verdicts.append({
            "book": bk,
            "corpus": books[bk]["corpus"],
            "verdict": verdict,
            "ceiling_forced": ceiling,
            "random_significant": random_sig,
            "delta_s_observed": delta_obs,
            "delta_s_random_mean": delta_rand,
        })

    verdict_counts = {}
    for v in verdicts:
        verdict_counts[v["verdict"]] = verdict_counts.get(v["verdict"], 0) + 1

    # Dimension sensitivity: does ΔS get less negative with higher d?
    dim_means = [dim_summary[str(d)]["mean_delta_s"] for d in sorted(dim_configs.keys())]
    dim_trend = "less_negative_with_d" if all(dim_means[i] <= dim_means[i+1] for i in range(len(dim_means)-1)) else "non_monotonic"

    # High dim: does sign change?
    sign_changes_d50 = high_dim_summary["any_positive"]
    ot_nt_persists_d50 = high_dim_summary["ot_nt_significant"]

    global_verdict = {
        "per_book": verdicts,
        "counts": verdict_counts,
        "dimension_sensitivity": {
            "trend": dim_trend,
            "mean_delta_by_d": {str(d): dim_summary[str(d)]["mean_delta_s"] for d in sorted(dim_configs.keys())},
        },
        "high_dim_d50": {
            "sign_changes": sign_changes_d50,
            "ot_nt_difference_persists": ot_nt_persists_d50,
        },
        "global_conclusion": "",
    }

    # Build conclusion
    n_real = verdict_counts.get("real", 0) + verdict_counts.get("real_beyond_ceiling", 0)
    n_artefacto = verdict_counts.get("artefacto", 0)
    n_indet = verdict_counts.get("indeterminado", 0)

    conclusion_parts = []
    conclusion_parts.append(
        f"De {n_books} libros: {n_real} con efecto REAL, "
        f"{n_artefacto} ARTEFACTO puro, {n_indet} indeterminados."
    )

    n_forced_pct = round(100 * sum(1 for r in theoretical_results if r["ceiling_forces_negative"]) / n_books, 1)
    conclusion_parts.append(
        f"Test 1 (cota teórica): {n_forced_pct}% de libros tienen S_Shannon > log2(9) → "
        f"ΔS negativo es parcialmente inevitable por la baja dimensión."
    )

    n_sig_pct = round(100 * sum(1 for r in random_results if r["significant_bonferroni"]) / n_books, 1)
    conclusion_parts.append(
        f"Test 2 (control aleatorio): {n_sig_pct}% de libros tienen ΔS observado "
        f"significativamente diferente del aleatorio (Bonferroni)."
    )

    conclusion_parts.append(
        f"Test 3 (sensibilidad): ΔS medio por d: "
        + ", ".join(f"d={d}→{dim_summary[str(d)]['mean_delta_s']:.3f}" for d in sorted(dim_configs.keys()))
        + f". Tendencia: {dim_trend}."
    )

    conclusion_parts.append(
        f"Test 4 (d=50 lemas): {'El signo NO cambia' if not sign_changes_d50 else 'Algunos positivos'}. "
        f"Diferencia AT/NT {'PERSISTE' if ot_nt_persists_d50 else 'desaparece'} (p={u_pval:.4f})."
    )

    global_verdict["global_conclusion"] = " | ".join(conclusion_parts)
    log.info(f"  {global_verdict['global_conclusion']}")

    with open(OUT / "verdict.json", "w", encoding="utf-8") as f:
        json.dump(global_verdict, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t0
    log.info(f"=== DONE en {elapsed:.1f}s ===")
    print(f"[control_delta_s] DONE — {elapsed:.1f}s")


if __name__ == "__main__":
    main()
