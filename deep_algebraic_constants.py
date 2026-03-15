#!/usr/bin/env python3
"""
deep_algebraic_constants.py — Fase 3, Investigación 2
¿Los ratios AT/NT ≈ √5, √2 sobreviven corrección de Bonferroni?
Búsqueda sistemática en espacio de constantes algebraicas.
"""
import json, logging, math, time, itertools
from collections import Counter
from pathlib import Path
import numpy as np
from scipy import stats as sp_stats

BASE = Path(__file__).parent
CORPUS = BASE / "bible_unified.json"
OUT = BASE / "results" / "deep_algebraic"
OUT.mkdir(parents=True, exist_ok=True)
LOG = BASE / "logs"
LOG.mkdir(parents=True, exist_ok=True)

logging.basicConfig(filename=LOG / "deep_algebraic_constants.log", level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("deep_algebraic_constants")


def build_constant_space():
    """Build systematic space of algebraic/transcendental constants."""
    constants = {}

    # Square roots: √n for n in 1..20
    for n in range(1, 21):
        constants[f"sqrt({n})"] = math.sqrt(n)

    # Cube roots: n^(1/3) for n in 1..20
    for n in range(1, 21):
        constants[f"cbrt({n})"] = n ** (1/3)

    # Combinations: (√n + √m)/k for n,m in 1..10, k in 1..5
    for n in range(1, 11):
        for m in range(n, 11):  # n <= m to avoid duplicates
            for k in range(1, 6):
                val = (math.sqrt(n) + math.sqrt(m)) / k
                constants[f"(sqrt({n})+sqrt({m}))/{k}"] = val

    # Transcendental constants
    constants["pi"] = math.pi
    constants["e"] = math.e
    constants["phi"] = (1 + math.sqrt(5)) / 2
    constants["ln(2)"] = math.log(2)
    constants["ln(3)"] = math.log(3)
    constants["1/pi"] = 1 / math.pi
    constants["1/e"] = 1 / math.e
    constants["pi/e"] = math.pi / math.e
    constants["e/pi"] = math.e / math.pi
    constants["2*pi"] = 2 * math.pi
    constants["pi^2"] = math.pi ** 2
    constants["e^2"] = math.e ** 2

    # Remove duplicates (same value within 1e-10)
    unique = {}
    for name, val in sorted(constants.items()):
        is_dup = False
        for existing_val in unique.values():
            if abs(val - existing_val) < 1e-10:
                is_dup = True
                break
        if not is_dup:
            unique[name] = val

    return unique


def main():
    log.info("=== DEEP ALGEBRAIC CONSTANTS — INICIO ===")
    t0 = time.time()

    with open(CORPUS, encoding="utf-8") as f:
        words = json.load(f)
    log.info(f"Corpus: {len(words)} palabras")

    # Compute observed ratios
    ot = [w for w in words if w["corpus"] == "OT"]
    nt = [w for w in words if w["corpus"] == "NT"]

    ot_books = set(w["book"] for w in ot)
    nt_books = set(w["book"] for w in nt)

    ot_chapters = set((w["book"], w["chapter"]) for w in ot)
    nt_chapters = set((w["book"], w["chapter"]) for w in nt)

    ot_verses = set((w["book"], w["chapter"], w["verse"]) for w in ot)
    nt_verses = set((w["book"], w["chapter"], w["verse"]) for w in nt)

    observed_ratios = {
        "words": len(ot) / len(nt),
        "verses": len(ot_verses) / len(nt_verses),
        "chapters": len(ot_chapters) / len(nt_chapters),
        "books": len(ot_books) / len(nt_books),
    }
    log.info(f"Ratios observados: {observed_ratios}")

    # Build constant space
    constants = build_constant_space()
    n_constants = len(constants)
    n_ratios = len(observed_ratios)
    n_comparisons = n_constants * n_ratios
    log.info(f"Espacio de constantes: {n_constants}, comparaciones totales: {n_comparisons}")

    # === 1. All distances ===
    log.info("Calculando todas las distancias...")
    all_distances = {}

    for ratio_name, ratio_val in observed_ratios.items():
        distances = []
        for const_name, const_val in constants.items():
            dist = abs(ratio_val - const_val)
            distances.append({
                "constant": const_name,
                "constant_value": round(const_val, 8),
                "distance": round(dist, 8),
            })
        distances.sort(key=lambda x: x["distance"])
        all_distances[ratio_name] = {
            "observed": round(ratio_val, 8),
            "top10_nearest": distances[:10],
        }

    # === 2. Bonferroni correction ===
    log.info("Aplicando corrección de Bonferroni...")
    bonferroni_results = {}

    for ratio_name, ratio_val in observed_ratios.items():
        # For significance: we need the bootstrap distribution of each ratio
        # to compute a p-value for each distance
        # We'll do this in the bootstrap section and add Bonferroni there
        nearest = all_distances[ratio_name]["top10_nearest"][0]
        bonferroni_results[ratio_name] = {
            "observed": round(ratio_val, 8),
            "nearest_constant": nearest["constant"],
            "nearest_value": nearest["constant_value"],
            "raw_distance": nearest["distance"],
            "n_comparisons": n_comparisons,
            "bonferroni_note": "p-values computed via bootstrap below",
        }

    # === 3. Bootstrap stability ===
    log.info("Bootstrap (n=10000) para estabilidad de ratios...")
    np.random.seed(42)
    n_bootstrap = 10000

    # Bootstrap: resample books, recompute word counts
    ot_book_words = {}
    for w in ot:
        ot_book_words.setdefault(w["book"], []).append(w)
    nt_book_words = {}
    for w in nt:
        nt_book_words.setdefault(w["book"], []).append(w)

    ot_book_list = list(ot_book_words.keys())
    nt_book_list = list(nt_book_words.keys())

    # Precompute per-book stats
    ot_book_stats = {}
    for bk in ot_book_list:
        bw = ot_book_words[bk]
        ot_book_stats[bk] = {
            "words": len(bw),
            "verses": len(set((w["chapter"], w["verse"]) for w in bw)),
            "chapters": len(set(w["chapter"] for w in bw)),
        }

    nt_book_stats = {}
    for bk in nt_book_list:
        bw = nt_book_words[bk]
        nt_book_stats[bk] = {
            "words": len(bw),
            "verses": len(set((w["chapter"], w["verse"]) for w in bw)),
            "chapters": len(set(w["chapter"] for w in bw)),
        }

    bootstrap_ratios = {"words": [], "verses": [], "chapters": [], "books": []}

    for i in range(n_bootstrap):
        # Resample books with replacement
        ot_sample = np.random.choice(ot_book_list, size=len(ot_book_list), replace=True)
        nt_sample = np.random.choice(nt_book_list, size=len(nt_book_list), replace=True)

        for metric in ["words", "verses", "chapters"]:
            ot_sum = sum(ot_book_stats[bk][metric] for bk in ot_sample)
            nt_sum = sum(nt_book_stats[bk][metric] for bk in nt_sample)
            if nt_sum > 0:
                bootstrap_ratios[metric].append(ot_sum / nt_sum)

        bootstrap_ratios["books"].append(len(set(ot_sample)) / len(set(nt_sample)))

        if (i + 1) % 2000 == 0:
            log.info(f"  Bootstrap {i+1}/{n_bootstrap}")

    bootstrap_stability = {}
    for ratio_name, ratio_val in observed_ratios.items():
        samples = np.array(bootstrap_ratios[ratio_name])
        ci_low, ci_high = np.percentile(samples, [2.5, 97.5])
        mean_boot = np.mean(samples)
        std_boot = np.std(samples)

        # P-value for distance to nearest constant
        nearest = all_distances[ratio_name]["top10_nearest"][0]
        nearest_val = nearest["constant_value"]
        # How often does the bootstrap ratio get as close or closer?
        distances_to_const = np.abs(samples - nearest_val)
        observed_dist = abs(ratio_val - nearest_val)
        p_raw = float(np.mean(distances_to_const <= observed_dist))
        p_bonferroni = min(1.0, p_raw * n_comparisons)

        bootstrap_stability[ratio_name] = {
            "observed": round(ratio_val, 8),
            "bootstrap_mean": round(float(mean_boot), 8),
            "bootstrap_std": round(float(std_boot), 8),
            "bootstrap_ci_95": [round(float(ci_low), 8), round(float(ci_high), 8)],
            "nearest_constant": nearest["constant"],
            "nearest_value": nearest["constant_value"],
            "raw_distance": nearest["distance"],
            "p_value_raw": round(p_raw, 6),
            "p_value_bonferroni": round(p_bonferroni, 6),
            "significant_after_bonferroni": p_bonferroni < 0.05,
        }
        log.info(f"  {ratio_name}: nearest={nearest['constant']}, "
                 f"d={nearest['distance']:.6f}, p_raw={p_raw:.4f}, p_bonf={p_bonferroni:.4f}")

        # Update bonferroni results
        bonferroni_results[ratio_name]["p_value_raw"] = round(p_raw, 6)
        bonferroni_results[ratio_name]["p_value_bonferroni"] = round(p_bonferroni, 6)
        bonferroni_results[ratio_name]["significant"] = p_bonferroni < 0.05

    # === 4. Progression test ===
    log.info("Test de progresión geométrica/aritmética...")
    ratio_values = [observed_ratios["books"], observed_ratios["words"],
                    observed_ratios["verses"], observed_ratios["chapters"]]
    ratio_names = ["books", "words", "verses", "chapters"]

    # Test arithmetic progression: constant differences?
    diffs = [ratio_values[i+1] - ratio_values[i] for i in range(len(ratio_values)-1)]
    arith_cv = float(np.std(diffs) / abs(np.mean(diffs))) if np.mean(diffs) != 0 else float('inf')

    # Test geometric progression: constant ratios?
    geo_ratios = [ratio_values[i+1] / ratio_values[i] for i in range(len(ratio_values)-1)
                  if ratio_values[i] != 0]
    geo_cv = float(np.std(geo_ratios) / abs(np.mean(geo_ratios))) if geo_ratios and np.mean(geo_ratios) != 0 else float('inf')

    progression_test = {
        "ratios_ordered": {n: round(v, 6) for n, v in zip(ratio_names, ratio_values)},
        "order": ratio_names,
        "arithmetic": {
            "differences": [round(d, 6) for d in diffs],
            "cv": round(arith_cv, 4),
            "is_arithmetic": arith_cv < 0.1,
        },
        "geometric": {
            "successive_ratios": [round(r, 6) for r in geo_ratios],
            "cv": round(geo_cv, 4),
            "is_geometric": geo_cv < 0.1,
        },
        "conclusion": (
            "geometric progression" if geo_cv < 0.1 else
            "arithmetic progression" if arith_cv < 0.1 else
            "neither arithmetic nor geometric"
        ),
    }
    log.info(f"Progresión: arith_cv={arith_cv:.4f}, geo_cv={geo_cv:.4f}")

    # === Save all results ===
    log.info("Guardando resultados...")
    with open(OUT / "all_distances.json", "w", encoding="utf-8") as f:
        json.dump({"n_constants": n_constants, "n_comparisons": n_comparisons,
                    "distances": all_distances}, f, ensure_ascii=False, indent=2)
    with open(OUT / "bonferroni_results.json", "w", encoding="utf-8") as f:
        json.dump({"n_comparisons": n_comparisons, "results": bonferroni_results},
                  f, ensure_ascii=False, indent=2)
    with open(OUT / "bootstrap_stability.json", "w", encoding="utf-8") as f:
        json.dump(bootstrap_stability, f, ensure_ascii=False, indent=2)
    with open(OUT / "progression_test.json", "w", encoding="utf-8") as f:
        json.dump(progression_test, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t0
    log.info(f"=== DONE en {elapsed:.1f}s ===")
    print(f"[deep_algebraic_constants] DONE — {elapsed:.1f}s")


if __name__ == "__main__":
    main()
