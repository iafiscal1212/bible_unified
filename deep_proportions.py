#!/usr/bin/env python3
"""
deep_proportions.py — Investigación 5: ¿Hay constantes proporcionales que se repitan
en distintos niveles del texto?
- Ratios jerárquicos
- Distancia a constantes irracionales (φ, √2, π/2, e/2)
- Distribución de tamaños de libros (power-law, log-normal)
- Test de autosimilaridad
"""
import json, logging, time, math
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy import stats as sp_stats

BASE = Path(__file__).parent
CORPUS = BASE / "bible_unified.json"
OUT = BASE / "results" / "deep_proportions"
OUT.mkdir(parents=True, exist_ok=True)
LOG = BASE / "logs"
LOG.mkdir(parents=True, exist_ok=True)

logging.basicConfig(filename=LOG / "deep_proportions.log", level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("deep_prop")

# Constantes irracionales
IRRATIONALS = {
    "phi": (1 + math.sqrt(5)) / 2,     # 1.61803...
    "sqrt2": math.sqrt(2),              # 1.41421...
    "pi_over_2": math.pi / 2,          # 1.57080...
    "e_over_2": math.e / 2,            # 1.35914...
    "sqrt3": math.sqrt(3),              # 1.73205...
    "ln2": math.log(2),                 # 0.69315...
    "pi": math.pi,                      # 3.14159...
    "e": math.e,                        # 2.71828...
    "sqrt5": math.sqrt(5),              # 2.23607...
}


def fit_distribution(data, name=""):
    """Ajusta varias distribuciones y compara con KS test."""
    data = np.array(data, dtype=float)
    results = {}

    # Log-normal
    try:
        shape, loc, scale = sp_stats.lognorm.fit(data, floc=0)
        ks, p = sp_stats.kstest(data, 'lognorm', args=(shape, loc, scale))
        results["lognormal"] = {
            "shape": round(float(shape), 4),
            "scale": round(float(scale), 4),
            "ks_stat": round(float(ks), 6),
            "p_value": round(float(p), 6),
        }
    except Exception:
        pass

    # Exponential
    try:
        loc, scale = sp_stats.expon.fit(data)
        ks, p = sp_stats.kstest(data, 'expon', args=(loc, scale))
        results["exponential"] = {
            "scale": round(float(scale), 4),
            "ks_stat": round(float(ks), 6),
            "p_value": round(float(p), 6),
        }
    except Exception:
        pass

    # Normal
    try:
        mu, sigma = sp_stats.norm.fit(data)
        ks, p = sp_stats.kstest(data, 'norm', args=(mu, sigma))
        results["normal"] = {
            "mu": round(float(mu), 4),
            "sigma": round(float(sigma), 4),
            "ks_stat": round(float(ks), 6),
            "p_value": round(float(p), 6),
        }
    except Exception:
        pass

    # Gamma
    try:
        a, loc, scale = sp_stats.gamma.fit(data, floc=0)
        ks, p = sp_stats.kstest(data, 'gamma', args=(a, loc, scale))
        results["gamma"] = {
            "shape": round(float(a), 4),
            "scale": round(float(scale), 4),
            "ks_stat": round(float(ks), 6),
            "p_value": round(float(p), 6),
        }
    except Exception:
        pass

    # Best fit
    best = None
    best_p = -1
    for dist_name, dist_data in results.items():
        if dist_data["p_value"] > best_p:
            best_p = dist_data["p_value"]
            best = dist_name
    results["best_fit"] = best

    return results


def main():
    log.info("Cargando corpus...")
    t0 = time.time()
    with open(CORPUS, encoding="utf-8") as f:
        words = json.load(f)
    log.info(f"Corpus: {len(words)} palabras")

    # Construir estructura jerárquica
    books_data = {}
    for w in words:
        key = w["book"]
        if key not in books_data:
            books_data[key] = {"corpus": w["corpus"], "book_num": w["book_num"],
                               "chapters": defaultdict(lambda: defaultdict(int)),
                               "total_words": 0}
        books_data[key]["chapters"][w["chapter"]][w["verse"]] += 1
        books_data[key]["total_words"] += 1

    # === 1. Ratios por nivel ===
    log.info("Calculando ratios por nivel...")

    # Palabras por versículo
    ot_wpv, nt_wpv = [], []
    # Versículos por capítulo
    ot_vpc, nt_vpc = [], []
    # Palabras por capítulo
    ot_wpc, nt_wpc = [], []
    # Capítulos por libro
    ot_cpb, nt_cpb = [], []
    # Palabras por libro
    ot_wpb, nt_wpb = [], []

    for bname, bd in books_data.items():
        corpus = bd["corpus"]
        n_chapters = len(bd["chapters"])
        target_cpb = ot_cpb if corpus == "OT" else nt_cpb
        target_cpb.append(n_chapters)
        target_wpb = ot_wpb if corpus == "OT" else nt_wpb
        target_wpb.append(bd["total_words"])

        for ch, verses in bd["chapters"].items():
            n_verses = len(verses)
            words_in_ch = sum(verses.values())
            target_vpc = ot_vpc if corpus == "OT" else nt_vpc
            target_vpc.append(n_verses)
            target_wpc = ot_wpc if corpus == "OT" else nt_wpc
            target_wpc.append(words_in_ch)

            for vs, wcount in verses.items():
                target_wpv = ot_wpv if corpus == "OT" else nt_wpv
                target_wpv.append(wcount)

    level_ratios = {
        "words_per_verse": {
            "ot_mean": round(float(np.mean(ot_wpv)), 4),
            "nt_mean": round(float(np.mean(nt_wpv)), 4),
            "ratio_means": round(float(np.mean(ot_wpv) / np.mean(nt_wpv)), 6) if nt_wpv else None,
            "ot_cv": round(float(np.std(ot_wpv) / np.mean(ot_wpv)), 4) if ot_wpv else None,
            "nt_cv": round(float(np.std(nt_wpv) / np.mean(nt_wpv)), 4) if nt_wpv else None,
        },
        "verses_per_chapter": {
            "ot_mean": round(float(np.mean(ot_vpc)), 4),
            "nt_mean": round(float(np.mean(nt_vpc)), 4),
            "ratio_means": round(float(np.mean(ot_vpc) / np.mean(nt_vpc)), 6) if nt_vpc else None,
            "ot_cv": round(float(np.std(ot_vpc) / np.mean(ot_vpc)), 4),
            "nt_cv": round(float(np.std(nt_vpc) / np.mean(nt_vpc)), 4),
        },
        "words_per_chapter": {
            "ot_mean": round(float(np.mean(ot_wpc)), 4),
            "nt_mean": round(float(np.mean(nt_wpc)), 4),
            "ratio_means": round(float(np.mean(ot_wpc) / np.mean(nt_wpc)), 6) if nt_wpc else None,
        },
        "chapters_per_book": {
            "ot_mean": round(float(np.mean(ot_cpb)), 4),
            "nt_mean": round(float(np.mean(nt_cpb)), 4),
            "ratio_means": round(float(np.mean(ot_cpb) / np.mean(nt_cpb)), 6) if nt_cpb else None,
        },
        "words_per_book": {
            "ot_mean": round(float(np.mean(ot_wpb)), 4),
            "nt_mean": round(float(np.mean(nt_wpb)), 4),
            "ratio_means": round(float(np.mean(ot_wpb) / np.mean(nt_wpb)), 6) if nt_wpb else None,
        },
        "total_ratio": {
            "total_words": round(sum(ot_wpb) / sum(nt_wpb), 6) if nt_wpb else None,
            "total_verses": round(len(ot_wpv) / len(nt_wpv), 6) if nt_wpv else None,
            "total_chapters": round(len(ot_wpc) / len(nt_wpc), 6) if nt_wpc else None,
            "total_books": round(39 / 27, 6),
        },
    }

    # === 2. Distancia a irracionales ===
    log.info("Calculando distancias a constantes irracionales...")
    observed_ratios = {
        "words_AT/NT": sum(ot_wpb) / sum(nt_wpb),
        "verses_AT/NT": len(ot_wpv) / len(nt_wpv),
        "chapters_AT/NT": len(ot_wpc) / len(nt_wpc),
        "books_AT/NT": 39 / 27,
        "mean_wpv_AT/NT": np.mean(ot_wpv) / np.mean(nt_wpv),
        "mean_vpc_AT/NT": np.mean(ot_vpc) / np.mean(nt_vpc),
    }

    irrational_distances = {}
    for ratio_name, ratio_val in observed_ratios.items():
        distances = {}
        for const_name, const_val in IRRATIONALS.items():
            dist = abs(ratio_val - const_val)
            rel_dist = dist / const_val  # relative distance
            distances[const_name] = {
                "constant_value": round(const_val, 6),
                "absolute_distance": round(dist, 6),
                "relative_distance": round(rel_dist, 6),
            }
        # Sort by distance
        nearest = min(distances.items(), key=lambda x: x[1]["absolute_distance"])
        irrational_distances[ratio_name] = {
            "observed_value": round(ratio_val, 6),
            "nearest_constant": nearest[0],
            "nearest_value": nearest[1]["constant_value"],
            "nearest_distance": nearest[1]["absolute_distance"],
            "all_distances": distances,
        }

    # === 3. Distribución de tamaños de libros ===
    log.info("Ajustando distribución de tamaños de libros...")
    all_book_sizes = [bd["total_words"] for bd in books_data.values()]
    ot_book_sizes = [bd["total_words"] for bd in books_data.values() if bd["corpus"] == "OT"]
    nt_book_sizes = [bd["total_words"] for bd in books_data.values() if bd["corpus"] == "NT"]

    book_dist = {
        "all_books": fit_distribution(all_book_sizes, "all"),
        "ot_books": fit_distribution(ot_book_sizes, "ot"),
        "nt_books": fit_distribution(nt_book_sizes, "nt"),
        "book_sizes": sorted([(bname, bd["total_words"], bd["corpus"])
                               for bname, bd in books_data.items()],
                              key=lambda x: -x[1]),
    }

    # === 4. Test de autosimilaridad ===
    log.info("Test de autosimilaridad...")

    # Distribución normalizada de longitudes de versículo dentro de capítulos
    # vs distribución normalizada de longitudes de capítulo dentro de libros
    verse_in_chapter_normalized = []  # normalized verse lengths within chapters
    chapter_in_book_normalized = []   # normalized chapter lengths within books

    for bname, bd in books_data.items():
        ch_sizes = []
        for ch, verses in bd["chapters"].items():
            ch_total = sum(verses.values())
            ch_sizes.append(ch_total)
            # Normalize verse lengths within this chapter
            if len(verses) >= 3:
                vl = list(verses.values())
                ch_mean = np.mean(vl)
                if ch_mean > 0:
                    verse_in_chapter_normalized.extend([v / ch_mean for v in vl])

        # Normalize chapter lengths within this book
        if len(ch_sizes) >= 3:
            book_mean = np.mean(ch_sizes)
            if book_mean > 0:
                chapter_in_book_normalized.extend([c / book_mean for c in ch_sizes])

    # KS test between the two normalized distributions
    if len(verse_in_chapter_normalized) >= 10 and len(chapter_in_book_normalized) >= 10:
        ks_stat, ks_p = sp_stats.ks_2samp(verse_in_chapter_normalized,
                                           chapter_in_book_normalized)
        selfsim = {
            "ks_stat": round(float(ks_stat), 6),
            "p_value": float(f"{ks_p:.2e}"),
            "n_verse_samples": len(verse_in_chapter_normalized),
            "n_chapter_samples": len(chapter_in_book_normalized),
            "verse_normalized_mean": round(float(np.mean(verse_in_chapter_normalized)), 4),
            "verse_normalized_std": round(float(np.std(verse_in_chapter_normalized)), 4),
            "chapter_normalized_mean": round(float(np.mean(chapter_in_book_normalized)), 4),
            "chapter_normalized_std": round(float(np.std(chapter_in_book_normalized)), 4),
            "interpretation": (
                "If p > 0.05, the two distributions are statistically indistinguishable, "
                "suggesting self-similarity across hierarchical levels."
            ),
            "self_similar": bool(ks_p > 0.05),
        }
    else:
        selfsim = {"error": "insufficient data"}

    # Save
    with open(OUT / "level_ratios.json", "w") as f:
        json.dump(level_ratios, f, indent=2)
    with open(OUT / "irrational_distances.json", "w") as f:
        json.dump(irrational_distances, f, indent=2)
    with open(OUT / "book_size_distribution.json", "w") as f:
        json.dump(book_dist, f, indent=2, ensure_ascii=False)
    with open(OUT / "selfsimilarity_test.json", "w") as f:
        json.dump(selfsim, f, indent=2)

    elapsed = time.time() - t0
    log.info(f"DONE en {elapsed:.1f}s")
    print(f"[deep_proportions] DONE — {elapsed:.1f}s")


if __name__ == "__main__":
    main()
