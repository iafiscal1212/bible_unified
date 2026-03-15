#!/usr/bin/env python3
"""
deep_numerical.py — Investigación 2: ¿El ratio gematría/isopsefia ≈ 0.991 es trivial o significativo?
- Permutation test (10,000)
- Bootstrap CI
- Ratio teórico vs observado
- Ratio por libro
"""
import json, logging, time, unicodedata
from collections import defaultdict
from pathlib import Path
import numpy as np

BASE = Path(__file__).parent
CORPUS = BASE / "bible_unified.json"
OUT = BASE / "results" / "deep_numerical"
OUT.mkdir(parents=True, exist_ok=True)
LOG = BASE / "logs"
LOG.mkdir(parents=True, exist_ok=True)

logging.basicConfig(filename=LOG / "deep_numerical.log", level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("deep_num")

HEBREW_GEMATRIA = {
    'א': 1, 'ב': 2, 'ג': 3, 'ד': 4, 'ה': 5, 'ו': 6, 'ז': 7, 'ח': 8, 'ט': 9,
    'י': 10, 'כ': 20, 'ל': 30, 'מ': 40, 'נ': 50, 'ס': 60, 'ע': 70, 'פ': 80, 'צ': 90,
    'ק': 100, 'ר': 200, 'ש': 300, 'ת': 400,
    'ך': 20, 'ם': 40, 'ן': 50, 'ף': 80, 'ץ': 90,
}

GREEK_ISOPSEPHY = {
    'α': 1, 'β': 2, 'γ': 3, 'δ': 4, 'ε': 5, 'ϛ': 6, 'ζ': 7, 'η': 8, 'θ': 9,
    'ι': 10, 'κ': 20, 'λ': 30, 'μ': 40, 'ν': 50, 'ξ': 60, 'ο': 70, 'π': 80, 'ϟ': 90,
    'ρ': 100, 'σ': 200, 'τ': 300, 'υ': 400, 'φ': 500, 'χ': 600, 'ψ': 700, 'ω': 800,
    'ς': 200,
}


def word_value(text, lang):
    if lang == "heb":
        return sum(HEBREW_GEMATRIA.get(ch, 0) for ch in text)
    elif lang == "grc":
        clean = ''.join(unicodedata.normalize('NFD', ch)[0].lower() for ch in text)
        return sum(GREEK_ISOPSEPHY.get(ch, 0) for ch in clean)
    return 0


def main():
    log.info("Cargando corpus...")
    t0 = time.time()
    with open(CORPUS, encoding="utf-8") as f:
        words = json.load(f)
    log.info(f"Corpus: {len(words)} palabras")

    # Calcular valor de cada palabra
    log.info("Calculando valores numéricos...")
    ot_vals = []
    nt_vals = []
    book_totals = defaultdict(lambda: {"total": 0, "n_words": 0, "corpus": ""})

    for w in words:
        v = word_value(w["text"], w["lang"])
        if w["corpus"] == "OT":
            ot_vals.append(v)
        else:
            nt_vals.append(v)
        book_totals[w["book"]]["total"] += v
        book_totals[w["book"]]["n_words"] += 1
        book_totals[w["book"]]["corpus"] = w["corpus"]

    ot_vals = np.array(ot_vals, dtype=np.float64)
    nt_vals = np.array(nt_vals, dtype=np.float64)

    ot_total = float(ot_vals.sum())
    nt_total = float(nt_vals.sum())
    observed_ratio = ot_total / nt_total
    log.info(f"OT total={ot_total:.0f}, NT total={nt_total:.0f}, ratio={observed_ratio:.6f}")

    # === 1. Permutation test ===
    log.info("Permutation test (10,000 permutaciones)...")
    n_perms = 10000
    all_vals = np.concatenate([ot_vals, nt_vals])
    n_ot = len(ot_vals)
    n_nt = len(nt_vals)
    rng = np.random.default_rng(42)

    perm_ratios = np.empty(n_perms)
    obs_distance = abs(observed_ratio - 1.0)

    for i in range(n_perms):
        perm = rng.permutation(all_vals)
        perm_ot = perm[:n_ot].sum()
        perm_nt = perm[n_ot:].sum()
        perm_ratios[i] = perm_ot / perm_nt if perm_nt != 0 else np.inf
        if (i + 1) % 2000 == 0:
            log.info(f"  Permutación {i+1}/{n_perms}")

    perm_distances = np.abs(perm_ratios - 1.0)
    p_value = float(np.mean(perm_distances <= obs_distance))

    perm_result = {
        "observed_ratio": round(observed_ratio, 8),
        "observed_distance_from_1": round(obs_distance, 8),
        "n_permutations": n_perms,
        "p_value": round(p_value, 6),
        "perm_ratio_mean": round(float(np.mean(perm_ratios)), 6),
        "perm_ratio_std": round(float(np.std(perm_ratios)), 6),
        "perm_ratio_min": round(float(np.min(perm_ratios)), 6),
        "perm_ratio_max": round(float(np.max(perm_ratios)), 6),
        "interpretation": (
            f"p={p_value:.4f}. "
            "If p < 0.05, the observed ratio is significantly closer to 1.0 "
            "than expected by chance (the quasi-equality is non-trivial)."
        ),
    }
    log.info(f"Permutation p-value: {p_value:.6f}")

    # === 2. Bootstrap CI ===
    log.info("Bootstrap CI (10,000 remuestreos)...")
    n_boot = 10000
    boot_ratios = np.empty(n_boot)
    for i in range(n_boot):
        boot_ot = rng.choice(ot_vals, size=n_ot, replace=True).sum()
        boot_nt = rng.choice(nt_vals, size=n_nt, replace=True).sum()
        boot_ratios[i] = boot_ot / boot_nt if boot_nt != 0 else np.inf
        if (i + 1) % 2000 == 0:
            log.info(f"  Bootstrap {i+1}/{n_boot}")

    ci_95 = (round(float(np.percentile(boot_ratios, 2.5)), 8),
             round(float(np.percentile(boot_ratios, 97.5)), 8))
    ci_99 = (round(float(np.percentile(boot_ratios, 0.5)), 8),
             round(float(np.percentile(boot_ratios, 99.5)), 8))

    bootstrap_result = {
        "observed_ratio": round(observed_ratio, 8),
        "n_bootstrap": n_boot,
        "ci_95": ci_95,
        "ci_99": ci_99,
        "boot_mean": round(float(np.mean(boot_ratios)), 8),
        "boot_std": round(float(np.std(boot_ratios)), 8),
        "contains_1": ci_95[0] <= 1.0 <= ci_95[1],
    }
    log.info(f"Bootstrap 95% CI: {ci_95}")

    # === 3. Ratio teórico ===
    log.info("Calculando ratio teórico...")
    # Si cada letra fuera equiprobable dentro de su alfabeto
    heb_mean = np.mean(list(HEBREW_GEMATRIA.values()))  # Media de valores hebreos
    grc_mean = np.mean(list(GREEK_ISOPSEPHY.values()))  # Media de valores griegos
    theoretical_ratio = (heb_mean * n_ot) / (grc_mean * n_nt)

    # Mean letter value per word (observed)
    ot_mean_per_word = float(ot_vals.mean())
    nt_mean_per_word = float(nt_vals.mean())

    theoretical_result = {
        "heb_alphabet_mean_value": round(float(heb_mean), 2),
        "grc_alphabet_mean_value": round(float(grc_mean), 2),
        "n_ot_words": int(n_ot),
        "n_nt_words": int(n_nt),
        "theoretical_ratio_uniform": round(float(theoretical_ratio), 8),
        "observed_ratio": round(observed_ratio, 8),
        "deviation_from_theoretical": round(abs(observed_ratio - theoretical_ratio), 8),
        "ot_observed_mean_per_word": round(ot_mean_per_word, 2),
        "nt_observed_mean_per_word": round(nt_mean_per_word, 2),
    }
    log.info(f"Theoretical ratio: {theoretical_ratio:.6f}, observed: {observed_ratio:.6f}")

    # === 4. Ratio por libro ===
    log.info("Ratio por libro...")
    ratio_by_book = []
    for bname, bdata in sorted(book_totals.items(), key=lambda x: x[0]):
        mean_val = bdata["total"] / bdata["n_words"] if bdata["n_words"] else 0
        ratio_by_book.append({
            "book": bname,
            "corpus": bdata["corpus"],
            "total_value": bdata["total"],
            "n_words": bdata["n_words"],
            "mean_value_per_word": round(mean_val, 2),
        })

    # ¿Qué libros contribuyen más al equilibrio?
    # Calcular contribución: si removemos cada libro, ¿cómo cambia el ratio?
    sensitivity = []
    for bname, bdata in book_totals.items():
        if bdata["corpus"] == "OT":
            new_ot = ot_total - bdata["total"]
            new_ratio = new_ot / nt_total if nt_total else 0
        else:
            new_nt = nt_total - bdata["total"]
            new_ratio = ot_total / new_nt if new_nt else 0
        delta_ratio = abs(new_ratio - observed_ratio)
        sensitivity.append({
            "book": bname,
            "corpus": bdata["corpus"],
            "ratio_without_book": round(new_ratio, 8),
            "delta_ratio": round(delta_ratio, 8),
        })
    sensitivity.sort(key=lambda x: -x["delta_ratio"])

    # Save
    with open(OUT / "permutation_test.json", "w") as f:
        json.dump(perm_result, f, indent=2)
    with open(OUT / "bootstrap_ci.json", "w") as f:
        json.dump(bootstrap_result, f, indent=2)
    with open(OUT / "theoretical_vs_observed.json", "w") as f:
        json.dump(theoretical_result, f, indent=2)
    with open(OUT / "ratio_by_book.json", "w") as f:
        json.dump({"ratio_by_book": ratio_by_book, "sensitivity_top20": sensitivity[:20]}, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    log.info(f"DONE en {elapsed:.1f}s")
    print(f"[deep_numerical] DONE — {elapsed:.1f}s")


if __name__ == "__main__":
    main()
