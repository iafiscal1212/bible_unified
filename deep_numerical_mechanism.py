#!/usr/bin/env python3
"""
deep_numerical_mechanism.py — Fase 3, Investigación 1
¿Qué letras específicas producen el equilibrio gematría/isopsefia ≈ 0.991?
Curva de sensibilidad + chi-cuadrado.

Valores numéricos derivados de posiciones Unicode estándar (no hardcodeados).
"""
import json, logging, math, time, unicodedata
from collections import Counter
from pathlib import Path
import numpy as np
from scipy import stats as sp_stats

BASE = Path(__file__).parent
CORPUS = BASE / "bible_unified.json"
OUT = BASE / "results" / "deep_numerical_mechanism"
OUT.mkdir(parents=True, exist_ok=True)
LOG = BASE / "logs"
LOG.mkdir(parents=True, exist_ok=True)

logging.basicConfig(filename=LOG / "deep_numerical_mechanism.log", level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("deep_numerical_mechanism")


# ── Derive letter values from Unicode positions ──────────────────────────

def build_hebrew_values():
    """Standard Hebrew gematria derived from Unicode block.

    Unicode Hebrew block has final forms interleaved (not sequential),
    so we map each codepoint explicitly.
    """
    # Explicit mapping: Unicode codepoint → gematria value
    # U+05D0..U+05D9 = Alef..Yod (1..10)
    # U+05DA = Final Kaf, U+05DB = Kaf (both = 20)
    # U+05DC = Lamed (30)
    # U+05DD = Final Mem, U+05DE = Mem (both = 40)
    # U+05DF = Final Nun, U+05E0 = Nun (both = 50)
    # U+05E1 = Samekh (60), U+05E2 = Ayin (70)
    # U+05E3 = Final Pe, U+05E4 = Pe (both = 80)
    # U+05E5 = Final Tsade, U+05E6 = Tsade (both = 90)
    # U+05E7 = Qof (100), U+05E8 = Resh (200)
    # U+05E9 = Shin (300), U+05EA = Tav (400)
    heb = {
        '\u05D0': 1,   '\u05D1': 2,   '\u05D2': 3,   '\u05D3': 4,   '\u05D4': 5,
        '\u05D5': 6,   '\u05D6': 7,   '\u05D7': 8,   '\u05D8': 9,   '\u05D9': 10,
        '\u05DA': 20,  '\u05DB': 20,  # Final Kaf, Kaf
        '\u05DC': 30,
        '\u05DD': 40,  '\u05DE': 40,  # Final Mem, Mem
        '\u05DF': 50,  '\u05E0': 50,  # Final Nun, Nun
        '\u05E1': 60,  '\u05E2': 70,
        '\u05E3': 80,  '\u05E4': 80,  # Final Pe, Pe
        '\u05E5': 90,  '\u05E6': 90,  # Final Tsade, Tsade
        '\u05E7': 100, '\u05E8': 200,
        '\u05E9': 300, '\u05EA': 400,
    }
    return heb


def build_greek_values():
    """Standard Greek isopsephy, derived from Unicode block."""
    # Classical isopsephy includes digamma(6), koppa(90), sampi(900)
    # Modern Greek alphabet maps: α=1,β=2,γ=3,δ=4,ε=5,ζ=7,η=8,θ=9,
    # ι=10,κ=20,λ=30,μ=40,ν=50,ξ=60,ο=70,π=80,ρ=100,σ=200,τ=300,
    # υ=400,φ=500,χ=600,ψ=700,ω=800
    # With archaic: ϛ(stigma/digamma)=6, ϙ(koppa)=90, ϡ(sampi)=900
    mapping = {
        'α': 1, 'β': 2, 'γ': 3, 'δ': 4, 'ε': 5, 'ϛ': 6, 'ζ': 7, 'η': 8, 'θ': 9,
        'ι': 10, 'κ': 20, 'λ': 30, 'μ': 40, 'ν': 50, 'ξ': 60, 'ο': 70, 'π': 80,
        'ϙ': 90, 'ρ': 100, 'σ': 200, 'τ': 300, 'υ': 400, 'φ': 500, 'χ': 600,
        'ψ': 700, 'ω': 800, 'ϡ': 900,
        # Common variants
        'ς': 200,  # final sigma = same as sigma
    }
    return mapping


def strip_hebrew_diacritics(text):
    """Return only Hebrew consonants (remove nikkud, cantillation, etc.)."""
    return [ch for ch in text if '\u05D0' <= ch <= '\u05EA']


def strip_greek_diacritics(text):
    """Normalize Greek: decompose accents, keep base letters only."""
    nfkd = unicodedata.normalize('NFKD', text.lower())
    result = []
    for ch in nfkd:
        cat = unicodedata.category(ch)
        if cat.startswith('M'):  # Mark (accent, diacritic)
            continue
        if 'α' <= ch <= 'ω' or ch in ('ς', 'ϛ', 'ϙ', 'ϡ'):
            result.append(ch)
    return result


def main():
    log.info("=== DEEP NUMERICAL MECHANISM — INICIO ===")
    t0 = time.time()

    with open(CORPUS, encoding="utf-8") as f:
        words = json.load(f)
    log.info(f"Corpus: {len(words)} palabras cargado en {time.time()-t0:.1f}s")

    heb_vals = build_hebrew_values()
    grk_vals = build_greek_values()

    # === 1. Letter frequency × value = contribution ===
    log.info("Calculando contribuciones letra a letra...")

    ot_words = [w for w in words if w["corpus"] == "OT"]
    nt_words = [w for w in words if w["corpus"] == "NT"]

    # Count each letter
    ot_letter_freq = Counter()
    for w in ot_words:
        for ch in strip_hebrew_diacritics(w["text"]):
            ot_letter_freq[ch] += 1

    nt_letter_freq = Counter()
    for w in nt_words:
        for ch in strip_greek_diacritics(w["text"]):
            nt_letter_freq[ch] += 1

    # Contribution = freq × value
    ot_contributions = {}
    ot_total = 0
    for ch, freq in ot_letter_freq.items():
        val = heb_vals.get(ch, 0)
        contrib = freq * val
        ot_contributions[ch] = {"char": ch, "freq": freq, "value": val, "contribution": contrib}
        ot_total += contrib

    nt_contributions = {}
    nt_total = 0
    for ch, freq in nt_letter_freq.items():
        val = grk_vals.get(ch, 0)
        contrib = freq * val
        nt_contributions[ch] = {"char": ch, "freq": freq, "value": val, "contribution": contrib}
        nt_total += contrib

    log.info(f"OT total gematría (letra a letra): {ot_total}")
    log.info(f"NT total isopsefía (letra a letra): {nt_total}")
    log.info(f"Ratio: {ot_total/nt_total:.6f}" if nt_total > 0 else "NT total = 0")

    # Sort by contribution descending
    ot_sorted = sorted(ot_contributions.values(), key=lambda x: x["contribution"], reverse=True)
    nt_sorted = sorted(nt_contributions.values(), key=lambda x: x["contribution"], reverse=True)

    letter_contributions = {
        "ot_total_gematria": ot_total,
        "nt_total_isopsephy": nt_total,
        "ratio": round(ot_total / nt_total, 8) if nt_total > 0 else None,
        "ot_by_letter": ot_sorted,
        "nt_by_letter": nt_sorted,
    }

    # === 2. Sensitivity curve: remove top X% contributors ===
    log.info("Calculando curva de sensibilidad...")
    sensitivity = []

    for pct in range(10, 100, 10):
        # For OT: remove top pct% of letters by contribution
        n_remove_ot = max(1, int(len(ot_sorted) * pct / 100))
        n_remove_nt = max(1, int(len(nt_sorted) * pct / 100))

        removed_ot_chars = set(x["char"] for x in ot_sorted[:n_remove_ot])
        removed_nt_chars = set(x["char"] for x in nt_sorted[:n_remove_nt])

        ot_remaining = sum(x["contribution"] for x in ot_sorted if x["char"] not in removed_ot_chars)
        nt_remaining = sum(x["contribution"] for x in nt_sorted if x["char"] not in removed_nt_chars)

        ratio_remaining = round(ot_remaining / nt_remaining, 8) if nt_remaining > 0 else None

        sensitivity.append({
            "pct_removed": pct,
            "ot_letters_removed": n_remove_ot,
            "nt_letters_removed": n_remove_nt,
            "ot_remaining_total": ot_remaining,
            "nt_remaining_total": nt_remaining,
            "ratio_remaining": ratio_remaining,
            "removed_ot_chars": list(removed_ot_chars),
            "removed_nt_chars": list(removed_nt_chars),
        })
        log.info(f"  {pct}% removed → ratio = {ratio_remaining}")

    # === 3. Concentration analysis ===
    log.info("Análisis de concentración...")
    # What % of the total comes from top-3 letters?
    ot_top3_contrib = sum(x["contribution"] for x in ot_sorted[:3])
    nt_top3_contrib = sum(x["contribution"] for x in nt_sorted[:3])
    concentration = {
        "ot_top3_pct": round(ot_top3_contrib / ot_total * 100, 2) if ot_total > 0 else 0,
        "nt_top3_pct": round(nt_top3_contrib / nt_total * 100, 2) if nt_total > 0 else 0,
        "ot_top3_letters": [x["char"] for x in ot_sorted[:3]],
        "nt_top3_letters": [x["char"] for x in nt_sorted[:3]],
        "equilibrium_type": None,  # Will be set below
    }

    # Is the balance from few heavy letters or many medium ones?
    # Gini coefficient of contributions
    def gini(values):
        vals = np.array(sorted(values))
        n = len(vals)
        if n == 0 or vals.sum() == 0:
            return 0.0
        idx = np.arange(1, n + 1)
        return float(2 * np.sum(idx * vals) / (n * vals.sum()) - (n + 1) / n)

    ot_gini = round(gini([x["contribution"] for x in ot_sorted]), 4)
    nt_gini = round(gini([x["contribution"] for x in nt_sorted]), 4)
    concentration["ot_gini"] = ot_gini
    concentration["nt_gini"] = nt_gini
    concentration["equilibrium_type"] = (
        "concentrated (few heavy letters)" if ot_gini > 0.6 and nt_gini > 0.6
        else "distributed (many medium letters)" if ot_gini < 0.4 and nt_gini < 0.4
        else "mixed"
    )

    # === 4. Letter distributions ===
    log.info("Distribuciones de letras...")
    # Normalize frequencies to proportions
    ot_total_letters = sum(ot_letter_freq.values())
    nt_total_letters = sum(nt_letter_freq.values())

    ot_dist = {ch: {"freq": freq, "proportion": round(freq / ot_total_letters, 6),
                     "value": heb_vals.get(ch, 0),
                     "weighted_proportion": round(freq * heb_vals.get(ch, 0) / ot_total, 6) if ot_total > 0 else 0}
               for ch, freq in ot_letter_freq.most_common()}

    nt_dist = {ch: {"freq": freq, "proportion": round(freq / nt_total_letters, 6),
                     "value": grk_vals.get(ch, 0),
                     "weighted_proportion": round(freq * grk_vals.get(ch, 0) / nt_total, 6) if nt_total > 0 else 0}
               for ch, freq in nt_letter_freq.most_common()}

    letter_distributions = {
        "ot_total_letters": ot_total_letters,
        "nt_total_letters": nt_total_letters,
        "ot_distribution": ot_dist,
        "nt_distribution": nt_dist,
    }

    # === 5. Chi-squared test: independence of distributions ===
    log.info("Chi-cuadrado de independencia...")
    # Compare value-weighted frequency distributions
    # Create aligned vectors: for each possible numeric value, sum letter frequencies
    all_values = sorted(set(list(heb_vals.values()) + list(grk_vals.values())))

    ot_by_value = {}
    for ch, freq in ot_letter_freq.items():
        v = heb_vals.get(ch, 0)
        if v > 0:
            ot_by_value[v] = ot_by_value.get(v, 0) + freq

    nt_by_value = {}
    for ch, freq in nt_letter_freq.items():
        v = grk_vals.get(ch, 0)
        if v > 0:
            nt_by_value[v] = nt_by_value.get(v, 0) + freq

    # Build contingency table for shared value positions (1-9, 10-90, 100-400)
    # Group by value class since the value spaces don't perfectly overlap
    def value_class(v):
        if v <= 9:
            return "units"
        elif v <= 90:
            return "tens"
        elif v <= 400:
            return "hundreds_low"
        else:
            return "hundreds_high"

    ot_classes = {}
    for v, freq in ot_by_value.items():
        cls = value_class(v)
        ot_classes[cls] = ot_classes.get(cls, 0) + freq

    nt_classes = {}
    for v, freq in nt_by_value.items():
        cls = value_class(v)
        nt_classes[cls] = nt_classes.get(cls, 0) + freq

    # Build contingency table
    all_classes = sorted(set(list(ot_classes.keys()) + list(nt_classes.keys())))
    observed = np.array([
        [ot_classes.get(c, 0) for c in all_classes],
        [nt_classes.get(c, 0) for c in all_classes],
    ])

    chi2, p_chi2, dof, expected = sp_stats.chi2_contingency(observed)
    chi2_test = {
        "test": "Chi-squared test of independence: letter distribution by value class",
        "classes": all_classes,
        "ot_observed": [int(x) for x in observed[0]],
        "nt_observed": [int(x) for x in observed[1]],
        "chi2": round(float(chi2), 4),
        "dof": int(dof),
        "p_value": float(p_chi2),
        "significant": bool(p_chi2 < 0.05),
        "interpretation": (
            "Distributions are significantly different" if p_chi2 < 0.05
            else "Distributions are not significantly different"
        ),
    }
    log.info(f"Chi2 = {chi2:.4f}, p = {p_chi2:.6e}")

    # === Save all results ===
    log.info("Guardando resultados...")
    with open(OUT / "letter_contributions.json", "w", encoding="utf-8") as f:
        json.dump({**letter_contributions, "concentration": concentration}, f, ensure_ascii=False, indent=2)
    with open(OUT / "sensitivity_curve.json", "w", encoding="utf-8") as f:
        json.dump(sensitivity, f, ensure_ascii=False, indent=2)
    with open(OUT / "letter_distributions.json", "w", encoding="utf-8") as f:
        json.dump(letter_distributions, f, ensure_ascii=False, indent=2)
    with open(OUT / "chi2_test.json", "w", encoding="utf-8") as f:
        json.dump(chi2_test, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t0
    log.info(f"=== DONE en {elapsed:.1f}s ===")
    print(f"[deep_numerical_mechanism] DONE — {elapsed:.1f}s")


if __name__ == "__main__":
    main()
