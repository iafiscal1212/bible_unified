#!/usr/bin/env python3
"""
Fase 18 — Pre-test: Robustez del umbral de delay.

Pregunta: ¿La separación AT-like / NT-like por control_delay_years
es robusta para un rango amplio de umbrales, o solo funciona con
un umbral cherry-picked?

Método:
1. Barrer umbral de 1 a 500 años en incrementos de 1.
2. En cada umbral t: clasificar corpus con delay < t como "temprano",
   delay >= t como "tardío".
3. Computar Fisher exact (temprano × AT-like) y accuracy.
4. Reportar:
   - Rango de umbrales con separación perfecta (accuracy=1.0)
   - Rango con p < 0.05
   - Gap en los datos (¿hay corpora entre 20 y 200?)
   - Advertencia sobre circularidad si el rango perfecto es estrecho.

Usa SOLO corpora con clasificación AT/NT disponible (P_AT no null).
Opcionalmente repite con H > 0.7 como proxy para los no clasificados.
"""

import json
import logging
import numpy as np
from pathlib import Path
from scipy import stats as sp_stats

BASE = Path(__file__).resolve().parent
RESULTS_DIR = BASE / "results" / "robustness"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)


def load_data():
    """Load the 9-corpus data matrix."""
    data_file = BASE / "results" / "transmission_origin" / "data_matrix.json"
    with open(data_file) as f:
        return json.load(f)


def fisher_exact_2x2(early_at, early_nt, late_at, late_nt):
    """Fisher exact test on 2x2 table."""
    table = [[early_at, early_nt], [late_at, late_nt]]
    odds, p = sp_stats.fisher_exact(table)
    return odds, p


def sweep_thresholds(corpora, use_h_proxy=False):
    """
    Sweep delay threshold from 1 to 500.
    Returns list of {threshold, accuracy, fisher_p, ...} dicts.
    """
    results = []

    for t in range(1, 501):
        early_at = 0
        early_nt = 0
        late_at = 0
        late_nt = 0

        for c in corpora:
            delay = c["control_delay_years"]

            if use_h_proxy:
                # Use H > 0.7 as AT-like proxy for all corpora
                at_like = c["H"] > 0.7
            else:
                # Use classifier prediction (skip if null)
                if c["predicted"] is None:
                    continue
                at_like = c["predicted"] == "AT"

            if delay < t:
                if at_like:
                    early_at += 1
                else:
                    early_nt += 1
            else:
                if at_like:
                    late_at += 1
                else:
                    late_nt += 1

        total = early_at + early_nt + late_at + late_nt
        if total == 0:
            continue

        correct = early_at + late_nt
        accuracy = correct / total

        # Fisher exact (need at least 1 in each margin to be meaningful)
        if (early_at + early_nt) > 0 and (late_at + late_nt) > 0:
            odds, p = fisher_exact_2x2(early_at, early_nt, late_at, late_nt)
        else:
            odds, p = float("inf"), 1.0

        results.append({
            "threshold": t,
            "early_at": early_at,
            "early_nt": early_nt,
            "late_at": late_at,
            "late_nt": late_nt,
            "total": total,
            "accuracy": round(accuracy, 4),
            "fisher_p": round(p, 6),
            "odds_ratio": odds if odds != float("inf") else "infinity",
        })

    return results


def analyze_robustness(sweep_results, label):
    """Analyze the sweep to find ranges of perfect/significant separation."""
    perfect_range = [r["threshold"] for r in sweep_results if r["accuracy"] == 1.0]
    sig_range = [r["threshold"] for r in sweep_results if r["fisher_p"] < 0.05]

    analysis = {
        "label": label,
        "n_thresholds_tested": len(sweep_results),
    }

    if perfect_range:
        analysis["perfect_accuracy"] = {
            "min_threshold": min(perfect_range),
            "max_threshold": max(perfect_range),
            "range_width": max(perfect_range) - min(perfect_range) + 1,
            "pct_of_sweep": round(len(perfect_range) / len(sweep_results) * 100, 1),
        }
    else:
        analysis["perfect_accuracy"] = {"range_width": 0}

    if sig_range:
        analysis["significant_p05"] = {
            "min_threshold": min(sig_range),
            "max_threshold": max(sig_range),
            "range_width": max(sig_range) - min(sig_range) + 1,
            "pct_of_sweep": round(len(sig_range) / len(sweep_results) * 100, 1),
        }
    else:
        analysis["significant_p05"] = {"range_width": 0}

    # Best threshold (lowest Fisher p)
    best = min(sweep_results, key=lambda r: r["fisher_p"])
    analysis["best_threshold"] = {
        "threshold": best["threshold"],
        "fisher_p": best["fisher_p"],
        "accuracy": best["accuracy"],
    }

    return analysis


def main():
    log.info("=" * 70)
    log.info("TEST DE ROBUSTEZ: Umbral de delay")
    log.info("=" * 70)

    corpora = load_data()

    # ── Datos disponibles ────────────────────────────────────────────
    log.info("\nCorpora y delays:")
    delays_with_class = []
    delays_all = []
    for c in corpora:
        pred = c.get("predicted", "?")
        delay = c["control_delay_years"]
        log.info(f"  {c['corpus']:15s}  delay={delay:4d}  predicted={pred}  H={c['H']:.4f}")
        delays_all.append(delay)
        if pred is not None:
            delays_with_class.append(delay)

    unique_delays = sorted(set(delays_all))
    log.info(f"\n  Delays únicos: {unique_delays}")

    # Identificar gaps
    gaps = []
    for i in range(len(unique_delays) - 1):
        gap = unique_delays[i + 1] - unique_delays[i]
        if gap > 50:
            gaps.append({
                "from": unique_delays[i],
                "to": unique_delays[i + 1],
                "width": gap,
            })
    log.info(f"  Gaps >50 años: {gaps}")

    # ── Sweep 1: Solo corpora con clasificación ──────────────────────
    log.info("\n--- Sweep 1: Corpora con clasificación (7 de 9) ---")
    sweep1 = sweep_thresholds(corpora, use_h_proxy=False)
    analysis1 = analyze_robustness(sweep1, "classified_only")
    log.info(f"  Accuracy perfecta: thresholds {analysis1['perfect_accuracy']}")
    log.info(f"  Fisher p<0.05: thresholds {analysis1['significant_p05']}")
    log.info(f"  Mejor umbral: {analysis1['best_threshold']}")

    # ── Sweep 2: Todos los corpora con H>0.7 como proxy ──────────────
    log.info("\n--- Sweep 2: Todos (9), AT-like = H > 0.7 ---")
    sweep2 = sweep_thresholds(corpora, use_h_proxy=True)
    analysis2 = analyze_robustness(sweep2, "h_proxy_all")
    log.info(f"  Accuracy perfecta: thresholds {analysis2['perfect_accuracy']}")
    log.info(f"  Fisher p<0.05: thresholds {analysis2['significant_p05']}")
    log.info(f"  Mejor umbral: {analysis2['best_threshold']}")

    # ── Sweep 3: Todos con H>0.65 (más conservador) ─────────────────
    log.info("\n--- Sweep 3: Todos (9), AT-like = H > 0.65 ---")
    sweep3_results = []
    for t in range(1, 501):
        early_at = early_nt = late_at = late_nt = 0
        for c in corpora:
            at_like = c["H"] > 0.65
            if c["control_delay_years"] < t:
                if at_like:
                    early_at += 1
                else:
                    early_nt += 1
            else:
                if at_like:
                    late_at += 1
                else:
                    late_nt += 1
        total = early_at + early_nt + late_at + late_nt
        correct = early_at + late_nt
        accuracy = correct / total if total > 0 else 0
        if (early_at + early_nt) > 0 and (late_at + late_nt) > 0:
            odds, p = fisher_exact_2x2(early_at, early_nt, late_at, late_nt)
        else:
            odds, p = float("inf"), 1.0
        sweep3_results.append({
            "threshold": t,
            "early_at": early_at, "early_nt": early_nt,
            "late_at": late_at, "late_nt": late_nt,
            "accuracy": round(accuracy, 4),
            "fisher_p": round(p, 6),
        })

    analysis3 = analyze_robustness(sweep3_results, "h_proxy_065")
    log.info(f"  Accuracy perfecta: thresholds {analysis3['perfect_accuracy']}")
    log.info(f"  Fisher p<0.05: thresholds {analysis3['significant_p05']}")

    # ── Diagnóstico de circularidad ──────────────────────────────────
    log.info("\n--- Diagnóstico de circularidad ---")

    # The critical question: is the "perfect" range just the gap in the data?
    gap_20_200 = {"from": 20, "to": 200, "width": 180}
    perf = analysis1.get("perfect_accuracy", {})
    perf_min = perf.get("min_threshold", 0)
    perf_max = perf.get("max_threshold", 0)
    perf_width = perf.get("range_width", 0)

    circularity = {
        "data_gap": gap_20_200,
        "perfect_range_classified": {
            "min": perf_min,
            "max": perf_max,
            "width": perf_width,
        },
        "overlap_with_gap": perf_min >= 21 and perf_max <= 200,
        "interpretation": "",
    }

    if perf_width == 0:
        circularity["interpretation"] = (
            "NO perfect separation at any threshold — hypothesis is weak."
        )
        circularity["verdict"] = "WEAK"
    elif perf_width < 30:
        circularity["interpretation"] = (
            f"Perfect separation in narrow range ({perf_width} years). "
            "Hypothesis is FRAGILE — depends on precise threshold choice."
        )
        circularity["verdict"] = "FRAGILE"
    elif circularity["overlap_with_gap"]:
        circularity["interpretation"] = (
            f"Perfect separation range ({perf_min}-{perf_max}) falls "
            f"entirely within the data gap (21-199). "
            "Cannot distinguish genuine robustness from absence of data. "
            "Need a corpus with delay ~50-150 to break the degeneracy."
        )
        circularity["verdict"] = "INDETERMINATE — DATA GAP"
    else:
        circularity["interpretation"] = (
            f"Perfect separation in wide range ({perf_width} years, "
            f"{perf_min}-{perf_max}). Range extends BEYOND the data gap. "
            "Hypothesis is ROBUST."
        )
        circularity["verdict"] = "ROBUST"

    log.info(f"  Rango perfecto: {perf_min}-{perf_max} ({perf_width} años)")
    log.info(f"  Gap en datos: 21-199")
    log.info(f"  ¿Rango ⊂ gap?: {circularity['overlap_with_gap']}")
    log.info(f"  Veredicto: {circularity['verdict']}")
    log.info(f"  {circularity['interpretation']}")

    # ── What corpus would break the degeneracy? ──────────────────────
    log.info("\n--- Corpora que romperían la ambigüedad ---")
    needed = {
        "description": (
            "To resolve whether the threshold is real or an artifact of the "
            "data gap, we need corpora with control_delay in [50, 200] years."
        ),
        "candidates": [
            {
                "corpus": "Tosefta",
                "estimated_delay": 100,
                "rationale": (
                    "Compiled ~250 CE, traditions from ~100 CE. "
                    "~150 years of free transmission. Same language/genre as Mishnah."
                ),
            },
            {
                "corpus": "Didache",
                "estimated_delay": 50,
                "rationale": (
                    "Early Christian text, ~50-120 CE. "
                    "Short delay before community standardization."
                ),
            },
            {
                "corpus": "Zoroastrian_Yasna",
                "estimated_delay": 100,
                "rationale": (
                    "Gathic Avestan core oral ~1000 BCE, written ~500 CE. "
                    "But liturgical use may have controlled transmission early."
                ),
            },
        ],
        "prediction_if_H4_prime_true": (
            "If H4' is correct: delay ~50 should be AT-like, "
            "delay ~150 should be borderline or NT-like. "
            "If delay ~100 is AT-like, threshold is >100. "
            "If delay ~100 is NT-like, threshold is <100."
        ),
    }
    for cand in needed["candidates"]:
        log.info(f"  {cand['corpus']} (delay~{cand['estimated_delay']}): {cand['rationale']}")

    # ── Spearman rank correlation (non-parametric, avoids threshold) ─
    log.info("\n--- Correlación Spearman (delay vs H, sin umbral) ---")
    delays_h = [(c["control_delay_years"], c["H"]) for c in corpora]
    d_arr = np.array([x[0] for x in delays_h])
    h_arr = np.array([x[1] for x in delays_h])

    rho, p_spearman = sp_stats.spearmanr(d_arr, h_arr)
    log.info(f"  Spearman ρ(delay, H) = {rho:.4f}, p = {p_spearman:.4f}")

    # Kendall tau (more robust with ties)
    tau, p_kendall = sp_stats.kendalltau(d_arr, h_arr)
    log.info(f"  Kendall τ(delay, H) = {tau:.4f}, p = {p_kendall:.4f}")

    # With AC1 (only corpora that have it)
    delays_ac1 = [(c["control_delay_years"], c["AC1"]) for c in corpora if c["AC1"] is not None]
    if len(delays_ac1) >= 3:
        d_ac = np.array([x[0] for x in delays_ac1])
        ac_arr = np.array([x[1] for x in delays_ac1])
        rho_ac, p_ac = sp_stats.spearmanr(d_ac, ac_arr)
        tau_ac, p_tau_ac = sp_stats.kendalltau(d_ac, ac_arr)
        log.info(f"  Spearman ρ(delay, AC1) = {rho_ac:.4f}, p = {p_ac:.4f}")
        log.info(f"  Kendall τ(delay, AC1) = {tau_ac:.4f}, p = {p_tau_ac:.4f}")
    else:
        rho_ac = p_ac = tau_ac = p_tau_ac = None

    # ── Save results ─────────────────────────────────────────────────
    output = {
        "description": (
            "Robustness test for control_delay_years threshold. "
            "Tests whether the AT/NT-like separation is robust to threshold choice "
            "or depends on a specific cherry-picked value."
        ),
        "data_summary": {
            "unique_delays": unique_delays,
            "gaps_over_50_years": gaps,
            "n_classified": sum(1 for c in corpora if c.get("predicted") is not None),
            "n_total": len(corpora),
        },
        "sweep_classified_only": {
            "analysis": analysis1,
            "sample_thresholds": [r for r in sweep1 if r["threshold"] in [1, 10, 20, 21, 50, 100, 150, 199, 200, 250, 300, 350, 400, 450, 500]],
        },
        "sweep_h_proxy_07": {
            "analysis": analysis2,
            "sample_thresholds": [r for r in sweep2 if r["threshold"] in [1, 10, 20, 21, 50, 100, 150, 199, 200, 250, 300, 350, 400, 450, 500]],
        },
        "sweep_h_proxy_065": {
            "analysis": analysis3,
        },
        "nonparametric_correlations": {
            "spearman_delay_H": {"rho": round(rho, 4), "p": round(p_spearman, 4)},
            "kendall_delay_H": {"tau": round(tau, 4), "p": round(p_kendall, 4)},
            "spearman_delay_AC1": (
                {"rho": round(rho_ac, 4), "p": round(p_ac, 4)}
                if rho_ac is not None else None
            ),
            "kendall_delay_AC1": (
                {"tau": round(tau_ac, 4), "p": round(p_tau_ac, 4)}
                if tau_ac is not None else None
            ),
        },
        "circularity_diagnostic": circularity,
        "needed_corpora": needed,
    }

    with open(RESULTS_DIR / "threshold_robustness.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    log.info(f"\nResultados en {RESULTS_DIR / 'threshold_robustness.json'}")

    # ── Final verdict ────────────────────────────────────────────────
    log.info(f"\n{'=' * 70}")
    log.info("VEREDICTO FINAL DE ROBUSTEZ")
    log.info(f"{'=' * 70}")
    log.info(f"  Rango de accuracy perfecta: {perf_min}-{perf_max} ({perf_width} años)")
    log.info(f"  Gap en datos: 21-199 ({180} años sin datos)")
    log.info(f"  Spearman ρ(delay, H) = {rho:.4f} (p={p_spearman:.4f})")
    log.info(f"  Diagnóstico: {circularity['verdict']}")
    log.info(f"  {circularity['interpretation']}")


if __name__ == "__main__":
    main()
