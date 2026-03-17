#!/usr/bin/env python3
"""
Fase 19 — Script 3: h4prime_retest.py

Retest H4' hypothesis with CORRECTED classifier on expanded corpus set.
- Build 14-corpus table from corrected classifications + gap corpora
- Fisher exact test (control_from_origin × AT-like)
- Threshold sweep 25–200 years
- Logistic regression (delay → P_AT)
- Spearman/Kendall rank correlations
- LOO sensitivity analysis
- Bootstrap CI for Didache (n=101)
- Verdict: exactly one of CONFIRMADA / SUGERIDA / INDETERMINADA

Yasna INVALIDADO — NO incluir.
"""

import json
import logging
import numpy as np
from pathlib import Path
from scipy import stats as sp_stats
from scipy.optimize import minimize

BASE = Path(__file__).resolve().parent
RESULTS_DIR = BASE / "results" / "h4prime_retest"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# Delay lookup: years from origin of tradition to formal control
# ═══════════════════════════════════════════════════════════════

DELAY_LOOKUP = {
    "AT": 0,
    "Rig_Veda": 0,
    "Book_of_Dead": 0,
    "Pali_Canon": 0,
    "Corán": 20,
    "1_Clemente": 65,
    "Didache": 70,
    "Policarpo": 80,
    "Tosefta": 170,
    "Heródoto": 200,
    "NT": 300,
    "Mishnah": 400,
    "Homero": 400,
}


# ═══════════════════════════════════════════════════════════════
# Hurst R/S for bootstrap
# ═══════════════════════════════════════════════════════════════

def hurst_exponent_rs(series):
    series = np.asarray(series, dtype=float)
    n = len(series)
    if n < 20:
        return np.nan
    max_k, min_k = n // 2, 10
    ns, rs = [], []
    for k in range(min_k, max_k + 1):
        nc = n // k
        if nc < 1:
            continue
        rv = []
        for i in range(nc):
            chunk = series[i*k:(i+1)*k]
            m = np.mean(chunk)
            cum = np.cumsum(chunk - m)
            R = np.max(cum) - np.min(cum)
            S = np.std(chunk, ddof=1)
            if S > 0:
                rv.append(R / S)
        if rv:
            ns.append(k); rs.append(np.mean(rv))
    if len(ns) < 3:
        return np.nan
    slope, *_ = sp_stats.linregress(np.log(ns), np.log(rs))
    return round(slope, 4)


# ═══════════════════════════════════════════════════════════════
# Build corpus table
# ═══════════════════════════════════════════════════════════════

def build_corpus_table():
    """Load all classified corpora from corrected classifier + gap corpora."""
    table = []
    seen = set()

    # 1. Load corrected reclassification (AT, NT, Corán, Rig_Veda, etc.)
    reclass_file = BASE / "results" / "classifier_corrected" / "reclassification_all_corpora.json"
    if reclass_file.exists():
        with open(reclass_file) as f:
            reclassified = json.load(f)
        for name, data in reclassified.items():
            if data.get("status") != "classified":
                continue
            delay = DELAY_LOOKUP.get(name)
            if delay is None:
                log.warning(f"  No delay for {name}, skipping")
                continue
            table.append({
                "corpus": name,
                "delay": delay,
                "predicted": data["predicted_class"],
                "P_AT": data["P_AT"],
                "H": data["features"].get("H"),
                "AC1": data["features"].get("AC1"),
                "DFA": data["features"].get("DFA"),
                "CV": data["features"].get("CV"),
                "source": "corrected_classifier",
            })
            seen.add(name)

    # 2. Load gap corpora (1 Clemente, Policarpo, Tosefta corrected)
    gap_dir = BASE / "results" / "gap_corpora"
    gap_files = {
        "1_Clemente": "1_clemente_metrics.json",
        "Policarpo": "policarpo_metrics.json",
    }

    for corpus_name, fname in gap_files.items():
        fpath = gap_dir / fname
        if not fpath.exists():
            continue
        with open(fpath) as f:
            data = json.load(f)

        if data.get("status") == "excluded":
            log.info(f"  {corpus_name}: excluded ({data.get('reason')})")
            continue

        if corpus_name in seen:
            continue

        delay = DELAY_LOOKUP.get(corpus_name) or data.get("delay_years")
        if delay is None:
            continue

        pred = data.get("predicted_class")
        p_at = data.get("P_AT")
        if pred is None:
            continue

        table.append({
            "corpus": corpus_name,
            "delay": delay,
            "predicted": pred,
            "P_AT": p_at,
            "H": data.get("H"),
            "AC1": data.get("AC1"),
            "DFA": data.get("DFA"),
            "CV": data.get("CV"),
            "source": "gap_corpora",
        })
        seen.add(corpus_name)

    # Tosefta corrected: update existing entry or add new
    tosefta_file = gap_dir / "tosefta_corrected.json"
    if tosefta_file.exists():
        with open(tosefta_file) as f:
            tdata = json.load(f)
        if "Tosefta" in seen:
            # Update with corrected classification
            for c in table:
                if c["corpus"] == "Tosefta":
                    if tdata.get("predicted_class_corrected"):
                        c["predicted"] = tdata["predicted_class_corrected"]
                        c["P_AT"] = tdata.get("P_AT_corrected", c["P_AT"])
                    c["delay"] = DELAY_LOOKUP.get("Tosefta", 170)
                    break
        elif "error" not in tdata:
            pred = tdata.get("predicted_class_corrected") or tdata.get("predicted_class")
            p_at = tdata.get("P_AT_corrected") or tdata.get("P_AT")
            if pred:
                table.append({
                    "corpus": "Tosefta",
                    "delay": DELAY_LOOKUP.get("Tosefta", 170),
                    "predicted": pred,
                    "P_AT": p_at,
                    "H": tdata.get("H"),
                    "AC1": tdata.get("AC1"),
                    "DFA": tdata.get("DFA"),
                    "CV": tdata.get("CV"),
                    "source": "gap_corpora (corrected)",
                })

    table.sort(key=lambda x: x["delay"])
    return table


# ═══════════════════════════════════════════════════════════════
# Fisher exact test
# ═══════════════════════════════════════════════════════════════

def fisher_exact_test(table, threshold):
    """2×2 Fisher exact: early/late × AT/NT."""
    early_at = sum(1 for c in table if c["delay"] <= threshold and c["predicted"] == "AT")
    early_nt = sum(1 for c in table if c["delay"] <= threshold and c["predicted"] == "NT")
    late_at = sum(1 for c in table if c["delay"] > threshold and c["predicted"] == "AT")
    late_nt = sum(1 for c in table if c["delay"] > threshold and c["predicted"] == "NT")

    contingency = [[early_at, early_nt], [late_at, late_nt]]
    odds, p = sp_stats.fisher_exact(contingency, alternative="two-sided")

    total = early_at + early_nt + late_at + late_nt
    accuracy = (early_at + late_nt) / total if total > 0 else 0

    return {
        "threshold": int(threshold),
        "early_at": int(early_at), "early_nt": int(early_nt),
        "late_at": int(late_at), "late_nt": int(late_nt),
        "accuracy": round(float(accuracy), 4),
        "fisher_p": round(float(p), 6),
        "odds_ratio": round(float(odds), 4) if np.isfinite(odds) else "inf",
    }


# ═══════════════════════════════════════════════════════════════
# Threshold sweep
# ═══════════════════════════════════════════════════════════════

def threshold_sweep(table):
    """Sweep thresholds 25–200 in steps of 5."""
    results = []
    for t in range(25, 201, 5):
        results.append(fisher_exact_test(table, t))

    best = min(results, key=lambda x: x["fisher_p"])

    perfect = [r for r in results if r["accuracy"] == 1.0]
    perfect_range = None
    if perfect:
        perfect_range = {
            "min": perfect[0]["threshold"],
            "max": perfect[-1]["threshold"],
            "width": perfect[-1]["threshold"] - perfect[0]["threshold"],
        }

    significant = [r for r in results if r["fisher_p"] < 0.05]
    sig_range = None
    if significant:
        sig_range = {
            "min": significant[0]["threshold"],
            "max": significant[-1]["threshold"],
            "width": significant[-1]["threshold"] - significant[0]["threshold"],
        }

    return {
        "sweep_range": [25, 200],
        "step": 5,
        "n_thresholds": len(results),
        "best_threshold": best,
        "perfect_range": perfect_range,
        "significant_range": sig_range,
        "all_thresholds": results,
    }


# ═══════════════════════════════════════════════════════════════
# Logistic regression
# ═══════════════════════════════════════════════════════════════

def logistic_regression_delay(table):
    """Logistic regression: P(AT-like) = σ(β₀ + β₁·delay)."""
    delays = np.array([c["delay"] for c in table], dtype=float)
    labels = np.array([1 if c["predicted"] == "AT" else 0 for c in table])

    def neg_log_likelihood(params):
        b0, b1 = params
        z = b0 + b1 * delays
        z = np.clip(z, -500, 500)
        p = 1 / (1 + np.exp(-z))
        p = np.clip(p, 1e-10, 1 - 1e-10)
        return -np.sum(labels * np.log(p) + (1 - labels) * np.log(1 - p))

    res = minimize(neg_log_likelihood, [0, 0], method="Nelder-Mead")
    b0, b1 = res.x

    delay_50 = -b0 / b1 if b1 != 0 else None

    return {
        "intercept": round(float(b0), 4),
        "slope": round(float(b1), 6),
        "delay_at_P50": round(float(delay_50), 1) if delay_50 is not None else None,
        "slope_interpretation": (
            "negative → more delay = less AT-like" if b1 < 0 else
            "positive or zero → no degradation signal"
        ),
        "converged": bool(res.success),
    }


# ═══════════════════════════════════════════════════════════════
# Spearman / Kendall correlations
# ═══════════════════════════════════════════════════════════════

def rank_correlations(table):
    """Spearman and Kendall correlations: delay vs P_AT, delay vs H."""
    delays = [c["delay"] for c in table]
    p_ats = [c["P_AT"] for c in table]
    hs = [c["H"] for c in table if c["H"] is not None]
    delays_h = [c["delay"] for c in table if c["H"] is not None]

    rho_pat, p_pat = sp_stats.spearmanr(delays, p_ats)
    tau_pat, p_tau_pat = sp_stats.kendalltau(delays, p_ats)

    result = {
        "delay_vs_P_AT": {
            "spearman_rho": round(float(rho_pat), 4),
            "spearman_p": round(float(p_pat), 6),
            "kendall_tau": round(float(tau_pat), 4),
            "kendall_p": round(float(p_tau_pat), 6),
        },
    }

    if len(hs) >= 5:
        rho_h, p_h = sp_stats.spearmanr(delays_h, hs)
        tau_h, p_tau_h = sp_stats.kendalltau(delays_h, hs)
        result["delay_vs_H"] = {
            "spearman_rho": round(float(rho_h), 4),
            "spearman_p": round(float(p_h), 6),
            "kendall_tau": round(float(tau_h), 4),
            "kendall_p": round(float(p_tau_h), 6),
        }

    return result


# ═══════════════════════════════════════════════════════════════
# LOO sensitivity
# ═══════════════════════════════════════════════════════════════

def loo_sensitivity(table):
    """Remove each corpus, check if Fisher significance changes."""
    # Find best threshold from full sweep
    sweep = threshold_sweep(table)
    best_t = sweep["best_threshold"]["threshold"]

    base = fisher_exact_test(table, best_t)
    base_sig = bool(base["fisher_p"] < 0.05)

    loo_results = []
    for i in range(len(table)):
        reduced = table[:i] + table[i+1:]
        r = fisher_exact_test(reduced, best_t)
        flipped = bool((r["fisher_p"] < 0.05) != base_sig)
        loo_results.append({
            "removed": table[i]["corpus"],
            "delay": table[i]["delay"],
            "accuracy": float(r["accuracy"]),
            "fisher_p": float(r["fisher_p"]),
            "significance_flipped": flipped,
        })

    critical = [r for r in loo_results if r["significance_flipped"]]

    return {
        "base_threshold": int(best_t),
        "base_fisher_p": float(base["fisher_p"]),
        "base_significant": base_sig,
        "n_corpora": len(table),
        "n_critical": len(critical),
        "critical_corpora": [r["removed"] for r in critical],
        "details": loo_results,
    }


# ═══════════════════════════════════════════════════════════════
# Bootstrap CI for Didache
# ═══════════════════════════════════════════════════════════════

def bootstrap_didache(n_boot=2000):
    """Bootstrap CI for Didache H (n=101 segments, small corpus)."""
    didache_file = BASE / "results" / "didache" / "didache_metrics.json"
    if not didache_file.exists():
        return None

    with open(didache_file) as f:
        dm = json.load(f)

    n_seg = dm.get("n_segments", 101)
    mean_len = dm.get("mean_verse_len", 21.77)
    std_len = dm.get("std_verse_len", 13.31)
    h_point = dm.get("H", 0.7096)

    # Simulate verse lengths from lognormal (positive, right-skewed)
    rng = np.random.RandomState(42)
    sigma_ln = np.sqrt(np.log(1 + (std_len / mean_len) ** 2))
    mu_ln = np.log(mean_len) - sigma_ln ** 2 / 2

    boot_H = []
    for _ in range(n_boot):
        sample = rng.lognormal(mu_ln, sigma_ln, size=n_seg).astype(int)
        sample = np.maximum(sample, 1)
        h = hurst_exponent_rs(sample)
        if not np.isnan(h):
            boot_H.append(h)

    if len(boot_H) < 100:
        return {"status": "insufficient_bootstrap_samples", "n_valid": len(boot_H)}

    ci_lo = round(float(np.percentile(boot_H, 2.5)), 4)
    ci_hi = round(float(np.percentile(boot_H, 97.5)), 4)

    return {
        "H_point": h_point,
        "n_segments": n_seg,
        "n_bootstrap": n_boot,
        "n_valid": len(boot_H),
        "H_bootstrap_mean": round(float(np.mean(boot_H)), 4),
        "H_bootstrap_std": round(float(np.std(boot_H)), 4),
        "H_CI_95": [ci_lo, ci_hi],
        "CI_width": round(ci_hi - ci_lo, 4),
        "classification_robust": bool(h_point > ci_lo),
    }


# ═══════════════════════════════════════════════════════════════
# Verdict
# ═══════════════════════════════════════════════════════════════

def determine_verdict(table, sweep, fisher_best, correlations, loo, bootstrap):
    """Exactly one of: CONFIRMADA, SUGERIDA, INDETERMINADA."""

    fisher_sig = fisher_best["fisher_p"] < 0.05

    delay_vs_pat = correlations.get("delay_vs_P_AT", {})
    spearman_monotone = (
        delay_vs_pat.get("spearman_rho", 0) < -0.3 and
        delay_vs_pat.get("spearman_p", 1) < 0.1
    )

    loo_robust = loo["n_critical"] == 0

    perfect_range_width = 0
    if sweep.get("perfect_range"):
        perfect_range_width = sweep["perfect_range"]["width"]

    sig_range_width = 0
    if sweep.get("significant_range"):
        sig_range_width = sweep["significant_range"]["width"]

    criteria = {
        "fisher_p_lt_0.05": bool(fisher_sig),
        "spearman_monotone_decrease": bool(spearman_monotone),
        "loo_robust": bool(loo_robust),
        "perfect_range_gt_30": bool(perfect_range_width > 30),
        "significant_range_gt_50": bool(sig_range_width > 50),
    }
    n_met = int(sum(criteria.values()))

    if n_met >= 4:
        verdict = "CONFIRMADA"
        reasoning = (
            f"H4' se confirma: {n_met}/5 criterios cumplidos. "
            "El delay de transmisión predice significativamente la clasificación AT/NT "
            "con el clasificador corregido (sin sesgo de mean_verse_len)."
        )
    elif n_met >= 2:
        verdict = "SUGERIDA"
        reasoning = (
            f"H4' sugerida pero no robusta: {n_met}/5 criterios cumplidos. "
            "La tendencia existe pero la evidencia estadística no es concluyente."
        )
    else:
        verdict = "INDETERMINADA"
        reasoning = (
            f"H4' indeterminada: solo {n_met}/5 criterios cumplidos. "
            "Los datos no permiten confirmar ni descartar la hipótesis."
        )

    return {
        "verdict": verdict,
        "criteria": criteria,
        "n_criteria_met": n_met,
        "n_criteria_total": 5,
        "reasoning": reasoning,
        "n_corpora_used": len(table),
    }


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    log.info("=" * 70)
    log.info("FASE 19 — Script 3: H4' Retest con clasificador corregido")
    log.info("=" * 70)

    # 1. Build corpus table
    log.info("\n[1] Construyendo tabla de corpus...")
    table = build_corpus_table()
    log.info(f"  {len(table)} corpus en tabla:")
    for c in table:
        log.info(f"    {c['corpus']:>15}: delay={c['delay']:>3}, "
                 f"pred={c['predicted']}, P_AT={c['P_AT']:.4f}, H={c.get('H')}")

    with open(RESULTS_DIR / "corpus_table.json", "w") as f:
        json.dump(table, f, indent=2, ensure_ascii=False)

    if len(table) < 6:
        log.error(f"  Insuficientes corpus ({len(table)} < 6)")
        with open(RESULTS_DIR / "verdict.json", "w") as f:
            json.dump({"verdict": "INDETERMINADA",
                        "reasoning": "Insuficientes corpus"}, f, indent=2)
        return

    # 2. Fisher exact at best threshold
    log.info("\n[2] Fisher exact test...")
    # Test at several key thresholds
    for t in [25, 50, 65, 70, 100, 170]:
        r = fisher_exact_test(table, t)
        log.info(f"  threshold={t}: acc={r['accuracy']}, p={r['fisher_p']}")

    # 3. Threshold sweep
    log.info("\n[3] Threshold sweep 25–200...")
    sweep = threshold_sweep(table)
    best = sweep["best_threshold"]
    log.info(f"  Best: threshold={best['threshold']}, p={best['fisher_p']}, "
             f"acc={best['accuracy']}")
    if sweep.get("perfect_range"):
        pr = sweep["perfect_range"]
        log.info(f"  Perfect range: {pr['min']}–{pr['max']} ({pr['width']} años)")
    if sweep.get("significant_range"):
        sr = sweep["significant_range"]
        log.info(f"  Significant range: {sr['min']}–{sr['max']} ({sr['width']} años)")

    with open(RESULTS_DIR / "threshold_sweep.json", "w") as f:
        json.dump(sweep, f, indent=2, ensure_ascii=False)

    # 4. Logistic regression
    log.info("\n[4] Logistic regression delay → P(AT)...")
    logit = logistic_regression_delay(table)
    log.info(f"  β₁={logit['slope']}, delay@P50={logit['delay_at_P50']}")

    with open(RESULTS_DIR / "logistic_regression.json", "w") as f:
        json.dump(logit, f, indent=2, ensure_ascii=False)

    # 5. Rank correlations
    log.info("\n[5] Spearman/Kendall correlations...")
    correlations = rank_correlations(table)
    dp = correlations.get("delay_vs_P_AT", {})
    log.info(f"  delay vs P_AT: ρ={dp.get('spearman_rho')} (p={dp.get('spearman_p')})")
    dh = correlations.get("delay_vs_H", {})
    if dh:
        log.info(f"  delay vs H: ρ={dh.get('spearman_rho')} (p={dh.get('spearman_p')})")

    with open(RESULTS_DIR / "correlations.json", "w") as f:
        json.dump(correlations, f, indent=2, ensure_ascii=False)

    # 6. LOO sensitivity
    log.info("\n[6] LOO sensitivity analysis...")
    loo = loo_sensitivity(table)
    log.info(f"  {loo['n_critical']} critical corpora: {loo['critical_corpora']}")

    with open(RESULTS_DIR / "loo_sensitivity.json", "w") as f:
        json.dump(loo, f, indent=2, ensure_ascii=False)

    # 7. Bootstrap Didache
    log.info("\n[7] Bootstrap CI for Didache...")
    bootstrap = bootstrap_didache()
    if bootstrap and "H_CI_95" in bootstrap:
        log.info(f"  H_point={bootstrap['H_point']}, "
                 f"95% CI={bootstrap['H_CI_95']}, width={bootstrap['CI_width']}")
    else:
        log.info("  Didache bootstrap skipped or failed")

    with open(RESULTS_DIR / "didache_bootstrap.json", "w") as f:
        json.dump(bootstrap, f, indent=2, ensure_ascii=False)

    # 8. Verdict
    log.info("\n[8] Determinando veredicto...")
    verdict = determine_verdict(table, sweep, best, correlations, loo, bootstrap)

    # Attach all analysis results
    verdict["fisher_best"] = best
    verdict["sweep_summary"] = {
        "perfect_range": sweep.get("perfect_range"),
        "significant_range": sweep.get("significant_range"),
    }
    verdict["logistic"] = logit
    verdict["correlations"] = correlations
    verdict["loo_summary"] = {
        "n_critical": loo["n_critical"],
        "critical_corpora": loo["critical_corpora"],
        "base_threshold": loo["base_threshold"],
    }
    verdict["didache_bootstrap"] = bootstrap

    with open(RESULTS_DIR / "verdict.json", "w") as f:
        json.dump(verdict, f, indent=2, ensure_ascii=False)

    log.info(f"\n  ╔══════════════════════════════════════════╗")
    log.info(f"  ║  VEREDICTO H4': {verdict['verdict']:>24} ║")
    log.info(f"  ╚══════════════════════════════════════════╝")
    log.info(f"  Criterios: {verdict['n_criteria_met']}/{verdict['n_criteria_total']}")
    log.info(f"  {verdict['reasoning']}")

    log.info(f"\n{'=' * 70}")
    log.info("Script 3 completado.")


if __name__ == "__main__":
    main()
