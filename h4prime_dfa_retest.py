#!/usr/bin/env python3
"""
Fase 20 — Script 4: h4prime_dfa_retest.py

H4' retest using DFA as primary metric:
1. Table of ~12 corpus with delay and DFA
2. Mann-Whitney: control_from_origin=Yes vs No in DFA
3. Spearman ρ: delay vs DFA
4. Linear regression: DFA ~ delay
5. Threshold sweep T ∈ {25, 50, 75, 100, 150, 200}: MW for each
6. Genre-controlled: repeat with narrative corpora only
7. LOO sensitivity
8. Verdict: CONFIRMADA / SUGERIDA / INDETERMINADA
"""

import json
import logging
import numpy as np
from pathlib import Path
from scipy import stats as sp_stats

BASE = Path(__file__).resolve().parent
RESULTS_DIR = BASE / "results" / "h4prime_dfa"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Genre classification for narrative filtering
# ═══════════════════════════════════════════════════════════════

NARRATIVE_CORPORA = {
    "AT_corpus", "NT_corpus", "Homer", "Herodotus",
}

# ═══════════════════════════════════════════════════════════════
# Load DFA data from results/
# ═══════════════════════════════════════════════════════════════

def load_corpus_table():
    """Build corpus table with delay and DFA. Skip unavailable."""
    corpora = []

    # data_matrix.json has the main corpora
    dm_file = BASE / "results" / "transmission_origin" / "data_matrix.json"
    if dm_file.exists():
        with open(dm_file) as f:
            dm = json.load(f)
        name_map = {
            "AT": "AT_corpus", "NT": "NT_corpus", "Corán": "Quran",
            "Rig_Veda": "Rig_Veda", "Homero": "Homer",
            "Heródoto": "Herodotus", "Book_of_Dead": "Book_of_Dead",
            "Pali_Canon": "Pali_Canon", "Mishnah": "Mishnah",
        }
        for entry in dm:
            name = name_map.get(entry["corpus"], entry["corpus"])
            if entry.get("DFA") is not None:
                corpora.append({
                    "corpus": name,
                    "DFA": float(entry["DFA"]),
                    "delay": int(entry.get("control_delay_years", 0)),
                    "control_from_origin": bool(entry.get("control_from_origin", False)),
                    "type": entry.get("transmission_type", ""),
                    "is_narrative": name in NARRATIVE_CORPORA,
                })

    # Individual metric files
    extras = [
        ("Didache", "didache/didache_metrics.json", 80, False, "early Christian text"),
        ("1_Clemente", "gap_corpora/1_clemente_metrics.json", 65, False,
         "early Christian letter"),
        ("Tosefta", "tosefta/tosefta_metrics.json", 170, False,
         "rabbinic supplement to Mishnah"),
    ]
    already = {c["corpus"] for c in corpora}
    for name, rel, delay, ctrl, ttype in extras:
        if name in already:
            continue
        fpath = BASE / "results" / rel
        if fpath.exists():
            with open(fpath) as f:
                data = json.load(f)
            dfa = data.get("DFA")
            if dfa is not None:
                corpora.append({
                    "corpus": name,
                    "DFA": float(dfa),
                    "delay": delay,
                    "control_from_origin": ctrl,
                    "type": ttype,
                    "is_narrative": False,
                })

    return corpora


# ═══════════════════════════════════════════════════════════════
# Main analysis
# ═══════════════════════════════════════════════════════════════

def main():
    log.info("=" * 70)
    log.info("FASE 20 — Script 4: H4' DFA Retest")
    log.info("=" * 70)

    # 1. Load corpus table
    corpora = load_corpus_table()
    log.info(f"  Loaded {len(corpora)} corpora")
    for c in corpora:
        log.info(f"    {c['corpus']:>15}: DFA={c['DFA']:.4f}, "
                 f"delay={c['delay']}, control={c['control_from_origin']}")

    # 2. Mann-Whitney: control=Yes vs No
    log.info("\n--- Mann-Whitney: Control vs No-Control ---")
    ctrl_dfa = [c["DFA"] for c in corpora if c["control_from_origin"]]
    noctrl_dfa = [c["DFA"] for c in corpora if not c["control_from_origin"]]

    mw_results = {}
    if len(ctrl_dfa) >= 2 and len(noctrl_dfa) >= 2:
        u, p = sp_stats.mannwhitneyu(ctrl_dfa, noctrl_dfa, alternative="two-sided")
        cohen_d = float((np.mean(ctrl_dfa) - np.mean(noctrl_dfa)) / np.sqrt(
            (np.var(ctrl_dfa, ddof=1) + np.var(noctrl_dfa, ddof=1)) / 2))
        mw_results = {
            "test": "Mann-Whitney: control_from_origin=Yes vs No",
            "n_control": len(ctrl_dfa),
            "n_no_control": len(noctrl_dfa),
            "control_mean_DFA": round(float(np.mean(ctrl_dfa)), 4),
            "no_control_mean_DFA": round(float(np.mean(noctrl_dfa)), 4),
            "U": round(float(u), 1),
            "p": round(float(p), 6),
            "cohen_d": round(cohen_d, 2),
            "significant": bool(p < 0.05),
            "control_corpora": [c["corpus"] for c in corpora
                                if c["control_from_origin"]],
            "no_control_corpora": [c["corpus"] for c in corpora
                                   if not c["control_from_origin"]],
        }
        log.info(f"  Control mean={np.mean(ctrl_dfa):.4f}, "
                 f"NoControl mean={np.mean(noctrl_dfa):.4f}")
        log.info(f"  U={u:.1f}, p={p:.6f}, Cohen d={cohen_d:.2f}")
    else:
        mw_results = {"status": "insufficient_data",
                      "n_control": len(ctrl_dfa),
                      "n_no_control": len(noctrl_dfa)}
        log.warning("  Insufficient data for Mann-Whitney")

    with open(RESULTS_DIR / "mann_whitney_results.json", "w") as f:
        json.dump(mw_results, f, indent=2, ensure_ascii=False)
    log.info("  Saved mann_whitney_results.json")

    # 3. Spearman: delay vs DFA
    log.info("\n--- Spearman: delay vs DFA ---")
    delays = np.array([c["delay"] for c in corpora], dtype=float)
    dfas = np.array([c["DFA"] for c in corpora], dtype=float)

    # Only use corpora with delay > 0 for correlation
    mask_nonzero = delays > 0
    corr_results = {}
    if np.sum(mask_nonzero) >= 4:
        rho, rho_p = sp_stats.spearmanr(delays[mask_nonzero], dfas[mask_nonzero])
        corr_results["delay_nonzero"] = {
            "n": int(np.sum(mask_nonzero)),
            "spearman_rho": round(float(rho), 4),
            "spearman_p": round(float(rho_p), 6),
            "significant": bool(rho_p < 0.05),
            "direction": "negative" if rho < 0 else "positive",
        }
        log.info(f"  delay>0: rho={rho:.4f}, p={rho_p:.6f}")

    # All corpora including delay=0
    if len(corpora) >= 4:
        rho_all, rho_p_all = sp_stats.spearmanr(delays, dfas)
        corr_results["all_corpora"] = {
            "n": len(corpora),
            "spearman_rho": round(float(rho_all), 4),
            "spearman_p": round(float(rho_p_all), 6),
            "significant": bool(rho_p_all < 0.05),
        }
        log.info(f"  All: rho={rho_all:.4f}, p={rho_p_all:.6f}")

    with open(RESULTS_DIR / "correlation_delay_dfa.json", "w") as f:
        json.dump(corr_results, f, indent=2, ensure_ascii=False)
    log.info("  Saved correlation_delay_dfa.json")

    # 4. Linear regression: DFA ~ delay
    log.info("\n--- Linear Regression: DFA ~ delay ---")
    if len(corpora) >= 4:
        slope, intercept, r, p_reg, se = sp_stats.linregress(delays, dfas)
        regression = {
            "slope": round(float(slope), 6),
            "intercept": round(float(intercept), 4),
            "r_squared": round(float(r ** 2), 4),
            "p_value": round(float(p_reg), 6),
            "std_error": round(float(se), 6),
            "n": len(corpora),
            "interpretation": (
                f"DFA = {intercept:.4f} + {slope:.6f} * delay, "
                f"R²={r**2:.4f}, p={p_reg:.4f}"
            ),
        }
        log.info(f"  DFA = {intercept:.4f} + {slope:.6f}*delay, "
                 f"R²={r**2:.4f}, p={p_reg:.6f}")
    else:
        regression = {"status": "insufficient_data"}

    # 5. Threshold sweep
    log.info("\n--- Threshold Sweep ---")
    thresholds = [25, 50, 75, 100, 150, 200]
    sweep_results = []

    for T in thresholds:
        early = [c["DFA"] for c in corpora if c["delay"] <= T]
        late = [c["DFA"] for c in corpora if c["delay"] > T]
        entry = {
            "threshold_years": T,
            "n_early": len(early),
            "n_late": len(late),
        }
        if len(early) >= 2 and len(late) >= 2:
            u, p = sp_stats.mannwhitneyu(early, late, alternative="two-sided")
            entry["early_mean_DFA"] = round(float(np.mean(early)), 4)
            entry["late_mean_DFA"] = round(float(np.mean(late)), 4)
            entry["U"] = round(float(u), 1)
            entry["p"] = round(float(p), 6)
            entry["significant"] = bool(p < 0.05)
            log.info(f"  T={T}: early={np.mean(early):.4f} (n={len(early)}), "
                     f"late={np.mean(late):.4f} (n={len(late)}), p={p:.4f}")
        else:
            entry["status"] = "insufficient_data"
            log.info(f"  T={T}: insufficient data "
                     f"(early={len(early)}, late={len(late)})")
        sweep_results.append(entry)

    with open(RESULTS_DIR / "threshold_sweep_dfa.json", "w") as f:
        json.dump(sweep_results, f, indent=2, ensure_ascii=False)
    log.info("  Saved threshold_sweep_dfa.json")

    # 6. Genre-controlled: narrative corpora only
    log.info("\n--- Genre-Controlled (Narrative Only) ---")
    narr_corpora = [c for c in corpora if c["is_narrative"]]
    gc_results = {"n_narrative": len(narr_corpora)}

    if len(narr_corpora) >= 4:
        narr_delays = np.array([c["delay"] for c in narr_corpora], dtype=float)
        narr_dfas = np.array([c["DFA"] for c in narr_corpora], dtype=float)

        rho, rho_p = sp_stats.spearmanr(narr_delays, narr_dfas)
        gc_results["spearman_rho"] = round(float(rho), 4)
        gc_results["spearman_p"] = round(float(rho_p), 6)
        gc_results["corpora"] = [c["corpus"] for c in narr_corpora]
        log.info(f"  Narrative: rho={rho:.4f}, p={rho_p:.6f}, n={len(narr_corpora)}")

        narr_ctrl = [c["DFA"] for c in narr_corpora if c["control_from_origin"]]
        narr_noctrl = [c["DFA"] for c in narr_corpora
                       if not c["control_from_origin"]]
        if len(narr_ctrl) >= 2 and len(narr_noctrl) >= 2:
            u, p = sp_stats.mannwhitneyu(narr_ctrl, narr_noctrl,
                                         alternative="two-sided")
            gc_results["mann_whitney_U"] = round(float(u), 1)
            gc_results["mann_whitney_p"] = round(float(p), 6)
            gc_results["significant"] = bool(p < 0.05)
        else:
            gc_results["mann_whitney"] = "insufficient_data"
    else:
        gc_results["status"] = "insufficient_narrative_corpora"
        log.info(f"  Only {len(narr_corpora)} narrative corpora — skipping")

    with open(RESULTS_DIR / "genre_controlled_dfa.json", "w") as f:
        json.dump(gc_results, f, indent=2, ensure_ascii=False)
    log.info("  Saved genre_controlled_dfa.json")

    # 7. LOO sensitivity
    log.info("\n--- LOO Sensitivity ---")
    loo_results = []
    for i, left_out in enumerate(corpora):
        remaining = [c for j, c in enumerate(corpora) if j != i]
        rem_delays = np.array([c["delay"] for c in remaining], dtype=float)
        rem_dfas = np.array([c["DFA"] for c in remaining], dtype=float)

        entry = {"left_out": left_out["corpus"]}
        if len(remaining) >= 4:
            rho, rho_p = sp_stats.spearmanr(rem_delays, rem_dfas)
            entry["spearman_rho"] = round(float(rho), 4)
            entry["spearman_p"] = round(float(rho_p), 6)
            entry["significant"] = bool(rho_p < 0.05)
        else:
            entry["status"] = "too_few_corpora"

        # MW test without this corpus
        rem_ctrl = [c["DFA"] for c in remaining if c["control_from_origin"]]
        rem_noctrl = [c["DFA"] for c in remaining if not c["control_from_origin"]]
        if len(rem_ctrl) >= 2 and len(rem_noctrl) >= 2:
            u, p = sp_stats.mannwhitneyu(rem_ctrl, rem_noctrl,
                                         alternative="two-sided")
            entry["mw_p"] = round(float(p), 6)
            entry["mw_significant"] = bool(p < 0.05)

        loo_results.append(entry)
        log.info(f"  LOO {left_out['corpus']:>15}: "
                 f"rho={entry.get('spearman_rho', '?')}, "
                 f"p={entry.get('spearman_p', '?')}")

    # 8. Verdict
    log.info("\n--- H4' DFA Verdict ---")

    # Criteria
    criteria = []

    # C1: MW control vs no-control is significant
    c1 = mw_results.get("significant", False)
    criteria.append({"name": "MW_control_significant", "met": bool(c1)})

    # C2: Spearman delay vs DFA is significant negative
    c2_data = corr_results.get("all_corpora", {})
    c2 = (c2_data.get("significant", False) and
          c2_data.get("spearman_rho", 0) < 0)
    criteria.append({"name": "Spearman_negative_significant", "met": bool(c2)})

    # C3: At least one threshold sweep is significant
    c3 = any(s.get("significant", False) for s in sweep_results)
    criteria.append({"name": "threshold_sweep_any_significant", "met": bool(c3)})

    # C4: LOO robust (>80% of LOO tests keep MW significant)
    n_loo_sig = sum(1 for r in loo_results if r.get("mw_significant", False))
    c4 = (n_loo_sig / len(loo_results)) >= 0.8 if loo_results else False
    criteria.append({
        "name": "LOO_robust_80pct",
        "met": bool(c4),
        "detail": f"{n_loo_sig}/{len(loo_results)} LOO MW tests significant"
    })

    n_met = sum(1 for c in criteria if c["met"])
    n_total = len(criteria)

    if n_met >= 3:
        verdict = "CONFIRMADA"
    elif n_met >= 2:
        verdict = "SUGERIDA"
    else:
        verdict = "INDETERMINADA"

    verdict_data = {
        "verdict": verdict,
        "n_criteria_met": n_met,
        "n_criteria_total": n_total,
        "criteria": criteria,
        "n_corpora_used": len(corpora),
        "corpora_table": [
            {
                "corpus": c["corpus"],
                "DFA": round(c["DFA"], 4),
                "delay": c["delay"],
                "control": bool(c["control_from_origin"]),
            }
            for c in corpora
        ],
        "mann_whitney": mw_results,
        "correlation": corr_results,
        "regression": regression,
        "threshold_sweep": sweep_results,
        "genre_controlled": gc_results,
        "loo_sensitivity": loo_results,
        "reasoning": (
            f"H4' DFA retest: {n_met}/{n_total} criteria met → {verdict}. "
            f"MW p={mw_results.get('p', '?')}, "
            f"Spearman rho={c2_data.get('spearman_rho', '?')} "
            f"(p={c2_data.get('spearman_p', '?')}), "
            f"LOO: {n_loo_sig}/{len(loo_results)} robust."
        ),
    }

    with open(RESULTS_DIR / "h4prime_dfa_verdict.json", "w") as f:
        json.dump(verdict_data, f, indent=2, ensure_ascii=False)
    log.info("  Saved h4prime_dfa_verdict.json")

    # Final
    log.info(f"\n{'=' * 70}")
    log.info(f"Script 4 completado.")
    log.info(f"  ╔══════════════════════════════════════════╗")
    log.info(f"  ║  H4' DFA VERDICT: {verdict:>22} ║")
    log.info(f"  ║  Criteria: {n_met}/{n_total}"
             f"{'':>29}║")
    log.info(f"  ╚══════════════════════════════════════════╝")


if __name__ == "__main__":
    main()
