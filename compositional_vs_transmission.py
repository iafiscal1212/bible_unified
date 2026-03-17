#!/usr/bin/env python3
"""
Fase 21 - Script 4: compositional_vs_transmission.py

Compare compositional vs transmission models for cross-corpus DFA:
1. Build 12-corpus table with DFA, AC1, H, delay, control
2. Model TRANS: DFA ~ control + delay
3. Model COMP: DFA ~ AC1 + H (10 corpora with AC1)
4. LOO predictions
5. Anomaly analysis (Mishnah, Quran, Rig Veda)
6. Verdict: COMPOSICIONAL / MIXTA / TRANSMISION
"""

import json
import logging
import numpy as np
from pathlib import Path
from scipy import stats as sp_stats

BASE = Path(__file__).resolve().parent
RESULTS_DIR = BASE / "results" / "compositional_hypothesis"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)


def ols_fit(X, y):
    """OLS fit returning beta, R2, RMSE, AIC, predictions."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    X_wi = np.column_stack([np.ones(n), X])
    k = X_wi.shape[1]

    try:
        beta = np.linalg.solve(X_wi.T @ X_wi, X_wi.T @ y)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(X_wi, y, rcond=None)[0]

    y_pred = X_wi @ beta
    residuals = y - y_pred
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    rmse = float(np.sqrt(ss_res / n))
    aic = float(n * np.log(ss_res / n) + 2 * k) if ss_res > 0 else float('inf')

    # Standard errors
    mse = ss_res / max(n - k, 1)
    try:
        var_beta = mse * np.linalg.inv(X_wi.T @ X_wi)
    except np.linalg.LinAlgError:
        var_beta = mse * np.linalg.pinv(X_wi.T @ X_wi)
    se_beta = np.sqrt(np.abs(np.diag(var_beta)))

    t_stats = beta / np.where(se_beta > 0, se_beta, 1e-10)
    p_values = 2 * sp_stats.t.sf(np.abs(t_stats), df=max(n - k, 1))

    return {
        "beta": beta,
        "predictions": y_pred,
        "residuals": residuals,
        "R2": float(r2),
        "RMSE": rmse,
        "AIC": aic,
        "se": se_beta,
        "t_stats": t_stats,
        "p_values": p_values,
        "n": n,
        "k": k,
    }


def loo_predictions(X, y):
    """Leave-one-out predictions."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    preds = np.zeros(n)

    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        X_train = X[mask]
        y_train = y[mask]

        if X_train.ndim == 1:
            X_train = X_train.reshape(-1, 1)

        X_wi = np.column_stack([np.ones(n - 1), X_train])
        try:
            beta = np.linalg.solve(X_wi.T @ X_wi, X_wi.T @ y_train)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(X_wi, y_train, rcond=None)[0]

        x_test = X[i]
        if x_test.ndim == 0:
            x_test = np.array([x_test])
        preds[i] = float(np.dot(np.concatenate([[1], x_test]), beta))

    return preds


def main():
    log.info("=" * 70)
    log.info("FASE 21 - Script 4: Compositional vs Transmission Models")
    log.info("=" * 70)

    # ──────────────────────────────────────────────────────────
    # 1. Build 12-corpus table
    # ──────────────────────────────────────────────────────────
    log.info("\n--- Building corpus table ---")

    with open(BASE / "results" / "transmission_origin" / "data_matrix.json") as f:
        dm = json.load(f)

    # Index by corpus name
    corpus_data = {}
    for entry in dm:
        corpus_data[entry["corpus"]] = entry

    # Add extra corpora from individual files
    extras = {
        "Didache": {
            "file": "didache/didache_metrics.json",
            "delay": 50, "control": False,
        },
        "1_Clemente": {
            "file": "gap_corpora/1_clemente_metrics.json",
            "delay": 65, "control": False,
        },
        "Tosefta": {
            "file": "tosefta/tosefta_metrics.json",
            "delay": 170, "control": False,
        },
    }

    for name, info in extras.items():
        fpath = BASE / "results" / info["file"]
        if fpath.exists():
            with open(fpath) as f:
                data = json.load(f)
            corpus_data[name] = {
                "corpus": name,
                "H": data.get("H"),
                "AC1": data.get("AC1"),
                "DFA": data.get("DFA"),
                "control_from_origin": info["control"],
                "control_delay_years": info["delay"],
            }

    # Normalize field names
    all_corpora = []
    for name, entry in corpus_data.items():
        row = {
            "corpus": name,
            "DFA": entry.get("DFA"),
            "AC1": entry.get("AC1"),
            "H": entry.get("H"),
            "delay": entry.get("control_delay_years", 0),
            "control": 1 if entry.get("control_from_origin", False) else 0,
        }
        if row["DFA"] is not None:
            all_corpora.append(row)

    log.info(f"  {len(all_corpora)} corpora total")
    for c in all_corpora:
        log.info(f"    {c['corpus']}: DFA={c['DFA']}, AC1={c['AC1']}, "
                 f"delay={c['delay']}, control={c['control']}")

    # Separate: all (for TRANS) vs those with AC1 (for COMP)
    corpora_all = all_corpora
    corpora_with_ac1 = [c for c in all_corpora if c["AC1"] is not None]

    n_all = len(corpora_all)
    n_ac1 = len(corpora_with_ac1)
    log.info(f"  TRANS model: {n_all} corpora")
    log.info(f"  COMP model: {n_ac1} corpora (with AC1)")

    # ──────────────────────────────────────────────────────────
    # 2. Model TRANS: DFA ~ control + delay
    # ──────────────────────────────────────────────────────────
    log.info("\n--- Model TRANS: DFA ~ control + delay ---")

    y_all = np.array([c["DFA"] for c in corpora_all])
    X_trans = np.column_stack([
        [c["control"] for c in corpora_all],
        [c["delay"] for c in corpora_all],
    ])

    fit_trans = ols_fit(X_trans, y_all)
    log.info(f"  R2={fit_trans['R2']:.4f}, RMSE={fit_trans['RMSE']:.4f}")
    log.info(f"  beta_control={fit_trans['beta'][1]:.4f} (p={fit_trans['p_values'][1]:.4f})")
    log.info(f"  beta_delay={fit_trans['beta'][2]:.6f} (p={fit_trans['p_values'][2]:.4f})")

    # ──────────────────────────────────────────────────────────
    # 3. Model COMP: DFA ~ AC1 + H
    # ──────────────────────────────────────────────────────────
    log.info("\n--- Model COMP: DFA ~ AC1 + H ---")

    y_ac1 = np.array([c["DFA"] for c in corpora_with_ac1])
    X_comp = np.column_stack([
        [c["AC1"] for c in corpora_with_ac1],
        [c["H"] for c in corpora_with_ac1],
    ])

    fit_comp = ols_fit(X_comp, y_ac1)
    log.info(f"  R2={fit_comp['R2']:.4f}, RMSE={fit_comp['RMSE']:.4f}")
    log.info(f"  beta_AC1={fit_comp['beta'][1]:.4f} (p={fit_comp['p_values'][1]:.4f})")
    log.info(f"  beta_H={fit_comp['beta'][2]:.4f} (p={fit_comp['p_values'][2]:.4f})")

    # Model COMP_simple: DFA ~ AC1 only
    log.info("\n--- Model COMP_simple: DFA ~ AC1 ---")
    X_comp_simple = np.array([c["AC1"] for c in corpora_with_ac1])
    fit_comp_s = ols_fit(X_comp_simple, y_ac1)
    log.info(f"  R2={fit_comp_s['R2']:.4f}, RMSE={fit_comp_s['RMSE']:.4f}")

    # Model TRANS on same n (for fair comparison)
    corpora_ac1_names = {c["corpus"] for c in corpora_with_ac1}
    corpora_trans_subset = [c for c in corpora_all if c["corpus"] in corpora_ac1_names]
    y_trans_sub = np.array([c["DFA"] for c in corpora_trans_subset])
    X_trans_sub = np.column_stack([
        [c["control"] for c in corpora_trans_subset],
        [c["delay"] for c in corpora_trans_subset],
    ])
    fit_trans_sub = ols_fit(X_trans_sub, y_trans_sub)
    log.info(f"  TRANS (same {n_ac1} corpora): R2={fit_trans_sub['R2']:.4f}")

    # ──────────────────────────────────────────────────────────
    # 4. LOO predictions
    # ──────────────────────────────────────────────────────────
    log.info("\n--- LOO Predictions ---")

    # TRANS LOO (all corpora)
    loo_trans = loo_predictions(X_trans, y_all)
    mae_trans = float(np.mean(np.abs(y_all - loo_trans)))

    # COMP LOO (AC1 corpora only)
    loo_comp = loo_predictions(X_comp, y_ac1)
    mae_comp = float(np.mean(np.abs(y_ac1 - loo_comp)))

    # TRANS LOO on same subset
    loo_trans_sub = loo_predictions(X_trans_sub, y_trans_sub)
    mae_trans_sub = float(np.mean(np.abs(y_trans_sub - loo_trans_sub)))

    log.info(f"  TRANS LOO MAE (all {n_all}): {mae_trans:.4f}")
    log.info(f"  COMP LOO MAE ({n_ac1} with AC1): {mae_comp:.4f}")
    log.info(f"  TRANS LOO MAE (same {n_ac1}): {mae_trans_sub:.4f}")

    loo_results = {}
    for i, c in enumerate(corpora_all):
        loo_results[c["corpus"]] = {
            "DFA_actual": round(float(c["DFA"]), 4),
            "DFA_pred_TRANS": round(float(loo_trans[i]), 4),
            "residual_TRANS": round(float(c["DFA"] - loo_trans[i]), 4),
        }

    for i, c in enumerate(corpora_with_ac1):
        name = c["corpus"]
        if name in loo_results:
            loo_results[name]["DFA_pred_COMP"] = round(float(loo_comp[i]), 4)
            loo_results[name]["residual_COMP"] = round(float(c["DFA"] - loo_comp[i]), 4)

    loo_results["_summary"] = {
        "MAE_TRANS_all": round(mae_trans, 4),
        "MAE_COMP_ac1": round(mae_comp, 4),
        "MAE_TRANS_same_subset": round(mae_trans_sub, 4),
    }

    with open(RESULTS_DIR / "loo_predictions.json", "w") as f:
        json.dump(loo_results, f, indent=2, ensure_ascii=False)
    log.info("  Saved loo_predictions.json")

    # ──────────────────────────────────────────────────────────
    # 5. Save model fits
    # ──────────────────────────────────────────────────────────
    model_fits = {
        "TRANS_all": {
            "formula": "DFA ~ control + delay",
            "n": int(n_all),
            "R2": round(float(fit_trans["R2"]), 4),
            "RMSE": round(float(fit_trans["RMSE"]), 4),
            "AIC": round(float(fit_trans["AIC"]), 2),
            "intercept": round(float(fit_trans["beta"][0]), 4),
            "beta_control": round(float(fit_trans["beta"][1]), 4),
            "beta_delay": round(float(fit_trans["beta"][2]), 6),
            "p_control": round(float(fit_trans["p_values"][1]), 6),
            "p_delay": round(float(fit_trans["p_values"][2]), 6),
        },
        "COMP_full": {
            "formula": "DFA ~ AC1 + H",
            "n": int(n_ac1),
            "R2": round(float(fit_comp["R2"]), 4),
            "RMSE": round(float(fit_comp["RMSE"]), 4),
            "AIC": round(float(fit_comp["AIC"]), 2),
            "intercept": round(float(fit_comp["beta"][0]), 4),
            "beta_AC1": round(float(fit_comp["beta"][1]), 4),
            "beta_H": round(float(fit_comp["beta"][2]), 4),
            "p_AC1": round(float(fit_comp["p_values"][1]), 6),
            "p_H": round(float(fit_comp["p_values"][2]), 6),
        },
        "COMP_simple": {
            "formula": "DFA ~ AC1",
            "n": int(n_ac1),
            "R2": round(float(fit_comp_s["R2"]), 4),
            "RMSE": round(float(fit_comp_s["RMSE"]), 4),
            "AIC": round(float(fit_comp_s["AIC"]), 2),
            "intercept": round(float(fit_comp_s["beta"][0]), 4),
            "beta_AC1": round(float(fit_comp_s["beta"][1]), 4),
            "p_AC1": round(float(fit_comp_s["p_values"][1]), 6),
        },
        "TRANS_same_subset": {
            "formula": "DFA ~ control + delay (same n as COMP)",
            "n": int(n_ac1),
            "R2": round(float(fit_trans_sub["R2"]), 4),
            "RMSE": round(float(fit_trans_sub["RMSE"]), 4),
            "AIC": round(float(fit_trans_sub["AIC"]), 2),
        },
    }

    with open(RESULTS_DIR / "model_fits.json", "w") as f:
        json.dump(model_fits, f, indent=2, ensure_ascii=False)
    log.info("  Saved model_fits.json")

    # ──────────────────────────────────────────────────────────
    # 6. Anomaly analysis
    # ──────────────────────────────────────────────────────────
    log.info("\n--- Anomaly Analysis ---")

    anomaly = {}

    for target in ["Mishnah", "Corán", "Rig_Veda"]:
        # Find in results
        trans_entry = loo_results.get(target, {})
        comp_entry = loo_results.get(target, {})

        corpus_info = corpus_data.get(target, {})

        entry = {
            "DFA_actual": trans_entry.get("DFA_actual"),
            "DFA_pred_TRANS": trans_entry.get("DFA_pred_TRANS"),
            "DFA_pred_COMP": comp_entry.get("DFA_pred_COMP"),
            "residual_TRANS": trans_entry.get("residual_TRANS"),
            "residual_COMP": comp_entry.get("residual_COMP"),
            "AC1": corpus_info.get("AC1"),
            "H": corpus_info.get("H"),
            "delay": corpus_info.get("control_delay_years"),
            "control": corpus_info.get("control_from_origin"),
        }

        if entry.get("residual_COMP") is not None and entry.get("residual_TRANS") is not None:
            entry["COMP_better"] = bool(
                abs(entry["residual_COMP"]) < abs(entry["residual_TRANS"])
            )

        anomaly[target] = entry
        log.info(f"  {target}: res_TRANS={entry.get('residual_TRANS')}, "
                 f"res_COMP={entry.get('residual_COMP')}")

    with open(RESULTS_DIR / "anomaly_analysis.json", "w") as f:
        json.dump(anomaly, f, indent=2, ensure_ascii=False)
    log.info("  Saved anomaly_analysis.json")

    # ──────────────────────────────────────────────────────────
    # 7. Verdict
    # ──────────────────────────────────────────────────────────
    log.info("\n--- Hypothesis Verdict ---")

    # C1: R2 COMP > R2 TRANS (on same n, difference > 0.10)
    c1 = bool(fit_comp["R2"] - fit_trans_sub["R2"] > 0.10)

    # C2: LOO MAE COMP < LOO MAE TRANS (same subset)
    c2 = bool(mae_comp < mae_trans_sub)

    # C3: Mishnah better predicted by COMP
    mishnah_entry = anomaly.get("Mishnah", {})
    c3 = bool(mishnah_entry.get("COMP_better", False))

    # C4: Spearman rho(AC1, DFA) > |rho(delay, DFA)| on the AC1 corpora
    ac1_vals = np.array([c["AC1"] for c in corpora_with_ac1])
    dfa_vals = np.array([c["DFA"] for c in corpora_with_ac1])
    delay_vals = np.array([c["delay"] for c in corpora_with_ac1])

    rho_ac1_dfa, _ = sp_stats.spearmanr(ac1_vals, dfa_vals)
    rho_delay_dfa, _ = sp_stats.spearmanr(delay_vals, dfa_vals)
    c4 = bool(abs(rho_ac1_dfa) > abs(rho_delay_dfa))

    criteria_met = sum([c1, c2, c3, c4])

    if criteria_met >= 4:
        verdict = "COMPOSICIONAL"
    elif criteria_met == 3:
        verdict = "COMPOSICIONAL (sugerida)"
    elif criteria_met == 2:
        verdict = "MIXTA"
    else:
        verdict = "TRANSMISION"

    verdict_result = {
        "criteria": {
            "C1_R2_COMP_gt_TRANS": {
                "met": c1,
                "R2_COMP": round(float(fit_comp["R2"]), 4),
                "R2_TRANS": round(float(fit_trans_sub["R2"]), 4),
                "difference": round(float(fit_comp["R2"] - fit_trans_sub["R2"]), 4),
            },
            "C2_LOO_MAE_COMP_lt_TRANS": {
                "met": c2,
                "MAE_COMP": round(mae_comp, 4),
                "MAE_TRANS": round(mae_trans_sub, 4),
            },
            "C3_Mishnah_better_COMP": {
                "met": c3,
                "residual_COMP": mishnah_entry.get("residual_COMP"),
                "residual_TRANS": mishnah_entry.get("residual_TRANS"),
            },
            "C4_rho_AC1_gt_rho_delay": {
                "met": c4,
                "rho_AC1_DFA": round(float(rho_ac1_dfa), 4),
                "rho_delay_DFA": round(float(rho_delay_dfa), 4),
            },
        },
        "criteria_met": int(criteria_met),
        "verdict": verdict,
        "reasoning": (
            f"{criteria_met}/4 criteria met. "
            f"R2 COMP={fit_comp['R2']:.3f} vs TRANS={fit_trans_sub['R2']:.3f}. "
            f"LOO MAE COMP={mae_comp:.3f} vs TRANS={mae_trans_sub:.3f}. "
            f"rho(AC1,DFA)={rho_ac1_dfa:.3f} vs rho(delay,DFA)={rho_delay_dfa:.3f}."
        ),
    }

    log.info(f"  C1 (R2 COMP > TRANS + 0.10): {c1}")
    log.info(f"  C2 (LOO MAE COMP < TRANS):    {c2}")
    log.info(f"  C3 (Mishnah better COMP):     {c3}")
    log.info(f"  C4 (rho AC1 > rho delay):     {c4}")
    log.info(f"  VERDICT: {verdict} ({criteria_met}/4)")

    with open(RESULTS_DIR / "hypothesis_verdict.json", "w") as f:
        json.dump(verdict_result, f, indent=2, ensure_ascii=False)
    log.info("  Saved hypothesis_verdict.json")

    log.info(f"\n{'=' * 70}")
    log.info("Script 4 completado.")

    return 0


if __name__ == "__main__":
    exit(main())
