#!/usr/bin/env python3
"""
Fase 21 - Script 1: dfa_compositional_regression.py

Compositional features → DFA regression analysis:
1. Univariate correlations (Pearson + Spearman) of each feature vs DFA
2. Partial correlations controlling AC1
3. Forward selection: DFA ~ best features
4. Model comparison (5-fold CV): testament-only vs AC1-only vs compositional
"""

import json
import logging
import numpy as np
from pathlib import Path
from scipy import stats as sp_stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

BASE = Path(__file__).resolve().parent
RESULTS_DIR = BASE / "results" / "compositional_hypothesis"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Genre map (from genre_controlled_analysis.py)
# ═══════════════════════════════════════════════════════════════

GENRE_MAP = {
    "Psalms": "AT_poetry", "Job": "AT_poetry", "Proverbs": "AT_poetry",
    "Song of Songs": "AT_poetry", "Lamentations": "AT_poetry",
    "Ecclesiastes": "AT_poetry",
    "Leviticus": "AT_legal", "Numbers": "AT_legal", "Deuteronomy": "AT_legal",
    "Isaiah": "AT_prophetic", "Jeremiah": "AT_prophetic",
    "Ezekiel": "AT_prophetic", "Hosea": "AT_prophetic", "Joel": "AT_prophetic",
    "Amos": "AT_prophetic", "Obadiah": "AT_prophetic", "Jonah": "AT_prophetic",
    "Micah": "AT_prophetic", "Nahum": "AT_prophetic",
    "Habakkuk": "AT_prophetic", "Zephaniah": "AT_prophetic",
    "Haggai": "AT_prophetic", "Zechariah": "AT_prophetic",
    "Malachi": "AT_prophetic", "Daniel": "AT_prophetic",
    "Genesis": "AT_narrative", "Exodus": "AT_narrative",
    "Joshua": "AT_narrative", "Judges": "AT_narrative", "Ruth": "AT_narrative",
    "1 Samuel": "AT_narrative", "2 Samuel": "AT_narrative",
    "1 Kings": "AT_narrative", "2 Kings": "AT_narrative",
    "1 Chronicles": "AT_narrative", "2 Chronicles": "AT_narrative",
    "Ezra": "AT_narrative", "Nehemiah": "AT_narrative", "Esther": "AT_narrative",
    "Matthew": "NT_narrative", "Mark": "NT_narrative", "Luke": "NT_narrative",
    "John": "NT_narrative", "Acts": "NT_narrative",
    "Romans": "NT_epistolar", "1 Corinthians": "NT_epistolar",
    "2 Corinthians": "NT_epistolar", "Galatians": "NT_epistolar",
    "Ephesians": "NT_epistolar", "Philippians": "NT_epistolar",
    "Colossians": "NT_epistolar", "1 Thessalonians": "NT_epistolar",
    "2 Thessalonians": "NT_epistolar", "1 Timothy": "NT_epistolar",
    "2 Timothy": "NT_epistolar", "Titus": "NT_epistolar",
    "Philemon": "NT_epistolar", "Hebrews": "NT_epistolar",
    "James": "NT_epistolar", "1 Peter": "NT_epistolar",
    "2 Peter": "NT_epistolar", "1 John": "NT_epistolar",
    "2 John": "NT_epistolar", "3 John": "NT_epistolar", "Jude": "NT_epistolar",
    "Revelation": "NT_apocalyptic",
}


def partial_corr(x, y, z):
    """r(x,y|z) via residuals."""
    sl_xz = sp_stats.linregress(z, x)
    res_x = x - (sl_xz.slope * z + sl_xz.intercept)
    sl_yz = sp_stats.linregress(z, y)
    res_y = y - (sl_yz.slope * z + sl_yz.intercept)
    return sp_stats.pearsonr(res_x, res_y)


def compute_aic(n, k, rss):
    """AIC = n*ln(RSS/n) + 2*k."""
    if rss <= 0 or n <= 0:
        return float('inf')
    return float(n * np.log(rss / n) + 2 * k)


def compute_bic(n, k, rss):
    """BIC = n*ln(RSS/n) + k*ln(n)."""
    if rss <= 0 or n <= 0:
        return float('inf')
    return float(n * np.log(rss / n) + k * np.log(n))


def cv_rmse(X, y, n_splits=5):
    """5-fold cross-validated RMSE."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    errors = []
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        model = LinearRegression()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        errors.extend((pred - y_test) ** 2)
    return float(np.sqrt(np.mean(errors)))


def main():
    log.info("=" * 70)
    log.info("FASE 21 - Script 1: DFA Compositional Regression")
    log.info("=" * 70)

    # ──────────────────────────────────────────────────────────
    # 1. Load and merge features
    # ──────────────────────────────────────────────────────────
    log.info("\n--- Loading data ---")

    with open(BASE / "results" / "refined_classifier" / "book_features.json") as f:
        rc_data = json.load(f)

    with open(BASE / "results" / "composition" / "book_features.json") as f:
        comp_data = json.load(f)
    comp_by_book = {entry["book"]: entry for entry in comp_data}

    with open(BASE / "results" / "phi_mechanism" / "transition_distributions.json") as f:
        phi_data = json.load(f)
    phi_by_book = phi_data.get("by_book", {})

    # Merge all features for books with non-null DFA
    books = []
    for name, rc in rc_data.items():
        if rc.get("DFA") is None:
            continue

        row = {
            "book": name,
            "testament": rc["testament"],
            "genre": GENRE_MAP.get(name, "unknown"),
            "DFA": float(rc["DFA"]),
            "H": float(rc["H"]) if rc.get("H") is not None else None,
            "AC1": float(rc["AC1"]) if rc.get("AC1") is not None else None,
            "CV": float(rc["CV"]) if rc.get("CV") is not None else None,
            "mean_verse_len": float(rc.get("mean_verse_len", 0)),
            "std_verse_len": float(rc.get("std_verse_len", 0)),
            "skewness": float(rc.get("skewness", 0)),
            "pos_entropy": float(rc.get("pos_entropy", 0)),
        }

        # From composition features
        comp = comp_by_book.get(name)
        if comp and "features" in comp:
            feat = comp["features"]
            row["repetition_density"] = float(feat.get("repetition_density", 0))
            row["verb_density"] = float(feat.get("verb_density", 0))
            row["proper_ratio"] = float(feat.get("proper_ratio", 0))
            row["connector_density"] = float(feat.get("connector_density", 0))
        else:
            row["repetition_density"] = None
            row["verb_density"] = None
            row["proper_ratio"] = None
            row["connector_density"] = None

        # From phi_mechanism (mean_delta)
        phi = phi_by_book.get(name)
        if phi:
            row["mean_delta"] = float(phi.get("mean_delta", 0))
        else:
            row["mean_delta"] = None

        books.append(row)

    log.info(f"  Merged {len(books)} books with DFA")

    # Feature list
    FEATURE_NAMES = [
        "AC1", "CV", "repetition_density", "mean_verse_len", "verb_density",
        "proper_ratio", "connector_density", "pos_entropy", "skewness",
        "std_verse_len", "mean_delta",
    ]

    # ──────────────────────────────────────────────────────────
    # 2. Univariate correlations with DFA
    # ──────────────────────────────────────────────────────────
    log.info("\n--- Univariate Correlations with DFA ---")

    dfa_arr = np.array([b["DFA"] for b in books])
    univar_results = {}

    for feat in FEATURE_NAMES:
        vals = [b.get(feat) for b in books]
        mask = [v is not None and not np.isnan(v) for v in vals]
        x = np.array([v for v, m in zip(vals, mask) if m])
        y = np.array([d for d, m in zip(dfa_arr, mask) if m])

        if len(x) < 5:
            continue

        r_pearson, p_pearson = sp_stats.pearsonr(x, y)
        r_spearman, p_spearman = sp_stats.spearmanr(x, y)

        univar_results[feat] = {
            "n": int(len(x)),
            "pearson_r": round(float(r_pearson), 4),
            "pearson_p": round(float(p_pearson), 6),
            "spearman_rho": round(float(r_spearman), 4),
            "spearman_p": round(float(p_spearman), 6),
            "abs_pearson_r": round(float(abs(r_pearson)), 4),
        }
        log.info(f"  {feat}: r={r_pearson:.4f} (p={p_pearson:.4e}), "
                 f"rho={r_spearman:.4f} (p={p_spearman:.4e})")

    # Sort by |r|
    ranking = sorted(univar_results.keys(),
                     key=lambda f: univar_results[f]["abs_pearson_r"],
                     reverse=True)
    for i, feat in enumerate(ranking):
        univar_results[feat]["rank"] = i + 1

    univar_results["_ranking"] = ranking

    with open(RESULTS_DIR / "univariate_correlations_dfa.json", "w") as f:
        json.dump(univar_results, f, indent=2, ensure_ascii=False)
    log.info("  Saved univariate_correlations_dfa.json")

    # ──────────────────────────────────────────────────────────
    # 3. Partial correlations controlling AC1
    # ──────────────────────────────────────────────────────────
    log.info("\n--- Partial Correlations (controlling AC1) ---")

    # Get books with both AC1 and DFA
    ac1_arr = np.array([b["AC1"] for b in books if b.get("AC1") is not None])
    dfa_with_ac1 = np.array([b["DFA"] for b in books if b.get("AC1") is not None])
    books_with_ac1 = [b for b in books if b.get("AC1") is not None]

    top5 = ranking[:5]
    partial_results = {}

    for feat in top5:
        if feat == "AC1":
            partial_results[feat] = {
                "note": "AC1 is the control variable itself",
                "partial_r": None,
                "partial_p": None,
            }
            continue

        vals = [b.get(feat) for b in books_with_ac1]
        mask = [v is not None and not np.isnan(v) for v in vals]
        x = np.array([v for v, m in zip(vals, mask) if m])
        y = np.array([d for d, m in zip(dfa_with_ac1, mask) if m])
        z = np.array([a for a, m in zip(ac1_arr, mask) if m])

        if len(x) < 5:
            continue

        r_partial, p_partial = partial_corr(x, y, z)
        partial_results[feat] = {
            "n": int(len(x)),
            "partial_r": round(float(r_partial), 4),
            "partial_p": round(float(p_partial), 6),
            "original_r": univar_results[feat]["pearson_r"],
            "change": round(float(abs(r_partial) - univar_results[feat]["abs_pearson_r"]), 4),
        }
        log.info(f"  {feat}: r_partial={r_partial:.4f} (p={p_partial:.4e}), "
                 f"original r={univar_results[feat]['pearson_r']:.4f}")

    with open(RESULTS_DIR / "partial_correlations_dfa.json", "w") as f:
        json.dump(partial_results, f, indent=2, ensure_ascii=False)
    log.info("  Saved partial_correlations_dfa.json")

    # ──────────────────────────────────────────────────────────
    # 4. Forward selection
    # ──────────────────────────────────────────────────────────
    log.info("\n--- Forward Selection: DFA ~ best features ---")

    # Use only books with all features available
    complete_books = []
    for b in books:
        if all(b.get(f) is not None for f in FEATURE_NAMES):
            complete_books.append(b)
    log.info(f"  {len(complete_books)} books with complete features")

    y_fs = np.array([b["DFA"] for b in complete_books])
    X_all = {f: np.array([b[f] for b in complete_books]) for f in FEATURE_NAMES}

    selected = []
    remaining = list(FEATURE_NAMES)
    steps = []
    best_r2_adj = -np.inf

    for step in range(min(5, len(FEATURE_NAMES))):
        best_feat = None
        best_new_r2_adj = best_r2_adj

        for feat in remaining:
            trial = selected + [feat]
            X_trial = np.column_stack([X_all[f] for f in trial])
            model = LinearRegression().fit(X_trial, y_fs)
            y_pred = model.predict(X_trial)
            ss_res = np.sum((y_fs - y_pred) ** 2)
            ss_tot = np.sum((y_fs - np.mean(y_fs)) ** 2)
            r2 = 1 - ss_res / ss_tot
            n = len(y_fs)
            p = len(trial)
            r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)

            if r2_adj > best_new_r2_adj:
                best_new_r2_adj = r2_adj
                best_feat = feat

        if best_feat is None or (best_new_r2_adj - best_r2_adj) < 0.01:
            break

        selected.append(best_feat)
        remaining.remove(best_feat)
        best_r2_adj = best_new_r2_adj

        steps.append({
            "step": step + 1,
            "added_feature": best_feat,
            "features": list(selected),
            "R2_adj": round(float(best_r2_adj), 4),
        })
        log.info(f"  Step {step+1}: +{best_feat} -> R2_adj={best_r2_adj:.4f}")

    # Final model stats
    if selected:
        X_final = np.column_stack([X_all[f] for f in selected])
        model_final = LinearRegression().fit(X_final, y_fs)
        y_pred_final = model_final.predict(X_final)
        ss_res = np.sum((y_fs - y_pred_final) ** 2)
        ss_tot = np.sum((y_fs - np.mean(y_fs)) ** 2)
        r2_final = 1 - ss_res / ss_tot
        n = len(y_fs)
        p = len(selected)
        r2_adj_final = 1 - (1 - r2_final) * (n - 1) / (n - p - 1)

        forward_result = {
            "selected_features": selected,
            "n_books": int(n),
            "R2": round(float(r2_final), 4),
            "R2_adj": round(float(r2_adj_final), 4),
            "coefficients": {f: round(float(c), 6) for f, c in zip(selected, model_final.coef_)},
            "intercept": round(float(model_final.intercept_), 6),
            "steps": steps,
        }
    else:
        forward_result = {"selected_features": [], "steps": []}

    with open(RESULTS_DIR / "forward_selection_dfa.json", "w") as f:
        json.dump(forward_result, f, indent=2, ensure_ascii=False)
    log.info("  Saved forward_selection_dfa.json")

    # ──────────────────────────────────────────────────────────
    # 5. Model comparison (5-fold CV)
    # ──────────────────────────────────────────────────────────
    log.info("\n--- Model Comparison (5-fold CV) ---")

    n = len(complete_books)

    # Model A: DFA ~ testament (dummy 0/1)
    is_at = np.array([1.0 if b["testament"] == "AT" else 0.0 for b in complete_books])
    X_A = is_at.reshape(-1, 1)
    model_A = LinearRegression().fit(X_A, y_fs)
    pred_A = model_A.predict(X_A)
    ss_res_A = np.sum((y_fs - pred_A) ** 2)
    ss_tot = np.sum((y_fs - np.mean(y_fs)) ** 2)
    r2_A = 1 - ss_res_A / ss_tot
    r2_adj_A = 1 - (1 - r2_A) * (n - 1) / (n - 2)
    cv_rmse_A = cv_rmse(X_A, y_fs)
    aic_A = compute_aic(n, 2, ss_res_A)
    bic_A = compute_bic(n, 2, ss_res_A)

    # Model B: DFA ~ AC1
    X_B = np.array([b["AC1"] for b in complete_books]).reshape(-1, 1)
    model_B = LinearRegression().fit(X_B, y_fs)
    pred_B = model_B.predict(X_B)
    ss_res_B = np.sum((y_fs - pred_B) ** 2)
    r2_B = 1 - ss_res_B / ss_tot
    r2_adj_B = 1 - (1 - r2_B) * (n - 1) / (n - 2)
    cv_rmse_B = cv_rmse(X_B, y_fs)
    aic_B = compute_aic(n, 2, ss_res_B)
    bic_B = compute_bic(n, 2, ss_res_B)

    # Model C: DFA ~ best compositional features (from forward selection)
    if selected:
        X_C = np.column_stack([X_all[f] for f in selected])
        model_C = LinearRegression().fit(X_C, y_fs)
        pred_C = model_C.predict(X_C)
        ss_res_C = np.sum((y_fs - pred_C) ** 2)
        r2_C = 1 - ss_res_C / ss_tot
        k_C = len(selected) + 1
        r2_adj_C = 1 - (1 - r2_C) * (n - 1) / (n - k_C)
        cv_rmse_C = cv_rmse(X_C, y_fs)
        aic_C = compute_aic(n, k_C, ss_res_C)
        bic_C = compute_bic(n, k_C, ss_res_C)
    else:
        r2_C = r2_adj_C = cv_rmse_C = aic_C = bic_C = None

    model_comparison = {
        "n_books": int(n),
        "Model_A_testament": {
            "formula": "DFA ~ is_AT",
            "R2": round(float(r2_A), 4),
            "R2_adj": round(float(r2_adj_A), 4),
            "AIC": round(float(aic_A), 2),
            "BIC": round(float(bic_A), 2),
            "CV_RMSE": round(float(cv_rmse_A), 4),
            "coef_is_AT": round(float(model_A.coef_[0]), 4),
            "intercept": round(float(model_A.intercept_), 4),
        },
        "Model_B_AC1": {
            "formula": "DFA ~ AC1",
            "R2": round(float(r2_B), 4),
            "R2_adj": round(float(r2_adj_B), 4),
            "AIC": round(float(aic_B), 2),
            "BIC": round(float(bic_B), 2),
            "CV_RMSE": round(float(cv_rmse_B), 4),
            "coef_AC1": round(float(model_B.coef_[0]), 4),
            "intercept": round(float(model_B.intercept_), 4),
        },
        "Model_C_compositional": {
            "formula": f"DFA ~ {' + '.join(selected)}" if selected else "N/A",
            "features": selected,
            "R2": round(float(r2_C), 4) if r2_C is not None else None,
            "R2_adj": round(float(r2_adj_C), 4) if r2_adj_C is not None else None,
            "AIC": round(float(aic_C), 2) if aic_C is not None else None,
            "BIC": round(float(bic_C), 2) if bic_C is not None else None,
            "CV_RMSE": round(float(cv_rmse_C), 4) if cv_rmse_C is not None else None,
        },
        "best_by_AIC": min(
            [("A_testament", aic_A), ("B_AC1", aic_B)] +
            ([("C_compositional", aic_C)] if aic_C is not None else []),
            key=lambda x: x[1]
        )[0],
        "best_by_CV_RMSE": min(
            [("A_testament", cv_rmse_A), ("B_AC1", cv_rmse_B)] +
            ([("C_compositional", cv_rmse_C)] if cv_rmse_C is not None else []),
            key=lambda x: x[1]
        )[0],
        "compositional_beats_testament": bool(
            r2_adj_C is not None and r2_adj_C > r2_adj_A
        ),
    }

    log.info(f"  Model A (testament): R2={r2_A:.4f}, CV-RMSE={cv_rmse_A:.4f}")
    log.info(f"  Model B (AC1):       R2={r2_B:.4f}, CV-RMSE={cv_rmse_B:.4f}")
    if r2_C is not None:
        log.info(f"  Model C (compos.):   R2={r2_C:.4f}, CV-RMSE={cv_rmse_C:.4f}")

    with open(RESULTS_DIR / "model_comparison_intra.json", "w") as f:
        json.dump(model_comparison, f, indent=2, ensure_ascii=False)
    log.info("  Saved model_comparison_intra.json")

    log.info(f"\n{'=' * 70}")
    log.info("Script 1 completado.")
    log.info(f"  Top predictor: {ranking[0]} (r={univar_results[ranking[0]]['pearson_r']:.4f})")
    log.info(f"  Compositional > testament: {model_comparison['compositional_beats_testament']}")

    return 0


if __name__ == "__main__":
    exit(main())
