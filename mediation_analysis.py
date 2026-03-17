#!/usr/bin/env python3
"""
Fase 21 - Script 3: mediation_analysis.py

Mediation analysis: testament → AC1 → DFA
1. Baron-Kenny 4 steps
2. Bootstrap mediation (n=1000)
3. Sobel test
4. Multi-mediator (AC1 + repetition_density)
5. Narrative-only control
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


def ols_regression(X, y):
    """Simple OLS returning slope, intercept, SE, p-value, R2."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    # Add intercept
    X_with_int = np.column_stack([np.ones(n), X])
    k = X_with_int.shape[1]

    # OLS: beta = (X'X)^-1 X'y
    XtX = X_with_int.T @ X_with_int
    Xty = X_with_int.T @ y
    beta = np.linalg.solve(XtX, Xty)

    y_pred = X_with_int @ beta
    residuals = y - y_pred
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # SE of coefficients
    mse = ss_res / (n - k)
    try:
        var_beta = mse * np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        var_beta = mse * np.linalg.pinv(XtX)
    se_beta = np.sqrt(np.diag(var_beta))

    # t-statistics and p-values
    t_stats = beta / se_beta
    p_values = 2 * sp_stats.t.sf(np.abs(t_stats), df=n - k)

    return {
        "intercept": float(beta[0]),
        "coefficients": [float(b) for b in beta[1:]],
        "se_intercept": float(se_beta[0]),
        "se_coefficients": [float(s) for s in se_beta[1:]],
        "t_statistics": [float(t) for t in t_stats[1:]],
        "p_values": [float(p) for p in p_values[1:]],
        "R2": float(r2),
        "n": int(n),
    }


def main():
    log.info("=" * 70)
    log.info("FASE 21 - Script 3: Mediation Analysis")
    log.info("=" * 70)

    # ──────────────────────────────────────────────────────────
    # Load data
    # ──────────────────────────────────────────────────────────
    with open(BASE / "results" / "refined_classifier" / "book_features.json") as f:
        rc_data = json.load(f)

    with open(BASE / "results" / "composition" / "book_features.json") as f:
        comp_data = json.load(f)
    comp_by_book = {entry["book"]: entry for entry in comp_data}

    # Build dataset: books with DFA and AC1
    books = []
    for name, rc in rc_data.items():
        if rc.get("DFA") is None or rc.get("AC1") is None:
            continue

        rep_dens = None
        comp = comp_by_book.get(name)
        if comp and "features" in comp:
            rep_dens = comp["features"].get("repetition_density")

        books.append({
            "book": name,
            "testament": rc["testament"],
            "genre": GENRE_MAP.get(name, "unknown"),
            "DFA": float(rc["DFA"]),
            "AC1": float(rc["AC1"]),
            "repetition_density": float(rep_dens) if rep_dens is not None else None,
        })

    log.info(f"  {len(books)} books with DFA and AC1")

    is_AT = np.array([1.0 if b["testament"] == "AT" else 0.0 for b in books])
    dfa = np.array([b["DFA"] for b in books])
    ac1 = np.array([b["AC1"] for b in books])
    rep_dens = np.array([b["repetition_density"] if b["repetition_density"] is not None
                         else np.nan for b in books])

    # ──────────────────────────────────────────────────────────
    # Baron-Kenny 4 steps
    # ──────────────────────────────────────────────────────────
    log.info("\n--- Baron-Kenny Mediation (testament → AC1 → DFA) ---")

    # Step c: DFA ~ is_AT (total effect)
    step_c = ols_regression(is_AT, dfa)
    c = step_c["coefficients"][0]
    c_p = step_c["p_values"][0]
    log.info(f"  Step c (total): beta={c:.4f}, p={c_p:.6f}")

    # Step a: AC1 ~ is_AT
    step_a = ols_regression(is_AT, ac1)
    a = step_a["coefficients"][0]
    a_se = step_a["se_coefficients"][0]
    a_p = step_a["p_values"][0]
    log.info(f"  Step a (AT→AC1): beta={a:.4f}, p={a_p:.6f}")

    # Step b+c': DFA ~ is_AT + AC1
    X_bc = np.column_stack([is_AT, ac1])
    step_bc = ols_regression(X_bc, dfa)
    c_prime = step_bc["coefficients"][0]
    b = step_bc["coefficients"][1]
    b_se = step_bc["se_coefficients"][1]
    c_prime_p = step_bc["p_values"][0]
    b_p = step_bc["p_values"][1]
    log.info(f"  Step b (AC1→DFA|AT): beta={b:.4f}, p={b_p:.6f}")
    log.info(f"  Step c' (direct):    beta={c_prime:.4f}, p={c_prime_p:.6f}")

    indirect = a * b
    mediation_type = "complete" if c_prime_p > 0.05 else "partial"
    log.info(f"  Indirect effect a*b = {indirect:.4f}")
    log.info(f"  Mediation type: {mediation_type}")

    baron_kenny = {
        "step_c_total": {
            "formula": "DFA ~ is_AT",
            "beta": round(float(c), 4),
            "SE": round(float(step_c["se_coefficients"][0]), 4),
            "p": round(float(c_p), 6),
            "R2": round(float(step_c["R2"]), 4),
        },
        "step_a": {
            "formula": "AC1 ~ is_AT",
            "beta": round(float(a), 4),
            "SE": round(float(a_se), 4),
            "p": round(float(a_p), 6),
            "R2": round(float(step_a["R2"]), 4),
        },
        "step_bc": {
            "formula": "DFA ~ is_AT + AC1",
            "beta_is_AT": round(float(c_prime), 4),
            "beta_AC1": round(float(b), 4),
            "SE_is_AT": round(float(step_bc["se_coefficients"][0]), 4),
            "SE_AC1": round(float(b_se), 4),
            "p_is_AT": round(float(c_prime_p), 6),
            "p_AC1": round(float(b_p), 6),
            "R2": round(float(step_bc["R2"]), 4),
        },
        "indirect_effect": round(float(indirect), 4),
        "direct_effect": round(float(c_prime), 4),
        "total_effect": round(float(c), 4),
        "proportion_mediated": round(float(indirect / c), 4) if abs(c) > 1e-10 else None,
        "mediation_type": mediation_type,
        "n": int(len(books)),
    }

    with open(RESULTS_DIR / "baron_kenny_mediation.json", "w") as f:
        json.dump(baron_kenny, f, indent=2, ensure_ascii=False)
    log.info("  Saved baron_kenny_mediation.json")

    # ──────────────────────────────────────────────────────────
    # Bootstrap mediation
    # ──────────────────────────────────────────────────────────
    log.info("\n--- Bootstrap Mediation (n=1000) ---")

    rng = np.random.RandomState(42)
    n_boot = 1000
    boot_indirect = []
    boot_proportion = []

    for _ in range(n_boot):
        idx = rng.choice(len(books), size=len(books), replace=True)
        is_AT_b = is_AT[idx]
        dfa_b = dfa[idx]
        ac1_b = ac1[idx]

        # a path
        reg_a = sp_stats.linregress(is_AT_b, ac1_b)
        a_b = reg_a.slope

        # b path (controlling is_AT)
        X_b = np.column_stack([is_AT_b, ac1_b])
        X_bi = np.column_stack([np.ones(len(idx)), X_b])
        try:
            beta_b = np.linalg.solve(X_bi.T @ X_bi, X_bi.T @ dfa_b)
        except np.linalg.LinAlgError:
            continue
        b_b = beta_b[2]  # AC1 coefficient

        # c path (total)
        reg_c = sp_stats.linregress(is_AT_b, dfa_b)
        c_b = reg_c.slope

        ind = a_b * b_b
        boot_indirect.append(float(ind))
        if abs(c_b) > 1e-10:
            boot_proportion.append(float(ind / c_b))

    boot_indirect = np.array(boot_indirect)
    ci_low = float(np.percentile(boot_indirect, 2.5))
    ci_high = float(np.percentile(boot_indirect, 97.5))
    ci_crosses_zero = bool(ci_low <= 0 <= ci_high)

    boot_result = {
        "n_bootstraps": n_boot,
        "indirect_effect_mean": round(float(np.mean(boot_indirect)), 4),
        "indirect_effect_median": round(float(np.median(boot_indirect)), 4),
        "indirect_effect_std": round(float(np.std(boot_indirect)), 4),
        "CI_95": [round(ci_low, 4), round(ci_high, 4)],
        "CI_crosses_zero": ci_crosses_zero,
        "significant": not ci_crosses_zero,
    }

    if boot_proportion:
        prop_arr = np.array(boot_proportion)
        boot_result["proportion_mediated_mean"] = round(float(np.mean(prop_arr)), 4)
        boot_result["proportion_mediated_CI_95"] = [
            round(float(np.percentile(prop_arr, 2.5)), 4),
            round(float(np.percentile(prop_arr, 97.5)), 4),
        ]

    log.info(f"  Indirect effect: {np.mean(boot_indirect):.4f} "
             f"[{ci_low:.4f}, {ci_high:.4f}]")
    log.info(f"  CI crosses zero: {ci_crosses_zero}")

    # Sobel test
    z_sobel = (a * b) / np.sqrt(b**2 * a_se**2 + a**2 * b_se**2)
    p_sobel = 2 * sp_stats.norm.sf(abs(z_sobel))
    boot_result["sobel_z"] = round(float(z_sobel), 4)
    boot_result["sobel_p"] = round(float(p_sobel), 6)
    log.info(f"  Sobel test: z={z_sobel:.4f}, p={p_sobel:.6f}")

    with open(RESULTS_DIR / "bootstrap_mediation.json", "w") as f:
        json.dump(boot_result, f, indent=2, ensure_ascii=False)
    log.info("  Saved bootstrap_mediation.json")

    # ──────────────────────────────────────────────────────────
    # Multi-mediator: AC1 + repetition_density
    # ──────────────────────────────────────────────────────────
    log.info("\n--- Multi-Mediator Analysis ---")

    # Filter books with repetition_density
    mask_rd = ~np.isnan(rep_dens)
    if np.sum(mask_rd) >= 10:
        is_AT_mm = is_AT[mask_rd]
        dfa_mm = dfa[mask_rd]
        ac1_mm = ac1[mask_rd]
        rd_mm = rep_dens[mask_rd]
        n_mm = int(np.sum(mask_rd))

        # Path 1: is_AT → AC1
        reg_a1 = ols_regression(is_AT_mm, ac1_mm)
        a1 = reg_a1["coefficients"][0]
        a1_se = reg_a1["se_coefficients"][0]

        # Path 2: is_AT → repetition_density
        reg_a2 = ols_regression(is_AT_mm, rd_mm)
        a2 = reg_a2["coefficients"][0]
        a2_se = reg_a2["se_coefficients"][0]

        # DFA ~ is_AT + AC1 + repetition_density
        X_mm = np.column_stack([is_AT_mm, ac1_mm, rd_mm])
        reg_mm = ols_regression(X_mm, dfa_mm)
        c_prime_mm = reg_mm["coefficients"][0]
        b1 = reg_mm["coefficients"][1]  # AC1
        b2 = reg_mm["coefficients"][2]  # repetition_density
        b1_se = reg_mm["se_coefficients"][1]
        b2_se = reg_mm["se_coefficients"][2]

        indirect1 = a1 * b1
        indirect2 = a2 * b2
        total_indirect = indirect1 + indirect2

        multi_mediator = {
            "n": n_mm,
            "path1_via_AC1": {
                "a": round(float(a1), 4),
                "b": round(float(b1), 4),
                "indirect": round(float(indirect1), 4),
                "a_p": round(float(reg_a1["p_values"][0]), 6),
                "b_p": round(float(reg_mm["p_values"][1]), 6),
            },
            "path2_via_repetition_density": {
                "a": round(float(a2), 4),
                "b": round(float(b2), 4),
                "indirect": round(float(indirect2), 4),
                "a_p": round(float(reg_a2["p_values"][0]), 6),
                "b_p": round(float(reg_mm["p_values"][2]), 6),
            },
            "total_indirect": round(float(total_indirect), 4),
            "direct_effect": round(float(c_prime_mm), 4),
            "direct_p": round(float(reg_mm["p_values"][0]), 6),
            "R2_full_model": round(float(reg_mm["R2"]), 4),
        }
        log.info(f"  Path 1 (AC1): indirect={indirect1:.4f}")
        log.info(f"  Path 2 (rep_dens): indirect={indirect2:.4f}")
    else:
        multi_mediator = {"error": "insufficient data with repetition_density"}

    with open(RESULTS_DIR / "multi_mediator.json", "w") as f:
        json.dump(multi_mediator, f, indent=2, ensure_ascii=False)
    log.info("  Saved multi_mediator.json")

    # ──────────────────────────────────────────────────────────
    # Narrative-only mediation
    # ──────────────────────────────────────────────────────────
    log.info("\n--- Narrative-Only Mediation ---")

    narr_books = [b for b in books if b["genre"] in ("AT_narrative", "NT_narrative")]
    log.info(f"  {len(narr_books)} narrative books")

    if len(narr_books) >= 10:
        is_AT_n = np.array([1.0 if b["testament"] == "AT" else 0.0 for b in narr_books])
        dfa_n = np.array([b["DFA"] for b in narr_books])
        ac1_n = np.array([b["AC1"] for b in narr_books])

        # c: total
        reg_c_n = ols_regression(is_AT_n, dfa_n)
        c_n = reg_c_n["coefficients"][0]

        # a: AT → AC1
        reg_a_n = ols_regression(is_AT_n, ac1_n)
        a_n = reg_a_n["coefficients"][0]
        a_se_n = reg_a_n["se_coefficients"][0]

        # b + c': DFA ~ AT + AC1
        X_n = np.column_stack([is_AT_n, ac1_n])
        reg_bc_n = ols_regression(X_n, dfa_n)
        c_prime_n = reg_bc_n["coefficients"][0]
        b_n = reg_bc_n["coefficients"][1]
        b_se_n = reg_bc_n["se_coefficients"][1]

        indirect_n = a_n * b_n

        # Bootstrap
        boot_ind_n = []
        for _ in range(n_boot):
            idx = rng.choice(len(narr_books), size=len(narr_books), replace=True)
            at_b = is_AT_n[idx]
            dfa_b = dfa_n[idx]
            ac1_b = ac1_n[idx]
            # Skip if all same testament (linregress needs variance in x)
            if np.std(at_b) == 0:
                continue
            try:
                ra = sp_stats.linregress(at_b, ac1_b)
            except ValueError:
                continue
            X_bi = np.column_stack([np.ones(len(idx)), at_b, ac1_b])
            try:
                beta = np.linalg.solve(X_bi.T @ X_bi, X_bi.T @ dfa_b)
            except np.linalg.LinAlgError:
                continue
            boot_ind_n.append(float(ra.slope * beta[2]))

        boot_ind_n = np.array(boot_ind_n)
        ci_low_n = float(np.percentile(boot_ind_n, 2.5))
        ci_high_n = float(np.percentile(boot_ind_n, 97.5))

        narr_mediation = {
            "n": int(len(narr_books)),
            "total_effect_c": round(float(c_n), 4),
            "total_p": round(float(reg_c_n["p_values"][0]), 6),
            "a_path": round(float(a_n), 4),
            "a_p": round(float(reg_a_n["p_values"][0]), 6),
            "b_path": round(float(b_n), 4),
            "b_p": round(float(reg_bc_n["p_values"][1]), 6),
            "direct_c_prime": round(float(c_prime_n), 4),
            "direct_p": round(float(reg_bc_n["p_values"][0]), 6),
            "indirect_effect": round(float(indirect_n), 4),
            "proportion_mediated": round(float(indirect_n / c_n), 4) if abs(c_n) > 1e-10 else None,
            "bootstrap_CI_95": [round(ci_low_n, 4), round(ci_high_n, 4)],
            "CI_crosses_zero": bool(ci_low_n <= 0 <= ci_high_n),
            "R2_full": round(float(reg_bc_n["R2"]), 4),
            "mediation_type": "complete" if reg_bc_n["p_values"][0] > 0.05 else "partial",
            "books": [b["book"] for b in narr_books],
        }
        log.info(f"  Narrative indirect: {indirect_n:.4f} [{ci_low_n:.4f}, {ci_high_n:.4f}]")
    else:
        narr_mediation = {"error": "insufficient narrative books"}

    with open(RESULTS_DIR / "narrative_only_mediation.json", "w") as f:
        json.dump(narr_mediation, f, indent=2, ensure_ascii=False)
    log.info("  Saved narrative_only_mediation.json")

    log.info(f"\n{'=' * 70}")
    log.info("Script 3 completado.")
    log.info(f"  Mediation type: {mediation_type}")
    log.info(f"  Proportion mediated: {baron_kenny.get('proportion_mediated')}")
    log.info(f"  Bootstrap CI crosses 0: {ci_crosses_zero}")

    return 0


if __name__ == "__main__":
    exit(main())
