#!/usr/bin/env python3
"""
Fase 21 - Script 2: parallelism_quantification.py

Quantify parallelism in each biblical book and correlate with DFA:
1. Compute 4 parallelism metrics per book (AC2, smoothness, run_length, spectral_regularity)
2. PCA → Parallelism Index (PC1)
3. Correlate PI with DFA
4. Genre profiles + external corpus predictions
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


# ═══════════════════════════════════════════════════════════════
# Genre map
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

# External corpora with known AC1 and DFA
EXTERNAL_CORPORA = {
    "AT": {"AC1": 0.34, "DFA": 0.93},
    "NT": {"AC1": 0.09, "DFA": 0.83},
    "Quran": {"AC1": 0.47, "DFA": 0.93},
    "Rig_Veda": {"AC1": -0.02, "DFA": 0.51},
    "Book_of_Dead": {"AC1": 0.20, "DFA": 0.77},
    "Pali_Canon": {"AC1": 0.21, "DFA": 0.76},
    "Mishnah": {"AC1": 0.18, "DFA": 0.65},
    "Didache": {"AC1": 0.20, "DFA": 0.83},
    "1_Clemente": {"AC1": 0.24, "DFA": 0.61},
    "Tosefta": {"AC1": 0.28, "DFA": 0.78},
}


# ═══════════════════════════════════════════════════════════════
# Metric functions
# ═══════════════════════════════════════════════════════════════

def autocorr_lag(series, lag):
    """Autocorrelation at given lag."""
    s = np.asarray(series, dtype=float)
    if len(s) < lag + 3:
        return np.nan
    m, v = np.mean(s), np.var(s)
    if v == 0:
        return 0.0
    return float(np.sum((s[:-lag] - m) * (s[lag:] - m)) / (len(s) * v))


def smoothness(series):
    """Fraction of consecutive pairs with |L_i - L_{i+1}| <= 2."""
    s = np.asarray(series, dtype=float)
    if len(s) < 2:
        return np.nan
    diffs = np.abs(np.diff(s))
    return float(np.mean(diffs <= 2))


def mean_run_length(series):
    """Mean length of consecutive runs where |deltaL| <= 2."""
    s = np.asarray(series, dtype=float)
    if len(s) < 2:
        return np.nan
    diffs = np.abs(np.diff(s))
    smooth = diffs <= 2

    runs = []
    current = 0
    for v in smooth:
        if v:
            current += 1
        else:
            if current > 0:
                runs.append(current)
            current = 0
    if current > 0:
        runs.append(current)

    return float(np.mean(runs)) if runs else 1.0


def spectral_regularity(series):
    """Ratio of max power to mean power in FFT (periodicidad)."""
    s = np.asarray(series, dtype=float)
    if len(s) < 10:
        return np.nan
    centered = s - np.mean(s)
    fft_vals = np.abs(np.fft.rfft(centered))[1:]  # exclude DC
    if len(fft_vals) == 0 or np.mean(fft_vals) == 0:
        return 1.0
    return float(np.max(fft_vals) / np.mean(fft_vals))


def dfa_exponent(series):
    """DFA exponent using logspace bin sizes."""
    s = np.asarray(series, dtype=float)
    n = len(s)
    if n < 20:
        return np.nan
    y = np.cumsum(s - np.mean(s))
    mn, mx = 4, n // 4
    if mx < mn + 2:
        return np.nan
    bsz = np.unique(np.logspace(np.log10(mn), np.log10(mx), 20).astype(int))
    fl, sz = [], []
    for bs in bsz:
        nb = n // bs
        if nb < 1:
            continue
        f2 = []
        for i in range(nb):
            seg = y[i * bs:(i + 1) * bs]
            x = np.arange(bs)
            trend = np.polyval(np.polyfit(x, seg, 1), x)
            f2.append(np.mean((seg - trend) ** 2))
        if f2:
            fl.append(np.sqrt(np.mean(f2)))
            sz.append(bs)
    if len(sz) < 3:
        return np.nan
    slope, *_ = sp_stats.linregress(np.log(sz), np.log(fl))
    return round(slope, 4)


def build_verse_lengths_per_book(words):
    """Build verse lengths grouped by book."""
    books = {}
    for w in words:
        bname = w.get("book", "")
        if not bname:
            continue
        key = (w["book_num"], w["chapter"], w["verse"])
        if bname not in books:
            books[bname] = {}
        books[bname][key] = books[bname].get(key, 0) + 1

    result = {}
    for bname, verses in books.items():
        lengths = [v for _, v in sorted(verses.items())]
        result[bname] = np.array(lengths, dtype=float)
    return result


def main():
    log.info("=" * 70)
    log.info("FASE 21 - Script 2: Parallelism Quantification")
    log.info("=" * 70)

    # ──────────────────────────────────────────────────────────
    # 1. Load bible_unified.json and build verse lengths per book
    # ──────────────────────────────────────────────────────────
    log.info("\n--- Loading bible_unified.json ---")
    with open(BASE / "bible_unified.json") as f:
        words = json.load(f)
    log.info(f"  Loaded {len(words)} words")

    book_lengths = build_verse_lengths_per_book(words)
    log.info(f"  Built verse lengths for {len(book_lengths)} books")

    # Free memory
    del words

    # Load book features for DFA and AC1
    with open(BASE / "results" / "refined_classifier" / "book_features.json") as f:
        rc_data = json.load(f)

    # ──────────────────────────────────────────────────────────
    # 2. Compute parallelism metrics per book
    # ──────────────────────────────────────────────────────────
    log.info("\n--- Computing Parallelism Metrics ---")

    metrics_all = {}
    books_for_pca = []

    for book_name, rc in rc_data.items():
        if rc.get("DFA") is None:
            continue

        lengths = book_lengths.get(book_name)
        if lengths is None or len(lengths) < 20:
            continue

        ac1 = float(rc["AC1"]) if rc.get("AC1") is not None else None
        ac2 = autocorr_lag(lengths, 2)
        sm = smoothness(lengths)
        rl = mean_run_length(lengths)
        sr = spectral_regularity(lengths)

        entry = {
            "book": book_name,
            "testament": rc["testament"],
            "genre": GENRE_MAP.get(book_name, "unknown"),
            "DFA": float(rc["DFA"]),
            "AC1": ac1,
            "AC2": round(float(ac2), 4) if not np.isnan(ac2) else None,
            "smoothness": round(float(sm), 4) if not np.isnan(sm) else None,
            "run_length": round(float(rl), 4) if not np.isnan(rl) else None,
            "spectral_regularity": round(float(sr), 4) if not np.isnan(sr) else None,
            "n_verses": int(len(lengths)),
        }
        metrics_all[book_name] = entry

        if (ac1 is not None and entry["AC2"] is not None
                and entry["smoothness"] is not None
                and entry["spectral_regularity"] is not None):
            books_for_pca.append(entry)

        log.info(f"  {book_name}: AC2={entry['AC2']}, smooth={entry['smoothness']}, "
                 f"run_len={entry['run_length']}, spec_reg={entry['spectral_regularity']}")

    log.info(f"  {len(metrics_all)} books computed, {len(books_for_pca)} with all 4 metrics")

    # ──────────────────────────────────────────────────────────
    # 3. PCA → Parallelism Index
    # ──────────────────────────────────────────────────────────
    log.info("\n--- PCA → Parallelism Index ---")

    feat_names = ["AC1", "AC2", "smoothness", "spectral_regularity"]
    X_pca = np.array([[b[f] for f in feat_names] for b in books_for_pca])

    # Standardize
    means = X_pca.mean(axis=0)
    stds = X_pca.std(axis=0)
    stds = np.where(stds == 0, 1.0, stds)
    X_std = (X_pca - means) / stds

    # Covariance matrix PCA
    cov = np.cov(X_std.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort by descending eigenvalue
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # PC1 = Parallelism Index
    pc1 = X_std @ eigenvectors[:, 0]
    variance_explained = eigenvalues / eigenvalues.sum()

    # Ensure PC1 is positively correlated with AC1 (parallelism = positive)
    ac1_values = np.array([b["AC1"] for b in books_for_pca])
    if sp_stats.pearsonr(pc1, ac1_values)[0] < 0:
        pc1 = -pc1
        eigenvectors[:, 0] = -eigenvectors[:, 0]

    for i, b in enumerate(books_for_pca):
        b["parallelism_index"] = round(float(pc1[i]), 4)
        metrics_all[b["book"]]["parallelism_index"] = b["parallelism_index"]

    pca_info = {
        "features": feat_names,
        "eigenvalues": [round(float(e), 4) for e in eigenvalues],
        "variance_explained": [round(float(v), 4) for v in variance_explained],
        "PC1_loadings": {f: round(float(eigenvectors[i, 0]), 4) for i, f in enumerate(feat_names)},
        "n_books": int(len(books_for_pca)),
    }
    log.info(f"  PC1 explains {variance_explained[0]*100:.1f}% of variance")
    log.info(f"  Loadings: {pca_info['PC1_loadings']}")

    # ──────────────────────────────────────────────────────────
    # 4. Correlate PI with DFA
    # ──────────────────────────────────────────────────────────
    log.info("\n--- Correlations with DFA ---")

    dfa_vals = np.array([b["DFA"] for b in books_for_pca])

    correlations = {}

    # PI vs DFA
    r_pi, p_pi = sp_stats.pearsonr(pc1, dfa_vals)
    rho_pi, p_rho_pi = sp_stats.spearmanr(pc1, dfa_vals)
    correlations["PI_vs_DFA"] = {
        "pearson_r": round(float(r_pi), 4),
        "pearson_p": round(float(p_pi), 6),
        "spearman_rho": round(float(rho_pi), 4),
        "spearman_p": round(float(p_rho_pi), 6),
    }
    log.info(f"  PI vs DFA: r={r_pi:.4f} (p={p_pi:.4e})")

    # Individual metrics vs DFA
    for feat in feat_names + ["run_length"]:
        vals = np.array([b[feat] for b in books_for_pca
                         if b.get(feat) is not None])
        dfa_sub = np.array([b["DFA"] for b in books_for_pca
                            if b.get(feat) is not None])
        if len(vals) < 5:
            continue
        r, p = sp_stats.pearsonr(vals, dfa_sub)
        correlations[f"{feat}_vs_DFA"] = {
            "pearson_r": round(float(r), 4),
            "pearson_p": round(float(p), 6),
            "n": int(len(vals)),
        }
        log.info(f"  {feat} vs DFA: r={r:.4f} (p={p:.4e})")

    correlations["_pca"] = pca_info

    with open(RESULTS_DIR / "parallelism_dfa_correlation.json", "w") as f:
        json.dump(correlations, f, indent=2, ensure_ascii=False)
    log.info("  Saved parallelism_dfa_correlation.json")

    # Save full metrics
    with open(RESULTS_DIR / "parallelism_metrics.json", "w") as f:
        json.dump(metrics_all, f, indent=2, ensure_ascii=False, default=str)
    log.info("  Saved parallelism_metrics.json")

    # ──────────────────────────────────────────────────────────
    # 5. Genre profiles
    # ──────────────────────────────────────────────────────────
    log.info("\n--- Genre Parallelism Profiles ---")

    genre_groups = {}
    for b in books_for_pca:
        g = b["genre"]
        if g not in genre_groups:
            genre_groups[g] = []
        genre_groups[g].append(b)

    genre_profiles = {}
    for genre in sorted(genre_groups.keys()):
        group = genre_groups[genre]
        profile = {"n": len(group)}
        for feat in feat_names + ["parallelism_index", "DFA", "run_length"]:
            vals = [b[feat] for b in group if b.get(feat) is not None]
            if vals:
                profile[f"{feat}_mean"] = round(float(np.mean(vals)), 4)
                profile[f"{feat}_std"] = round(float(np.std(vals, ddof=1)), 4) if len(vals) > 1 else 0.0
        genre_profiles[genre] = profile
        log.info(f"  {genre}: n={len(group)}, PI={profile.get('parallelism_index_mean', 'N/A')}")

    # Mann-Whitney tests: AT_poetry vs AT_narrative on PI
    at_poetry_pi = [b["parallelism_index"] for b in books_for_pca
                    if b["genre"] == "AT_poetry"]
    at_narr_pi = [b["parallelism_index"] for b in books_for_pca
                  if b["genre"] == "AT_narrative"]

    if len(at_poetry_pi) >= 2 and len(at_narr_pi) >= 2:
        u, p = sp_stats.mannwhitneyu(at_poetry_pi, at_narr_pi, alternative="two-sided")
        genre_profiles["_AT_poetry_vs_AT_narrative"] = {
            "test": "Mann-Whitney",
            "U": round(float(u), 1),
            "p": round(float(p), 6),
            "AT_poetry_mean_PI": round(float(np.mean(at_poetry_pi)), 4),
            "AT_narrative_mean_PI": round(float(np.mean(at_narr_pi)), 4),
        }
        log.info(f"  AT_poetry vs AT_narrative PI: U={u:.1f}, p={p:.6f}")

    # AT vs NT on PI
    at_pi = [b["parallelism_index"] for b in books_for_pca if b["testament"] == "AT"]
    nt_pi = [b["parallelism_index"] for b in books_for_pca if b["testament"] == "NT"]
    if at_pi and nt_pi:
        u, p = sp_stats.mannwhitneyu(at_pi, nt_pi, alternative="two-sided")
        genre_profiles["_AT_vs_NT"] = {
            "test": "Mann-Whitney",
            "U": round(float(u), 1),
            "p": round(float(p), 6),
            "AT_mean_PI": round(float(np.mean(at_pi)), 4),
            "NT_mean_PI": round(float(np.mean(nt_pi)), 4),
        }
        log.info(f"  AT vs NT PI: U={u:.1f}, p={p:.6f}")

    with open(RESULTS_DIR / "genre_parallelism_profiles.json", "w") as f:
        json.dump(genre_profiles, f, indent=2, ensure_ascii=False)
    log.info("  Saved genre_parallelism_profiles.json")

    # ──────────────────────────────────────────────────────────
    # 6. External corpus predictions
    # ──────────────────────────────────────────────────────────
    log.info("\n--- External Corpus Predictions ---")

    # Build regression: DFA ~ PI (from Bible books)
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(pc1.reshape(-1, 1), dfa_vals)

    ext_predictions = {}
    for corpus, data in EXTERNAL_CORPORA.items():
        ac1_ext = data["AC1"]
        dfa_ext = data["DFA"]

        # We can only compute a partial PI proxy: standardize AC1 and use
        # the AC1 loading to approximate PI (since we only have AC1 for externals)
        ac1_std = (ac1_ext - means[0]) / stds[0]
        pi_approx = ac1_std * eigenvectors[0, 0]  # AC1 contribution to PC1

        dfa_pred = float(model.predict(np.array([[pi_approx]]))[0])

        ext_predictions[corpus] = {
            "AC1": ac1_ext,
            "DFA_actual": dfa_ext,
            "PI_approx": round(float(pi_approx), 4),
            "DFA_predicted": round(float(dfa_pred), 4),
            "residual": round(float(dfa_ext - dfa_pred), 4),
        }
        log.info(f"  {corpus}: DFA_actual={dfa_ext}, DFA_pred={dfa_pred:.4f}, "
                 f"residual={dfa_ext - dfa_pred:.4f}")

    ext_predictions["_regression"] = {
        "formula": "DFA ~ PI",
        "slope": round(float(model.coef_[0]), 4),
        "intercept": round(float(model.intercept_), 4),
        "R2": round(float(r_pi ** 2), 4),
    }

    with open(RESULTS_DIR / "external_parallelism_prediction.json", "w") as f:
        json.dump(ext_predictions, f, indent=2, ensure_ascii=False)
    log.info("  Saved external_parallelism_prediction.json")

    log.info(f"\n{'=' * 70}")
    log.info("Script 2 completado.")
    log.info(f"  r(PI, DFA) = {r_pi:.4f} (p={p_pi:.4e})")
    log.info(f"  PC1 explains {variance_explained[0]*100:.1f}% variance")

    return 0


if __name__ == "__main__":
    exit(main())
