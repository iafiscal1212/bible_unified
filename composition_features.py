#!/usr/bin/env python3
"""
Fase 11 — Script 1: Composition Features
¿Qué propiedades del proceso de composición producen H alto?

Calcula features compositivas por libro, correlaciona con H,
regresión múltiple, y test con corpus externos.
"""

import json
import logging
import os
import sys
import time
import numpy as np
from pathlib import Path
from scipy import stats
from collections import Counter, defaultdict

BASE = Path(__file__).resolve().parent
RESULTS_DIR = BASE / "results" / "composition"
LOG_DIR = BASE / "logs"
BIBLE_JSON = BASE / "bible_unified.json"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "fase11_composition.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


# ── Métricas ─────────────────────────────────────────────────────────────

def hurst_exponent_rs(series):
    series = np.asarray(series, dtype=float)
    n = len(series)
    if n < 20:
        return float("nan"), 0.0
    min_block, max_block = 10, n // 2
    sizes, rs_values = [], []
    block = min_block
    while block <= max_block:
        sizes.append(block)
        n_blocks = n // block
        rs_list = []
        for i in range(n_blocks):
            seg = series[i * block:(i + 1) * block]
            devs = np.cumsum(seg - seg.mean())
            R = devs.max() - devs.min()
            S = seg.std(ddof=1)
            if S > 0:
                rs_list.append(R / S)
        if rs_list:
            rs_values.append(np.mean(rs_list))
        block = int(block * 1.5)
        if block == sizes[-1]:
            block += 1
    if len(sizes) < 3:
        return float("nan"), 0.0
    slope, _, r, _, _ = stats.linregress(np.log(sizes), np.log(rs_values))
    return float(slope), float(r ** 2)


def dfa_exponent(series):
    series = np.asarray(series, dtype=float)
    n = len(series)
    if n < 20:
        return float("nan"), 0.0
    y = np.cumsum(series - series.mean())
    min_box, max_box = 4, n // 4
    sizes, flucts = [], []
    box = min_box
    while box <= max_box:
        sizes.append(box)
        n_boxes = n // box
        rms_list = []
        for i in range(n_boxes):
            seg = y[i * box:(i + 1) * box]
            coeffs = np.polyfit(np.arange(box), seg, 1)
            trend = np.polyval(coeffs, np.arange(box))
            rms_list.append(np.sqrt(np.mean((seg - trend) ** 2)))
        if rms_list:
            flucts.append(np.mean(rms_list))
        box = int(box * 1.5)
        if box == sizes[-1]:
            box += 1
    if len(sizes) < 3:
        return float("nan"), 0.0
    slope, _, r, _, _ = stats.linregress(np.log(sizes), np.log(flucts))
    return float(slope), float(r ** 2)


# ── Carga de datos ───────────────────────────────────────────────────────

def load_bible_by_book():
    """Carga bible_unified.json y agrupa por libro."""
    log.info("Cargando bible_unified.json...")
    with open(BIBLE_JSON, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    books = defaultdict(lambda: {"words": [], "verses": defaultdict(list)})
    for w in corpus:
        book = w.get("book", "")
        ch = w.get("chapter", 0)
        vs = w.get("verse", 0)
        books[book]["words"].append(w)
        books[book]["verses"][(ch, vs)].append(w)

    log.info(f"  {len(books)} libros cargados")
    return books


def load_external_corpora():
    """Carga datos de corpus externos desde results/."""
    comparison_file = BASE / "results" / "fase5_comparison.json"
    if comparison_file.exists():
        with open(comparison_file, "r") as f:
            return json.load(f)
    return []


# ── Features compositivas por libro ──────────────────────────────────────

def compute_book_features(book_name, book_data, corpus_type):
    """Calcula features compositivas de un libro."""
    words = book_data["words"]
    verses = book_data["verses"]
    n_words = len(words)

    if n_words < 50:
        return None

    # Serie temporal: palabras por versículo
    sorted_vkeys = sorted(verses.keys())
    verse_lens = [len(verses[k]) for k in sorted_vkeys]
    n_verses = len(verse_lens)

    if n_verses < 20:
        return None

    # H y alpha del libro
    h, h_r2 = hurst_exponent_rs(verse_lens)
    alpha, alpha_r2 = dfa_exponent(verse_lens)

    # Feature 1: Densidad de repetición (tipo/token ratio invertido)
    lemmas = [w.get("lemma", w.get("text", "")) for w in words]
    unique_lemmas = len(set(lemmas))
    type_token_ratio = unique_lemmas / n_words if n_words > 0 else 0
    repetition_density = 1 - type_token_ratio  # alto = mucha repetición

    # Feature 2: CV de longitud de versículo
    vl_array = np.array(verse_lens, dtype=float)
    cv_verse_len = float(vl_array.std() / vl_array.mean()) if vl_array.mean() > 0 else 0

    # Feature 3: Densidad verbal (verbos / total)
    pos_counts = Counter()
    for w in words:
        pos = w.get("pos", "other")
        pos_counts[pos] += 1
    verb_density = pos_counts.get("verb", 0) / n_words if n_words > 0 else 0

    # Feature 4: Ratio de nombres propios
    # En hebreo: pos contiene info, en OSHB morph: HNp = proper noun
    proper_count = 0
    for w in words:
        morph = w.get("morph", "")
        if len(morph) >= 3 and morph[1] == "N" and morph[2] == "p":
            proper_count += 1
        elif w.get("pos", "") in ("proper_noun", "nmpr"):
            proper_count += 1
    proper_ratio = proper_count / n_words if n_words > 0 else 0

    # Feature 5: Densidad de conectores (conjunciones + preposiciones)
    connector_count = sum(pos_counts.get(p, 0)
                          for p in ["conjunction", "preposition", "particle"])
    connector_density = connector_count / n_words if n_words > 0 else 0

    # Feature 6: Entropía POS (diversidad sintáctica)
    pos_probs = np.array(list(pos_counts.values()), dtype=float)
    pos_probs = pos_probs / pos_probs.sum()
    pos_entropy = float(-np.sum(pos_probs * np.log2(pos_probs + 1e-15)))

    # Feature 7: Autocorrelación lag-1 de longitudes de versículo
    if n_verses > 2:
        autocorr_1 = float(np.corrcoef(verse_lens[:-1], verse_lens[1:])[0, 1])
    else:
        autocorr_1 = 0.0

    # Feature 8: Skewness de longitudes de versículo
    skewness = float(stats.skew(vl_array))

    # Feature 9: Varianza de longitud (std absoluta)
    std_verse_len = float(vl_array.std())

    # Feature 10: Mean verse length
    mean_verse_len = float(vl_array.mean())

    return {
        "book": book_name,
        "corpus": corpus_type,
        "n_words": n_words,
        "n_verses": n_verses,
        "H": float(h) if not np.isnan(h) else None,
        "H_R2": float(h_r2),
        "alpha": float(alpha) if not np.isnan(alpha) else None,
        "features": {
            "repetition_density": round(repetition_density, 4),
            "cv_verse_len": round(cv_verse_len, 4),
            "verb_density": round(verb_density, 4),
            "proper_ratio": round(proper_ratio, 4),
            "connector_density": round(connector_density, 4),
            "pos_entropy": round(pos_entropy, 4),
            "autocorr_lag1": round(autocorr_1, 4) if not np.isnan(autocorr_1) else 0.0,
            "skewness": round(skewness, 4),
            "std_verse_len": round(std_verse_len, 4),
            "mean_verse_len": round(mean_verse_len, 4),
        },
    }


# ── Análisis de correlación ──────────────────────────────────────────────

def correlation_analysis(book_features):
    """Correlaciona cada feature con H."""
    log.info("\n── Correlación features vs H ──")

    valid = [b for b in book_features if b["H"] is not None and not np.isnan(b["H"])]
    log.info(f"  {len(valid)} libros con H válido")

    h_vals = np.array([b["H"] for b in valid])
    feature_names = list(valid[0]["features"].keys())

    correlations = []
    for fname in feature_names:
        fvals = np.array([b["features"][fname] for b in valid])
        # Filter NaN
        mask = ~np.isnan(fvals)
        if mask.sum() < 5:
            continue
        r, p = stats.pearsonr(fvals[mask], h_vals[mask])
        rho, p_rho = stats.spearmanr(fvals[mask], h_vals[mask])
        correlations.append({
            "feature": fname,
            "pearson_r": round(float(r), 4),
            "pearson_p": float(p),
            "spearman_rho": round(float(rho), 4),
            "spearman_p": float(p_rho),
            "abs_r": round(abs(float(r)), 4),
        })
        log.info(f"  {fname}: r={r:.4f} (p={p:.4f}), ρ={rho:.4f} (p={p_rho:.4f})")

    correlations.sort(key=lambda x: x["abs_r"], reverse=True)
    return correlations


def forward_selection_regression(book_features):
    """Forward selection regression with AIC."""
    log.info("\n── Regresión múltiple (forward selection) ──")

    valid = [b for b in book_features if b["H"] is not None and not np.isnan(b["H"])]
    h_vals = np.array([b["H"] for b in valid])
    feature_names = list(valid[0]["features"].keys())

    # Build feature matrix
    X_all = np.column_stack([
        np.array([b["features"][fn] for b in valid]) for fn in feature_names
    ])

    # Replace NaN with column means
    for col in range(X_all.shape[1]):
        mask = np.isnan(X_all[:, col])
        if mask.any():
            X_all[mask, col] = np.nanmean(X_all[:, col])

    n = len(h_vals)
    selected = []
    remaining = list(range(len(feature_names)))
    best_aic = np.inf

    # Null model AIC
    ss_null = np.sum((h_vals - h_vals.mean()) ** 2)
    aic_null = n * np.log(ss_null / n) + 2 * 1
    log.info(f"  Null model: AIC={aic_null:.2f}")

    models = [{"features": [], "aic": float(aic_null), "r2": 0.0}]

    for step in range(min(5, len(feature_names))):
        best_new_aic = np.inf
        best_new_idx = None

        for idx in remaining:
            trial = selected + [idx]
            X_trial = np.column_stack([X_all[:, i] for i in trial])
            X_trial = np.column_stack([np.ones(n), X_trial])

            try:
                beta = np.linalg.lstsq(X_trial, h_vals, rcond=None)[0]
                residuals = h_vals - X_trial @ beta
                ss_res = np.sum(residuals ** 2)
                k = len(trial) + 1  # +1 for intercept
                aic = n * np.log(ss_res / n) + 2 * k

                if aic < best_new_aic:
                    best_new_aic = aic
                    best_new_idx = idx
            except Exception:
                continue

        if best_new_aic < best_aic - 2:  # AIC improvement > 2
            selected.append(best_new_idx)
            remaining.remove(best_new_idx)
            best_aic = best_new_aic

            # Compute R2
            X_sel = np.column_stack([X_all[:, i] for i in selected])
            X_sel = np.column_stack([np.ones(n), X_sel])
            beta = np.linalg.lstsq(X_sel, h_vals, rcond=None)[0]
            ss_res = np.sum((h_vals - X_sel @ beta) ** 2)
            ss_tot = np.sum((h_vals - h_vals.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot

            model_features = [feature_names[i] for i in selected]
            log.info(f"  Step {step + 1}: +{feature_names[best_new_idx]}, "
                     f"AIC={best_aic:.2f}, R²={r2:.4f}")
            models.append({
                "features": model_features,
                "aic": float(best_aic),
                "r2": float(r2),
                "coefficients": {feature_names[selected[i]]: float(beta[i + 1])
                                  for i in range(len(selected))},
                "intercept": float(beta[0]),
            })
        else:
            log.info(f"  Step {step + 1}: no improvement, stopping")
            break

    # LOO cross-validation for best model
    if selected:
        loo_errors = []
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            X_train = np.column_stack([X_all[mask, j] for j in selected])
            X_train = np.column_stack([np.ones(mask.sum()), X_train])
            y_train = h_vals[mask]
            try:
                beta = np.linalg.lstsq(X_train, y_train, rcond=None)[0]
                x_test = np.array([1.0] + [X_all[i, j] for j in selected])
                pred = x_test @ beta
                loo_errors.append((h_vals[i] - pred) ** 2)
            except Exception:
                pass

        loo_rmse = float(np.sqrt(np.mean(loo_errors))) if loo_errors else None
        log.info(f"  LOO RMSE: {loo_rmse:.4f}")
        models[-1]["loo_rmse"] = loo_rmse

    return models


def synthetic_test(book_features, top_features, n_synthetic=500):
    """Test: ¿puede un texto sintético con features del AT reproducir H>0.9?"""
    log.info("\n── Test sintético ──")

    valid = [b for b in book_features if b["H"] is not None]
    ot_books = [b for b in valid if b["corpus"] == "OT"]
    nt_books = [b for b in valid if b["corpus"] == "NT"]

    # Get AT-like parameters
    if not ot_books:
        return {"error": "no OT books"}

    ot_h = np.mean([b["H"] for b in ot_books])
    ot_mean_vl = np.mean([b["features"]["mean_verse_len"] for b in ot_books])
    ot_cv = np.mean([b["features"]["cv_verse_len"] for b in ot_books])
    ot_autocorr = np.mean([b["features"]["autocorr_lag1"] for b in ot_books])
    ot_skew = np.mean([b["features"]["skewness"] for b in ot_books])

    nt_mean_vl = np.mean([b["features"]["mean_verse_len"] for b in nt_books]) if nt_books else 15
    nt_cv = np.mean([b["features"]["cv_verse_len"] for b in nt_books]) if nt_books else 0.5
    nt_autocorr = np.mean([b["features"]["autocorr_lag1"] for b in nt_books]) if nt_books else 0.0

    log.info(f"  OT params: mean_vl={ot_mean_vl:.1f}, CV={ot_cv:.3f}, "
             f"autocorr={ot_autocorr:.3f}, skew={ot_skew:.3f}")

    rng = np.random.default_rng(42)

    def generate_ar1_series(n, mean, std, phi):
        """Generate AR(1) series with specified autocorrelation."""
        innovations = rng.normal(0, std * np.sqrt(1 - phi ** 2), n)
        series = np.zeros(n)
        series[0] = mean + innovations[0]
        for i in range(1, n):
            series[i] = mean * (1 - phi) + phi * series[i - 1] + innovations[i]
        return np.maximum(series, 1).astype(int)

    results = {"ot_like": [], "nt_like": [], "iid": []}

    n_verses = 800  # typical book size

    for _ in range(n_synthetic):
        # OT-like: high autocorrelation
        ot_series = generate_ar1_series(n_verses, ot_mean_vl,
                                         ot_mean_vl * ot_cv,
                                         max(0.01, min(0.99, ot_autocorr)))
        h_ot, _ = hurst_exponent_rs(ot_series)
        if not np.isnan(h_ot):
            results["ot_like"].append(float(h_ot))

        # NT-like: lower autocorrelation
        nt_series = generate_ar1_series(n_verses, nt_mean_vl,
                                         nt_mean_vl * nt_cv,
                                         max(0.01, min(0.99, nt_autocorr)))
        h_nt, _ = hurst_exponent_rs(nt_series)
        if not np.isnan(h_nt):
            results["nt_like"].append(float(h_nt))

        # IID baseline
        iid_series = rng.normal(ot_mean_vl, ot_mean_vl * ot_cv, n_verses)
        iid_series = np.maximum(iid_series, 1).astype(int)
        h_iid, _ = hurst_exponent_rs(iid_series)
        if not np.isnan(h_iid):
            results["iid"].append(float(h_iid))

    summary = {}
    for label, h_list in results.items():
        if h_list:
            arr = np.array(h_list)
            summary[label] = {
                "n": len(h_list),
                "H_mean": round(float(arr.mean()), 4),
                "H_std": round(float(arr.std()), 4),
                "H_median": round(float(np.median(arr)), 4),
                "pct_above_0.9": round(float(np.mean(arr > 0.9) * 100), 1),
                "pct_above_0.8": round(float(np.mean(arr > 0.8) * 100), 1),
            }
            log.info(f"  {label}: H={arr.mean():.4f}±{arr.std():.4f}, "
                     f">{0.9}: {np.mean(arr > 0.9) * 100:.1f}%")

    # Conclusion
    can_reproduce = (summary.get("ot_like", {}).get("pct_above_0.9", 0) > 50)
    conclusion = ("AR(1) con autocorrelación AT-like PUEDE reproducir H>0.9"
                  if can_reproduce
                  else "AR(1) con autocorrelación AT-like NO reproduce H>0.9 — "
                       "se requiere memoria de orden superior")

    return {"synthetic_results": summary, "conclusion": conclusion,
            "ot_params": {"mean_vl": ot_mean_vl, "cv": ot_cv,
                          "autocorr_lag1": ot_autocorr}}


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    log.info("=" * 70)
    log.info("FASE 11 — Script 1: Composition Features")
    log.info("=" * 70)

    books = load_bible_by_book()

    # Determinar corpus (AT vs NT)
    ot_books_list = ["Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy",
                     "Joshua", "Judges", "Ruth", "1 Samuel", "2 Samuel",
                     "1 Kings", "2 Kings", "1 Chronicles", "2 Chronicles",
                     "Ezra", "Nehemiah", "Esther", "Job", "Psalms",
                     "Proverbs", "Ecclesiastes", "Song of Solomon",
                     "Isaiah", "Jeremiah", "Lamentations", "Ezekiel",
                     "Daniel", "Hosea", "Joel", "Amos", "Obadiah",
                     "Jonah", "Micah", "Nahum", "Habakkuk", "Zephaniah",
                     "Haggai", "Zechariah", "Malachi"]

    book_features = []
    for book_name, book_data in books.items():
        corpus_type = "OT" if book_name in ot_books_list else "NT"
        feats = compute_book_features(book_name, book_data, corpus_type)
        if feats:
            book_features.append(feats)
            log.info(f"  {book_name} ({corpus_type}): H={feats['H']}, "
                     f"n_verses={feats['n_verses']}")

    log.info(f"\n  Total: {len(book_features)} libros con features")

    # Save raw features
    with open(RESULTS_DIR / "book_features.json", "w") as f:
        json.dump(book_features, f, indent=2, ensure_ascii=False)

    # Correlations
    correlations = correlation_analysis(book_features)
    with open(RESULTS_DIR / "feature_correlations.json", "w") as f:
        json.dump(correlations, f, indent=2, ensure_ascii=False)

    # Feature ranking
    ranking = {"top_predictors": correlations[:5],
               "all_correlations": correlations}
    with open(RESULTS_DIR / "feature_ranking.json", "w") as f:
        json.dump(ranking, f, indent=2, ensure_ascii=False)

    # Regression
    models = forward_selection_regression(book_features)
    with open(RESULTS_DIR / "regression_model.json", "w") as f:
        json.dump(models, f, indent=2, ensure_ascii=False)

    # Compare OT vs NT features
    ot_feats = [b for b in book_features if b["corpus"] == "OT"]
    nt_feats = [b for b in book_features if b["corpus"] == "NT"]

    ot_vs_nt = {}
    if ot_feats and nt_feats:
        for fname in ot_feats[0]["features"].keys():
            ot_vals = [b["features"][fname] for b in ot_feats
                       if not np.isnan(b["features"][fname])]
            nt_vals = [b["features"][fname] for b in nt_feats
                       if not np.isnan(b["features"][fname])]
            if ot_vals and nt_vals:
                u, p = stats.mannwhitneyu(ot_vals, nt_vals, alternative="two-sided")
                ot_vs_nt[fname] = {
                    "ot_mean": round(float(np.mean(ot_vals)), 4),
                    "nt_mean": round(float(np.mean(nt_vals)), 4),
                    "mann_whitney_p": float(p),
                    "significant": p < 0.05,
                }
    log.info(f"\n  OT vs NT feature comparison: {len(ot_vs_nt)} features tested")
    for fname, info in sorted(ot_vs_nt.items(), key=lambda x: x[1]["mann_whitney_p"]):
        log.info(f"    {fname}: OT={info['ot_mean']:.4f}, NT={info['nt_mean']:.4f}, "
                 f"p={info['mann_whitney_p']:.4f}")

    # Synthetic test
    top_features = [c["feature"] for c in correlations[:3]] if correlations else []
    synthetic = synthetic_test(book_features, top_features)
    with open(RESULTS_DIR / "synthetic_test.json", "w") as f:
        json.dump(synthetic, f, indent=2, ensure_ascii=False)

    # External corpus comparison
    ext_corpora = load_external_corpora()
    ext_comparison = []
    if ext_corpora:
        for ec in ext_corpora:
            ext_comparison.append({
                "corpus": ec.get("corpus", "?"),
                "H": ec.get("hurst_H"),
                "mean_unit_len": ec.get("mean_unit_length"),
                "type": ec.get("type", "?"),
            })
        with open(RESULTS_DIR / "external_comparison.json", "w") as f:
            json.dump(ext_comparison, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    log.info(f"\n{'=' * 70}")
    log.info(f"Script 1 completado en {elapsed:.1f}s")
    log.info(f"Resultados en {RESULTS_DIR}")

    return {
        "n_books": len(book_features),
        "top_predictor": correlations[0] if correlations else None,
        "best_model": models[-1] if models else None,
        "synthetic_conclusion": synthetic.get("conclusion", ""),
        "ot_vs_nt": ot_vs_nt,
    }


if __name__ == "__main__":
    main()
