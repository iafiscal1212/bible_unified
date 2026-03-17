#!/usr/bin/env python3
"""
Fase 19 — Script 1: classifier_correction.py (BLOQUEANTE)

Corrige el sesgo del clasificador eliminando features dependientes de longitud
absoluta. Solo usa features temporales/adimensionales:
  H, DFA, AC1, CV

Features PROHIBIDAS: mean_verse_len, std_verse_len, skewness, pos_entropy
"""

import json
import logging
import pickle
import numpy as np
from pathlib import Path
from scipy import stats as sp_stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut

BASE = Path(__file__).resolve().parent
RESULTS_DIR = BASE / "results" / "classifier_corrected"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)

FEAT_NAMES = ["H", "DFA", "AC1", "CV"]


def load_training_data():
    """Load 66 biblical books, extract permitted features."""
    ff = BASE / "results" / "refined_classifier" / "book_features.json"
    with open(ff) as f:
        bf = json.load(f)
    if isinstance(bf, dict):
        bf = list(bf.values())

    books, X, y, meta = [], [], [], []
    skipped = []
    for b in bf:
        row = [b.get(fn) for fn in FEAT_NAMES]
        if any(v is None for v in row):
            skipped.append(b["book"])
            continue
        books.append(b["book"])
        X.append(row)
        y.append(0 if b["testament"] == "AT" else 1)
        meta.append({
            "book": b["book"], "testament": b["testament"],
            "mean_verse_len": b.get("mean_verse_len"),
        })

    log.info(f"  Training data: {len(X)} books ({len([v for v in y if v==0])} AT, "
             f"{len([v for v in y if v==1])} NT)")
    if skipped:
        log.info(f"  Skipped (null DFA): {skipped}")

    return books, np.array(X), np.array(y), meta, skipped


def train_and_evaluate(X, y, books, meta):
    """Train 3 classifiers with LOO CV, select best."""
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    classifiers = {
        "LogisticRegression": LogisticRegression(max_iter=2000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(probability=True, kernel="rbf", random_state=42),
    }

    results = {}
    loo = LeaveOneOut()

    for name, clf in classifiers.items():
        preds = np.zeros(len(y))
        probas = np.zeros(len(y))
        misclassified = []

        for train_idx, test_idx in loo.split(X_s):
            clf.fit(X_s[train_idx], y[train_idx])
            pred = clf.predict(X_s[test_idx])[0]
            prob = clf.predict_proba(X_s[test_idx])[0]
            preds[test_idx[0]] = pred
            probas[test_idx[0]] = prob[0]  # P(AT)

            if pred != y[test_idx[0]]:
                misclassified.append({
                    "book": books[test_idx[0]],
                    "true": "AT" if y[test_idx[0]] == 0 else "NT",
                    "predicted": "AT" if pred == 0 else "NT",
                    "P_AT": round(float(prob[0]), 4),
                })

        acc = np.mean(preds == y)
        results[name] = {
            "accuracy_loo": round(float(acc), 4),
            "n_misclassified": len(misclassified),
            "misclassified": misclassified,
            "probas_AT": probas.tolist(),
        }
        log.info(f"  {name}: LOO accuracy = {acc:.4f} "
                 f"({len(misclassified)} misclassified)")

    # Select best by LOO accuracy
    best_name = max(results, key=lambda k: results[k]["accuracy_loo"])
    log.info(f"  Best classifier: {best_name}")

    # Retrain best on full data
    best_clf = classifiers[best_name].__class__(
        **classifiers[best_name].get_params())
    best_clf.fit(X_s, y)

    return best_name, best_clf, scaler, results


def bias_check(X, y, books, meta, results, best_name):
    """Check correlation between mean_verse_len and P_AT predicted."""
    probas = results[best_name]["probas_AT"]
    mvl = [m["mean_verse_len"] for m in meta]

    # Pearson correlation
    r, p = sp_stats.pearsonr(mvl, probas)
    log.info(f"  Bias check: r(mean_verse_len, P_AT) = {r:.4f} (p={p:.4f})")
    biased = abs(r) >= 0.3

    if biased:
        log.warning(f"  WARNING: |r| = {abs(r):.3f} >= 0.3 — bias may persist")
    else:
        log.info(f"  OK: |r| = {abs(r):.3f} < 0.3 — no systematic bias")

    return {
        "r_mean_verse_len_vs_P_AT": round(float(r), 4),
        "p_value": round(float(p), 4),
        "abs_r": round(float(abs(r)), 4),
        "bias_threshold": 0.3,
        "bias_detected": bool(biased),
    }


def load_external_metrics():
    """Load metrics for all external corpora from saved result files."""
    externals = {}

    # AT, NT, Corán, Rig_Veda from fitted_params.json
    fp_file = BASE / "results" / "unified_model" / "fitted_params.json"
    if fp_file.exists():
        with open(fp_file) as f:
            fp = json.load(f)
        for name, data in fp.items():
            t = data.get("target", {})
            externals[name] = {
                "H": t.get("H"), "DFA": t.get("DFA"),
                "AC1": t.get("AC1"), "CV": t.get("cv"),
                "source": "fitted_params.json (corpus-level)",
            }

    # Individual corpus files
    corpus_files = {
        "Mishnah": "mishnah/mishnah_metrics.json",
        "Tosefta": "tosefta/tosefta_metrics.json",
        "Book_of_Dead": "book_of_dead/corpus_metrics.json",
        "Pali_Canon": "pali_canon/combined_metrics.json",
        "Didache": "didache/didache_metrics.json",
    }

    for name, rel_path in corpus_files.items():
        fpath = BASE / "results" / rel_path
        if fpath.exists():
            with open(fpath) as f:
                data = json.load(f)
            if isinstance(data, dict) and "error" not in data:
                externals[name] = {
                    "H": data.get("H"), "DFA": data.get("DFA"),
                    "AC1": data.get("AC1"), "CV": data.get("CV"),
                    "source": rel_path,
                }

    # Homero from recitation + data_matrix
    homer_ac1_file = BASE / "results" / "recitation" / "homer_syllable_ac1.json"
    dm_file = BASE / "results" / "transmission_origin" / "data_matrix.json"
    if homer_ac1_file.exists() and dm_file.exists():
        with open(homer_ac1_file) as f:
            hac = json.load(f)
        with open(dm_file) as f:
            dm = json.load(f)
        homer_dm = next((c for c in dm if c["corpus"] == "Homero"), None)
        if homer_dm:
            externals["Homero"] = {
                "H": homer_dm.get("H"), "DFA": homer_dm.get("DFA"),
                "AC1": hac.get("ac1_words"),
                "CV": None,  # not available
                "source": "homer_syllable_ac1.json + data_matrix.json",
                "missing_features": ["CV"],
            }

    # Heródoto
    if dm_file.exists():
        with open(dm_file) as f:
            dm = json.load(f)
        herod = next((c for c in dm if c["corpus"] == "Heródoto"), None)
        if herod:
            externals["Heródoto"] = {
                "H": herod.get("H"), "DFA": herod.get("DFA"),
                "AC1": None, "CV": None,
                "source": "data_matrix.json",
                "missing_features": ["AC1", "CV"],
            }

    return externals


def reclassify_externals(externals, clf, scaler):
    """Reclassify all external corpora with corrected classifier."""
    results = {}

    for name, data in externals.items():
        features = [data.get(fn) for fn in FEAT_NAMES]

        if any(v is None for v in features):
            missing = [fn for fn, v in zip(FEAT_NAMES, features) if v is None]
            results[name] = {
                "status": "excluded",
                "reason": f"Missing features: {missing}",
                "available": {fn: data.get(fn) for fn in FEAT_NAMES if data.get(fn) is not None},
            }
            log.info(f"  {name}: EXCLUDED (missing {missing})")
            continue

        X_new = scaler.transform([features])
        pred = clf.predict(X_new)[0]
        proba = clf.predict_proba(X_new)[0]

        results[name] = {
            "status": "classified",
            "predicted_class": "AT" if pred == 0 else "NT",
            "P_AT": round(float(proba[0]), 4),
            "P_NT": round(float(proba[1]), 4),
            "features": {fn: round(v, 4) for fn, v in zip(FEAT_NAMES, features)},
            "source": data.get("source", "unknown"),
        }
        log.info(f"  {name}: {results[name]['predicted_class']}-like "
                 f"(P_AT={results[name]['P_AT']})")

    return results


def load_original_classifications():
    """Load original (biased) classifier results for comparison."""
    original = {}
    dirs = {
        "Mishnah": "mishnah", "Tosefta": "tosefta",
        "Book_of_Dead": "book_of_dead", "Pali_Canon": "pali_canon",
        "Didache": "didache",
    }
    for name, d in dirs.items():
        cf = BASE / "results" / d / "classifier_result.json"
        if cf.exists():
            with open(cf) as f:
                data = json.load(f)
            original[name] = {
                "predicted_class": data.get("predicted_class"),
                "P_AT": data.get("P_AT"),
            }
    return original


def main():
    log.info("=" * 70)
    log.info("FASE 19 — Script 1: Corrección del clasificador")
    log.info("Features: H, DFA, AC1, CV (sin mean_verse_len, pos_entropy)")
    log.info("=" * 70)

    # 1. Load training data
    log.info("\n[1] Cargando datos de entrenamiento...")
    books, X, y, meta, skipped = load_training_data()

    # 2. Train and evaluate
    log.info("\n[2] Entrenando 3 clasificadores con LOO CV...")
    best_name, best_clf, scaler, clf_results = train_and_evaluate(X, y, books, meta)

    # 3. Bias check
    log.info("\n[3] Verificación de sesgo...")
    bias = bias_check(X, y, books, meta, clf_results, best_name)

    # 4. Save model
    log.info("\n[4] Guardando modelo corregido...")
    with open(RESULTS_DIR / "model.pkl", "wb") as f:
        pickle.dump({"classifier": best_clf, "scaler": scaler,
                      "features": FEAT_NAMES, "best_name": best_name}, f)

    # Save classifier details
    clf_detail = {
        "features_used": FEAT_NAMES,
        "features_prohibited": ["mean_verse_len", "std_verse_len", "skewness", "pos_entropy"],
        "n_training_books": len(books),
        "skipped_books": skipped,
        "best_classifier": best_name,
        "classifiers": {},
        "bias_check": bias,
    }
    for cname, cr in clf_results.items():
        clf_detail["classifiers"][cname] = {
            "accuracy_loo": cr["accuracy_loo"],
            "n_misclassified": cr["n_misclassified"],
            "misclassified": cr["misclassified"],
        }

    # Per-book scores for best classifier
    book_scores = {}
    for i, book in enumerate(books):
        book_scores[book] = {
            "true": "AT" if y[i] == 0 else "NT",
            "P_AT_corrected": round(clf_results[best_name]["probas_AT"][i], 4),
        }
    clf_detail["book_scores"] = book_scores

    with open(RESULTS_DIR / "classifier_detail.json", "w") as f:
        json.dump(clf_detail, f, indent=2, ensure_ascii=False)

    # 5. Load original classifications for comparison
    log.info("\n[5] Cargando clasificaciones originales...")
    original = load_original_classifications()

    # 6. Reclassify externals
    log.info("\n[6] Reclasificando corpus externos...")
    externals = load_external_metrics()
    reclassified = reclassify_externals(externals, best_clf, scaler)

    # Add original for comparison
    for name in reclassified:
        if name in original:
            reclassified[name]["original_predicted"] = original[name].get("predicted_class")
            reclassified[name]["original_P_AT"] = original[name].get("P_AT")
            reclassified[name]["classification_changed"] = (
                reclassified[name].get("predicted_class") !=
                original[name].get("predicted_class")
            )

    with open(RESULTS_DIR / "reclassification_all_corpora.json", "w") as f:
        json.dump(reclassified, f, indent=2, ensure_ascii=False)

    # 7. Mishnah verdict
    log.info("\n[7] Generando mishnah_verdict.json...")
    mish = reclassified.get("Mishnah", {})
    mish_orig = original.get("Mishnah", {})
    mishnah_verdict = {
        "corpus": "Mishnah",
        "P_AT_original": mish_orig.get("P_AT"),
        "P_AT_corrected": mish.get("P_AT"),
        "original_class": mish_orig.get("predicted_class"),
        "corrected_class": mish.get("predicted_class"),
        "classification_changed": mish.get("classification_changed", False),
        "features": mish.get("features"),
        "classifier": best_name,
        "classifier_loo_accuracy": clf_results[best_name]["accuracy_loo"],
    }
    with open(RESULTS_DIR / "mishnah_verdict.json", "w") as f:
        json.dump(mishnah_verdict, f, indent=2, ensure_ascii=False)

    log.info(f"\n  MISHNAH VERDICT:")
    log.info(f"    Original: {mishnah_verdict['original_class']}-like "
             f"(P_AT={mishnah_verdict['P_AT_original']})")
    log.info(f"    Corrected: {mishnah_verdict['corrected_class']}-like "
             f"(P_AT={mishnah_verdict['P_AT_corrected']})")
    log.info(f"    Changed: {mishnah_verdict['classification_changed']}")

    # 8. Summary of changes
    log.info("\n[8] Resumen de cambios:")
    changes = []
    for name, data in reclassified.items():
        if data.get("classification_changed"):
            changes.append({
                "corpus": name,
                "original": data.get("original_predicted"),
                "corrected": data.get("predicted_class"),
                "P_AT_original": data.get("original_P_AT"),
                "P_AT_corrected": data.get("P_AT"),
            })
            log.info(f"  CHANGED: {name}: {data.get('original_predicted')} → "
                     f"{data.get('predicted_class')} "
                     f"(P_AT: {data.get('original_P_AT')} → {data.get('P_AT')})")

    if not changes:
        log.info("  No classification changes")

    with open(RESULTS_DIR / "changes_summary.json", "w") as f:
        json.dump({
            "n_changes": len(changes),
            "changes": changes,
            "best_classifier": best_name,
            "loo_accuracy": clf_results[best_name]["accuracy_loo"],
            "bias_check": bias,
        }, f, indent=2, ensure_ascii=False)

    log.info(f"\n{'=' * 70}")
    log.info(f"Script 1 completado. Resultados en {RESULTS_DIR}")
    log.info(f"Best: {best_name}, LOO acc={clf_results[best_name]['accuracy_loo']}")
    log.info(f"Bias |r|={bias['abs_r']} ({'BIASED' if bias['bias_detected'] else 'OK'})")
    log.info(f"Mishnah: {mishnah_verdict['corrected_class']}-like")
    log.info(f"Changes: {len(changes)}")


if __name__ == "__main__":
    main()
