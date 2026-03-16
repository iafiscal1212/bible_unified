#!/usr/bin/env python3
"""
Fase 14 — Script 3: Refined Authenticity Tool
Clasificador multi-feature con sklearn para distinguir AT de NT.

Features: H, DFA_α, AC1, CV, mean_verse_len, std_verse_len, pos_entropy
Clasificadores: LogisticRegression, RandomForest, SVM
Evaluación: Leave-One-Out Cross Validation
Scoring: probabilidad de pertenencia a cada testamento para los 66 libros.
"""

import json
import logging
import time
import re
import numpy as np
from pathlib import Path
from scipy import stats as sp_stats
from collections import defaultdict

BASE = Path(__file__).resolve().parent
RESULTS_DIR = BASE / "results" / "refined_classifier"
LOG_DIR = BASE / "logs"
BIBLE_JSON = BASE / "bible_unified.json"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "fase14_refined_classifier.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


# ── Reusable functions ────────────────────────────────────────────

def hurst_exponent_rs(series):
    series = np.asarray(series, dtype=float)
    n = len(series)
    if n < 20:
        return float("nan")
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
        return float("nan")
    slope, _, _, _, _ = sp_stats.linregress(np.log(sizes), np.log(rs_values))
    return float(slope)


def dfa_exponent(series):
    series = np.asarray(series, dtype=float)
    n = len(series)
    if n < 50:
        return float("nan")
    y = np.cumsum(series - series.mean())
    sizes, flucts = [], []
    s = 10
    while s <= n // 4:
        sizes.append(s)
        n_segs = n // s
        f2 = []
        for i in range(n_segs):
            seg = y[i * s:(i + 1) * s]
            x = np.arange(s)
            coef = np.polyfit(x, seg, 1)
            trend = np.polyval(coef, x)
            f2.append(np.mean((seg - trend) ** 2))
        flucts.append(np.sqrt(np.mean(f2)))
        s = int(s * 1.5)
        if s == sizes[-1]:
            s += 1
    if len(sizes) < 3:
        return float("nan")
    slope, _, _, _, _ = sp_stats.linregress(np.log(sizes), np.log(flucts))
    return float(slope)


def autocorr_lag1(series):
    arr = np.asarray(series, dtype=float)
    if len(arr) < 10:
        return float("nan")
    return float(np.corrcoef(arr[:-1], arr[1:])[0, 1])


# ── Feature extraction ────────────────────────────────────────────

OT_BOOKS = {"Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy",
            "Joshua", "Judges", "Ruth", "1 Samuel", "2 Samuel",
            "1 Kings", "2 Kings", "1 Chronicles", "2 Chronicles",
            "Ezra", "Nehemiah", "Esther", "Job", "Psalms",
            "Proverbs", "Ecclesiastes", "Song of Solomon",
            "Isaiah", "Jeremiah", "Lamentations", "Ezekiel",
            "Daniel", "Hosea", "Joel", "Amos", "Obadiah",
            "Jonah", "Micah", "Nahum", "Habakkuk", "Zephaniah",
            "Haggai", "Zechariah", "Malachi"}

NT_BOOKS = {"Matthew", "Mark", "Luke", "John", "Acts",
            "Romans", "1 Corinthians", "2 Corinthians", "Galatians",
            "Ephesians", "Philippians", "Colossians",
            "1 Thessalonians", "2 Thessalonians",
            "1 Timothy", "2 Timothy", "Titus", "Philemon",
            "Hebrews", "James", "1 Peter", "2 Peter",
            "1 John", "2 John", "3 John", "Jude", "Revelation"}


def extract_book_features(corpus):
    """Extract per-book features for classification."""
    log.info("Extrayendo features por libro...")

    # Count words per verse per book
    book_verses = defaultdict(lambda: defaultdict(int))
    book_pos = defaultdict(lambda: defaultdict(list))

    for w in corpus:
        book = w.get("book", "")
        key = (book, w.get("chapter", 0), w.get("verse", 0))
        book_verses[book][key] += 1
        pos = w.get("pos", w.get("sp", ""))
        if pos:
            book_pos[book][key].append(pos)

    features = {}
    for book in sorted(set(list(OT_BOOKS) + list(NT_BOOKS))):
        if book not in book_verses:
            continue
        verses = book_verses[book]
        lens = np.array([verses[k] for k in sorted(verses.keys())], dtype=float)
        n = len(lens)
        if n < 20:
            log.info(f"  {book}: solo {n} versos, skipping")
            continue

        h = hurst_exponent_rs(lens)
        ac1 = autocorr_lag1(lens)
        dfa = dfa_exponent(lens) if n >= 50 else float("nan")
        mean_len = float(lens.mean())
        std_len = float(lens.std())
        cv_len = std_len / mean_len if mean_len > 0 else 0.0
        skewness = float(sp_stats.skew(lens))

        # POS entropy
        all_pos_for_book = []
        for key in sorted(book_pos[book].keys()):
            all_pos_for_book.extend(book_pos[book][key])
        pos_counts = defaultdict(int)
        for p in all_pos_for_book:
            pos_counts[p] += 1
        total_pos = sum(pos_counts.values())
        pos_ent = 0.0
        if total_pos > 0:
            for c in pos_counts.values():
                p = c / total_pos
                if p > 0:
                    pos_ent -= p * np.log2(p)

        testament = "AT" if book in OT_BOOKS else "NT"
        features[book] = {
            "book": book,
            "testament": testament,
            "n_verses": n,
            "H": round(h, 4) if not np.isnan(h) else None,
            "DFA": round(dfa, 4) if not np.isnan(dfa) else None,
            "AC1": round(ac1, 4) if not np.isnan(ac1) else None,
            "mean_verse_len": round(mean_len, 2),
            "std_verse_len": round(std_len, 2),
            "CV": round(cv_len, 4),
            "skewness": round(skewness, 4),
            "pos_entropy": round(pos_ent, 4),
        }
        log.info(f"  {book} ({testament}): H={h:.3f}, AC1={ac1:.3f}, n={n}")

    return features


# ── Classification ─────────────────────────────────────────────────

FEATURE_COLS = ["H", "DFA", "AC1", "mean_verse_len", "std_verse_len", "CV",
                "skewness", "pos_entropy"]


def build_matrix(features):
    """Build X, y from features dict. Replace None/NaN with column mean."""
    books = []
    X_raw = []
    y = []
    for book, feat in sorted(features.items()):
        row = []
        for col in FEATURE_COLS:
            v = feat.get(col)
            row.append(v if v is not None else float("nan"))
        X_raw.append(row)
        books.append(book)
        y.append(0 if feat["testament"] == "AT" else 1)

    X = np.array(X_raw, dtype=float)
    y = np.array(y, dtype=int)

    # Impute NaN with column mean
    for j in range(X.shape[1]):
        col = X[:, j]
        mask = np.isnan(col)
        if mask.any():
            col_mean = np.nanmean(col) if np.nanmean(col) != 0 else 0.0
            X[mask, j] = col_mean

    return books, X, y


def run_classifiers(books, X, y):
    """Run LogisticRegression, RandomForest, SVM with LOO CV."""
    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        from sklearn.model_selection import LeaveOneOut
    except ImportError:
        log.error("sklearn not installed! pip install scikit-learn --break-system-packages")
        return None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    classifiers = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42),
    }

    results = {}
    loo = LeaveOneOut()

    for clf_name, clf in classifiers.items():
        log.info(f"\n  Clasificador: {clf_name}")
        predictions = []
        probas = []
        y_true_list = []

        for train_idx, test_idx in loo.split(X_scaled):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)[0]
            predictions.append(int(pred))
            y_true_list.append(int(y_test[0]))

            if hasattr(clf, "predict_proba"):
                prob = clf.predict_proba(X_test)[0]
                probas.append(prob.tolist())
            else:
                probas.append([1 - pred, pred])

        # Accuracy
        correct = sum(1 for p, t in zip(predictions, y_true_list) if p == t)
        accuracy = correct / len(predictions)
        log.info(f"    LOO accuracy: {accuracy:.3f} ({correct}/{len(predictions)})")

        # Confusion matrix
        tp = sum(1 for p, t in zip(predictions, y_true_list) if p == 1 and t == 1)
        tn = sum(1 for p, t in zip(predictions, y_true_list) if p == 0 and t == 0)
        fp = sum(1 for p, t in zip(predictions, y_true_list) if p == 1 and t == 0)
        fn = sum(1 for p, t in zip(predictions, y_true_list) if p == 0 and t == 1)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # Feature importances (where available)
        clf.fit(X_scaled, y)  # Fit on all data for importance
        importances = None
        if hasattr(clf, "feature_importances_"):
            importances = {FEATURE_COLS[i]: round(float(v), 4)
                           for i, v in enumerate(clf.feature_importances_)}
        elif hasattr(clf, "coef_"):
            importances = {FEATURE_COLS[i]: round(float(abs(v)), 4)
                           for i, v in enumerate(clf.coef_[0])}

        # Per-book scores
        book_scores = {}
        for i, book in enumerate(books):
            book_scores[book] = {
                "true": "AT" if y_true_list[i] == 0 else "NT",
                "predicted": "AT" if predictions[i] == 0 else "NT",
                "correct": bool(predictions[i] == y_true_list[i]),
                "P_AT": round(probas[i][0], 4),
                "P_NT": round(probas[i][1], 4),
            }

        # Misclassified books
        misclassified = [books[i] for i in range(len(books))
                         if predictions[i] != y_true_list[i]]

        results[clf_name] = {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "confusion": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
            "feature_importances": importances,
            "misclassified": misclassified,
            "n_misclassified": len(misclassified),
            "book_scores": book_scores,
        }
        log.info(f"    Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
        log.info(f"    Misclassified ({len(misclassified)}): {', '.join(misclassified)}")

    return results


# ── Main ───────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    log.info("=" * 70)
    log.info("FASE 14 — Script 3: Refined Authenticity Tool")
    log.info("=" * 70)

    # Load corpus
    log.info("\nCargando corpus...")
    with open(BIBLE_JSON, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    log.info(f"  {len(corpus)} palabras")

    # Extract features
    features = extract_book_features(corpus)
    log.info(f"\n  {len(features)} libros con features")

    with open(RESULTS_DIR / "book_features.json", "w") as f:
        json.dump(features, f, indent=2, ensure_ascii=False)

    # Build matrix
    books, X, y = build_matrix(features)
    n_at = sum(1 for v in y if v == 0)
    n_nt = sum(1 for v in y if v == 1)
    log.info(f"\n  Matrix: {X.shape[0]} books × {X.shape[1]} features")
    log.info(f"  AT: {n_at}, NT: {n_nt}")

    # Run classifiers
    clf_results = run_classifiers(books, X, y)

    if clf_results is None:
        # sklearn not available — fallback to simple thresholding
        log.warning("sklearn no disponible — usando clasificación simple por umbral")
        clf_results = {}
        # Simple H threshold classifier
        h_threshold = 0.65
        predictions = []
        for book in books:
            h = features[book].get("H", 0.5)
            if h is None:
                h = 0.5
            predictions.append(0 if h > h_threshold else 1)

        correct = sum(1 for p, t in zip(predictions, y) if p == t)
        accuracy = correct / len(predictions)
        misclassified = [books[i] for i in range(len(books)) if predictions[i] != y[i]]

        clf_results["SimpleThreshold_H"] = {
            "accuracy": round(accuracy, 4),
            "threshold": h_threshold,
            "misclassified": misclassified,
            "n_misclassified": len(misclassified),
            "note": "sklearn not installed; using H > 0.65 → AT",
        }
        log.info(f"  Simple threshold: accuracy={accuracy:.3f}, "
                 f"misclassified={len(misclassified)}")

    with open(RESULTS_DIR / "classifier_results.json", "w") as f:
        json.dump(clf_results, f, indent=2, ensure_ascii=False)

    # Summary / verdict
    log.info("\n=== VEREDICTO ===")
    best_clf = max(clf_results.items(),
                   key=lambda x: x[1].get("accuracy", 0) if isinstance(x[1], dict) else 0)
    best_name, best_res = best_clf

    # Find most discriminating feature
    top_feature = None
    if best_res.get("feature_importances"):
        top_feature = max(best_res["feature_importances"].items(),
                          key=lambda x: x[1])

    # Ambiguous books: AT books predicted as NT or vice versa
    ambiguous = []
    if "book_scores" in best_res:
        for book, score in best_res["book_scores"].items():
            if not score["correct"]:
                ambiguous.append({
                    "book": book,
                    "true": score["true"],
                    "predicted": score["predicted"],
                    "P_AT": score["P_AT"],
                    "P_NT": score["P_NT"],
                })

    # Most uncertain: books with P_AT near 0.5
    uncertain = []
    if "book_scores" in best_res:
        for book, score in best_res["book_scores"].items():
            if abs(score["P_AT"] - 0.5) < 0.15:
                uncertain.append({
                    "book": book,
                    "P_AT": score["P_AT"],
                    "testament": score["true"],
                })
        uncertain.sort(key=lambda x: abs(x["P_AT"] - 0.5))

    verdict = {
        "best_classifier": best_name,
        "best_accuracy": best_res.get("accuracy"),
        "best_f1": best_res.get("f1"),
        "top_feature": top_feature[0] if top_feature else None,
        "top_feature_importance": top_feature[1] if top_feature else None,
        "n_ambiguous": len(ambiguous),
        "ambiguous_books": ambiguous,
        "n_uncertain": len(uncertain),
        "most_uncertain": uncertain[:5],
        "summary": (f"Mejor clasificador: {best_name} "
                    f"(accuracy={best_res.get('accuracy', 0):.1%}). "
                    f"Feature top: {top_feature[0] if top_feature else 'N/A'}. "
                    f"{len(ambiguous)} libros ambiguos."),
    }
    log.info(f"  {verdict['summary']}")
    log.info(f"  Libros ambiguos: {[a['book'] for a in ambiguous]}")
    if uncertain:
        log.info(f"  Más inciertos: {[u['book'] for u in uncertain[:5]]}")

    with open(RESULTS_DIR / "verdict.json", "w") as f:
        json.dump(verdict, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    log.info(f"\n{'=' * 70}")
    log.info(f"Script 3 completado en {elapsed:.1f}s")


if __name__ == "__main__":
    main()
