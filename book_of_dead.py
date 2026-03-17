#!/usr/bin/env python3
"""
Fase 17 — Script 1: book_of_dead.py
Libro de los Muertos egipcio (ORAEC corpus).

Secciones:
1. Descarga y parseo de textos ORAEC (Totenbuch + rituales)
2. Métricas corpus completo: H, DFA, AC1, MPS, CV
3. POS entropy (distribución de POS ORAEC)
4. Clasificador de Fase 14 (LogisticRegression reentrenado)
5. φ y d placement en espacio del modelo unificado
6. Comparación con Salmos (mismo género litúrgico)
"""

import json
import logging
import time
import urllib.request
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from scipy import stats as sp_stats
from scipy import linalg as la
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent
RESULTS_DIR = BASE / "results" / "book_of_dead"
LOG_DIR = BASE / "logs"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "fase17_book_of_dead.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


# ── Reusable functions ────────────────────────────────────────────────

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


def autocorr_lag1(series):
    arr = np.asarray(series, dtype=float)
    if len(arr) < 10:
        return float("nan")
    return float(np.corrcoef(arr[:-1], arr[1:])[0, 1])


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


def compute_bond_dimension(series, max_lag=256, threshold=0.99):
    n = min(max_lag, len(series) // 4)
    if n < 2:
        return 1, np.array([1.0])
    mean_val = np.mean(series)
    centered = series - mean_val
    acf = np.zeros(n)
    for lag in range(n):
        if lag >= len(centered):
            break
        acf[lag] = np.mean(centered[:len(centered)-lag] * centered[lag:])
    T = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            lag = abs(i - j)
            if lag < n:
                T[i, j] = acf[lag]
    _, sigma, _ = la.svd(T, full_matrices=False)
    total = np.sum(sigma ** 2)
    if total == 0:
        return 1, sigma
    cumulative = np.cumsum(sigma ** 2) / total
    chi = int(np.searchsorted(cumulative, threshold) + 1)
    return chi, sigma


# ── Download ORAEC ────────────────────────────────────────────────────

KNOWN_BOD_IDS = [67, 76, 101, 114, 129, 130, 136, 138, 148, 159]
RITUAL_KEYWORDS = ['totenbuch', 'tb ', 'ritual', 'hymne', 'osiris', 'totenb',
                   'book of the dead', 'book of dead']


def download_oraec(oraec_id):
    url = f"https://raw.githubusercontent.com/oraec/corpus_raw_data/main/oraec{oraec_id}.json"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def is_ritual_text(data, oraec_id):
    """Check if title contains ritual/religious keywords."""
    key = f"oraec{oraec_id}"
    if key not in data:
        return False
    title = data[key].get("title", "").lower()
    return any(kw in title for kw in RITUAL_KEYWORDS)


def parse_oraec_sentences(data, oraec_id):
    """Extract sentences with token counts and POS tags."""
    key = f"oraec{oraec_id}"
    if key not in data:
        return [], [], ""
    entry = data[key]
    title = entry.get("title", "")
    sentences = entry.get("sentences", [])

    sent_lengths = []
    all_pos = []

    for sent in sentences:
        tokens = sent.get("token", [])
        if not tokens:
            continue
        sent_lengths.append(len(tokens))
        for tok in tokens:
            pos = tok.get("pos", "undefined")
            if pos:
                all_pos.append(pos)

    return sent_lengths, all_pos, title


# ── Main pipeline ─────────────────────────────────────────────────────

def main():
    t0 = time.time()
    log.info("=" * 70)
    log.info("FASE 17 — Script 1: book_of_dead.py")
    log.info("Libro de los Muertos egipcio (ORAEC)")
    log.info("=" * 70)

    # ── Section 1: Download and parse ─────────────────────────────────
    log.info("\n=== Section 1: Download & Parse ===")

    all_lengths = []
    all_pos_tags = []
    texts_found = []

    # 1a. Download known Book of Dead texts
    log.info("  Downloading known Totenbuch texts...")
    for oid in KNOWN_BOD_IDS:
        data = download_oraec(oid)
        if data:
            lens, pos, title = parse_oraec_sentences(data, oid)
            if lens:
                texts_found.append({
                    "id": f"oraec{oid}",
                    "title": title,
                    "n_sentences": len(lens),
                    "n_tokens": sum(lens),
                    "source": "known_bod",
                })
                all_lengths.extend(lens)
                all_pos_tags.extend(pos)
                log.info(f"    oraec{oid}: {len(lens)} sentences, {sum(lens)} tokens — {title[:60]}")
            else:
                log.info(f"    oraec{oid}: no sentences parsed")
        else:
            log.info(f"    oraec{oid}: download failed")

    # 1b. Scan wider range for ritual/religious texts
    log.info("  Scanning oraec1-993 for ritual texts...")
    last_log = time.time()
    scan_found = 0
    for oid in range(1, 994):
        if oid in KNOWN_BOD_IDS:
            continue
        data = download_oraec(oid)
        if data and is_ritual_text(data, oid):
            lens, pos, title = parse_oraec_sentences(data, oid)
            if lens:
                texts_found.append({
                    "id": f"oraec{oid}",
                    "title": title,
                    "n_sentences": len(lens),
                    "n_tokens": sum(lens),
                    "source": "scan_ritual",
                })
                all_lengths.extend(lens)
                all_pos_tags.extend(pos)
                scan_found += 1
                log.info(f"    oraec{oid}: {len(lens)} sentences — {title[:60]}")

        now = time.time()
        if now - last_log >= 30:
            log.info(f"    [scan] oraec{oid}/993, found {scan_found} ritual texts so far")
            last_log = now

    log.info(f"  Total texts: {len(texts_found)}")
    log.info(f"  Total sentences: {len(all_lengths)}")
    log.info(f"  Total tokens: {sum(all_lengths)}")

    if len(all_lengths) < 20:
        log.error("  INSUFFICIENT DATA — too few sentences for analysis")
        # Save what we have and exit
        with open(RESULTS_DIR / "corpus_metrics.json", "w") as f:
            json.dump({"error": "insufficient_data", "n_sentences": len(all_lengths),
                        "texts_found": texts_found}, f, indent=2, ensure_ascii=False)
        print(f"[book_of_dead] DONE (insufficient data) — {time.time()-t0:.1f}s")
        return

    series = np.array(all_lengths, dtype=float)

    # ── Section 2: Corpus metrics ─────────────────────────────────────
    log.info("\n=== Section 2: Corpus Metrics ===")

    H = hurst_exponent_rs(series)
    dfa = dfa_exponent(series)
    ac1 = autocorr_lag1(series)
    mean_len = float(np.mean(series))
    std_len = float(np.std(series))
    cv = std_len / mean_len if mean_len > 0 else 0
    skew = float(sp_stats.skew(series))

    # MPS bond dimension with permutation test
    chi, sigma = compute_bond_dimension(series, max_lag=min(256, len(series) // 4),
                                         threshold=0.99)
    rng = np.random.default_rng(42)
    n_perm = 10000
    chi_perms = []
    last_log = time.time()
    for i in range(n_perm):
        shuffled = rng.permutation(series)
        chi_s, _ = compute_bond_dimension(shuffled, max_lag=min(64, len(series) // 4),
                                           threshold=0.99)
        chi_perms.append(chi_s)
        now = time.time()
        if now - last_log >= 30:
            log.info(f"    MPS permutation {i+1}/{n_perm}")
            last_log = now

    chi_real, _ = compute_bond_dimension(series, max_lag=min(64, len(series) // 4),
                                          threshold=0.99)
    p_mps = float(np.mean(np.array(chi_perms) <= chi_real))

    corpus_metrics = {
        "corpus": "Book of the Dead (ORAEC)",
        "language": "Egyptian (Middle/Late)",
        "n_texts": len(texts_found),
        "n_sentences": len(series),
        "n_tokens": int(np.sum(series)),
        "H": round(H, 4) if not np.isnan(H) else None,
        "DFA": round(dfa, 4) if not np.isnan(dfa) else None,
        "AC1": round(ac1, 4) if not np.isnan(ac1) else None,
        "mean_verse_len": round(mean_len, 2),
        "std_verse_len": round(std_len, 2),
        "CV": round(cv, 4),
        "skewness": round(skew, 4),
        "bond_dimension": chi,
        "bond_dimension_64": chi_real,
        "mps_perm_p": round(p_mps, 4),
        "mps_perm_mean": round(float(np.mean(chi_perms)), 2),
        "texts": texts_found,
    }

    log.info(f"  H={H:.4f}, DFA={dfa:.4f}, AC1={ac1:.4f}")
    log.info(f"  mean={mean_len:.2f}, std={std_len:.2f}, CV={cv:.4f}, skew={skew:.4f}")
    log.info(f"  MPS: chi={chi_real}, perm_mean={np.mean(chi_perms):.1f}, p={p_mps:.4f}")

    with open(RESULTS_DIR / "corpus_metrics.json", "w") as f:
        json.dump(corpus_metrics, f, indent=2, ensure_ascii=False)

    # ── Section 3: POS entropy ────────────────────────────────────────
    log.info("\n=== Section 3: POS Entropy ===")

    pos_counts = Counter(all_pos_tags)
    total_tags = sum(pos_counts.values())
    if total_tags > 0:
        probs = np.array(list(pos_counts.values()), dtype=float) / total_tags
        pos_ent = float(-np.sum(probs * np.log2(probs + 1e-15)))
    else:
        pos_ent = float("nan")

    pos_dist = {
        "pos_entropy": round(pos_ent, 4) if not np.isnan(pos_ent) else None,
        "n_tokens_tagged": total_tags,
        "n_pos_categories": len(pos_counts),
        "distribution": {k: v for k, v in pos_counts.most_common()},
        "note": (
            "POS tags from ORAEC morphological annotation (Egyptian). "
            "Categories: substantive, verb, adjective, adverb, preposition, "
            "pronoun, particle, numeral, undefined. "
            "NOT directly comparable with OSHB Hebrew POS — different tagsets."
        ),
    }

    log.info(f"  pos_entropy={pos_ent:.4f}, {len(pos_counts)} categories, {total_tags} tokens")
    for tag, count in pos_counts.most_common(5):
        log.info(f"    {tag}: {count} ({100*count/total_tags:.1f}%)")

    with open(RESULTS_DIR / "pos_distribution.json", "w") as f:
        json.dump(pos_dist, f, indent=2, ensure_ascii=False)

    # ── Section 4: Classifier ─────────────────────────────────────────
    log.info("\n=== Section 4: Classifier (LogReg from Fase 14) ===")

    features_file = BASE / "results" / "refined_classifier" / "book_features.json"
    with open(features_file, "r") as f:
        book_features = json.load(f)

    # Build training data
    feature_names = ["H", "DFA", "AC1", "mean_verse_len", "std_verse_len",
                     "CV", "skewness", "pos_entropy"]
    X_train = []
    y_train = []
    for book, feats in book_features.items():
        row = []
        valid = True
        for fn in feature_names:
            val = feats.get(fn)
            if val is None:
                valid = False
                break
            row.append(val)
        if valid and feats.get("testament") in ("AT", "NT"):
            X_train.append(row)
            y_train.append(0 if feats["testament"] == "AT" else 1)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_scaled, y_train)

    train_acc = float(np.mean(clf.predict(X_scaled) == y_train))
    log.info(f"  Classifier retrained: accuracy={train_acc:.4f} on {len(y_train)} books")

    # Apply to Book of Dead
    bod_features = [
        corpus_metrics.get("H") or 0.5,
        corpus_metrics.get("DFA") or 0.5,
        corpus_metrics.get("AC1") or 0.0,
        corpus_metrics.get("mean_verse_len") or 10.0,
        corpus_metrics.get("std_verse_len") or 5.0,
        corpus_metrics.get("CV") or 0.4,
        corpus_metrics.get("skewness") or 0.5,
        pos_dist.get("pos_entropy") or 2.5,
    ]

    bod_scaled = scaler.transform([bod_features])
    pred = clf.predict(bod_scaled)[0]
    proba = clf.predict_proba(bod_scaled)[0]

    classifier_result = {
        "predicted_class": "AT" if pred == 0 else "NT",
        "P_AT": round(float(proba[0]), 4),
        "P_NT": round(float(proba[1]), 4),
        "features_used": dict(zip(feature_names, [round(f, 4) for f in bod_features])),
        "classifier": "LogisticRegression (retrained on 63 biblical books)",
        "train_accuracy": round(train_acc, 4),
        "interpretation": (
            f"Book of the Dead is classified as "
            f"{'AT-like' if pred == 0 else 'NT-like'} "
            f"(P_AT={proba[0]:.3f}, P_NT={proba[1]:.3f})."
        ),
        "caveat": (
            "Different language (Egyptian vs Hebrew/Greek), different POS tagset. "
            "Classification is based on structural metrics (H, DFA, AC1, CV) "
            "which are language-independent, plus pos_entropy which is tagset-dependent."
        ),
    }

    log.info(f"  Prediction: {'AT' if pred == 0 else 'NT'}-like "
             f"(P_AT={proba[0]:.4f}, P_NT={proba[1]:.4f})")

    with open(RESULTS_DIR / "classifier_result.json", "w") as f:
        json.dump(classifier_result, f, indent=2, ensure_ascii=False)

    # ── Section 5: φ, d placement ─────────────────────────────────────
    log.info("\n=== Section 5: φ, d Placement ===")

    fitted_file = BASE / "results" / "unified_model" / "fitted_params.json"
    with open(fitted_file, "r") as f:
        fitted_params = json.load(f)

    # Extract reference corpora
    ref_points = {}
    for corpus, data in fitted_params.items():
        p = data.get("params", {})
        t = data.get("target", {})
        ref_points[corpus] = {
            "phi": p.get("phi"),
            "d": p.get("d"),
            "H": t.get("H"),
            "AC1": t.get("AC1"),
        }

    # Approximate φ and d for Book of Dead from AC1 and H
    bod_ac1 = corpus_metrics.get("AC1") or 0.0
    bod_h = corpus_metrics.get("H") or 0.5
    bod_d_approx = bod_h - 0.5

    # Find nearest corpus in (AC1, d_approx) space
    distances = {}
    for corpus, pt in ref_points.items():
        ref_ac1 = pt.get("AC1") or 0
        ref_d = pt.get("d") or 0
        dist = np.sqrt((bod_ac1 - ref_ac1)**2 + (bod_d_approx - ref_d)**2)
        distances[corpus] = round(float(dist), 4)

    nearest = min(distances, key=distances.get)

    phi_d_result = {
        "AC1": round(bod_ac1, 4),
        "d_approx": round(bod_d_approx, 4),
        "H": round(bod_h, 4),
        "reference_corpora": ref_points,
        "distances": distances,
        "nearest_corpus": nearest,
        "nearest_distance": distances[nearest],
        "interpretation": (
            f"In (AC1, d_approx) space, Book of the Dead is nearest to {nearest} "
            f"(distance={distances[nearest]:.4f})."
        ),
    }

    log.info(f"  AC1={bod_ac1:.4f}, d_approx={bod_d_approx:.4f}")
    log.info(f"  Nearest corpus: {nearest} (dist={distances[nearest]:.4f})")
    for c, d in sorted(distances.items(), key=lambda x: x[1]):
        log.info(f"    {c}: {d}")

    with open(RESULTS_DIR / "phi_d_placement.json", "w") as f:
        json.dump(phi_d_result, f, indent=2, ensure_ascii=False)

    # ── Section 6: Comparison with Psalms ─────────────────────────────
    log.info("\n=== Section 6: Comparison with Psalms ===")

    psalms_feats = book_features.get("Psalms", {})

    # Build per-text metrics for statistical comparison
    per_text_ac1 = []
    per_text_h = []
    for t in texts_found:
        # Recompute per-text (we need the raw data — approximate from stored info)
        pass  # We'll use the overall metrics for comparison

    # Direct comparison
    comparison = {
        "book_of_dead": {
            "H": corpus_metrics.get("H"),
            "AC1": corpus_metrics.get("AC1"),
            "DFA": corpus_metrics.get("DFA"),
            "mean_verse_len": corpus_metrics.get("mean_verse_len"),
            "CV": corpus_metrics.get("CV"),
            "pos_entropy": pos_dist.get("pos_entropy"),
        },
        "psalms": {
            "H": psalms_feats.get("H"),
            "AC1": psalms_feats.get("AC1"),
            "DFA": psalms_feats.get("DFA"),
            "mean_verse_len": psalms_feats.get("mean_verse_len"),
            "CV": psalms_feats.get("CV"),
            "pos_entropy": psalms_feats.get("pos_entropy"),
        },
    }

    # Differences
    diffs = {}
    for metric in ["H", "AC1", "DFA", "mean_verse_len", "CV", "pos_entropy"]:
        v_bod = comparison["book_of_dead"].get(metric)
        v_psa = comparison["psalms"].get(metric)
        if v_bod is not None and v_psa is not None:
            diffs[metric] = {
                "bod": round(v_bod, 4),
                "psalms": round(v_psa, 4),
                "diff": round(v_bod - v_psa, 4),
                "abs_diff": round(abs(v_bod - v_psa), 4),
            }

    # Are they similar? (within 1 SD of AT book distribution)
    at_books_feats = {b: f for b, f in book_features.items() if f.get("testament") == "AT"}
    similarity = {}
    for metric in ["H", "AC1", "CV"]:
        at_vals = [f[metric] for f in at_books_feats.values() if f.get(metric) is not None]
        if at_vals:
            at_std = np.std(at_vals)
            bod_val = comparison["book_of_dead"].get(metric)
            psa_val = comparison["psalms"].get(metric)
            if bod_val is not None and psa_val is not None and at_std > 0:
                similarity[metric] = {
                    "diff_in_sd": round(abs(bod_val - psa_val) / at_std, 4),
                    "similar": bool(abs(bod_val - psa_val) / at_std < 1.0),
                }

    comparison["differences"] = diffs
    comparison["similarity_within_1sd"] = similarity
    comparison["genre_match"] = "Both are liturgical/ritual texts"
    comparison["interpretation"] = (
        f"Book of the Dead vs Psalms: "
        f"{'structurally similar' if all(s.get('similar', False) for s in similarity.values()) else 'structurally different'} "
        f"in key metrics (H, AC1, CV)."
    )

    log.info(f"  BoD H={comparison['book_of_dead']['H']}, Psalms H={comparison['psalms']['H']}")
    log.info(f"  BoD AC1={comparison['book_of_dead']['AC1']}, Psalms AC1={comparison['psalms']['AC1']}")
    for m, s in similarity.items():
        log.info(f"    {m}: diff={s['diff_in_sd']:.2f} SD, {'SIMILAR' if s['similar'] else 'DIFFERENT'}")

    with open(RESULTS_DIR / "comparison_with_psalms.json", "w") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    # ── Summary ───────────────────────────────────────────────────────
    elapsed = time.time() - t0
    log.info(f"\n{'=' * 70}")
    log.info(f"book_of_dead.py completado en {elapsed:.1f}s")
    log.info(f"  Classification: {'AT' if pred == 0 else 'NT'}-like (P_AT={proba[0]:.3f})")
    log.info(f"  Nearest corpus: {nearest}")
    print(f"[book_of_dead] DONE — {elapsed:.1f}s")


if __name__ == "__main__":
    main()
