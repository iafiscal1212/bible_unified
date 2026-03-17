#!/usr/bin/env python3
"""
Fase 20 — Script 1: dfa_reclassification.py

DFA-only classification of AT/NT books:
1. Sweep threshold [0.30, 0.90] → optimal accuracy
2. Bootstrap CI for threshold
3. Classify external corpora as AT/NT/AMBIGUOUS
4. Compare with Fase 14 (original) and Fase 19 (corrected)
5. Mishnah DFA verdict
"""

import json
import logging
import numpy as np
from pathlib import Path
from scipy import stats as sp_stats

BASE = Path(__file__).resolve().parent
RESULTS_DIR = BASE / "results" / "dfa_classification"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Metric functions (copied from gap_corpora_f19.py:43-121)
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


def dfa_exponent(series):
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
            seg = y[i*bs:(i+1)*bs]
            x = np.arange(bs)
            trend = np.polyval(np.polyfit(x, seg, 1), x)
            f2.append(np.mean((seg - trend) ** 2))
        if f2:
            fl.append(np.sqrt(np.mean(f2))); sz.append(bs)
    if len(sz) < 3:
        return np.nan
    slope, *_ = sp_stats.linregress(np.log(sz), np.log(fl))
    return round(slope, 4)


def autocorr_lag1(series):
    s = np.asarray(series, dtype=float)
    if len(s) < 3:
        return np.nan
    m, v = np.mean(s), np.var(s)
    if v == 0:
        return 0.0
    return round(float(np.sum((s[:-1] - m) * (s[1:] - m)) / (len(s) * v)), 4)


def compute_metrics(lengths):
    s = np.array(lengths, dtype=float)
    return {
        "H": hurst_exponent_rs(s),
        "DFA": dfa_exponent(s),
        "AC1": autocorr_lag1(s),
        "CV": round(float(np.std(s) / np.mean(s)), 4) if np.mean(s) > 0 else 0,
        "mean_verse_len": round(float(np.mean(s)), 2),
        "std_verse_len": round(float(np.std(s)), 2),
        "n_segments": len(s),
        "n_words": int(np.sum(s)),
    }


# ═══════════════════════════════════════════════════════════════
# External corpus DFA values
# ═══════════════════════════════════════════════════════════════

def load_external_dfa():
    """Load DFA values for all available external corpora."""
    corpora = {}

    # data_matrix.json (transmission_origin)
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
                corpora[name] = {
                    "DFA": float(entry["DFA"]),
                    "source": "data_matrix.json",
                }

    # Individual metric files
    metric_files = {
        "Mishnah": "mishnah/mishnah_metrics.json",
        "Tosefta": "tosefta/tosefta_metrics.json",
        "Book_of_Dead": "book_of_dead/corpus_metrics.json",
        "Pali_Canon": "pali_canon/combined_metrics.json",
        "Didache": "didache/didache_metrics.json",
        "1_Clemente": "gap_corpora/1_clemente_metrics.json",
    }
    for name, rel_path in metric_files.items():
        fpath = BASE / "results" / rel_path
        if fpath.exists():
            with open(fpath) as f:
                data = json.load(f)
            dfa_val = data.get("DFA") or data.get("dfa_exponent")
            if dfa_val is not None:
                corpora[name] = {
                    "DFA": float(dfa_val),
                    "source": rel_path,
                }

    return corpora


# ═══════════════════════════════════════════════════════════════
# Main analysis
# ═══════════════════════════════════════════════════════════════

def main():
    log.info("=" * 70)
    log.info("FASE 20 — Script 1: DFA Reclassification")
    log.info("=" * 70)

    # 1. Load book features
    bf_file = BASE / "results" / "refined_classifier" / "book_features.json"
    with open(bf_file) as f:
        all_books = json.load(f)

    # Filter books with non-null DFA
    books = []
    for name, info in all_books.items():
        if info.get("DFA") is not None:
            books.append({
                "book": name,
                "testament": info["testament"],
                "DFA": float(info["DFA"]),
            })

    n_at = sum(1 for b in books if b["testament"] == "AT")
    n_nt = sum(1 for b in books if b["testament"] == "NT")
    log.info(f"  Books with DFA: {len(books)} ({n_at} AT, {n_nt} NT)")

    dfa_at = np.array([b["DFA"] for b in books if b["testament"] == "AT"])
    dfa_nt = np.array([b["DFA"] for b in books if b["testament"] == "NT"])
    labels = np.array([1 if b["testament"] == "AT" else 0 for b in books])
    dfa_all = np.array([b["DFA"] for b in books])

    # Mann-Whitney AT vs NT for DFA
    mw_stat, mw_p = sp_stats.mannwhitneyu(dfa_at, dfa_nt, alternative="two-sided")
    cohen_d = float((np.mean(dfa_at) - np.mean(dfa_nt)) / np.sqrt(
        (np.var(dfa_at, ddof=1) + np.var(dfa_nt, ddof=1)) / 2))
    log.info(f"  DFA AT mean={np.mean(dfa_at):.4f}, NT mean={np.mean(dfa_nt):.4f}")
    log.info(f"  Mann-Whitney U={mw_stat:.1f}, p={mw_p:.6f}, Cohen d={cohen_d:.2f}")

    # 2. Sweep threshold [0.30, 0.90]
    thresholds = np.arange(0.30, 0.91, 0.01)
    sweep_results = []
    best_f1 = -1
    best_thresh = None

    for thresh in thresholds:
        preds = (dfa_all >= thresh).astype(int)  # 1=AT, 0=NT
        tp = int(np.sum((preds == 1) & (labels == 1)))
        fp = int(np.sum((preds == 1) & (labels == 0)))
        tn = int(np.sum((preds == 0) & (labels == 0)))
        fn = int(np.sum((preds == 0) & (labels == 1)))
        acc = (tp + tn) / len(labels)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        sweep_results.append({
            "threshold": round(float(thresh), 2),
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "F1": round(float(f1), 4),
            "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        })

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(thresh)

    log.info(f"  Best threshold: {best_thresh:.2f} (F1={best_f1:.4f})")

    # 3. Bootstrap CI for threshold
    rng = np.random.RandomState(42)
    n_boot = 1000
    boot_thresholds = []

    for _ in range(n_boot):
        idx = rng.choice(len(books), size=len(books), replace=True)
        b_dfa = dfa_all[idx]
        b_labels = labels[idx]
        best_bt = best_thresh
        best_bf = -1
        for thresh in thresholds:
            preds = (b_dfa >= thresh).astype(int)
            tp = np.sum((preds == 1) & (b_labels == 1))
            fp = np.sum((preds == 1) & (b_labels == 0))
            fn = np.sum((preds == 0) & (b_labels == 1))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            if f1 > best_bf:
                best_bf = f1
                best_bt = float(thresh)
        boot_thresholds.append(best_bt)

    boot_thresholds = np.array(boot_thresholds)
    thresh_ci_lo = float(np.percentile(boot_thresholds, 2.5))
    thresh_ci_hi = float(np.percentile(boot_thresholds, 97.5))
    thresh_std = float(np.std(boot_thresholds))
    thresh_mean = float(np.mean(boot_thresholds))

    log.info(f"  Bootstrap threshold: mean={thresh_mean:.3f}, "
             f"CI=[{thresh_ci_lo:.2f}, {thresh_ci_hi:.2f}], SD={thresh_std:.3f}")

    # 4. Ambiguity zone
    amb_lo = best_thresh - thresh_std
    amb_hi = best_thresh + thresh_std
    log.info(f"  Ambiguity zone: [{amb_lo:.3f}, {amb_hi:.3f}]")

    threshold_analysis = {
        "n_books": len(books),
        "n_AT": int(n_at),
        "n_NT": int(n_nt),
        "DFA_AT_mean": round(float(np.mean(dfa_at)), 4),
        "DFA_AT_std": round(float(np.std(dfa_at, ddof=1)), 4),
        "DFA_NT_mean": round(float(np.mean(dfa_nt)), 4),
        "DFA_NT_std": round(float(np.std(dfa_nt, ddof=1)), 4),
        "mann_whitney_U": round(float(mw_stat), 1),
        "mann_whitney_p": round(float(mw_p), 6),
        "cohen_d": round(cohen_d, 2),
        "best_threshold": round(best_thresh, 2),
        "best_F1": round(best_f1, 4),
        "bootstrap_n": n_boot,
        "bootstrap_threshold_mean": round(thresh_mean, 3),
        "bootstrap_threshold_CI95": [round(thresh_ci_lo, 2), round(thresh_ci_hi, 2)],
        "bootstrap_threshold_SD": round(thresh_std, 3),
        "ambiguity_zone": [round(amb_lo, 3), round(amb_hi, 3)],
        "sweep": sweep_results,
    }

    with open(RESULTS_DIR / "threshold_analysis.json", "w") as f:
        json.dump(threshold_analysis, f, indent=2, ensure_ascii=False)
    log.info("  Saved threshold_analysis.json")

    # 5. Load external corpus DFA values
    ext_corpora = load_external_dfa()
    log.info(f"  Loaded {len(ext_corpora)} external corpora")

    # 6. Classify each corpus
    corpus_dfa_values = {}
    for name, info in ext_corpora.items():
        dfa_val = info["DFA"]
        if dfa_val >= amb_hi:
            classification = "AT"
        elif dfa_val <= amb_lo:
            classification = "NT"
        else:
            classification = "AMBIGUOUS"

        corpus_dfa_values[name] = {
            "DFA": round(dfa_val, 4),
            "classification": classification,
            "source": info["source"],
            "above_threshold": bool(dfa_val >= best_thresh),
            "in_ambiguity_zone": bool(amb_lo <= dfa_val <= amb_hi),
        }
        log.info(f"  {name}: DFA={dfa_val:.4f} → {classification}")

    with open(RESULTS_DIR / "corpus_dfa_values.json", "w") as f:
        json.dump(corpus_dfa_values, f, indent=2, ensure_ascii=False)
    log.info("  Saved corpus_dfa_values.json")

    # 7. Compare with Fase 14 (original) and Fase 19 (corrected)
    comparison = {"corpora": {}}

    # Load Fase 14 results if available
    f14_file = BASE / "results" / "transmission_origin" / "data_matrix.json"
    f14_data = {}
    if f14_file.exists():
        with open(f14_file) as f:
            f14 = json.load(f)
        for entry in f14:
            f14_data[entry["corpus"]] = entry.get("predicted")

    # Load Fase 19 gap_corpora results
    f19_gap = BASE / "results" / "gap_corpora" / "gap_summary.json"
    f19_data = {}
    if f19_gap.exists():
        with open(f19_gap) as f:
            f19 = json.load(f)
        for name, info in f19.get("corpora", {}).items():
            if isinstance(info, dict):
                f19_data[name] = info.get("predicted")

    for name, info in corpus_dfa_values.items():
        comparison["corpora"][name] = {
            "DFA": info["DFA"],
            "fase20_DFA_class": info["classification"],
            "fase14_class": f14_data.get(name) or f14_data.get(
                name.replace("_", " ")) or None,
            "fase19_class": f19_data.get(name),
        }

    with open(RESULTS_DIR / "reclassification_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    log.info("  Saved reclassification_comparison.json")

    # 8. Mishnah DFA verdict
    mishnah_dfa = ext_corpora.get("Mishnah", {}).get("DFA")
    mishnah_verdict = {
        "corpus": "Mishnah",
        "DFA": round(float(mishnah_dfa), 4) if mishnah_dfa else None,
        "threshold": round(best_thresh, 2),
        "ambiguity_zone": [round(amb_lo, 3), round(amb_hi, 3)],
    }
    if mishnah_dfa is not None:
        if mishnah_dfa >= amb_hi:
            mishnah_verdict["classification"] = "AT"
        elif mishnah_dfa <= amb_lo:
            mishnah_verdict["classification"] = "NT"
        else:
            mishnah_verdict["classification"] = "AMBIGUOUS"

        # Where does Mishnah sit relative to AT and NT distributions?
        z_at = float((mishnah_dfa - np.mean(dfa_at)) / np.std(dfa_at, ddof=1))
        z_nt = float((mishnah_dfa - np.mean(dfa_nt)) / np.std(dfa_nt, ddof=1))
        mishnah_verdict["z_score_vs_AT"] = round(z_at, 3)
        mishnah_verdict["z_score_vs_NT"] = round(z_nt, 3)
        mishnah_verdict["closer_to"] = "AT" if abs(z_at) < abs(z_nt) else "NT"
        mishnah_verdict["reasoning"] = (
            f"Mishnah DFA={mishnah_dfa:.4f} classified as "
            f"{mishnah_verdict['classification']} by DFA threshold. "
            f"z-score vs AT={z_at:.3f}, vs NT={z_nt:.3f}. "
            f"Closer to {'AT' if abs(z_at) < abs(z_nt) else 'NT'} distribution."
        )
        log.info(f"  Mishnah verdict: {mishnah_verdict['classification']} "
                 f"(closer to {mishnah_verdict['closer_to']})")
    else:
        mishnah_verdict["classification"] = "UNAVAILABLE"
        mishnah_verdict["reasoning"] = "Mishnah DFA value not found"

    with open(RESULTS_DIR / "mishnah_dfa_verdict.json", "w") as f:
        json.dump(mishnah_verdict, f, indent=2, ensure_ascii=False)
    log.info("  Saved mishnah_dfa_verdict.json")

    # Final summary
    log.info(f"\n{'=' * 70}")
    log.info("Script 1 completado.")
    log.info(f"  Best DFA threshold: {best_thresh:.2f}")
    log.info(f"  Ambiguity zone: [{amb_lo:.3f}, {amb_hi:.3f}]")
    log.info(f"  Mishnah: {mishnah_verdict['classification']}")
    log.info(f"  External corpora classified: {len(corpus_dfa_values)}")


if __name__ == "__main__":
    main()
