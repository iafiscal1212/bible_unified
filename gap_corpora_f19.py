#!/usr/bin/env python3
"""
Fase 19 — Script 2: gap_corpora_f19.py

Download and analyze gap corpora with CORRECTED classifier:
A. 1 Clemente (~96 CE, delay ~65 años)
B. Policarpo (~110 CE, delay ~80 años)
C. Tosefta (reuse existing metrics, delay ~170 años)

Yasna INVALIDADO — NO incluir.
"""

import json
import logging
import pickle
import re
import time
import urllib.request
import numpy as np
from pathlib import Path
from xml.etree import ElementTree as ET
from scipy import stats as sp_stats
from scipy.linalg import toeplitz

BASE = Path(__file__).resolve().parent
RESULTS_DIR = BASE / "results" / "gap_corpora"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)

FEAT_NAMES = ["H", "DFA", "AC1", "CV"]


# ═══════════════════════════════════════════════════════════════
# Metric functions
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


def autocorr_lag1(series):
    s = np.asarray(series, dtype=float)
    if len(s) < 3:
        return np.nan
    m, v = np.mean(s), np.var(s)
    if v == 0:
        return 0.0
    return round(float(np.sum((s[:-1]-m)*(s[1:]-m)) / (len(s)*v)), 4)


def dfa_exponent(series):
    s = np.asarray(series, dtype=float)
    n = len(s)
    if n < 20:
        return np.nan
    y = np.cumsum(s - np.mean(s))
    mn, mx = 4, n//4
    if mx < mn+2:
        return np.nan
    bsz = np.unique(np.logspace(np.log10(mn), np.log10(mx), 20).astype(int))
    fl, sz = [], []
    for bs in bsz:
        nb = n//bs
        if nb < 1:
            continue
        f2 = []
        for i in range(nb):
            seg = y[i*bs:(i+1)*bs]
            x = np.arange(bs)
            trend = np.polyval(np.polyfit(x, seg, 1), x)
            f2.append(np.mean((seg-trend)**2))
        if f2:
            fl.append(np.sqrt(np.mean(f2))); sz.append(bs)
    if len(sz) < 3:
        return np.nan
    slope, *_ = sp_stats.linregress(np.log(sz), np.log(fl))
    return round(slope, 4)


def compute_metrics(lengths):
    s = np.array(lengths, dtype=float)
    return {
        "H": hurst_exponent_rs(s),
        "DFA": dfa_exponent(s),
        "AC1": autocorr_lag1(s),
        "CV": round(float(np.std(s)/np.mean(s)), 4) if np.mean(s) > 0 else 0,
        "mean_verse_len": round(float(np.mean(s)), 2),
        "std_verse_len": round(float(np.std(s)), 2),
        "n_segments": len(s),
        "n_words": int(np.sum(s)),
    }


def classify_corrected(features):
    """Classify using the corrected model from Script 1."""
    model_file = BASE / "results" / "classifier_corrected" / "model.pkl"
    if not model_file.exists():
        log.warning("  Corrected classifier not found!")
        return None
    with open(model_file, "rb") as f:
        model = pickle.load(f)
    clf = model["classifier"]
    scaler = model["scaler"]
    feat_vals = [features.get(fn) for fn in FEAT_NAMES]
    if any(v is None for v in feat_vals):
        return None
    X = scaler.transform([feat_vals])
    pred = clf.predict(X)[0]
    proba = clf.predict_proba(X)[0]
    return {
        "predicted_class": "AT" if pred == 0 else "NT",
        "P_AT": round(float(proba[0]), 4),
        "P_NT": round(float(proba[1]), 4),
        "classifier": "corrected (H, DFA, AC1, CV only)",
    }


# ═══════════════════════════════════════════════════════════════
# TEI XML parser for First1KGreek
# ═══════════════════════════════════════════════════════════════

def parse_first1k_xml(xml_text):
    """Parse First1KGreek TEI XML, return list of text segments."""
    # Remove namespace if present
    xml_text = re.sub(r'xmlns="[^"]*"', '', xml_text)
    xml_text = re.sub(r'xmlns:[a-z]+="[^"]*"', '', xml_text)

    root = ET.fromstring(xml_text)

    segments = []

    # Find text body
    body = root.find('.//body')
    if body is None:
        body = root

    # Extract all text content from div/p/ab/seg elements
    for elem in body.iter():
        if elem.tag in ('p', 'ab', 'seg', 'l'):
            text = ''.join(elem.itertext()).strip()
            # Clean: remove brackets, numbers at start
            text = re.sub(r'\[.*?\]', '', text)
            text = re.sub(r'^\d+[\.\s]*', '', text)
            text = text.strip()
            if text:
                words = text.split()
                if len(words) >= 2:
                    segments.append(len(words))

    return segments


def download_and_parse_xml(url, name):
    """Download a First1KGreek XML and parse it."""
    log.info(f"  Downloading {name}...")
    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "Bible-Unified-Research/1.0")
        with urllib.request.urlopen(req, timeout=60) as resp:
            xml_text = resp.read().decode("utf-8")
        log.info(f"  Downloaded {len(xml_text)} bytes")
    except Exception as e:
        log.error(f"  Download failed: {e}")
        return None

    segments = parse_first1k_xml(xml_text)
    log.info(f"  Parsed {len(segments)} segments, {sum(segments)} words")
    return segments


# ═══════════════════════════════════════════════════════════════
# Corpus A: 1 Clemente
# ═══════════════════════════════════════════════════════════════

def analyze_1_clemente():
    log.info("\n" + "=" * 60)
    log.info("Corpus A: 1 Clemente (~96 CE, delay ~65 años)")
    log.info("=" * 60)

    url = ("https://raw.githubusercontent.com/OpenGreekAndLatin/"
           "First1KGreek/master/data/tlg1271/tlg001/"
           "tlg1271.tlg001.1st1K-grc1.xml")

    segments = download_and_parse_xml(url, "1 Clemente")
    if not segments or len(segments) < 20:
        log.error("  Insufficient data for 1 Clemente")
        return None

    metrics = compute_metrics(segments)
    metrics["corpus"] = "1 Clemente"
    metrics["language"] = "Koine Greek"
    metrics["delay_years"] = 65

    clf_result = classify_corrected(metrics)
    if clf_result:
        metrics.update(clf_result)
        log.info(f"  H={metrics['H']}, AC1={metrics['AC1']}, DFA={metrics['DFA']}")
        log.info(f"  Classification: {clf_result['predicted_class']}-like "
                 f"(P_AT={clf_result['P_AT']})")

    with open(RESULTS_DIR / "1_clemente_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    return metrics


# ═══════════════════════════════════════════════════════════════
# Corpus B: Policarpo
# ═══════════════════════════════════════════════════════════════

def analyze_policarpo():
    log.info("\n" + "=" * 60)
    log.info("Corpus B: Policarpo (~110 CE, delay ~80 años)")
    log.info("=" * 60)

    url = ("https://raw.githubusercontent.com/OpenGreekAndLatin/"
           "First1KGreek/master/data/tlg1622/tlg001/"
           "tlg1622.tlg001.1st1K-grc1.xml")

    segments = download_and_parse_xml(url, "Policarpo")

    if not segments or len(segments) < 50:
        n = len(segments) if segments else 0
        log.warning(f"  Policarpo has only {n} segments — EXCLUDED from formal analysis")
        result = {
            "corpus": "Policarpo",
            "status": "excluded",
            "reason": f"n_segments={n} < 50",
            "n_segments": n,
            "n_words": sum(segments) if segments else 0,
        }
        if segments and len(segments) >= 10:
            m = compute_metrics(segments)
            result["metrics_informal"] = m
            result["small_sample_warning"] = True
        with open(RESULTS_DIR / "policarpo_metrics.json", "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        return result

    metrics = compute_metrics(segments)
    metrics["corpus"] = "Policarpo"
    metrics["language"] = "Koine Greek"
    metrics["delay_years"] = 80

    # Bootstrap CI for H if n < 200
    if len(segments) < 200:
        log.info(f"  n={len(segments)} < 200: computing bootstrap CI for H...")
        boot_H = []
        rng = np.random.RandomState(42)
        for _ in range(1000):
            sample = rng.choice(segments, size=len(segments), replace=True)
            boot_H.append(hurst_exponent_rs(sample))
        boot_H = [h for h in boot_H if not np.isnan(h)]
        if boot_H:
            metrics["H_bootstrap_CI_95"] = [
                round(float(np.percentile(boot_H, 2.5)), 4),
                round(float(np.percentile(boot_H, 97.5)), 4),
            ]
            metrics["H_bootstrap_std"] = round(float(np.std(boot_H)), 4)
            log.info(f"  H bootstrap 95% CI: {metrics['H_bootstrap_CI_95']}")

    clf_result = classify_corrected(metrics)
    if clf_result:
        metrics.update(clf_result)
        log.info(f"  H={metrics['H']}, AC1={metrics['AC1']}, DFA={metrics['DFA']}")
        log.info(f"  Classification: {clf_result['predicted_class']}-like "
                 f"(P_AT={clf_result['P_AT']})")

    with open(RESULTS_DIR / "policarpo_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    return metrics


# ═══════════════════════════════════════════════════════════════
# Corpus C: Tosefta (reuse existing)
# ═══════════════════════════════════════════════════════════════

def analyze_tosefta():
    log.info("\n" + "=" * 60)
    log.info("Corpus C: Tosefta (reusing existing metrics, delay ~170 años)")
    log.info("=" * 60)

    mf = BASE / "results" / "tosefta" / "tosefta_metrics.json"
    if not mf.exists():
        log.error("  Tosefta metrics not found — skipping")
        return None

    with open(mf) as f:
        metrics = json.load(f)

    if "error" in metrics:
        log.error(f"  Tosefta had error: {metrics['error']}")
        return None

    metrics["delay_years"] = 170

    # Reclassify with corrected classifier
    clf_result = classify_corrected(metrics)
    if clf_result:
        metrics["predicted_class_corrected"] = clf_result["predicted_class"]
        metrics["P_AT_corrected"] = clf_result["P_AT"]
        log.info(f"  H={metrics['H']}, AC1={metrics['AC1']}, DFA={metrics['DFA']}")
        log.info(f"  Corrected classification: {clf_result['predicted_class']}-like "
                 f"(P_AT={clf_result['P_AT']})")

    with open(RESULTS_DIR / "tosefta_corrected.json", "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    return metrics


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    log.info("=" * 70)
    log.info("FASE 19 — Script 2: Gap Corpora (1 Clemente + Policarpo + Tosefta)")
    log.info("=" * 70)

    results = {}
    results["1_Clemente"] = analyze_1_clemente()
    results["Policarpo"] = analyze_policarpo()
    results["Tosefta"] = analyze_tosefta()

    # Summary
    summary = {"corpora": {}}
    for name, m in results.items():
        if m is None:
            summary["corpora"][name] = {"status": "failed"}
        elif m.get("status") == "excluded":
            summary["corpora"][name] = m
        else:
            summary["corpora"][name] = {
                "H": m.get("H"), "AC1": m.get("AC1"), "DFA": m.get("DFA"),
                "CV": m.get("CV"),
                "predicted": m.get("predicted_class") or m.get("predicted_class_corrected"),
                "P_AT": m.get("P_AT") or m.get("P_AT_corrected"),
                "delay": m.get("delay_years"),
                "n_segments": m.get("n_segments") or m.get("n_passages"),
            }

    with open(RESULTS_DIR / "gap_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    log.info(f"\n{'=' * 70}")
    log.info("Script 2 completado.")
    for name, m in results.items():
        if m and m.get("status") != "excluded":
            pred = m.get("predicted_class") or m.get("predicted_class_corrected", "?")
            log.info(f"  {name}: {pred}-like, H={m.get('H')}")
        elif m and m.get("status") == "excluded":
            log.info(f"  {name}: EXCLUDED ({m.get('reason')})")
        else:
            log.info(f"  {name}: FAILED")


if __name__ == "__main__":
    main()
