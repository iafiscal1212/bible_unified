#!/usr/bin/env python3
"""
Fase 8 — Script 5: h_authenticity_tool.py
¿Puede H usarse como criterio de autenticidad textual?
"""

import json
import logging
import time
import numpy as np
from pathlib import Path
from scipy import stats
from collections import defaultdict

BASE = Path(__file__).resolve().parent
RESULTS_DIR = BASE / "results" / "authenticity"
LOG_DIR = BASE / "logs"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "fase8_h_authenticity_tool.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


def hurst_exponent_rs(series):
    series = np.asarray(series, dtype=float)
    n = len(series)
    if n < 20:
        return float("nan"), 0.0
    min_block = 10
    max_block = n // 2
    sizes, rs_values = [], []
    block = min_block
    while block <= max_block:
        sizes.append(block)
        n_blocks = n // block
        rs_list = []
        for i in range(n_blocks):
            seg = series[i * block:(i + 1) * block]
            mean_seg = seg.mean()
            devs = np.cumsum(seg - mean_seg)
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
    log_n = np.log(sizes)
    log_rs = np.log(rs_values)
    slope, intercept, r, p, se = stats.linregress(log_n, log_rs)
    return float(slope), float(r ** 2)


def main():
    t0 = time.time()
    log.info("=== Script 5: h_authenticity_tool — START ===")

    # ── Load reference data ────────────────────────────────────────────
    log.info("Loading reference corpus data...")

    with open(BASE / "results" / "fase5_comparison.json") as f:
        comparison = json.load(f)

    with open(BASE / "results" / "rigveda" / "rigveda_metrics.json") as f:
        rigveda = json.load(f)

    with open(BASE / "results" / "deep_fractal" / "fractal_by_corpus.json") as f:
        fractal_corpus = json.load(f)

    with open(BASE / "results" / "deep_zipf_semantic" / "zipf_by_at_book.json") as f:
        zipf_by_book = json.load(f)

    # ── Build reference curve ──────────────────────────────────────────
    log.info("Building reference curve...")

    reference_corpora = []
    for c in comparison:
        corpus_name = c["corpus"]
        transmission = "controlled" if c["hurst_H"] > 0.9 and c.get("mps_significant", False) else "free"

        # Override with known transmission types
        known_controlled = ["AT (Hebreo)", "Corán (Árabe)"]
        known_free = ["Homero (Griego)", "Heródoto (Griego)", "NT (Griego)"]
        known_semi = ["Mishnah (Hebreo)"]

        if corpus_name in known_controlled:
            transmission = "controlled"
        elif corpus_name in known_free:
            transmission = "free"
        elif corpus_name in known_semi:
            transmission = "semi-controlled"

        family = "semitic" if c.get("lang") in ("heb", "ara") else "indo-european"
        if c.get("lang") == "grc":
            family = "indo-european"

        entry = {
            "corpus": corpus_name,
            "lang": c.get("lang"),
            "family": family,
            "transmission": transmission,
            "H": c["hurst_H"],
            "alpha": c.get("dfa_alpha"),
            "n_units": c["n_units"],
            "n_words": c["n_words"],
            "mean_unit_length": c.get("mean_unit_length"),
            "mps_significant": bool(c.get("mps_significant", False)),
            "bond_dim_chi": c.get("bond_dim_chi"),
            "composition_date_estimated": None
        }

        # Add estimated composition dates
        dates = {
            "AT (Hebreo)": -600,  # Average across all AT books
            "NT (Griego)": 70,
            "Corán (Árabe)": 632,
            "Homero (Griego)": -750,
            "Heródoto (Griego)": -440,
            "Mishnah (Hebreo)": 200,
        }
        entry["composition_date_estimated"] = dates.get(corpus_name)

        reference_corpora.append(entry)

    # Add Rig Veda
    reference_corpora.append({
        "corpus": "Rig Veda (Sánscrito)",
        "lang": "san",
        "family": "indo-european",
        "transmission": "controlled",
        "H": rigveda["hurst_H"],
        "alpha": rigveda["dfa_alpha"],
        "n_units": rigveda["n_padas"],
        "n_words": rigveda["n_words"],
        "mean_unit_length": rigveda["n_words"] / rigveda["n_padas"],
        "mps_significant": bool(rigveda["mps_significant"]),
        "bond_dim_chi": rigveda["bond_dim_chi"],
        "composition_date_estimated": -1200
    })

    log.info(f"Reference corpora: {len(reference_corpora)}")

    # ── Compute H distribution by transmission type ────────────────────
    log.info("Computing H distributions by transmission type...")

    h_by_transmission = defaultdict(list)
    for rc in reference_corpora:
        h_by_transmission[rc["transmission"]].append(rc["H"])

    transmission_stats = {}
    for trans, h_values in h_by_transmission.items():
        transmission_stats[trans] = {
            "n": len(h_values),
            "mean_H": float(np.mean(h_values)),
            "std_H": float(np.std(h_values, ddof=1)) if len(h_values) > 1 else 0.0,
            "min_H": float(np.min(h_values)),
            "max_H": float(np.max(h_values)),
            "corpora": [rc["corpus"] for rc in reference_corpora if rc["transmission"] == trans]
        }

    log.info(f"Transmission stats: {json.dumps(transmission_stats, indent=2)}")

    # ── Bootstrap confidence intervals ─────────────────────────────────
    log.info("Computing bootstrap CI for H by transmission type (n=10000)...")
    n_boot = 10000
    rng = np.random.RandomState(42)

    bootstrap_ci = {}
    for trans, h_values in h_by_transmission.items():
        h_arr = np.array(h_values)
        if len(h_arr) < 2:
            bootstrap_ci[trans] = {
                "mean": float(h_arr[0]) if len(h_arr) == 1 else None,
                "ci95_lower": None,
                "ci95_upper": None,
                "note": "Too few samples for bootstrap"
            }
            continue

        boot_means = []
        for _ in range(n_boot):
            sample = rng.choice(h_arr, size=len(h_arr), replace=True)
            boot_means.append(sample.mean())

        boot_means = np.array(boot_means)
        bootstrap_ci[trans] = {
            "mean": float(np.mean(boot_means)),
            "ci95_lower": float(np.percentile(boot_means, 2.5)),
            "ci95_upper": float(np.percentile(boot_means, 97.5)),
            "n_bootstrap": n_boot
        }

    # ── Per-book H for AT (using bible_unified data) ───────────────────
    log.info("Loading per-book data for AT anomaly analysis...")

    # We need per-book H values. These aren't directly in our results.
    # We have zipf_by_book (Zipf s by book) but not H by book.
    # Compute H by book using verse lengths from fractal_corpus data.
    # Actually, fractal_by_corpus only has global/OT/NT.
    # We need to load the raw data per book.

    # Try loading bible_unified.json or structure_analysis for per-book verse lengths
    structure_file = BASE / "results" / "structure" / "structure_analysis.json"
    per_book_data = None
    if structure_file.exists():
        with open(structure_file) as f:
            structure = json.load(f)
        # Check if it has per-book verse counts
        per_book_data = structure

    # Alternative: use the verse length distribution data
    # For now, compute anomaly based on Zipf s (not H) since we have that per-book
    # and note this limitation

    log.info("Computing per-book anomaly scores using Zipf s (H not available per-book)...")

    # Global reference: AT Zipf s from comparison
    at_zipf = None
    for c in comparison:
        if c["corpus"] == "AT (Hebreo)" and c.get("zipf_s_lemma") is not None:
            at_zipf = c["zipf_s_lemma"]
            break

    # If AT Zipf s not in comparison, compute from per-book data
    all_zipf_values = [b["zipf_s_lemma"] for b in zipf_by_book]
    mean_zipf = np.mean(all_zipf_values)
    std_zipf = np.std(all_zipf_values, ddof=1)

    book_anomalies = []
    for book_data in zipf_by_book:
        z_score = (book_data["zipf_s_lemma"] - mean_zipf) / std_zipf if std_zipf > 0 else 0
        book_anomalies.append({
            "book": book_data["book"],
            "book_num": book_data["book_num"],
            "n_tokens": book_data["n_tokens"],
            "zipf_s_lemma": book_data["zipf_s_lemma"],
            "z_score": round(float(z_score), 4),
            "anomalous": bool(abs(z_score) > 2),
            "direction": "low" if z_score < -2 else ("high" if z_score > 2 else "normal")
        })

    # Sort by z_score
    book_anomalies.sort(key=lambda x: x["z_score"])

    n_anomalous = sum(1 for b in book_anomalies if b["anomalous"])
    log.info(f"Anomalous books (|z| > 2): {n_anomalous}")
    for b in book_anomalies:
        if b["anomalous"]:
            log.info(f"  {b['book']}: z={b['z_score']:.2f}, s={b['zipf_s_lemma']:.4f}")

    # ── Prediction model: transmission type → expected H range ─────────
    log.info("Building prediction model...")

    # Simple model: for a given transmission type, what is the expected H?
    controlled = [rc for rc in reference_corpora if rc["transmission"] == "controlled"]
    free = [rc for rc in reference_corpora if rc["transmission"] == "free"]

    model = {
        "controlled": {
            "expected_H_range": [
                float(min(rc["H"] for rc in controlled)),
                float(max(rc["H"] for rc in controlled))
            ],
            "mean_H": float(np.mean([rc["H"] for rc in controlled])),
            "std_H": float(np.std([rc["H"] for rc in controlled], ddof=1)) if len(controlled) > 1 else 0,
            "n_corpora": len(controlled)
        },
        "free": {
            "expected_H_range": [
                float(min(rc["H"] for rc in free)),
                float(max(rc["H"] for rc in free))
            ],
            "mean_H": float(np.mean([rc["H"] for rc in free])),
            "std_H": float(np.std([rc["H"] for rc in free], ddof=1)) if len(free) > 1 else 0,
            "n_corpora": len(free)
        },
        "decision_boundary_H": None,
        "note": "Simple threshold model"
    }

    # Decision boundary: midpoint between means
    if controlled and free:
        model["decision_boundary_H"] = float(
            (model["controlled"]["mean_H"] + model["free"]["mean_H"]) / 2
        )

    # ── Leave-one-out cross-validation ─────────────────────────────────
    log.info("Leave-one-out cross-validation...")

    loo_results = []
    for i, rc in enumerate(reference_corpora):
        # Train on all except i
        train = [r for j, r in enumerate(reference_corpora) if j != i]
        train_controlled = [r["H"] for r in train if r["transmission"] == "controlled"]
        train_free = [r["H"] for r in train if r["transmission"] == "free"]

        if not train_controlled or not train_free:
            continue

        boundary = (np.mean(train_controlled) + np.mean(train_free)) / 2
        predicted = "controlled" if rc["H"] > boundary else "free"
        actual = rc["transmission"]

        # For semi-controlled, check if closer to controlled or free
        if actual == "semi-controlled":
            correct = None  # Cannot evaluate
        else:
            correct = predicted == actual

        loo_results.append({
            "corpus": rc["corpus"],
            "H": rc["H"],
            "actual_transmission": actual,
            "predicted_transmission": predicted,
            "boundary_used": round(float(boundary), 4),
            "correct": correct
        })

    n_testable = sum(1 for r in loo_results if r["correct"] is not None)
    n_correct = sum(1 for r in loo_results if r["correct"] is True)
    accuracy = n_correct / n_testable if n_testable > 0 else 0

    cross_validation = {
        "method": "Leave-one-out cross-validation",
        "feature": "Hurst H",
        "decision_rule": "H > boundary → controlled, H ≤ boundary → free",
        "n_testable": n_testable,
        "n_correct": n_correct,
        "accuracy": round(accuracy, 4),
        "results": loo_results
    }

    log.info(f"LOO accuracy: {accuracy:.1%} ({n_correct}/{n_testable})")

    # ── Method limitations ─────────────────────────────────────────────
    method_limitations = {
        "limitations": [
            {
                "id": "L1",
                "title": "Segmentation sensitivity",
                "description": (
                    "H depends critically on the unit of segmentation (verse, line, strophe). "
                    "As documented in Phase 7 correction, different units can produce ΔH up to 0.20. "
                    "Cross-corpus comparisons require functionally equivalent units."
                )
            },
            {
                "id": "L2",
                "title": "Corpus size effects",
                "description": (
                    "Smaller corpora (< 1,000 units) may show more variable H estimates. "
                    "The Mishnah (471 units) and Heródoto (1,555 units) have fewer data points "
                    "than the AT (23,213 units). Size-dependent bias is not corrected."
                )
            },
            {
                "id": "L3",
                "title": "Language confound",
                "description": (
                    "H may be influenced by language-specific properties (morphological complexity, "
                    "word order flexibility, compounding patterns). The current model does not "
                    "control for language family beyond grouping."
                )
            },
            {
                "id": "L4",
                "title": "Genre confound",
                "description": (
                    "Different literary genres (poetry, prose, legal, narrative) show different H. "
                    "Mixed-genre corpora (like the full AT) combine these. Genre composition "
                    "could explain some H variation independent of transmission."
                )
            },
            {
                "id": "L5",
                "title": "Small reference set",
                "description": (
                    f"Only {len(reference_corpora)} reference corpora are available. "
                    "The model is severely underdetermined. Statistical power is low."
                )
            },
            {
                "id": "L6",
                "title": "Anomalies are statistical, not historical",
                "description": (
                    "A book with anomalous z-score may reflect genre, size, or transmission "
                    "variation — not necessarily different authorship or dating. These anomalies "
                    "should be treated as hypotheses for further investigation, never as evidence."
                )
            }
        ],
        "overall_assessment": (
            "The H-based authenticity tool is INDICATIVE, not DEFINITIVE. "
            "It can flag statistical outliers for further investigation by textual scholars, "
            "but cannot make claims about authorship, dating, or textual integrity. "
            "The method's validity depends on controlling for segmentation, corpus size, "
            "language, and genre — all of which are imperfectly controlled in this study."
        )
    }

    # ── Save ───────────────────────────────────────────────────────────
    elapsed = time.time() - t0

    reference_curve = {
        "reference_corpora": reference_corpora,
        "transmission_stats": transmission_stats,
        "bootstrap_ci": bootstrap_ci,
        "prediction_model": model,
        "elapsed_seconds": elapsed
    }

    with open(RESULTS_DIR / "reference_curve.json", "w") as f:
        json.dump(reference_curve, f, indent=2, ensure_ascii=False)

    with open(RESULTS_DIR / "book_anomalies.json", "w") as f:
        json.dump({
            "metric": "zipf_s_lemma",
            "note": "H per book not available; using Zipf s as proxy. Books with anomalous Zipf exponent may have different linguistic or compositional properties.",
            "mean_zipf_s": round(float(mean_zipf), 4),
            "std_zipf_s": round(float(std_zipf), 4),
            "n_anomalous": n_anomalous,
            "anomaly_threshold": 2.0,
            "books": book_anomalies
        }, f, indent=2, ensure_ascii=False)

    with open(RESULTS_DIR / "cross_validation.json", "w") as f:
        json.dump(cross_validation, f, indent=2, ensure_ascii=False)

    with open(RESULTS_DIR / "method_limitations.json", "w") as f:
        json.dump(method_limitations, f, indent=2, ensure_ascii=False)

    log.info(f"=== Script 5: h_authenticity_tool — DONE ({elapsed:.1f}s) ===")
    return {
        "reference_curve": reference_curve,
        "book_anomalies": book_anomalies,
        "cross_validation": cross_validation,
        "method_limitations": method_limitations
    }


if __name__ == "__main__":
    main()
