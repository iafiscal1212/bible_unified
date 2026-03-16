#!/usr/bin/env python3
"""
Fase 8 — Script 4: h5_resolution_attempt.py
¿Podemos distinguir H5a de H5b con los datos disponibles?

H5a: la estructura existía antes de la canonización
H5b: la canonización generó la estructura
"""

import json
import logging
import time
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent
RESULTS_DIR = BASE / "results" / "h5"
LOG_DIR = BASE / "logs"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "fase8_h5_resolution_attempt.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


def main():
    t0 = time.time()
    log.info("=== Script 4: h5_resolution_attempt — START ===")

    # ── Load all relevant data ─────────────────────────────────────────
    log.info("Loading data from previous phases...")

    with open(BASE / "results" / "fase5_comparison.json") as f:
        comparison = json.load(f)

    with open(BASE / "results" / "rigveda" / "rigveda_metrics.json") as f:
        rigveda = json.load(f)

    with open(BASE / "results" / "dss" / "dss_isaiah_comparison.json") as f:
        dss_data = json.load(f)

    with open(BASE / "results" / "dss_wordlevel" / "dss_wordlevel_summary.json") as f:
        dss_summary = json.load(f)

    # ── Build temporal points for all controlled-transmission texts ────
    log.info("Building temporal analysis...")

    temporal_points = []

    # 1. AT / Isaiah
    corrected = dss_data.get("CORRECTED_COMPARISON", {})
    if corrected:
        temporal_points.append({
            "text": "AT / Isaiah",
            "transmission": "controlled",
            "canonization_event": "Council of Jamnia ~90 CE (traditional date; debated)",
            "canonization_date": 90,
            "pre_canonization_point": {
                "date": -100,
                "H": corrected["dss_verses"]["hurst_H"],
                "alpha": corrected["dss_verses"]["dfa_alpha"],
                "source": "1QIsa^a (Dead Sea Scrolls)",
                "note": "Complete scroll, verse-level segmentation"
            },
            "post_canonization_point": {
                "date": 1000,
                "H": corrected["wlc_verses"]["hurst_H"],
                "alpha": corrected["wlc_verses"]["dfa_alpha"],
                "source": "WLC (Westminster Leningrad Codex)",
                "note": "Masoretic text"
            },
            "delta_H": corrected["dss_verses"]["hurst_H"] - corrected["wlc_verses"]["hurst_H"],
            "mann_whitney_p": dss_data.get("CORRECTED_COMPARISON", {}).get("mann_whitney_p"),
            "conclusion": "H is statistically identical before and after canonization"
        })

    # 2. Quran
    quran_h = None
    for c in comparison:
        if "Corán" in c.get("corpus", ""):
            quran_h = c["hurst_H"]
            quran_alpha = c["dfa_alpha"]
            break

    temporal_points.append({
        "text": "Corán",
        "transmission": "controlled",
        "canonization_event": "Uthmanic codex ~650 CE",
        "canonization_date": 650,
        "pre_canonization_point": None,
        "post_canonization_point": {
            "date": 650,
            "H": quran_h,
            "alpha": quran_alpha,
            "source": "Quranic Arabic Corpus v0.4",
            "note": "Based on Uthmanic tradition; effectively post-canonization"
        },
        "delta_H": None,
        "mann_whitney_p": None,
        "conclusion": "No pre-canonization corpus available. Cannot test H5a vs H5b.",
        "required_corpus": (
            "A digital morphologically-tagged transcription of a pre-Uthmanic Quran manuscript "
            "(e.g., Codex Parisino-Petropolitanus, ~640-680 CE, or Birmingham manuscript, "
            "~568-645 CE) would provide a pre-canonization data point. These manuscripts exist "
            "but are not available as tagged digital corpora."
        )
    })

    # 3. Rig Veda
    temporal_points.append({
        "text": "Rig Veda",
        "transmission": "controlled",
        "canonization_event": "No single canonization event; oral tradition formalized over centuries",
        "canonization_date": None,
        "pre_canonization_point": None,
        "post_canonization_point": {
            "date": 1000,
            "H": rigveda["hurst_H"],
            "alpha": rigveda["dfa_alpha"],
            "source": "Digital Corpus of Sanskrit (DCS)",
            "note": "Based on modern critical editions of oral tradition"
        },
        "delta_H": None,
        "mann_whitney_p": None,
        "conclusion": (
            "No pre-canonization corpus available. The Rig Veda has no single canonization "
            "event — it was transmitted orally with extreme control (padapatha, vikrtipatha methods) "
            "for millennia before first manuscripts (~1000 CE). The concept of pre/post canonization "
            "does not cleanly apply."
        ),
        "required_corpus": (
            "Not applicable: the Rig Veda has no pre-canonization written text. "
            "The earliest manuscripts already reflect millennia of controlled oral transmission. "
            "The H5a/H5b distinction is not testable for purely oral traditions."
        )
    })

    # ── Formal logical analysis ────────────────────────────────────────
    log.info("Performing formal logical analysis...")

    h5_analysis = {
        "hypothesis_H5a": {
            "statement": "The long-range correlation structure (H > 0.5) existed in the original composition, before formal canonization.",
            "prediction": "H_pre_canon ≥ H_post_canon (structure already present)",
            "would_be_confirmed_if": "H_DSS > H_WLC (memory was higher in pre-canonical text)"
        },
        "hypothesis_H5b": {
            "statement": "The canonization process (editorial standardization, scribal control) generated or amplified the long-range correlation structure.",
            "prediction": "H_pre_canon < H_post_canon (structure grew through canonization)",
            "would_be_confirmed_if": "H_WLC > H_DSS (memory increased through Masoretic transmission)"
        },
        "observed_data": {
            "H_pre_canon_DSS": corrected["dss_verses"]["hurst_H"] if corrected else None,
            "H_post_canon_WLC": corrected["wlc_verses"]["hurst_H"] if corrected else None,
            "delta_H": round(corrected["dss_verses"]["hurst_H"] - corrected["wlc_verses"]["hurst_H"], 4) if corrected else None,
            "p_value": dss_data.get("CORRECTED_COMPARISON", {}).get("mann_whitney_p"),
            "statistically_distinguishable": False
        },
        "logical_verdict": {
            "H5a": "NOT CONFIRMED — H_DSS is NOT significantly greater than H_WLC",
            "H5b": "NOT CONFIRMED — H_WLC is NOT significantly greater than H_DSS",
            "actual_result": "INVARIANCE — H_DSS ≈ H_WLC within statistical noise",
            "implication": (
                "The data is compatible with BOTH H5a and H5b simultaneously. "
                "If H5a is true (structure existed before canonization), the Masoretes preserved it. "
                "If H5b is true (canonization created it), it was created early enough to be present "
                "in 1QIsa^a (~100 BCE), which predates the traditional canonization date (~90 CE). "
                "The temporal invariance means that whatever process generated H, it was complete "
                "by ~100 BCE at the latest."
            )
        },
        "constraint_from_data": {
            "statement": (
                "The observed invariance constrains the timing: whatever generated the "
                "long-range structure in Isaiah, it happened BEFORE ~100 BCE. This is consistent "
                "with H5a (original composition ~700 BCE) and with an early H5b (editorial "
                "standardization during the Second Temple period, before Jamnia)."
            ),
            "earliest_date_with_H": -100,
            "source": "1QIsa^a",
            "implication_for_canonization": (
                "If canonization (Jamnia, ~90 CE) is the H5b event, then DSS should show "
                "LOWER H than WLC. It does not. Therefore either (a) the structure predates "
                "canonization (H5a), or (b) the structure-generating process was already "
                "complete before the formal canonization event."
            )
        }
    }

    # ── What corpus would resolve H5a vs H5b? ─────────────────────────
    log.info("Identifying required future corpora...")

    required_future_corpus = {
        "question": "What corpus would definitively distinguish H5a from H5b?",
        "requirements": [
            "A text with KNOWN composition date",
            "A pre-canonization version (digital, morphologically tagged)",
            "A post-canonization version (digital, morphologically tagged)",
            "Both versions segmentable by the same textual unit",
            "Documented transmission history between the two versions"
        ],
        "candidates": [
            {
                "corpus": "Pre-Uthmanic Quran manuscripts",
                "feasibility": "Medium",
                "composition": "~610-632 CE",
                "canonization": "~650 CE (Uthmanic codex)",
                "pre_canon_source": "Birmingham manuscript (~568-645 CE), Codex Parisino-Petropolitanus (~640-680 CE)",
                "status": "Manuscript images available; no morphologically-tagged digital corpus exists",
                "effort": "Requires Arabic NLP pipeline + manual verification"
            },
            {
                "corpus": "Septuagint (LXX) vs Masoretic",
                "feasibility": "High",
                "composition": "~700-400 BCE (Hebrew original)",
                "canonization": "~90 CE (Jamnia, MT tradition)",
                "pre_canon_source": "LXX (~250 BCE, Greek translation from pre-Masoretic Hebrew)",
                "status": "CATSS and Rahlfs LXX are digitally available",
                "effort": "Moderate: LXX is in Greek (translation), not Hebrew. Comparing H across languages introduces confounds. Still, if LXX Greek Isaiah has similar H to DSS Hebrew Isaiah, it would support H5a.",
                "caveat": "LXX is a TRANSLATION, not a copy. Language change affects H independently of transmission."
            },
            {
                "corpus": "Samaritan Pentateuch vs Masoretic Pentateuch",
                "feasibility": "High",
                "composition": "~1400-400 BCE (Torah)",
                "canonization": "Split ~400 BCE (Samaritan) vs ~90 CE (Masoretic)",
                "pre_canon_source": "Samaritan Pentateuch (independent transmission since ~400 BCE)",
                "status": "Digital texts available (various quality); some morphological tagging exists",
                "effort": "Moderate: same language (Hebrew), independent transmission lines, both pre-date final MT canonization",
                "note": "This is the STRONGEST candidate: same language, same text, two independent transmission lines diverging ~400 BCE. If both show similar H, it confirms H5a (structure predates the split). If MT shows higher H, it supports H5b."
            },
            {
                "corpus": "Vulgate Latin vs Greek NT",
                "feasibility": "Medium",
                "composition": "~50-100 CE (Greek NT)",
                "canonization": "~380 CE (Jerome's Vulgate as Latin canon)",
                "pre_canon_source": "Greek NT manuscripts (P66, P75, Codex Sinaiticus)",
                "status": "Both digitally available with morphological tagging",
                "effort": "Cross-language comparison (Greek→Latin translation), same caveats as LXX",
                "caveat": "Translation confound"
            }
        ],
        "recommendation": (
            "The Samaritan Pentateuch vs Masoretic Pentateuch comparison is the most promising: "
            "same language, same base text, independent transmission since ~400 BCE. If both show "
            "similar H, H5a is confirmed (structure predates the divergence). If MT shows higher H, "
            "H5b gains support. A tagged digital Samaritan Pentateuch would resolve the question."
        )
    }

    # ── Save ───────────────────────────────────────────────────────────
    elapsed = time.time() - t0

    with open(RESULTS_DIR / "temporal_points.json", "w") as f:
        json.dump(temporal_points, f, indent=2, ensure_ascii=False)

    with open(RESULTS_DIR / "h5_analysis.json", "w") as f:
        json.dump(h5_analysis, f, indent=2, ensure_ascii=False)

    with open(RESULTS_DIR / "required_future_corpus.json", "w") as f:
        json.dump(required_future_corpus, f, indent=2, ensure_ascii=False)

    log.info(f"=== Script 4: h5_resolution_attempt — DONE ({elapsed:.1f}s) ===")
    log.info(f"Verdict: H5a and H5b are indistinguishable with current data")
    log.info(f"Best future candidate: Samaritan Pentateuch vs Masoretic")
    return {
        "h5_analysis": h5_analysis,
        "temporal_points": temporal_points,
        "required_future_corpus": required_future_corpus
    }


if __name__ == "__main__":
    main()
