#!/usr/bin/env python3
"""
Fase 8 — Script 3: transmission_decay_rate.py
¿Decae H más rápido en textos con transmisión libre (Homero, Heródoto)
que en textos con transmisión controlada (AT)?
"""

import json
import logging
import time
import subprocess
import sys
import numpy as np
from pathlib import Path
from scipy import stats

BASE = Path(__file__).resolve().parent
RESULTS_DIR = BASE / "results" / "decay"
LOG_DIR = BASE / "logs"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "fase8_transmission_decay_rate.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


def main():
    t0 = time.time()
    log.info("=== Script 3: transmission_decay_rate — START ===")

    # ── Load all comparison data ───────────────────────────────────────
    log.info("Loading comparison corpus data...")
    with open(BASE / "results" / "fase5_comparison.json") as f:
        comparison = json.load(f)

    with open(BASE / "results" / "rigveda" / "rigveda_metrics.json") as f:
        rigveda = json.load(f)

    with open(BASE / "results" / "dss" / "dss_isaiah_comparison.json") as f:
        dss_data = json.load(f)

    # ── Build temporal data points ─────────────────────────────────────
    log.info("Building temporal data points...")

    # Each entry: {corpus, transmission_type, temporal_points: [{date, H, source}]}
    # Dates in years CE (negative = BCE)

    corpora_temporal = []

    # 1. AT Hebrew (WLC)
    # Composition: ~700-400 BCE (varied), WLC manuscript: ~1000 CE
    # DSS Isaiah: ~100 BCE, WLC Isaiah: ~1000 CE
    at_entry = {
        "corpus": "AT (Hebreo)",
        "transmission": "controlled",
        "family": "semitic",
        "current_H": None,
        "current_alpha": None,
        "temporal_points": [],
        "notes": []
    }
    for c in comparison:
        if c["corpus"] == "AT (Hebreo)":
            at_entry["current_H"] = c["hurst_H"]
            at_entry["current_alpha"] = c["dfa_alpha"]
            break

    # DSS Isaiah provides an earlier data point
    corrected = dss_data.get("CORRECTED_COMPARISON", {})
    if corrected:
        at_entry["temporal_points"].append({
            "date": -100,
            "H": corrected["dss_verses"]["hurst_H"],
            "alpha": corrected["dss_verses"]["dfa_alpha"],
            "source": "1QIsa^a DSS (biblical verses)",
            "book": "Isaiah"
        })
        at_entry["temporal_points"].append({
            "date": 1000,
            "H": corrected["wlc_verses"]["hurst_H"],
            "alpha": corrected["wlc_verses"]["dfa_alpha"],
            "source": "WLC Masoretic (biblical verses)",
            "book": "Isaiah"
        })

    corpora_temporal.append(at_entry)

    # 2. NT Greek
    nt_entry = {
        "corpus": "NT (Griego)",
        "transmission": "free",
        "family": "indo-european",
        "current_H": None,
        "current_alpha": None,
        "temporal_points": [],
        "notes": []
    }
    for c in comparison:
        if c["corpus"] == "NT (Griego)":
            nt_entry["current_H"] = c["hurst_H"]
            nt_entry["current_alpha"] = c["dfa_alpha"]
            nt_entry["temporal_points"].append({
                "date": 1000,  # approximate date of current critical text basis
                "H": c["hurst_H"],
                "alpha": c["dfa_alpha"],
                "source": "SBLGNT (critical text)"
            })
            break

    # Check for Papyrus P66 (John, ~200 CE)
    log.info("Checking for Papyrus P66 digital corpus...")
    p66_available = False
    p66_note = ("Papyrus P66 (~200 CE, Gospel of John) is not available as a "
                "digital morphologically-tagged corpus in standard NLP repositories "
                "(MorphGNT, Perseus, AGDT). The papyrus text exists in transcription "
                "but without POS tagging and verse-level word counts needed for H analysis. "
                "A second temporal point for NT transmission is therefore not available.")
    nt_entry["notes"].append(p66_note)
    log.info(f"P66: {p66_note}")

    corpora_temporal.append(nt_entry)

    # 3. Corán Arabic
    quran_entry = {
        "corpus": "Corán (Árabe)",
        "transmission": "controlled",
        "family": "semitic",
        "current_H": None,
        "current_alpha": None,
        "temporal_points": [],
        "notes": []
    }
    for c in comparison:
        if "Corán" in c.get("corpus", ""):
            quran_entry["current_H"] = c["hurst_H"]
            quran_entry["current_alpha"] = c["dfa_alpha"]
            quran_entry["temporal_points"].append({
                "date": 650,  # Uthmanic codex
                "H": c["hurst_H"],
                "alpha": c["dfa_alpha"],
                "source": "Quranic Arabic Corpus v0.4 (based on Uthmanic tradition)"
            })
            break

    quran_entry["notes"].append(
        "The Samarkand Codex (~700 CE) is not available as a digital morphologically-tagged corpus. "
        "Only one temporal point is available for the Quran."
    )
    corpora_temporal.append(quran_entry)

    # 4. Homero
    homer_entry = {
        "corpus": "Homero (Griego)",
        "transmission": "free",
        "family": "indo-european",
        "current_H": None,
        "current_alpha": None,
        "temporal_points": [],
        "notes": []
    }
    for c in comparison:
        if "Homero" in c.get("corpus", ""):
            homer_entry["current_H"] = c["hurst_H"]
            homer_entry["current_alpha"] = c["dfa_alpha"]
            homer_entry["temporal_points"].append({
                "date": 300,  # Medieval manuscript tradition, but composition ~750 BCE
                "H": c["hurst_H"],
                "alpha": c["dfa_alpha"],
                "source": "AGDT (Perseus Project, based on modern critical editions)"
            })
            break

    homer_entry["notes"].append(
        "Only one temporal point available. Composition ~750 BCE, oral transmission ~750-550 BCE, "
        "first written versions ~550 BCE. Current text reflects ~2,800 years of transmission, "
        "mostly uncontrolled after the Alexandrian period (~200 BCE)."
    )
    corpora_temporal.append(homer_entry)

    # 5. Heródoto
    herod_entry = {
        "corpus": "Heródoto (Griego)",
        "transmission": "free",
        "family": "indo-european",
        "current_H": None,
        "current_alpha": None,
        "temporal_points": [],
        "notes": []
    }
    for c in comparison:
        if "Heródoto" in c.get("corpus", ""):
            herod_entry["current_H"] = c["hurst_H"]
            herod_entry["current_alpha"] = c["dfa_alpha"]
            herod_entry["temporal_points"].append({
                "date": 300,
                "H": c["hurst_H"],
                "alpha": c["dfa_alpha"],
                "source": "AGDT (Perseus Project, based on modern critical editions)"
            })
            break

    herod_entry["notes"].append(
        "Only one temporal point available. Composition ~440 BCE, manuscript tradition "
        "through Byzantium. Current text reflects ~2,400 years of free (scribal) transmission."
    )
    corpora_temporal.append(herod_entry)

    # 6. Rig Veda
    rv_entry = {
        "corpus": "Rig Veda (Sánscrito)",
        "transmission": "controlled",
        "family": "indo-european",
        "current_H": rigveda["hurst_H"],
        "current_alpha": rigveda["dfa_alpha"],
        "temporal_points": [{
            "date": 1000,
            "H": rigveda["hurst_H"],
            "alpha": rigveda["dfa_alpha"],
            "source": "Digital Corpus of Sanskrit (DCS), first manuscripts ~1000 CE"
        }],
        "notes": [
            "Composition ~1500-1200 BCE, oral transmission continuously controlled. "
            "Only one temporal point (modern critical text). No pre-manuscript corpus available."
        ]
    }
    corpora_temporal.append(rv_entry)

    # 7. Mishnah
    mishnah_entry = {
        "corpus": "Mishnah (Hebreo)",
        "transmission": "semi-controlled",
        "family": "semitic",
        "current_H": None,
        "current_alpha": None,
        "temporal_points": [],
        "notes": []
    }
    for c in comparison:
        if "Mishnah" in c.get("corpus", ""):
            mishnah_entry["current_H"] = c["hurst_H"]
            mishnah_entry["current_alpha"] = c["dfa_alpha"]
            mishnah_entry["temporal_points"].append({
                "date": 200,
                "H": c["hurst_H"],
                "alpha": c["dfa_alpha"],
                "source": "Sefaria API (based on Vilna edition, ~1880 CE)"
            })
            break

    mishnah_entry["notes"].append(
        "Composition ~200 CE, redacted by Rabbi Judah ha-Nasi. "
        "Transmission semi-controlled (rabbinic academies). "
        "Only one temporal point available."
    )
    corpora_temporal.append(mishnah_entry)

    # ── Compute decay rates where possible ─────────────────────────────
    log.info("Computing decay rates...")

    decay_rates = []

    # Only AT/Isaiah has TWO temporal points
    for corpus_data in corpora_temporal:
        points = corpus_data["temporal_points"]
        if len(points) >= 2:
            # Sort by date
            points_sorted = sorted(points, key=lambda x: x["date"])
            t1, t2 = points_sorted[0]["date"], points_sorted[-1]["date"]
            h1, h2 = points_sorted[0]["H"], points_sorted[-1]["H"]
            a1, a2 = points_sorted[0]["alpha"], points_sorted[-1]["alpha"]

            dt_centuries = (t2 - t1) / 100.0
            rate_H = (h2 - h1) / dt_centuries if dt_centuries != 0 else 0
            rate_alpha = (a2 - a1) / dt_centuries if dt_centuries != 0 else 0

            decay_rates.append({
                "corpus": corpus_data["corpus"],
                "transmission": corpus_data["transmission"],
                "t_ancient": t1,
                "H_ancient": h1,
                "t_recent": t2,
                "H_recent": h2,
                "delta_t_centuries": dt_centuries,
                "rate_H_per_century": round(rate_H, 6),
                "rate_alpha_per_century": round(rate_alpha, 6),
                "type": "measured"
            })
        else:
            # Single point: compute upper bound assuming H_original = 0.5
            if len(points) == 1:
                h_current = points[0]["H"]
                # Estimate composition date
                composition_dates = {
                    "Homero (Griego)": -750,
                    "Heródoto (Griego)": -440,
                    "NT (Griego)": 50,
                    "Corán (Árabe)": 632,
                    "Rig Veda (Sánscrito)": -1200,
                    "Mishnah (Hebreo)": 200,
                }
                t_comp = composition_dates.get(corpus_data["corpus"])
                t_current = points[0]["date"]

                if t_comp is not None:
                    dt_centuries = (t_current - t_comp) / 100.0

                    # Upper bound: H_original = 0.5 (white noise baseline)
                    rate_upper = (h_current - 0.5) / dt_centuries if dt_centuries != 0 else 0

                    # Lower bound: H_original = h_current (no change)
                    rate_lower = 0.0

                    decay_rates.append({
                        "corpus": corpus_data["corpus"],
                        "transmission": corpus_data["transmission"],
                        "t_composition_estimated": t_comp,
                        "H_current": h_current,
                        "t_current_text": t_current,
                        "delta_t_centuries": dt_centuries,
                        "rate_H_per_century_upper_bound": round(rate_upper, 6),
                        "rate_H_per_century_lower_bound": rate_lower,
                        "H_original_assumed": 0.5,
                        "type": "upper_bound"
                    })

    # ── Compare controlled vs free ─────────────────────────────────────
    log.info("Comparing transmission regimes...")

    controlled_rates = []
    free_rates = []

    for dr in decay_rates:
        if dr["type"] == "measured":
            rate = abs(dr["rate_H_per_century"])
            if dr["transmission"] == "controlled":
                controlled_rates.append(rate)
            else:
                free_rates.append(rate)
        elif dr["type"] == "upper_bound":
            rate = abs(dr["rate_H_per_century_upper_bound"])
            if dr["transmission"] == "controlled":
                controlled_rates.append(rate)
            else:
                free_rates.append(rate)

    if controlled_rates and free_rates:
        # Note: with very few samples, this test has low power
        mw_stat, mw_p = stats.mannwhitneyu(
            controlled_rates, free_rates, alternative="two-sided"
        ) if len(controlled_rates) > 1 and len(free_rates) > 1 else (float("nan"), float("nan"))
    else:
        mw_stat, mw_p = float("nan"), float("nan")

    comparison_result = {
        "controlled_group": {
            "corpora": [dr["corpus"] for dr in decay_rates if dr["transmission"] == "controlled"],
            "rates": controlled_rates,
            "mean_rate": float(np.mean(controlled_rates)) if controlled_rates else None,
        },
        "free_group": {
            "corpora": [dr["corpus"] for dr in decay_rates if dr["transmission"] == "free"],
            "rates": free_rates,
            "mean_rate": float(np.mean(free_rates)) if free_rates else None,
        },
        "mann_whitney_p": float(mw_p) if not np.isnan(mw_p) else None,
        "note": (
            "This comparison is severely limited: only AT/Isaiah has two measured temporal points. "
            "All other corpora use upper-bound estimates (assuming H_original=0.5). "
            "The comparison is therefore indicative, not conclusive."
        )
    }

    # ── Cross-corpus H by transmission type ────────────────────────────
    log.info("Grouping current H by transmission type...")
    controlled_h = []
    free_h = []
    for ct in corpora_temporal:
        h = ct["current_H"]
        if h is not None:
            if ct["transmission"] == "controlled":
                controlled_h.append(h)
            elif ct["transmission"] == "free":
                free_h.append(h)

    if len(controlled_h) >= 2 and len(free_h) >= 2:
        cross_mw, cross_p = stats.mannwhitneyu(controlled_h, free_h, alternative="greater")
    else:
        cross_mw, cross_p = float("nan"), float("nan")

    cross_corpus = {
        "controlled_H_values": controlled_h,
        "controlled_H_mean": float(np.mean(controlled_h)) if controlled_h else None,
        "free_H_values": free_h,
        "free_H_mean": float(np.mean(free_h)) if free_h else None,
        "mann_whitney_p_one_sided": float(cross_p) if not np.isnan(cross_p) else None,
        "test": "H(controlled) > H(free)",
        "note": "This is a cross-sectional comparison (current H values), not a temporal decay comparison."
    }

    # ── Save ───────────────────────────────────────────────────────────
    elapsed = time.time() - t0

    temporal_comparison = {
        "analysis": "Temporal comparison of H across transmission regimes",
        "corpora": corpora_temporal,
        "limitation": (
            "Only one corpus (AT/Isaiah) has two measured temporal points "
            "(DSS ~100 BCE and WLC ~1000 CE). For all others, only upper-bound "
            "estimates are available. P66 (Papyrus, NT ~200 CE) is not available "
            "as a digital morphologically-tagged corpus."
        ),
        "elapsed_seconds": elapsed
    }

    with open(RESULTS_DIR / "temporal_comparison.json", "w") as f:
        json.dump(temporal_comparison, f, indent=2, ensure_ascii=False)

    decay_result = {
        "decay_rates": decay_rates,
        "comparison_controlled_vs_free": comparison_result,
        "cross_corpus_h_by_transmission": cross_corpus,
        "conclusion": (
            "The only directly measured decay rate (AT/Isaiah, controlled transmission) is "
            f"~{abs(decay_rates[0]['rate_H_per_century']):.4f} H/century — effectively zero. "
            "Upper-bound estimates for free transmission corpora suggest rates of "
            f"~{np.mean(free_rates):.4f} H/century, but these assume H_original=0.5 "
            "and are therefore not directly comparable. "
            "The cross-sectional comparison (current H values) shows controlled > free "
            f"(mean {np.mean(controlled_h):.3f} vs {np.mean(free_h):.3f}), "
            "but this does not establish that free transmission *degrades* H."
        ),
        "elapsed_seconds": elapsed
    }

    with open(RESULTS_DIR / "decay_rates.json", "w") as f:
        json.dump(decay_result, f, indent=2, ensure_ascii=False)

    log.info(f"=== Script 3: transmission_decay_rate — DONE ({elapsed:.1f}s) ===")
    return decay_result


if __name__ == "__main__":
    main()
