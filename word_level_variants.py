#!/usr/bin/env python3
"""
Fase 8 — Script 1: word_level_variants.py
¿Qué tipo de variante textual DSS/WLC impacta más la estructura de correlaciones local?
"""

import json
import logging
import time
import numpy as np
from pathlib import Path
from scipy import stats
from collections import Counter, defaultdict

BASE = Path(__file__).resolve().parent
RESULTS_DIR = BASE / "results" / "word_variants"
CACHE_DIR = BASE / "results" / "dss_wordlevel" / "_cache"
ALIGNMENT_FILE = BASE / "results" / "dss_wordlevel" / "alignment_variants.json"
LOG_DIR = BASE / "logs"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "fase8_word_level_variants.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ── Métricas (copiadas de analyze_dss.py) ──────────────────────────────

def hurst_exponent_rs(series):
    """Hurst exponent via Rescaled Range (R/S) analysis."""
    series = np.asarray(series, dtype=float)
    n = len(series)
    if n < 20:
        return float("nan"), 0.0
    min_block = 10
    max_block = n // 2
    sizes = []
    rs_values = []
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


def dfa_exponent(series):
    """Detrended Fluctuation Analysis (DFA)."""
    series = np.asarray(series, dtype=float)
    n = len(series)
    if n < 20:
        return float("nan"), 0.0
    y = np.cumsum(series - series.mean())
    min_box = 4
    max_box = n // 4
    sizes = []
    flucts = []
    box = min_box
    while box <= max_box:
        sizes.append(box)
        n_boxes = n // box
        rms_list = []
        for i in range(n_boxes):
            seg = y[i * box:(i + 1) * box]
            x_ax = np.arange(box)
            coeffs = np.polyfit(x_ax, seg, 1)
            trend = np.polyval(coeffs, x_ax)
            rms_list.append(np.sqrt(np.mean((seg - trend) ** 2)))
        if rms_list:
            flucts.append(np.mean(rms_list))
        box = int(box * 1.5)
        if box == sizes[-1]:
            box += 1
    if len(sizes) < 3:
        return float("nan"), 0.0
    log_s = np.log(sizes)
    log_f = np.log(flucts)
    slope, intercept, r, p, se = stats.linregress(log_s, log_f)
    return float(slope), float(r ** 2)


def local_hurst_sliding(verse_lengths, window=50, stride=1):
    """Compute local Hurst exponent with sliding window."""
    n = len(verse_lengths)
    results = []
    for start in range(0, n - window + 1, stride):
        seg = verse_lengths[start:start + window]
        h, r2 = hurst_exponent_rs(seg)
        results.append({
            "center": start + window // 2,
            "H": h,
            "R2": r2
        })
    return results


def main():
    t0 = time.time()
    log.info("=== Script 1: word_level_variants — START ===")

    # ── Load data ──────────────────────────────────────────────────────
    log.info("Loading alignment variants...")
    with open(ALIGNMENT_FILE) as f:
        alignment = json.load(f)

    log.info("Loading DSS cache...")
    with open(CACHE_DIR / "dss_full.json") as f:
        dss_data = json.load(f)

    log.info("Loading WLC cache...")
    with open(CACHE_DIR / "wlc_isaiah.json") as f:
        wlc_data = json.load(f)

    # ── Build verse-level word count series ────────────────────────────
    # DSS: group words by (chapter, verse)
    dss_words = dss_data.get("words", [])
    if not dss_words:
        # Reconstruct from lines if words not directly available
        # Use the lines data and map through chapter/verse
        log.info("Reconstructing DSS verse lengths from lines data...")
        # Need to load from Text-Fabric or use cached verse mapping
        # Fall back: compute from alignment
        pass

    # Build DSS verse lengths from words grouped by (chapter, verse)
    dss_verse_counts = defaultdict(int)
    for w in dss_words:
        ch = w.get("chapter")
        vs = w.get("verse")
        if ch is not None and vs is not None:
            dss_verse_counts[(ch, vs)] += 1

    # Build WLC verse lengths
    wlc_verses = wlc_data["verses"]
    wlc_verse_map = {}
    for v in wlc_verses:
        key = (v["chapter"], v["verse"])
        wlc_verse_map[key] = v["n_words"]

    # Get ordered list of common verses
    common_keys = sorted(set(dss_verse_counts.keys()) & set(wlc_verse_map.keys()))
    log.info(f"Common verses: {len(common_keys)}")

    dss_lengths = [dss_verse_counts[k] for k in common_keys]
    wlc_lengths = [wlc_verse_map[k] for k in common_keys]

    n_verses = len(common_keys)

    # ── Compute local H with sliding window ────────────────────────────
    log.info("Computing local Hurst exponents (window=50, stride=1)...")
    window = 50
    stride = 1

    dss_local = local_hurst_sliding(dss_lengths, window=window, stride=stride)
    wlc_local = local_hurst_sliding(wlc_lengths, window=window, stride=stride)

    # Compute ΔH_local for each position
    delta_h_by_center = {}
    for d, w in zip(dss_local, wlc_local):
        center = d["center"]
        dh = d["H"] - w["H"]
        delta_h_by_center[center] = {
            "H_dss": d["H"],
            "H_wlc": w["H"],
            "delta_H": dh
        }

    log.info(f"Local H computed for {len(delta_h_by_center)} positions")

    # ── Build variant density by verse ─────────────────────────────────
    log.info("Computing variant density per verse...")
    variants = alignment["variants"]

    # Count variants per WLC verse
    verse_variant_counts = defaultdict(lambda: defaultdict(int))
    verse_variant_pos = defaultdict(list)

    for var in variants:
        # Get verse reference
        if var.get("wlc") and var["wlc"].get("ch") is not None:
            ch, vs = var["wlc"]["ch"], var["wlc"]["vs"]
        elif var.get("dss") and var["dss"].get("line") is not None:
            # For deletions (wlc=null), try to infer verse from DSS word
            # Skip these for now - they don't have verse mapping in variant data
            continue
        else:
            continue

        key = (ch, vs)
        vtype = var["type"]
        verse_variant_counts[key][vtype] += 1
        verse_variant_counts[key]["total"] += 1

        # Track POS of variant words
        if var.get("dss"):
            verse_variant_pos[key].append(var["dss"].get("pos", "unknown"))
        if var.get("wlc"):
            verse_variant_pos[key].append(var["wlc"].get("pos", "unknown"))

    # Compute density = n_variants / n_words for each verse
    verse_densities = {}
    for i, key in enumerate(common_keys):
        n_words = wlc_verse_map[key]
        n_vars = verse_variant_counts[key]["total"] if key in verse_variant_counts else 0
        density = n_vars / n_words if n_words > 0 else 0.0
        dominant_type = "none"
        if key in verse_variant_counts:
            type_counts = {k: v for k, v in verse_variant_counts[key].items() if k != "total"}
            if type_counts:
                dominant_type = max(type_counts, key=type_counts.get)
        verse_densities[i] = {
            "key": list(key),
            "n_words": n_words,
            "n_variants": n_vars,
            "density": density,
            "dominant_type": dominant_type,
            "type_breakdown": dict(verse_variant_counts.get(key, {}))
        }

    # ── Correlate ΔH_local with variant density ────────────────────────
    log.info("Correlating ΔH_local with variant density...")

    # For each center position in the sliding window, compute average variant density
    # in the window
    half_w = window // 2
    correlation_data = []

    for center_idx, dh_info in delta_h_by_center.items():
        start = center_idx - half_w
        end = center_idx + half_w
        # Average density in window
        densities_in_window = []
        types_in_window = Counter()
        for vi in range(max(0, start), min(n_verses, end)):
            if vi in verse_densities:
                densities_in_window.append(verse_densities[vi]["density"])
                for t, c in verse_densities[vi].get("type_breakdown", {}).items():
                    if t != "total":
                        types_in_window[t] += c

        avg_density = np.mean(densities_in_window) if densities_in_window else 0.0
        dominant_in_window = types_in_window.most_common(1)[0][0] if types_in_window else "none"

        correlation_data.append({
            "center": center_idx,
            "delta_H": dh_info["delta_H"],
            "H_dss": dh_info["H_dss"],
            "H_wlc": dh_info["H_wlc"],
            "avg_density": avg_density,
            "dominant_type": dominant_in_window,
            "n_variants_window": sum(types_in_window.values())
        })

    # Pearson correlation: ΔH vs density
    delta_h_arr = np.array([d["delta_H"] for d in correlation_data])
    density_arr = np.array([d["avg_density"] for d in correlation_data])

    # Filter out NaN
    valid = ~(np.isnan(delta_h_arr) | np.isnan(density_arr))
    if valid.sum() > 10:
        r_corr, p_corr = stats.pearsonr(delta_h_arr[valid], density_arr[valid])
        spearman_r, spearman_p = stats.spearmanr(delta_h_arr[valid], density_arr[valid])
    else:
        r_corr, p_corr = float("nan"), float("nan")
        spearman_r, spearman_p = float("nan"), float("nan")

    log.info(f"Pearson r(ΔH, density) = {r_corr:.4f}, p = {p_corr:.4e}")
    log.info(f"Spearman ρ(ΔH, density) = {spearman_r:.4f}, p = {spearman_p:.4e}")

    local_h_correlation = {
        "pearson_r": float(r_corr),
        "pearson_p": float(p_corr),
        "spearman_rho": float(spearman_r),
        "spearman_p": float(spearman_p),
        "n_windows": int(valid.sum()),
        "interpretation": "Positive r means higher variant density → higher ΔH (DSS more structured locally)"
    }

    # ── ANOVA: variant type vs ΔH_local ────────────────────────────────
    log.info("ANOVA: variant type vs ΔH_local...")

    # Group ΔH by dominant variant type in window
    type_groups = defaultdict(list)
    for d in correlation_data:
        if not np.isnan(d["delta_H"]):
            type_groups[d["dominant_type"]].append(d["delta_H"])

    # Only keep groups with enough data
    anova_groups = {k: v for k, v in type_groups.items() if len(v) >= 10}
    anova_labels = sorted(anova_groups.keys())

    if len(anova_labels) >= 2:
        anova_arrays = [np.array(anova_groups[k]) for k in anova_labels]
        f_stat, p_anova = stats.f_oneway(*anova_arrays)

        # Per-group stats
        group_stats = {}
        for label in anova_labels:
            arr = np.array(anova_groups[label])
            group_stats[label] = {
                "n": len(arr),
                "mean_delta_H": float(arr.mean()),
                "std_delta_H": float(arr.std()),
                "median_delta_H": float(np.median(arr))
            }
    else:
        f_stat, p_anova = float("nan"), float("nan")
        group_stats = {}

    log.info(f"ANOVA F={f_stat:.4f}, p={p_anova:.4e}")
    for label, gs in group_stats.items():
        log.info(f"  {label}: n={gs['n']}, mean_ΔH={gs['mean_delta_H']:.4f}")

    # ── Per-verse variant impact ───────────────────────────────────────
    # Also compute per-verse ΔH (not windowed) using verse-level differences
    log.info("Computing per-verse ΔH...")
    verse_delta = []
    for i in range(n_verses):
        dl = dss_lengths[i] - wlc_lengths[i]
        vd = verse_densities.get(i, {"density": 0, "dominant_type": "none", "n_variants": 0})
        verse_delta.append({
            "idx": i,
            "key": list(common_keys[i]),
            "dss_words": dss_lengths[i],
            "wlc_words": wlc_lengths[i],
            "word_diff": dl,
            "variant_density": vd["density"],
            "n_variants": vd["n_variants"],
            "dominant_type": vd["dominant_type"]
        })

    # Per-verse ANOVA on word_diff by dominant variant type
    type_word_diff = defaultdict(list)
    for vd in verse_delta:
        if vd["n_variants"] > 0:
            type_word_diff[vd["dominant_type"]].append(vd["word_diff"])

    verse_anova_groups = {k: v for k, v in type_word_diff.items() if len(v) >= 5}
    verse_anova_labels = sorted(verse_anova_groups.keys())

    if len(verse_anova_labels) >= 2:
        verse_f, verse_p = stats.f_oneway(*[np.array(verse_anova_groups[k]) for k in verse_anova_labels])
        verse_type_stats = {}
        for label in verse_anova_labels:
            arr = np.array(verse_anova_groups[label])
            verse_type_stats[label] = {
                "n_verses": len(arr),
                "mean_word_diff": float(arr.mean()),
                "std_word_diff": float(arr.std()),
                "abs_mean_diff": float(np.abs(arr).mean())
            }
    else:
        verse_f, verse_p = float("nan"), float("nan")
        verse_type_stats = {}

    log.info(f"Per-verse ANOVA (word_diff by type): F={verse_f:.4f}, p={verse_p:.4e}")

    # ── Top 10 most perturbed verses (by |word_diff|) ──────────────────
    log.info("Finding top 10 most perturbed verses...")
    sorted_by_diff = sorted(verse_delta, key=lambda x: abs(x["word_diff"]), reverse=True)

    top_positive = [v for v in sorted(verse_delta, key=lambda x: x["word_diff"], reverse=True)[:10]]
    top_negative = [v for v in sorted(verse_delta, key=lambda x: x["word_diff"])[:10]]

    top_perturbed = {
        "top_10_positive_delta": [
            {"chapter": v["key"][0], "verse": v["key"][1],
             "dss_words": v["dss_words"], "wlc_words": v["wlc_words"],
             "word_diff": v["word_diff"], "n_variants": v["n_variants"],
             "dominant_type": v["dominant_type"]}
            for v in top_positive
        ],
        "top_10_negative_delta": [
            {"chapter": v["key"][0], "verse": v["key"][1],
             "dss_words": v["dss_words"], "wlc_words": v["wlc_words"],
             "word_diff": v["word_diff"], "n_variants": v["n_variants"],
             "dominant_type": v["dominant_type"]}
            for v in top_negative
        ],
        "top_10_abs_delta": [
            {"chapter": v["key"][0], "verse": v["key"][1],
             "dss_words": v["dss_words"], "wlc_words": v["wlc_words"],
             "word_diff": v["word_diff"], "n_variants": v["n_variants"],
             "dominant_type": v["dominant_type"]}
            for v in sorted_by_diff[:10]
        ]
    }

    # ── POS breakdown of variants ──────────────────────────────────────
    log.info("POS breakdown of variants...")
    pos_by_type = defaultdict(Counter)
    for var in variants:
        vtype = var["type"]
        if var.get("dss") and var["dss"].get("pos"):
            pos_by_type[vtype][var["dss"]["pos"]] += 1
        if var.get("wlc") and var["wlc"].get("pos"):
            pos_by_type[vtype][var["wlc"]["pos"]] += 1

    pos_breakdown = {t: dict(c.most_common()) for t, c in pos_by_type.items()}

    # ── Save results ───────────────────────────────────────────────────
    elapsed = time.time() - t0
    log.info(f"Saving results... (elapsed: {elapsed:.1f}s)")

    variant_impact = {
        "analysis": "Word-level variant impact on local H structure",
        "window_size": window,
        "stride": stride,
        "n_common_verses": n_verses,
        "global_H_dss": float(hurst_exponent_rs(dss_lengths)[0]),
        "global_H_wlc": float(hurst_exponent_rs(wlc_lengths)[0]),
        "anova_windowed": {
            "F_statistic": float(f_stat),
            "p_value": float(p_anova),
            "groups": group_stats,
            "question": "Does dominant variant TYPE in a window predict ΔH?"
        },
        "anova_per_verse": {
            "F_statistic": float(verse_f),
            "p_value": float(verse_p),
            "groups": verse_type_stats,
            "question": "Does dominant variant TYPE predict word-count difference?"
        },
        "pos_breakdown_by_variant_type": pos_breakdown,
        "total_variants_by_type": dict(alignment["counts"]),
        "elapsed_seconds": elapsed
    }

    with open(RESULTS_DIR / "variant_impact.json", "w") as f:
        json.dump(variant_impact, f, indent=2, ensure_ascii=False)

    with open(RESULTS_DIR / "local_h_correlation.json", "w") as f:
        json.dump(local_h_correlation, f, indent=2, ensure_ascii=False)

    with open(RESULTS_DIR / "top_perturbed_verses.json", "w") as f:
        json.dump(top_perturbed, f, indent=2, ensure_ascii=False)

    log.info(f"=== Script 1: word_level_variants — DONE ({elapsed:.1f}s) ===")
    return variant_impact


if __name__ == "__main__":
    main()
