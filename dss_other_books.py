#!/usr/bin/env python3
"""
Fase 8 — Script 2: dss_other_books.py
¿Es la invarianza temporal de H una propiedad general del AT o específica de Isaías?
"""

import json
import logging
import time
import subprocess
import sys
import os
import numpy as np
from pathlib import Path
from scipy import stats
from collections import defaultdict

BASE = Path(__file__).resolve().parent
RESULTS_DIR = BASE / "results" / "dss_books"
LOG_DIR = BASE / "logs"
DSS_REPO = Path.home() / "github" / "ETCBC" / "dss"
BHSA_REPO = Path.home() / "github" / "ETCBC" / "bhsa"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "fase8_dss_other_books.log"),
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


def dfa_exponent(series):
    series = np.asarray(series, dtype=float)
    n = len(series)
    if n < 20:
        return float("nan"), 0.0
    y = np.cumsum(series - series.mean())
    min_box = 4
    max_box = n // 4
    sizes, flucts = [], []
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


def ensure_text_fabric():
    try:
        import tf
        return True
    except ImportError:
        pass
    try:
        from tf.app import use
        return True
    except ImportError:
        pass
    subprocess.check_call([sys.executable, "-m", "pip", "install", "text-fabric", "-q"])
    return True


def clone_repo(repo_url, target):
    if not target.exists():
        log.info(f"Cloning {repo_url} → {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(["git", "clone", "--depth", "1", repo_url, str(target)])
    return target


def main():
    t0 = time.time()
    log.info("=== Script 2: dss_other_books — START ===")

    ensure_text_fabric()
    from tf.fabric import Fabric

    # ── Clone repos if needed ──────────────────────────────────────────
    clone_repo("https://github.com/ETCBC/dss", DSS_REPO)
    clone_repo("https://github.com/ETCBC/bhsa", BHSA_REPO)

    # ── Load DSS corpus ────────────────────────────────────────────────
    log.info("Loading DSS corpus via Text-Fabric...")
    # Find the latest version directory
    tf_dir = DSS_REPO / "tf"
    versions = sorted([d.name for d in tf_dir.iterdir() if d.is_dir()]) if tf_dir.exists() else []
    version = versions[-1] if versions else "1.8.1"
    log.info(f"Using DSS version: {version}")

    TF_dss = Fabric(locations=str(tf_dir / version), silent="deep")
    api_dss = TF_dss.load("scroll sp lex line fragment biblical book chapter verse g_cons", silent="deep")
    F_dss = api_dss.F
    L_dss = api_dss.L
    T_dss = api_dss.T

    # ── Load BHSA (WLC) corpus ─────────────────────────────────────────
    log.info("Loading BHSA corpus via Text-Fabric...")
    tf_bhsa_dir = BHSA_REPO / "tf"
    bhsa_versions = sorted([d.name for d in tf_bhsa_dir.iterdir() if d.is_dir()]) if tf_bhsa_dir.exists() else []
    bhsa_version = bhsa_versions[-1] if bhsa_versions else "2021"
    log.info(f"Using BHSA version: {bhsa_version}")

    TF_bhsa = Fabric(locations=str(tf_bhsa_dir / bhsa_version), silent="deep")
    api_bhsa = TF_bhsa.load("book chapter verse g_cons sp", silent="deep")
    F_bhsa = api_bhsa.F
    L_bhsa = api_bhsa.L

    # ── Find all scrolls with biblical content and verse mapping ───────
    log.info("Scanning all DSS scrolls for biblical content...")

    # Get all scroll nodes
    all_scrolls = list(F_dss.otype.s("scroll"))
    log.info(f"Total scrolls in corpus: {len(all_scrolls)}")

    last_log = time.time()
    scroll_info = []

    for scroll_node in all_scrolls:
        scroll_name = F_dss.scroll.v(scroll_node)
        # Get all words in this scroll
        words = L_dss.d(scroll_node, otype="word")
        if not words:
            continue

        # Check biblical content
        n_biblical = 0
        n_total = len(words)
        verse_words = defaultdict(list)  # (book, chapter, verse) → [word_nodes]

        for w in words:
            bib = F_dss.biblical.v(w)
            if bib and str(bib).lower() not in ("0", "false", "none", ""):
                n_biblical += 1
                book = F_dss.book.v(w)
                ch = F_dss.chapter.v(w)
                vs = F_dss.verse.v(w)
                if book and ch and vs:
                    try:
                        verse_words[(str(book), int(ch), int(vs))].append(w)
                    except (ValueError, TypeError):
                        pass

        if n_biblical == 0 or not verse_words:
            continue

        # Group by book
        books_in_scroll = defaultdict(lambda: defaultdict(list))
        for (bk, ch, vs), wds in verse_words.items():
            books_in_scroll[bk][(ch, vs)] = wds

        for book_name, verses_dict in books_in_scroll.items():
            n_verses = len(verses_dict)
            n_words = sum(len(wds) for wds in verses_dict.values())

            scroll_info.append({
                "scroll": scroll_name,
                "book": book_name,
                "n_verses_dss": n_verses,
                "n_words_dss": n_words,
                "verse_keys": sorted(verses_dict.keys())
            })

        now = time.time()
        if now - last_log > 30:
            log.info(f"  Scanned {all_scrolls.index(scroll_node)+1}/{len(all_scrolls)} scrolls...")
            last_log = now

    log.info(f"Found {len(scroll_info)} scroll-book combinations with verse mapping")

    # ── Get WLC book data ──────────────────────────────────────────────
    log.info("Extracting WLC book data...")
    bhsa_books = list(F_bhsa.otype.s("book"))
    bhsa_book_data = {}

    for book_node in bhsa_books:
        book_name = F_bhsa.book.v(book_node)
        verses = L_bhsa.d(book_node, otype="verse")
        verse_data = {}
        for v_node in verses:
            ch = F_bhsa.chapter.v(v_node)
            vs = F_bhsa.verse.v(v_node)
            words = L_bhsa.d(v_node, otype="word")
            try:
                verse_data[(int(ch), int(vs))] = len(words)
            except (ValueError, TypeError):
                pass
        bhsa_book_data[book_name] = verse_data

    # ── Book name normalization ────────────────────────────────────────
    # DSS and BHSA may use different book names
    dss_to_bhsa_map = {}
    bhsa_names = list(bhsa_book_data.keys())
    bhsa_names_lower = {n.lower(): n for n in bhsa_names}

    for si in scroll_info:
        dss_book = si["book"]
        if dss_book in bhsa_book_data:
            dss_to_bhsa_map[dss_book] = dss_book
        elif dss_book.lower() in bhsa_names_lower:
            dss_to_bhsa_map[dss_book] = bhsa_names_lower[dss_book.lower()]
        else:
            # Try partial matches
            for bn in bhsa_names:
                if dss_book.lower() in bn.lower() or bn.lower() in dss_book.lower():
                    dss_to_bhsa_map[dss_book] = bn
                    break

    log.info(f"Book name mappings found: {len(dss_to_bhsa_map)}")
    for dk, bv in sorted(dss_to_bhsa_map.items()):
        log.info(f"  DSS '{dk}' → BHSA '{bv}'")

    # ── Compare H for each book with sufficient coverage ───────────────
    log.info("Computing H comparisons for books with ≥50% verse coverage...")

    # Aggregate: for each book, merge all scroll fragments
    book_aggregated = defaultdict(lambda: defaultdict(int))
    for si in scroll_info:
        dss_book = si["book"]
        for (ch, vs) in si["verse_keys"]:
            # Count words per verse from the verse_words data
            key = (ch, vs)
            n_words_in_verse = len([w for w in si["verse_keys"] if w == key])
            # We need actual word counts, not just presence
            # Use n_words / n_verses as approximation or recount
            pass

    # Better approach: recount from Text-Fabric directly
    book_dss_verses = defaultdict(lambda: defaultdict(int))
    for si in scroll_info:
        dss_book = si["book"]
        for (ch, vs), wds in zip(si["verse_keys"],
                                  [si["verse_keys"]] * len(si["verse_keys"])):
            pass

    # Actually, let's directly recount from Text-Fabric
    log.info("Recounting DSS words per verse from Text-Fabric...")
    book_verse_wordcounts_dss = defaultdict(lambda: defaultdict(int))

    for scroll_node in all_scrolls:
        words = L_dss.d(scroll_node, otype="word")
        for w in words:
            bib = F_dss.biblical.v(w)
            if not bib or str(bib).lower() in ("0", "false", "none", ""):
                continue
            book = F_dss.book.v(w)
            ch = F_dss.chapter.v(w)
            vs = F_dss.verse.v(w)
            if book and ch and vs:
                try:
                    book_verse_wordcounts_dss[str(book)][(int(ch), int(vs))] += 1
                except (ValueError, TypeError):
                    pass

    log.info(f"DSS books with verse data: {list(book_verse_wordcounts_dss.keys())}")

    comparisons = []
    coverage_summary = []

    for dss_book, dss_verses in sorted(book_verse_wordcounts_dss.items()):
        bhsa_book = dss_to_bhsa_map.get(dss_book)
        if not bhsa_book or bhsa_book not in bhsa_book_data:
            log.info(f"  {dss_book}: no BHSA mapping, skipping")
            coverage_summary.append({
                "dss_book": dss_book,
                "bhsa_book": None,
                "status": "no_bhsa_mapping"
            })
            continue

        bhsa_verses = bhsa_book_data[bhsa_book]
        n_bhsa_total = len(bhsa_verses)

        # Find common verses
        common = sorted(set(dss_verses.keys()) & set(bhsa_verses.keys()))
        coverage = len(common) / n_bhsa_total if n_bhsa_total > 0 else 0.0

        coverage_summary.append({
            "dss_book": dss_book,
            "bhsa_book": bhsa_book,
            "n_dss_verses": len(dss_verses),
            "n_bhsa_verses": n_bhsa_total,
            "n_common_verses": len(common),
            "coverage": round(coverage, 4)
        })

        if coverage < 0.5:
            log.info(f"  {dss_book} ({bhsa_book}): coverage {coverage:.1%} < 50%, skipping")
            continue

        if len(common) < 30:
            log.info(f"  {dss_book} ({bhsa_book}): only {len(common)} common verses, need ≥30, skipping")
            continue

        # Build series
        dss_lengths = [dss_verses[k] for k in common]
        bhsa_lengths = [bhsa_verses[k] for k in common]

        # Compute metrics
        h_dss, r2_dss = hurst_exponent_rs(dss_lengths)
        h_bhsa, r2_bhsa = hurst_exponent_rs(bhsa_lengths)
        a_dss, ar2_dss = dfa_exponent(dss_lengths)
        a_bhsa, ar2_bhsa = dfa_exponent(bhsa_lengths)

        # Statistical tests
        mw_stat, mw_p = stats.mannwhitneyu(dss_lengths, bhsa_lengths, alternative="two-sided")
        ks_stat, ks_p = stats.ks_2samp(dss_lengths, bhsa_lengths)

        delta_h = h_dss - h_bhsa
        delta_a = a_dss - a_bhsa

        result = {
            "dss_book": dss_book,
            "bhsa_book": bhsa_book,
            "n_common_verses": len(common),
            "coverage": round(coverage, 4),
            "dss": {
                "H": round(h_dss, 4) if not np.isnan(h_dss) else None,
                "H_R2": round(r2_dss, 4),
                "alpha": round(a_dss, 4) if not np.isnan(a_dss) else None,
                "alpha_R2": round(ar2_dss, 4),
                "n_words": sum(dss_lengths),
                "mean_verse_len": round(np.mean(dss_lengths), 2)
            },
            "wlc": {
                "H": round(h_bhsa, 4) if not np.isnan(h_bhsa) else None,
                "H_R2": round(r2_bhsa, 4),
                "alpha": round(a_bhsa, 4) if not np.isnan(a_bhsa) else None,
                "alpha_R2": round(ar2_bhsa, 4),
                "n_words": sum(bhsa_lengths),
                "mean_verse_len": round(np.mean(bhsa_lengths), 2)
            },
            "delta_H": round(delta_h, 4) if not np.isnan(delta_h) else None,
            "delta_alpha": round(delta_a, 4) if not np.isnan(delta_a) else None,
            "mann_whitney_p": round(mw_p, 6),
            "ks_p": round(ks_p, 6),
            "statistically_distinguishable": bool(mw_p < 0.05)
        }

        comparisons.append(result)
        log.info(f"  {dss_book} ({bhsa_book}): {len(common)} verses, "
                 f"coverage={coverage:.1%}, "
                 f"H_DSS={h_dss:.4f}, H_WLC={h_bhsa:.4f}, "
                 f"ΔH={delta_h:.4f}, p={mw_p:.4f}")

    # ── Summary ────────────────────────────────────────────────────────
    elapsed = time.time() - t0

    n_compared = len(comparisons)
    n_significant = sum(1 for c in comparisons if c["statistically_distinguishable"])
    avg_delta_h = np.mean([c["delta_H"] for c in comparisons if c["delta_H"] is not None]) if comparisons else float("nan")

    summary = {
        "analysis": "DSS vs WLC comparison across all biblical books with verse coverage ≥ 50%",
        "n_books_compared": n_compared,
        "n_significantly_different": n_significant,
        "avg_abs_delta_H": round(float(np.mean([abs(c["delta_H"]) for c in comparisons if c["delta_H"] is not None])), 4) if comparisons else None,
        "avg_delta_H": round(float(avg_delta_h), 4) if not np.isnan(avg_delta_h) else None,
        "conclusion": (
            f"Of {n_compared} books compared, {n_significant} show statistically significant H differences (p<0.05). "
            f"{'Invariance appears general across books.' if n_significant == 0 else 'Some books show significant differences — see details.'}"
        ),
        "books": comparisons,
        "elapsed_seconds": elapsed
    }

    with open(RESULTS_DIR / "all_books_comparison.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(RESULTS_DIR / "coverage_summary.json", "w") as f:
        json.dump(coverage_summary, f, indent=2, ensure_ascii=False)

    log.info(f"=== Script 2: dss_other_books — DONE ({elapsed:.1f}s) ===")
    log.info(f"Books compared: {n_compared}, significantly different: {n_significant}")
    return summary


if __name__ == "__main__":
    main()
