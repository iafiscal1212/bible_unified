#!/usr/bin/env python3
"""
Fase 14 — Script 4: Excluded Canon
Textos cristianos tempranos excluidos del canon: Didaché, Pastor de Hermas, Justino Mártir.

Fuente: First1KGreek (OpenGreekAndLatin/First1KGreek en GitHub)
- Didaché: TLG1311/tlg001
- Pastor de Hermas: TLG1419/tlg001
- Justino Mártir (1ª Apología): TLG0645/tlg001

Parsear XML TEI epidoc → secuencia de longitudes por sección/capítulo.
Comparar H, AC1, DFA con NT canónico.
"""

import json
import logging
import time
import re
import subprocess
import sys
import os
import numpy as np
from pathlib import Path
from scipy import stats as sp_stats
from collections import defaultdict
import xml.etree.ElementTree as ET

BASE = Path(__file__).resolve().parent
RESULTS_DIR = BASE / "results" / "excluded_canon"
LOG_DIR = BASE / "logs"
BIBLE_JSON = BASE / "bible_unified.json"
DOWNLOAD_DIR = Path("/tmp/first1kgreek")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "fase14_excluded_canon.log"),
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


def count_greek_words(text):
    """Count words in Greek text (split on whitespace/punctuation)."""
    words = re.findall(r'[\u0370-\u03FF\u1F00-\u1FFF]+', text)
    return len(words)


# ── Download First1KGreek texts ─────────────────────────────────────

TEXTS = {
    "Didache": {
        "tlg": "tlg1311",
        "work": "tlg001",
        "name": "Didaché",
        "urls": [
            "https://raw.githubusercontent.com/OpenGreekAndLatin/First1KGreek/master/data/{tlg}/tlg001/__cts__.xml",
            "https://raw.githubusercontent.com/OpenGreekAndLatin/First1KGreek/master/data/{tlg}/{work}/{tlg}.{work}.1st1K-grc1.xml",
            "https://raw.githubusercontent.com/OpenGreekAndLatin/First1KGreek/master/data/{tlg}/{work}/{tlg}.{work}.opp-grc1.xml",
        ],
    },
    "Hermas": {
        "tlg": "tlg1419",
        "work": "tlg001",
        "name": "Pastor de Hermas",
        "urls": [
            "https://raw.githubusercontent.com/OpenGreekAndLatin/First1KGreek/master/data/{tlg}/{work}/{tlg}.{work}.1st1K-grc1.xml",
            "https://raw.githubusercontent.com/OpenGreekAndLatin/First1KGreek/master/data/{tlg}/{work}/{tlg}.{work}.opp-grc1.xml",
        ],
    },
    "JustinMartyr": {
        "tlg": "tlg0645",
        "work": "tlg001",
        "name": "Justino Mártir — 1ª Apología",
        "urls": [
            "https://raw.githubusercontent.com/OpenGreekAndLatin/First1KGreek/master/data/{tlg}/{work}/{tlg}.{work}.1st1K-grc1.xml",
            "https://raw.githubusercontent.com/OpenGreekAndLatin/First1KGreek/master/data/{tlg}/{work}/{tlg}.{work}.opp-grc1.xml",
        ],
    },
}


def download_text(key, info):
    """Download text from First1KGreek, trying multiple URL patterns."""
    dl_dir = DOWNLOAD_DIR / key
    dl_dir.mkdir(parents=True, exist_ok=True)

    tlg, work = info["tlg"], info["work"]

    # Try direct URL patterns
    for url_template in info["urls"]:
        url = url_template.format(tlg=tlg, work=work)
        out_file = dl_dir / url.split("/")[-1]
        if out_file.exists() and out_file.stat().st_size > 100:
            log.info(f"  {key}: ya existe {out_file.name}")
            return out_file

        log.info(f"  {key}: descargando {url}")
        try:
            result = subprocess.run(
                ["wget", "-q", "-O", str(out_file), url],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0 and out_file.exists() and out_file.stat().st_size > 100:
                log.info(f"  {key}: descargado ({out_file.stat().st_size} bytes)")
                return out_file
        except Exception as e:
            log.warning(f"  {key}: error descargando {url}: {e}")

    # Try cloning the entire repo (shallow) if direct download fails
    repo_dir = DOWNLOAD_DIR / "First1KGreek"
    if not repo_dir.exists():
        log.info(f"  Clonando First1KGreek (sparse)...")
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
                 "https://github.com/OpenGreekAndLatin/First1KGreek.git",
                 str(repo_dir)],
                capture_output=True, text=True, timeout=120, cwd=str(DOWNLOAD_DIR)
            )
            # Sparse checkout the data dirs we need
            for t in TEXTS.values():
                sparse_path = f"data/{t['tlg']}/{t['work']}"
                subprocess.run(
                    ["git", "sparse-checkout", "add", sparse_path],
                    capture_output=True, text=True, timeout=30, cwd=str(repo_dir)
                )
        except Exception as e:
            log.warning(f"  Error clonando First1KGreek: {e}")

    # Search cloned repo for files
    if repo_dir.exists():
        import glob as gl
        pattern = str(repo_dir / "data" / tlg / work / "*.xml")
        files = sorted(gl.glob(pattern))
        for f in files:
            if "cts" not in os.path.basename(f).lower():
                log.info(f"  {key}: encontrado en repo: {os.path.basename(f)}")
                return Path(f)

    log.warning(f"  {key}: NO encontrado")
    return None


# ── XML TEI epidoc parsing ──────────────────────────────────────────

def parse_tei_xml(filepath, key):
    """Parse TEI XML and extract text sections with word counts."""
    log.info(f"  Parseando {key}: {filepath.name}")

    # Read file
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Remove namespace for easier parsing
    content = re.sub(r'\sxmlns[^"]*"[^"]*"', '', content)
    content = re.sub(r'\sxmlns="[^"]*"', '', content)

    try:
        root = ET.fromstring(content)
    except ET.ParseError as e:
        log.warning(f"  {key}: XML parse error: {e}")
        # Try to fix common issues
        content = re.sub(r'&(?!amp;|lt;|gt;|apos;|quot;)', '&amp;', content)
        try:
            root = ET.fromstring(content)
        except ET.ParseError:
            log.error(f"  {key}: cannot parse XML")
            return None

    # Extract text from <body> or <text>
    body = root.find(".//body")
    if body is None:
        body = root.find(".//text")
    if body is None:
        body = root

    # Strategy 1: Extract by <div> sections (chapters/sections)
    section_lens = []
    divs = body.findall(".//div")
    if not divs:
        divs = body.findall(".//div1")
    if not divs:
        divs = body.findall(".//div2")

    if divs:
        for div in divs:
            text = ET.tostring(div, encoding="unicode", method="text")
            text = text.strip()
            if len(text) > 10:
                wc = count_greek_words(text)
                if wc > 0:
                    section_lens.append(wc)

    # Strategy 2: If no divs, extract by <p> paragraphs
    if not section_lens:
        for p in body.findall(".//p"):
            text = ET.tostring(p, encoding="unicode", method="text").strip()
            if len(text) > 5:
                wc = count_greek_words(text)
                if wc > 0:
                    section_lens.append(wc)

    # Strategy 3: If still nothing, extract by <l> (lines) or <seg>
    if not section_lens:
        for tag in ["l", "seg", "s", "ab"]:
            for elem in body.findall(f".//{tag}"):
                text = ET.tostring(elem, encoding="unicode", method="text").strip()
                if len(text) > 3:
                    wc = count_greek_words(text)
                    if wc > 0:
                        section_lens.append(wc)
            if section_lens:
                break

    # Strategy 4: Split full text into sentences
    if not section_lens:
        full_text = ET.tostring(body, encoding="unicode", method="text")
        sentences = re.split(r'[.;·]+', full_text)
        for sent in sentences:
            sent = sent.strip()
            wc = count_greek_words(sent)
            if wc >= 3:
                section_lens.append(wc)

    log.info(f"  {key}: {len(section_lens)} secciones extraídas")
    return np.array(section_lens, dtype=float) if section_lens else None


# ── NT canonical reference ───────────────────────────────────────────

def load_nt_reference():
    """Load NT canonical stats for comparison."""
    with open(BIBLE_JSON, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    NT_BOOKS = {"Matthew", "Mark", "Luke", "John", "Acts",
                "Romans", "1 Corinthians", "2 Corinthians", "Galatians",
                "Ephesians", "Philippians", "Colossians",
                "1 Thessalonians", "2 Thessalonians",
                "1 Timothy", "2 Timothy", "Titus", "Philemon",
                "Hebrews", "James", "1 Peter", "2 Peter",
                "1 John", "2 John", "3 John", "Jude", "Revelation"}

    book_verses = defaultdict(lambda: defaultdict(int))
    for w in corpus:
        book = w.get("book", "")
        if book in NT_BOOKS:
            key = (book, w.get("chapter", 0), w.get("verse", 0))
            book_verses[book][key] += 1

    nt_stats = {}
    for book in sorted(NT_BOOKS):
        if book in book_verses:
            lens = np.array([book_verses[book][k]
                             for k in sorted(book_verses[book].keys())], dtype=float)
            if len(lens) >= 20:
                h = hurst_exponent_rs(lens)
                ac1 = autocorr_lag1(lens)
                dfa = dfa_exponent(lens) if len(lens) >= 50 else float("nan")
                nt_stats[book] = {
                    "n_verses": len(lens),
                    "H": round(h, 4) if not np.isnan(h) else None,
                    "AC1": round(ac1, 4) if not np.isnan(ac1) else None,
                    "DFA": round(dfa, 4) if not np.isnan(dfa) else None,
                    "mean_len": round(float(lens.mean()), 2),
                }

    # Global NT stats
    all_nt = []
    for book in sorted(NT_BOOKS):
        if book in book_verses:
            for k in sorted(book_verses[book].keys()):
                all_nt.append(book_verses[book][k])
    nt_all = np.array(all_nt, dtype=float)
    nt_global = {
        "n_verses": len(nt_all),
        "H": round(hurst_exponent_rs(nt_all), 4),
        "AC1": round(autocorr_lag1(nt_all), 4),
        "DFA": round(dfa_exponent(nt_all), 4),
        "mean_len": round(float(nt_all.mean()), 2),
    }

    return nt_stats, nt_global


# ── Main ───────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    log.info("=" * 70)
    log.info("FASE 14 — Script 4: Excluded Canon")
    log.info("=" * 70)

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # Download texts
    log.info("\n=== Descargando textos ===")
    downloaded = {}
    for key, info in TEXTS.items():
        fpath = download_text(key, info)
        downloaded[key] = fpath

    # Parse downloaded texts
    log.info("\n=== Parseando textos ===")
    parsed = {}
    availability = {}
    for key, info in TEXTS.items():
        fpath = downloaded.get(key)
        if fpath is not None and fpath.exists():
            series = parse_tei_xml(fpath, key)
            if series is not None and len(series) >= 10:
                parsed[key] = series
                availability[key] = {
                    "found": True,
                    "name": info["name"],
                    "file": str(fpath),
                    "n_sections": len(series),
                    "mean_words": round(float(series.mean()), 2),
                }
                log.info(f"  {key}: {len(series)} secciones, "
                         f"mean={series.mean():.1f} words")
            else:
                n = len(series) if series is not None else 0
                availability[key] = {
                    "found": True,
                    "name": info["name"],
                    "n_sections": n,
                    "note": "too few sections for analysis",
                }
                log.warning(f"  {key}: solo {n} secciones — insuficiente")
        else:
            availability[key] = {
                "found": False,
                "name": info["name"],
                "note": "file not found",
            }
            log.warning(f"  {key}: archivo no encontrado")

    with open(RESULTS_DIR / "corpus_availability.json", "w") as f:
        json.dump(availability, f, indent=2, ensure_ascii=False)

    # Load NT reference
    log.info("\n=== Cargando referencia NT ===")
    nt_book_stats, nt_global = load_nt_reference()
    log.info(f"  NT global: H={nt_global['H']}, AC1={nt_global['AC1']}")

    # Analyze excluded texts
    log.info("\n=== Análisis de textos excluidos ===")
    excluded_stats = {}
    for key, series in parsed.items():
        h = hurst_exponent_rs(series)
        ac1 = autocorr_lag1(series)
        dfa = dfa_exponent(series)
        cv_val = float(series.std() / series.mean()) if series.mean() > 0 else 0

        excluded_stats[key] = {
            "name": TEXTS[key]["name"],
            "n_sections": len(series),
            "H": round(h, 4) if not np.isnan(h) else None,
            "AC1": round(ac1, 4) if not np.isnan(ac1) else None,
            "DFA": round(dfa, 4) if not np.isnan(dfa) else None,
            "CV": round(cv_val, 4),
            "mean_len": round(float(series.mean()), 2),
        }
        log.info(f"  {key}: H={h:.3f}, AC1={ac1:.3f}")

    # Comparison with NT
    log.info("\n=== Comparación con NT canónico ===")
    comparison = {}
    for key, stats in excluded_stats.items():
        if stats["H"] is None:
            continue
        # Find closest NT book by H
        closest_book = None
        min_dist = 1e10
        for nt_book, nt_s in nt_book_stats.items():
            if nt_s.get("H") is not None:
                dist = abs(stats["H"] - nt_s["H"])
                if dist < min_dist:
                    min_dist = dist
                    closest_book = nt_book

        # NT range
        nt_hs = [s["H"] for s in nt_book_stats.values() if s.get("H") is not None]
        nt_ac1s = [s["AC1"] for s in nt_book_stats.values() if s.get("AC1") is not None]

        in_nt_range = bool(min(nt_hs) <= stats["H"] <= max(nt_hs)) if nt_hs else None
        ac1_in_range = (bool(min(nt_ac1s) <= stats["AC1"] <= max(nt_ac1s))
                        if nt_ac1s and stats["AC1"] is not None else None)

        comparison[key] = {
            "name": TEXTS[key]["name"],
            "H": stats["H"],
            "AC1": stats["AC1"],
            "closest_NT_book": closest_book,
            "closest_NT_H": nt_book_stats[closest_book]["H"] if closest_book else None,
            "delta_H_from_NT_mean": round(stats["H"] - nt_global["H"], 4),
            "in_NT_H_range": in_nt_range,
            "in_NT_AC1_range": ac1_in_range,
            "NT_H_range": [round(min(nt_hs), 4), round(max(nt_hs), 4)] if nt_hs else None,
            "classification": ("NT-like" if in_nt_range else
                               "AT-like" if stats["H"] and stats["H"] > max(nt_hs) else
                               "anomalous"),
        }
        log.info(f"  {key}: {comparison[key]['classification']} "
                 f"(closest={closest_book}, ΔH={comparison[key]['delta_H_from_NT_mean']:.3f})")

    with open(RESULTS_DIR / "excluded_vs_canonical.json", "w") as f:
        json.dump({
            "excluded": excluded_stats,
            "nt_global": nt_global,
            "comparison": comparison,
        }, f, indent=2, ensure_ascii=False)

    # Proxy analysis if not enough excluded texts found
    log.info("\n=== Análisis proxy (si textos insuficientes) ===")
    n_analyzed = len(excluded_stats)
    proxy = {}
    if n_analyzed < 2:
        log.info("  Insuficientes textos excluidos — usando proxy canónico")

        # Group NT by early (pre-70 CE) vs late (post-100 CE)
        early = ["Mark", "1 Thessalonians", "Galatians", "1 Corinthians",
                 "2 Corinthians", "Romans", "Philippians", "Philemon"]
        late = ["1 Timothy", "2 Timothy", "Titus", "2 Peter", "Jude"]
        disputed = ["Ephesians", "Colossians", "2 Thessalonians", "Hebrews"]

        for group_name, group_books in [("early_NT", early), ("late_NT", late),
                                         ("disputed_NT", disputed)]:
            hs = [nt_book_stats[b]["H"] for b in group_books
                  if b in nt_book_stats and nt_book_stats[b].get("H") is not None]
            ac1s = [nt_book_stats[b]["AC1"] for b in group_books
                    if b in nt_book_stats and nt_book_stats[b].get("AC1") is not None]
            proxy[group_name] = {
                "books": group_books,
                "n_with_H": len(hs),
                "H_mean": round(float(np.mean(hs)), 4) if hs else None,
                "H_std": round(float(np.std(hs)), 4) if hs else None,
                "AC1_mean": round(float(np.mean(ac1s)), 4) if ac1s else None,
            }
            if hs:
                log.info(f"  {group_name}: H={np.mean(hs):.3f}±{np.std(hs):.3f} "
                         f"({len(hs)} books)")

        # Test: early vs late
        early_hs = [nt_book_stats[b]["H"] for b in early
                    if b in nt_book_stats and nt_book_stats[b].get("H") is not None]
        late_hs = [nt_book_stats[b]["H"] for b in late
                   if b in nt_book_stats and nt_book_stats[b].get("H") is not None]
        if len(early_hs) >= 2 and len(late_hs) >= 2:
            stat, p = sp_stats.mannwhitneyu(early_hs, late_hs, alternative="two-sided")
            proxy["early_vs_late_test"] = {
                "statistic": round(float(stat), 4),
                "p_value": round(float(p), 4),
                "significant": bool(p < 0.05),
                "early_H_mean": round(float(np.mean(early_hs)), 4),
                "late_H_mean": round(float(np.mean(late_hs)), 4),
                "delta_H": round(float(np.mean(early_hs) - np.mean(late_hs)), 4),
            }
            log.info(f"  Early vs Late: p={p:.4f}, ΔH={np.mean(early_hs) - np.mean(late_hs):.3f}")

    with open(RESULTS_DIR / "proxy_analysis.json", "w") as f:
        json.dump(proxy, f, indent=2, ensure_ascii=False)

    # Verdict
    log.info("\n=== VEREDICTO ===")
    verdict = {
        "n_texts_downloaded": sum(1 for v in availability.values() if v.get("found")),
        "n_texts_analyzed": n_analyzed,
        "excluded_texts": list(excluded_stats.keys()),
    }

    if excluded_stats:
        # Are excluded texts NT-like or AT-like?
        classifications = [c.get("classification", "unknown") for c in comparison.values()]
        nt_like = sum(1 for c in classifications if c == "NT-like")
        verdict["n_NT_like"] = nt_like
        verdict["n_AT_like"] = sum(1 for c in classifications if c == "AT-like")
        verdict["n_anomalous"] = sum(1 for c in classifications if c == "anomalous")
        verdict["conclusion"] = (
            f"{nt_like}/{len(classifications)} textos excluidos son NT-like en H. "
            f"Exclusión canónica {'NO' if nt_like == len(classifications) else ''} "
            f"correlaciona con diferencias en estructura de memoria."
        )
    else:
        verdict["conclusion"] = ("Textos no disponibles digitalmente en formato parseable. "
                                 "Proxy early/late NT analizado.")

    if proxy.get("early_vs_late_test"):
        verdict["proxy_conclusion"] = (
            f"NT temprano vs tardío: ΔH={proxy['early_vs_late_test']['delta_H']:.3f}, "
            f"p={proxy['early_vs_late_test']['p_value']:.4f}"
        )

    log.info(f"  {verdict.get('conclusion', '')}")
    if "proxy_conclusion" in verdict:
        log.info(f"  {verdict['proxy_conclusion']}")

    with open(RESULTS_DIR / "verdict.json", "w") as f:
        json.dump(verdict, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    log.info(f"\n{'=' * 70}")
    log.info(f"Script 4 completado en {elapsed:.1f}s")


if __name__ == "__main__":
    main()
