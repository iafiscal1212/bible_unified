#!/usr/bin/env python3
"""
Fase 13 — Script 4: Apocryphal Gospels
¿Los evangelios excluidos del canon difieren en H de los canónicos?

1. Buscar corpus apócrifos disponibles (Thomas, Peter, Mary, Didache)
2. Si disponibles: calcular H, DFA α, MPS
3. Si no disponibles: proxy analysis (Marcos temprano vs Pastorales tardías)
4. Comparar con evangelios canónicos individuales
"""

import json
import logging
import time
import os
import re
import subprocess
import numpy as np
from pathlib import Path
from scipy import stats as sp_stats
from collections import defaultdict

BASE = Path(__file__).resolve().parent
RESULTS_DIR = BASE / "results" / "apocryphal"
LOG_DIR = BASE / "logs"
BIBLE_JSON = BASE / "bible_unified.json"
SOURCES_DIR = BASE / "sources" / "apocryphal"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
SOURCES_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "fase13_apocryphal.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


# ── Core metrics ─────────────────────────────────────────────────────────

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
    slope, _, r, _, _ = sp_stats.linregress(np.log(sizes), np.log(rs_values))
    return float(slope)


def dfa_exponent(series):
    """Detrended Fluctuation Analysis."""
    series = np.asarray(series, dtype=float)
    n = len(series)
    if n < 20:
        return float("nan")
    y = np.cumsum(series - series.mean())
    sizes = []
    flucts = []
    block = 10
    max_block = n // 4
    while block <= max_block:
        sizes.append(block)
        n_blocks = n // block
        f_list = []
        for i in range(n_blocks):
            seg = y[i * block:(i + 1) * block]
            x = np.arange(block)
            slope, intercept = np.polyfit(x, seg, 1)
            trend = slope * x + intercept
            f_list.append(np.sqrt(np.mean((seg - trend) ** 2)))
        if f_list:
            flucts.append(np.mean(f_list))
        block = int(block * 1.5)
        if block == sizes[-1]:
            block += 1
    if len(sizes) < 3:
        return float("nan")
    slope, _, r, _, _ = sp_stats.linregress(np.log(sizes), np.log(flucts))
    return float(slope)


def autocorr_lag1(series):
    arr = np.asarray(series, dtype=float)
    if len(arr) < 10:
        return float("nan")
    return float(np.corrcoef(arr[:-1], arr[1:])[0, 1])


def compute_mps_significance(series, n_perm=50):
    arr = np.asarray(series, dtype=float)
    n = len(arr)
    if n < 50:
        return None, None
    d = min(9, n // 5)
    if d < 3:
        return None, None
    m = n - d + 1
    traj = np.zeros((m, d))
    for i in range(m):
        traj[i] = arr[i:i + d]
    U, s, Vt = np.linalg.svd(traj, full_matrices=False)
    energy = np.cumsum(s ** 2) / np.sum(s ** 2)
    chi_obs = int(np.searchsorted(energy, 0.99) + 1)
    rng = np.random.default_rng(42)
    chi_rand = []
    for _ in range(n_perm):
        shuffled = arr.copy()
        rng.shuffle(shuffled)
        traj_s = np.zeros((m, d))
        for i in range(m):
            traj_s[i] = shuffled[i:i + d]
        _, s_s, _ = np.linalg.svd(traj_s, full_matrices=False)
        e_s = np.cumsum(s_s ** 2) / np.sum(s_s ** 2)
        chi_rand.append(int(np.searchsorted(e_s, 0.99) + 1))
    p = float(np.mean(np.array(chi_rand) <= chi_obs))
    return chi_obs, p


def compute_full_metrics(lens, label):
    """Compute H, DFA α, AC(1), MPS for a verse-length series."""
    arr = np.asarray(lens, dtype=float)
    h = hurst_exponent_rs(arr)
    alpha = dfa_exponent(arr)
    ac1 = autocorr_lag1(arr)
    chi, p = compute_mps_significance(arr, n_perm=100)
    return {
        "label": label,
        "n_units": len(arr),
        "mean_len": round(float(arr.mean()), 2),
        "std_len": round(float(arr.std()), 2),
        "H": round(float(h), 4) if not np.isnan(h) else None,
        "DFA_alpha": round(float(alpha), 4) if not np.isnan(alpha) else None,
        "AC1": round(float(ac1), 4) if not np.isnan(ac1) else None,
        "MPS_chi": chi,
        "MPS_p": round(float(p), 4) if p is not None else None,
        "MPS_significant": bool(p < 0.05) if p is not None else None,
    }


# ── Corpus search and loading ────────────────────────────────────────────

def try_download(url, dest, timeout=60):
    """Try to download a file, return True if successful."""
    try:
        result = subprocess.run(
            ["wget", "-q", "-O", str(dest), url],
            timeout=timeout, capture_output=True)
        if result.returncode == 0 and dest.exists() and dest.stat().st_size > 100:
            return True
    except Exception:
        pass
    if dest.exists() and dest.stat().st_size <= 100:
        dest.unlink()
    return False


def search_gospel_of_thomas():
    """Search for Gospel of Thomas text."""
    log.info("  Buscando Evangelio de Tomás...")

    # Try scrollmapper bible_databases
    urls = [
        "https://raw.githubusercontent.com/scrollmapper/bible_databases/master/other/gospel_of_thomas.txt",
        "https://raw.githubusercontent.com/scrollmapper/bible_databases/master/csv/gospel_of_thomas.csv",
    ]
    for url in urls:
        dest = SOURCES_DIR / "thomas.txt"
        if try_download(url, dest):
            log.info(f"    Encontrado: {url}")
            return dest, "scrollmapper"

    # Try Perseus
    perseus_urls = [
        "https://raw.githubusercontent.com/PerseusDL/canonical-greekLit/master/data/tlg0032/tlg001/tlg0032.tlg001.perseus-grc1.xml",
    ]
    for url in perseus_urls:
        dest = SOURCES_DIR / "thomas_perseus.xml"
        if try_download(url, dest):
            log.info(f"    Encontrado en Perseus: {url}")
            return dest, "perseus_xml"

    log.info("    No encontrado")
    return None, None


def search_gospel_of_peter():
    """Search for Gospel of Peter text."""
    log.info("  Buscando Evangelio de Pedro...")
    urls = [
        "https://raw.githubusercontent.com/PerseusDL/canonical-greekLit/master/data/tlg0031/tlg026/tlg0031.tlg026.perseus-grc1.xml",
    ]
    for url in urls:
        dest = SOURCES_DIR / "peter_perseus.xml"
        if try_download(url, dest):
            log.info(f"    Encontrado: {url}")
            return dest, "perseus_xml"
    log.info("    No encontrado")
    return None, None


def search_didache():
    """Search for Didache text."""
    log.info("  Buscando Didaché...")
    urls = [
        "https://raw.githubusercontent.com/PerseusDL/canonical-greekLit/master/data/tlg1311/tlg001/tlg1311.tlg001.perseus-grc1.xml",
    ]
    for url in urls:
        dest = SOURCES_DIR / "didache_perseus.xml"
        if try_download(url, dest):
            log.info(f"    Encontrado: {url}")
            return dest, "perseus_xml"
    log.info("    No encontrado")
    return None, None


def parse_plain_text_logia(fpath):
    """Parse plain text file with numbered logia/sections.
    Each logion = 1 unit. Count words per logion."""
    lens = []
    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
        current_words = 0
        for line in f:
            line = line.strip()
            if not line:
                if current_words > 0:
                    lens.append(current_words)
                    current_words = 0
                continue
            # Check for logion/saying number
            if re.match(r'^\d+\.?\s', line) or re.match(r'^\(\d+\)', line):
                if current_words > 0:
                    lens.append(current_words)
                current_words = len(line.split())
            else:
                current_words += len(line.split())
        if current_words > 0:
            lens.append(current_words)
    return np.array(lens, dtype=float) if lens else None


def parse_perseus_xml_sentences(fpath):
    """Parse Perseus XML, extract sentences/verses by splitting on periods."""
    try:
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception:
        return None

    # Strip XML tags
    text = re.sub(r'<[^>]+>', ' ', content)
    text = re.sub(r'\s+', ' ', text).strip()

    # Split by sentence-ending punctuation
    sentences = re.split(r'[.;·]', text)
    lens = []
    for s in sentences:
        words = s.strip().split()
        if len(words) >= 2:
            lens.append(len(words))
    return np.array(lens, dtype=float) if lens else None


# ── Canonical gospel loading ─────────────────────────────────────────────

def load_canonical_gospels():
    """Load canonical gospel verse lengths from bible_unified.json."""
    log.info("Cargando evangelios canónicos...")
    with open(BIBLE_JSON, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    canonical = ["Matthew", "Mark", "Luke", "John"]
    early_nt = ["Mark"]  # ~70 CE
    late_nt = ["1 Timothy", "2 Timothy", "Titus"]  # ~100-120 CE (Pastorals)
    all_nt_books = set()

    book_verses = defaultdict(lambda: defaultdict(int))
    for w in corpus:
        book = w.get("book", "")
        key = (w.get("chapter", 0), w.get("verse", 0))
        book_verses[book][key] += 1
        if book in {"Matthew", "Mark", "Luke", "John", "Acts",
                     "Romans", "1 Corinthians", "2 Corinthians", "Galatians",
                     "Ephesians", "Philippians", "Colossians",
                     "1 Thessalonians", "2 Thessalonians",
                     "1 Timothy", "2 Timothy", "Titus", "Philemon",
                     "Hebrews", "James", "1 Peter", "2 Peter",
                     "1 John", "2 John", "3 John", "Jude", "Revelation"}:
            all_nt_books.add(book)

    results = {}
    for book in canonical + early_nt + late_nt:
        if book in book_verses and book not in results:
            bv = book_verses[book]
            lens = np.array([bv[k] for k in sorted(bv.keys())], dtype=float)
            results[book] = {
                "lens": lens,
                "metrics": compute_full_metrics(lens, book),
            }
            log.info(f"  {book}: {len(lens)} versículos, "
                     f"H={results[book]['metrics']['H']}")

    return results, book_verses


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    log.info("=" * 70)
    log.info("FASE 13 — Script 4: Apocryphal Gospels")
    log.info("=" * 70)

    # Load canonical
    canonical_data, all_book_verses = load_canonical_gospels()

    # 1. Search for apocryphal texts
    log.info("\n=== PARTE 1: Búsqueda de corpus apócrifos ===")
    availability = {}
    apocryphal_data = {}

    # Gospel of Thomas
    thomas_path, thomas_fmt = search_gospel_of_thomas()
    if thomas_path:
        if thomas_fmt == "scrollmapper":
            thomas_lens = parse_plain_text_logia(thomas_path)
        elif thomas_fmt == "perseus_xml":
            thomas_lens = parse_perseus_xml_sentences(thomas_path)
        else:
            thomas_lens = None

        if thomas_lens is not None and len(thomas_lens) >= 20:
            availability["Gospel_of_Thomas"] = {
                "found": True, "source": thomas_fmt,
                "n_units": len(thomas_lens),
                "unit_type": "logion" if thomas_fmt == "scrollmapper" else "sentence",
            }
            apocryphal_data["Gospel_of_Thomas"] = {
                "lens": thomas_lens,
                "metrics": compute_full_metrics(thomas_lens, "Gospel of Thomas"),
            }
            log.info(f"  Thomas: {len(thomas_lens)} unidades")
        else:
            availability["Gospel_of_Thomas"] = {
                "found": True, "source": thomas_fmt,
                "usable": False, "reason": "too_short or parse_failed",
            }
    else:
        availability["Gospel_of_Thomas"] = {"found": False}

    # Gospel of Peter
    peter_path, peter_fmt = search_gospel_of_peter()
    if peter_path:
        peter_lens = parse_perseus_xml_sentences(peter_path)
        if peter_lens is not None and len(peter_lens) >= 20:
            availability["Gospel_of_Peter"] = {
                "found": True, "source": peter_fmt,
                "n_units": len(peter_lens),
                "note": "fragmentary (~60 verses)",
            }
            apocryphal_data["Gospel_of_Peter"] = {
                "lens": peter_lens,
                "metrics": compute_full_metrics(peter_lens, "Gospel of Peter"),
            }
        else:
            availability["Gospel_of_Peter"] = {
                "found": True, "usable": False,
                "reason": "too_short or parse_failed",
            }
    else:
        availability["Gospel_of_Peter"] = {"found": False}

    # Didache
    didache_path, didache_fmt = search_didache()
    if didache_path:
        didache_lens = parse_perseus_xml_sentences(didache_path)
        if didache_lens is not None and len(didache_lens) >= 20:
            availability["Didache"] = {
                "found": True, "source": didache_fmt,
                "n_units": len(didache_lens),
            }
            apocryphal_data["Didache"] = {
                "lens": didache_lens,
                "metrics": compute_full_metrics(didache_lens, "Didache"),
            }
        else:
            availability["Didache"] = {
                "found": True, "usable": False,
                "reason": "too_short or parse_failed",
            }
    else:
        availability["Didache"] = {"found": False}

    # Gospel of Mary — unlikely to find, document
    availability["Gospel_of_Mary"] = {
        "found": False,
        "note": "Primarily Coptic with Greek fragments; no digital corpus found",
    }

    with open(RESULTS_DIR / "corpus_availability.json", "w") as f:
        json.dump(availability, f, indent=2, ensure_ascii=False)
    log.info(f"  Disponibilidad guardada. Apócrifos encontrados: {len(apocryphal_data)}")

    # 2. Compare canonical vs apocryphal
    log.info("\n=== PARTE 2: Comparación H ===")
    comparison = {"canonical": {}, "apocryphal": {}}

    for book, data in canonical_data.items():
        comparison["canonical"][book] = data["metrics"]

    for name, data in apocryphal_data.items():
        comparison["apocryphal"][name] = data["metrics"]

    # Cluster assignment
    for section in ["canonical", "apocryphal"]:
        for name, metrics in comparison[section].items():
            h = metrics.get("H")
            mps_sig = metrics.get("MPS_significant")
            if h is not None:
                if h > 0.85 and mps_sig:
                    cluster = "AT-like"
                elif h > 0.85:
                    cluster = "high-H (no MPS)"
                elif h > 0.7:
                    cluster = "intermediate"
                else:
                    cluster = "NT-like"
                metrics["cluster"] = cluster

    with open(RESULTS_DIR / "h_comparison_canonical_vs_apocryphal.json", "w") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    # 3. Proxy analysis: early vs late canonical
    log.info("\n=== PARTE 3: Proxy analysis (temprano vs tardío) ===")
    proxy = {}

    # Mark (~70 CE) vs Pastorals (~100-120 CE)
    early_books = ["Mark"]
    late_books = ["1 Timothy", "2 Timothy", "Titus"]

    for book in early_books + late_books:
        if book in canonical_data:
            proxy[book] = canonical_data[book]["metrics"]
            proxy[book]["era"] = "early (~70 CE)" if book in early_books else "late (~100-120 CE)"

    # Combined pastorals
    pastoral_lens = []
    for book in late_books:
        if book in all_book_verses:
            bv = all_book_verses[book]
            pastoral_lens.extend([bv[k] for k in sorted(bv.keys())])
    if pastoral_lens:
        proxy["Pastorals_combined"] = compute_full_metrics(
            np.array(pastoral_lens, dtype=float), "Pastorals combined")
        proxy["Pastorals_combined"]["era"] = "late (~100-120 CE)"

    # Statistical comparison
    if "Mark" in canonical_data and pastoral_lens:
        mark_lens = canonical_data["Mark"]["lens"]
        past_arr = np.array(pastoral_lens, dtype=float)

        # Compare verse length distributions
        ks_stat, ks_p = sp_stats.ks_2samp(mark_lens, past_arr)
        proxy["early_vs_late_test"] = {
            "Mark_H": proxy.get("Mark", {}).get("H"),
            "Pastorals_H": proxy.get("Pastorals_combined", {}).get("H"),
            "delta_H": round(
                (proxy.get("Mark", {}).get("H") or 0) -
                (proxy.get("Pastorals_combined", {}).get("H") or 0), 4),
            "KS_statistic": round(float(ks_stat), 4),
            "KS_p": round(float(ks_p), 6),
            "length_dist_different": bool(ks_p < 0.05),
        }
        log.info(f"  Mark H={proxy.get('Mark', {}).get('H')}, "
                 f"Pastorals H={proxy.get('Pastorals_combined', {}).get('H')}")

    # Also do all 4 gospel comparison
    gospel_metrics = []
    for book in ["Matthew", "Mark", "Luke", "John"]:
        if book in canonical_data:
            m = canonical_data[book]["metrics"]
            gospel_metrics.append({
                "book": book,
                "H": m.get("H"),
                "DFA_alpha": m.get("DFA_alpha"),
                "AC1": m.get("AC1"),
                "MPS_p": m.get("MPS_p"),
                "n_verses": m.get("n_units"),
            })
    proxy["canonical_gospels_summary"] = gospel_metrics

    with open(RESULTS_DIR / "proxy_analysis_if_needed.json", "w") as f:
        json.dump(proxy, f, indent=2, ensure_ascii=False)

    # Summary
    log.info("\n=== RESUMEN ===")
    n_apocryphal_found = sum(1 for v in availability.values() if v.get("found"))
    n_usable = len(apocryphal_data)
    log.info(f"  Apócrifos buscados: {len(availability)}")
    log.info(f"  Encontrados: {n_apocryphal_found}")
    log.info(f"  Usables para análisis: {n_usable}")

    if n_usable == 0:
        log.info("  → Usando proxy analysis (Marcos temprano vs Pastorales tardías)")

    elapsed = time.time() - t0
    log.info(f"\n{'=' * 70}")
    log.info(f"Script 4 completado en {elapsed:.1f}s")


if __name__ == "__main__":
    main()
