#!/usr/bin/env python3
"""
Fase 13 — Script 3: OT Quotes in NT
¿Las citas del AT en el NT heredan estructura AT-like localmente?

1. Identificar versículos NT que citan AT (TAGNT + cross_references)
2. H_local en ventanas ±10 con citas vs sin citas
3. Libros NT con citas densas que elevan H_local
4. Bonus: longitud de cita NT vs versículo AT original
"""

import json
import logging
import time
import re
import os
import subprocess
import numpy as np
from pathlib import Path
from scipy import stats as sp_stats
from collections import defaultdict

BASE = Path(__file__).resolve().parent
RESULTS_DIR = BASE / "results" / "ot_quotes"
LOG_DIR = BASE / "logs"
BIBLE_JSON = BASE / "bible_unified.json"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "fase13_ot_quotes.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

NT_BOOKS = {"Matthew", "Mark", "Luke", "John", "Acts",
            "Romans", "1 Corinthians", "2 Corinthians", "Galatians",
            "Ephesians", "Philippians", "Colossians",
            "1 Thessalonians", "2 Thessalonians",
            "1 Timothy", "2 Timothy", "Titus", "Philemon",
            "Hebrews", "James", "1 Peter", "2 Peter",
            "1 John", "2 John", "3 John", "Jude", "Revelation"}

OT_BOOKS = {"Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy",
            "Joshua", "Judges", "Ruth", "1 Samuel", "2 Samuel",
            "1 Kings", "2 Kings", "1 Chronicles", "2 Chronicles",
            "Ezra", "Nehemiah", "Esther", "Job", "Psalms",
            "Proverbs", "Ecclesiastes", "Song of Solomon",
            "Isaiah", "Jeremiah", "Lamentations", "Ezekiel",
            "Daniel", "Hosea", "Joel", "Amos", "Obadiah",
            "Jonah", "Micah", "Nahum", "Habakkuk", "Zephaniah",
            "Haggai", "Zechariah", "Malachi"}

# Book name normalization for cross-references
BOOK_ABBREVS = {
    # scrollmapper/openbible format
    "Gen": "Genesis", "Exod": "Exodus", "Lev": "Leviticus",
    "Num": "Numbers", "Deut": "Deuteronomy", "Josh": "Joshua",
    "Judg": "Judges", "Ruth": "Ruth", "1Sam": "1 Samuel", "2Sam": "2 Samuel",
    "1Kgs": "1 Kings", "2Kgs": "2 Kings", "1Chr": "1 Chronicles",
    "2Chr": "2 Chronicles", "Ezra": "Ezra", "Neh": "Nehemiah",
    "Esth": "Esther", "Job": "Job", "Ps": "Psalms", "Prov": "Proverbs",
    "Eccl": "Ecclesiastes", "Song": "Song of Solomon",
    "Isa": "Isaiah", "Jer": "Jeremiah", "Lam": "Lamentations",
    "Ezek": "Ezekiel", "Dan": "Daniel", "Hos": "Hosea", "Joel": "Joel",
    "Amos": "Amos", "Obad": "Obadiah", "Jonah": "Jonah", "Mic": "Micah",
    "Nah": "Nahum", "Hab": "Habakkuk", "Zeph": "Zephaniah",
    "Hag": "Haggai", "Zech": "Zechariah", "Mal": "Malachi",
    "Matt": "Matthew", "Mark": "Mark", "Luke": "Luke", "John": "John",
    "Acts": "Acts", "Rom": "Romans", "1Cor": "1 Corinthians",
    "2Cor": "2 Corinthians", "Gal": "Galatians", "Eph": "Ephesians",
    "Phil": "Philippians", "Col": "Colossians",
    "1Thess": "1 Thessalonians", "2Thess": "2 Thessalonians",
    "1Tim": "1 Timothy", "2Tim": "2 Timothy", "Titus": "Titus",
    "Phlm": "Philemon", "Heb": "Hebrews", "Jas": "James",
    "1Pet": "1 Peter", "2Pet": "2 Peter", "1John": "1 John",
    "2John": "2 John", "3John": "3 John", "Jude": "Jude", "Rev": "Revelation",
    # TAGNT format aliases
    "Mat": "Matthew", "Mar": "Mark", "Luk": "Luke", "Joh": "John",
    "Act": "Acts", "1Co": "1 Corinthians", "2Co": "2 Corinthians",
    "Phi": "Philippians", "1Th": "1 Thessalonians", "2Th": "2 Thessalonians",
    "1Ti": "1 Timothy", "2Ti": "2 Timothy", "Tit": "Titus",
    "Phm": "Philemon", "Jam": "James", "1Pe": "1 Peter", "2Pe": "2 Peter",
    "1Jo": "1 John", "2Jo": "2 John", "3Jo": "3 John", "Jud": "Jude",
    "Psa": "Psalms", "Pro": "Proverbs", "Ecc": "Ecclesiastes",
    "Exo": "Exodus", "Deu": "Deuteronomy", "Jos": "Joshua",
    "Jdg": "Judges", "Rut": "Ruth", "1Sa": "1 Samuel", "2Sa": "2 Samuel",
    "1Ki": "1 Kings", "2Ki": "2 Kings", "1Ch": "1 Chronicles",
    "2Ch": "2 Chronicles", "Ezr": "Ezra", "Est": "Esther",
    "Sol": "Song of Solomon", "Eze": "Ezekiel", "Joe": "Joel",
    "Amo": "Amos", "Oba": "Obadiah", "Jon": "Jonah",
    "Zep": "Zephaniah", "Zec": "Zechariah",
}


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


def autocorr_lag1(series):
    arr = np.asarray(series, dtype=float)
    if len(arr) < 10:
        return float("nan")
    return float(np.corrcoef(arr[:-1], arr[1:])[0, 1])


# ── TAGNT parsing ────────────────────────────────────────────────────────

def download_tagnt():
    """Download TAGNT files if not present."""
    tagnt_dir = BASE / "sources" / "tagnt"
    tagnt_dir.mkdir(parents=True, exist_ok=True)

    urls = [
        ("https://raw.githubusercontent.com/STEPBible/STEPBible-Data/master/Older%20Formats/TAGNT%20Mat-Jhn%20-%20Translators%20Amalgamated%20Greek%20NT%20-%20STEPBible.org%20CC-BY.txt",
         "TAGNT_Mat-Jhn.txt"),
        ("https://raw.githubusercontent.com/STEPBible/STEPBible-Data/master/Older%20Formats/TAGNT%20Act-Rev%20-%20Translators%20Amalgamated%20Greek%20NT%20-%20STEPBible.org%20CC-BY.txt",
         "TAGNT_Act-Rev.txt"),
    ]

    for url, fname in urls:
        fpath = tagnt_dir / fname
        if fpath.exists() and fpath.stat().st_size > 1000:
            log.info(f"  TAGNT ya existe: {fname}")
            continue
        log.info(f"  Descargando {fname}...")
        try:
            subprocess.run(["wget", "-q", "-O", str(fpath), url],
                           timeout=120, check=True)
            log.info(f"    OK: {fpath.stat().st_size} bytes")
        except Exception as e:
            log.warning(f"    Error descargando {fname}: {e}")

    return tagnt_dir


def parse_tagnt(tagnt_dir):
    """Parse TAGNT files to find verses with OT references (H#### codes)."""
    log.info("Parseando TAGNT...")
    ot_quote_verses = defaultdict(int)  # (book, chapter, verse) → count of H#### words

    h_pattern = re.compile(r'H\d{4,5}')

    for fname in sorted(tagnt_dir.glob("TAGNT*.txt")):
        log.info(f"  Leyendo {fname.name}...")
        with open(fname, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # TAGNT format: tab-separated, first field has book.chapter.verse
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                # Look for H#### in any field (indicates Hebrew/OT reference)
                h_matches = h_pattern.findall(line)
                if not h_matches:
                    continue
                # Extract verse reference from first field
                ref = parts[0].strip()
                # Format varies: "Mat.1.1" or similar
                ref_match = re.match(r'(\d?\w+)\.(\d+)\.(\d+)', ref)
                if ref_match:
                    book_abbr = ref_match.group(1)
                    chapter = int(ref_match.group(2))
                    verse = int(ref_match.group(3))
                    book = BOOK_ABBREVS.get(book_abbr, book_abbr)
                    if book in NT_BOOKS:
                        ot_quote_verses[(book, chapter, verse)] += len(h_matches)

    log.info(f"  TAGNT: {len(ot_quote_verses)} versículos NT con ref H####")
    return ot_quote_verses


# ── Cross-references parsing ─────────────────────────────────────────────

def parse_cross_references(min_votes=50):
    """Parse cross_references.txt for NT→OT references.
    Format: 'Gen.1.1\\tProv.8.22-Prov.8.30\\t59'"""
    log.info(f"Parseando cross_references (votes>{min_votes})...")

    xref_file = Path("/tmp/cross_references.txt")
    if not xref_file.exists():
        for alt in [BASE / "sources" / "cross_references.txt",
                    BASE / "cross_references.txt"]:
            if alt.exists():
                xref_file = alt
                break

    if not xref_file.exists():
        log.warning("  cross_references.txt no encontrado")
        return {}

    nt_ot_refs = defaultdict(int)  # (book, chapter, verse) → count

    # Format: "Book.chapter.verse" or "Book.chapter.verse-Book.chapter.verse"
    ref_pat = re.compile(r'^(\d?\w+)\.(\d+)\.(\d+)')

    with open(xref_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("From") or line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            from_verse = parts[0].strip()
            to_verse = parts[1].strip()
            try:
                votes = int(parts[2].strip())
            except ValueError:
                continue

            if votes < min_votes:
                continue

            from_match = ref_pat.match(from_verse)
            to_match = ref_pat.match(to_verse)
            if not from_match or not to_match:
                continue

            from_book = BOOK_ABBREVS.get(from_match.group(1), from_match.group(1))
            to_book = BOOK_ABBREVS.get(to_match.group(1), to_match.group(1))

            # Keep only NT→OT references
            if from_book in NT_BOOKS and to_book in OT_BOOKS:
                chapter = int(from_match.group(2))
                verse = int(from_match.group(3))
                nt_ot_refs[(from_book, chapter, verse)] += 1

    log.info(f"  Cross-refs NT→OT: {len(nt_ot_refs)} versículos NT")
    return nt_ot_refs


# ── Main analysis ────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    log.info("=" * 70)
    log.info("FASE 13 — Script 3: OT Quotes in NT")
    log.info("=" * 70)

    # Load NT verse lengths
    log.info("Cargando NT...")
    with open(BIBLE_JSON, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    nt_verses = defaultdict(int)
    ot_verses = defaultdict(int)
    for w in corpus:
        book = w.get("book", "")
        key = (book, w.get("chapter", 0), w.get("verse", 0))
        if book in NT_BOOKS:
            nt_verses[key] += 1
        elif book in OT_BOOKS:
            ot_verses[key] += 1

    # Ordered list of NT verses
    nt_keys_ordered = sorted(nt_verses.keys())
    nt_lens = np.array([nt_verses[k] for k in nt_keys_ordered], dtype=float)
    log.info(f"  NT: {len(nt_lens)} versículos")

    # Create index mapping
    key_to_idx = {k: i for i, k in enumerate(nt_keys_ordered)}

    # 1. Identify OT quotes in NT
    log.info("\n=== PARTE 1: Identificación de citas ===")

    # Method A: TAGNT
    tagnt_dir = download_tagnt()
    tagnt_refs = parse_tagnt(tagnt_dir) if tagnt_dir.exists() else {}

    # Filter TAGNT: ≥3 H#### words per verse
    tagnt_quote_keys = {k for k, v in tagnt_refs.items() if v >= 3}
    log.info(f"  TAGNT con ≥3 refs: {len(tagnt_quote_keys)} versículos")

    # Method B: Cross-references
    xref_refs = parse_cross_references(min_votes=50)
    xref_quote_keys = set(xref_refs.keys())
    log.info(f"  Cross-refs votes>50: {len(xref_quote_keys)} versículos")

    # Intersection for high confidence
    if tagnt_quote_keys and xref_quote_keys:
        high_conf_keys = tagnt_quote_keys & xref_quote_keys
        method = "intersection"
    elif tagnt_quote_keys:
        high_conf_keys = tagnt_quote_keys
        method = "TAGNT_only"
    elif xref_quote_keys:
        high_conf_keys = xref_quote_keys
        method = "cross_refs_only"
    else:
        high_conf_keys = set()
        method = "none"

    # Union for broader coverage
    union_keys = tagnt_quote_keys | xref_quote_keys

    log.info(f"  Método: {method}")
    log.info(f"  Alta confianza: {len(high_conf_keys)} versículos")
    log.info(f"  Unión: {len(union_keys)} versículos")

    # Use union if intersection is too small
    quote_keys = high_conf_keys if len(high_conf_keys) >= 30 else union_keys
    log.info(f"  Usando: {len(quote_keys)} versículos "
             f"({'intersección' if len(high_conf_keys) >= 30 else 'unión'})")

    quote_identification = {
        "tagnt_total": len(tagnt_refs),
        "tagnt_ge3_refs": len(tagnt_quote_keys),
        "cross_refs_total": len(xref_refs),
        "cross_refs_votes_gt50": len(xref_quote_keys),
        "intersection": len(tagnt_quote_keys & xref_quote_keys),
        "union": len(union_keys),
        "method_used": method,
        "final_quote_count": len(quote_keys),
    }

    with open(RESULTS_DIR / "quote_identification.json", "w") as f:
        json.dump(quote_identification, f, indent=2, ensure_ascii=False)

    if not quote_keys:
        log.warning("  No se identificaron citas. Terminando.")
        return

    # 2. H_local comparison: windows with quotes vs without
    log.info("\n=== PARTE 2: H_local comparison ===")
    window = 10  # ±10 verses

    quote_indices = set()
    for k in quote_keys:
        if k in key_to_idx:
            quote_indices.add(key_to_idx[k])

    log.info(f"  {len(quote_indices)} citas mapeadas a índices NT")

    # H_local for windows centered on quote verses
    h_with_quotes = []
    h_without_quotes = []

    # Mark all indices within window of a quote
    near_quote = set()
    for idx in quote_indices:
        for offset in range(-window, window + 1):
            near_quote.add(idx + offset)

    # Compute H_local for windows centered on each verse
    n_total = len(nt_lens)
    for center in range(window, n_total - window):
        w_start = center - window
        w_end = center + window + 1
        local_lens = nt_lens[w_start:w_end]
        h_local = hurst_exponent_rs(local_lens)
        if np.isnan(h_local):
            continue
        if center in quote_indices:
            h_with_quotes.append(float(h_local))
        elif center not in near_quote:
            h_without_quotes.append(float(h_local))

    h_with = np.array(h_with_quotes)
    h_without = np.array(h_without_quotes)

    comparison = {
        "window_size": window,
        "n_quote_windows": len(h_with),
        "n_non_quote_windows": len(h_without),
    }

    if len(h_with) > 5 and len(h_without) > 5:
        u_stat, u_p = sp_stats.mannwhitneyu(h_with, h_without,
                                             alternative='greater')
        comparison.update({
            "H_local_with_quotes_mean": round(float(h_with.mean()), 4),
            "H_local_with_quotes_std": round(float(h_with.std()), 4),
            "H_local_without_quotes_mean": round(float(h_without.mean()), 4),
            "H_local_without_quotes_std": round(float(h_without.std()), 4),
            "delta_H": round(float(h_with.mean() - h_without.mean()), 4),
            "mann_whitney_U": float(u_stat),
            "mann_whitney_p": round(float(u_p), 6),
            "significant": bool(u_p < 0.05),
            "quotes_elevate_H": bool(h_with.mean() > h_without.mean() and u_p < 0.05),
        })
        log.info(f"  H_local con citas: {h_with.mean():.4f}±{h_with.std():.4f}")
        log.info(f"  H_local sin citas: {h_without.mean():.4f}±{h_without.std():.4f}")
        log.info(f"  Mann-Whitney p={u_p:.6f}, significativo={u_p < 0.05}")
    else:
        comparison["note"] = "Insufficient data for comparison"
        log.warning("  Datos insuficientes para comparación")

    with open(RESULTS_DIR / "local_h_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    # 3. Dense quote books
    log.info("\n=== PARTE 3: Libros con citas densas ===")
    book_quote_density = defaultdict(lambda: {"total": 0, "quotes": 0})
    for k in nt_keys_ordered:
        book = k[0]
        book_quote_density[book]["total"] += 1
        if k in quote_keys:
            book_quote_density[book]["quotes"] += 1

    dense_books = {}
    for book, counts in sorted(book_quote_density.items()):
        density = counts["quotes"] / counts["total"] if counts["total"] > 0 else 0
        # Compute H for this book alone
        book_lens = np.array([nt_verses[k] for k in nt_keys_ordered
                              if k[0] == book], dtype=float)
        h_book = hurst_exponent_rs(book_lens)
        ac1_book = autocorr_lag1(book_lens)

        dense_books[book] = {
            "n_verses": counts["total"],
            "n_quotes": counts["quotes"],
            "quote_density": round(density, 4),
            "H": round(float(h_book), 4) if not np.isnan(h_book) else None,
            "AC1": round(float(ac1_book), 4) if not np.isnan(ac1_book) else None,
        }

    # Correlation: quote density vs H
    densities = []
    h_values = []
    for book, data in dense_books.items():
        if data["H"] is not None and data["n_verses"] > 30:
            densities.append(data["quote_density"])
            h_values.append(data["H"])

    correlation = {}
    if len(densities) > 5:
        r, p = sp_stats.pearsonr(densities, h_values)
        rho, rho_p = sp_stats.spearmanr(densities, h_values)
        correlation = {
            "pearson_r": round(float(r), 4),
            "pearson_p": round(float(p), 6),
            "spearman_rho": round(float(rho), 4),
            "spearman_p": round(float(rho_p), 6),
            "n_books": len(densities),
            "significant": bool(p < 0.05),
        }
        log.info(f"  Correlación densidad-H: r={r:.4f} (p={p:.6f})")

    dense_result = {
        "by_book": dense_books,
        "density_H_correlation": correlation,
    }

    with open(RESULTS_DIR / "dense_quote_books.json", "w") as f:
        json.dump(dense_result, f, indent=2, ensure_ascii=False)

    # 4. Bonus: verse length similarity
    log.info("\n=== PARTE 4: Similitud de longitud cita-original ===")
    # Compare length of NT verse (quoting) vs mean AT verse length
    ot_mean_len = float(np.mean([ot_verses[k] for k in ot_verses]))
    nt_quote_lens = [nt_verses[k] for k in quote_keys if k in nt_verses]
    nt_nonquote_lens = [nt_verses[k] for k in nt_keys_ordered
                        if k not in quote_keys]

    if nt_quote_lens and nt_nonquote_lens:
        q_arr = np.array(nt_quote_lens, dtype=float)
        nq_arr = np.array(nt_nonquote_lens, dtype=float)
        length_comparison = {
            "ot_mean_verse_len": round(ot_mean_len, 2),
            "nt_quote_mean_len": round(float(q_arr.mean()), 2),
            "nt_nonquote_mean_len": round(float(nq_arr.mean()), 2),
            "quote_closer_to_ot": bool(
                abs(q_arr.mean() - ot_mean_len) < abs(nq_arr.mean() - ot_mean_len)),
            "t_stat": round(float(sp_stats.ttest_ind(q_arr, nq_arr).statistic), 4),
            "t_p": round(float(sp_stats.ttest_ind(q_arr, nq_arr).pvalue), 6),
        }
        log.info(f"  Longitud media citas NT: {q_arr.mean():.2f} vs no-citas: {nq_arr.mean():.2f}")
        log.info(f"  Media AT: {ot_mean_len:.2f}")
    else:
        length_comparison = {"note": "Insufficient data"}

    with open(RESULTS_DIR / "length_similarity.json", "w") as f:
        json.dump(length_comparison, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    log.info(f"\n{'=' * 70}")
    log.info(f"Script 3 completado en {elapsed:.1f}s")


if __name__ == "__main__":
    main()
