#!/usr/bin/env python3
"""
Fase 16 — Script 2: book_of_mormon.py
Book of Mormon como control moderno (s.XIX).

Secciones:
1. Descarga y parseo de BOM, D&C, OT_eng, NT_eng
2. Métricas BOM: H, DFA, AC1, MPS, CV, mean, std, skewness
3. Métricas D&C: mismas
4. Comparación con OT/NT inglés
5. POS entropy con NLTK
6. Confound de citas AT (KJV formulas)
7. Clasificación: BOM vs rangos AT/NT
"""

import json
import logging
import time
import urllib.request
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from scipy import stats as sp_stats
from scipy import linalg as la

BASE = Path(__file__).resolve().parent
RESULTS_DIR = BASE / "results" / "bom"
LOG_DIR = BASE / "logs"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "fase16_book_of_mormon.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


# ── Reusable functions ────────────────────────────────────────────────

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


def compute_bond_dimension(series, max_lag=256, threshold=0.99):
    n = min(max_lag, len(series) // 4)
    if n < 2:
        return 1, np.array([1.0])
    mean = np.mean(series)
    centered = series - mean
    acf = np.zeros(n)
    for lag in range(n):
        if lag >= len(centered):
            break
        acf[lag] = np.mean(centered[:len(centered)-lag] * centered[lag:])
    T = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            lag = abs(i - j)
            if lag < n:
                T[i, j] = acf[lag]
    _, sigma, _ = la.svd(T, full_matrices=False)
    total = np.sum(sigma ** 2)
    if total == 0:
        return 1, sigma
    cumulative = np.cumsum(sigma ** 2) / total
    chi = int(np.searchsorted(cumulative, threshold) + 1)
    return chi, sigma


def compute_full_metrics(series, label="", n_perm=1000):
    """Compute all metrics for a verse-length series."""
    arr = np.asarray(series, dtype=float)
    n = len(arr)

    H = hurst_exponent_rs(arr)
    dfa = dfa_exponent(arr)
    ac1 = autocorr_lag1(arr)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    cv = std / mean if mean > 0 else 0
    skew = float(sp_stats.skew(arr))

    # MPS bond dimension with permutation test
    chi, sigma = compute_bond_dimension(arr, max_lag=min(256, n // 4), threshold=0.99)
    rng = np.random.default_rng(42)
    chi_perms = []
    for _ in range(n_perm):
        shuffled = rng.permutation(arr)
        chi_s, _ = compute_bond_dimension(shuffled, max_lag=min(64, n // 4), threshold=0.99)
        chi_perms.append(chi_s)
    chi_real, _ = compute_bond_dimension(arr, max_lag=min(64, n // 4), threshold=0.99)
    p_mps = float(np.mean(np.array(chi_perms) <= chi_real))

    return {
        "label": label,
        "n_verses": n,
        "H": round(H, 4) if not np.isnan(H) else None,
        "DFA": round(dfa, 4) if not np.isnan(dfa) else None,
        "AC1": round(ac1, 4) if not np.isnan(ac1) else None,
        "mean_verse_len": round(mean, 2),
        "std_verse_len": round(std, 2),
        "CV": round(cv, 4),
        "skewness": round(skew, 4),
        "bond_dimension": chi,
        "bond_dimension_64": chi_real,
        "mps_perm_p": round(p_mps, 4),
    }


# ── Download and parse ────────────────────────────────────────────────

URLS = {
    "bom": "https://raw.githubusercontent.com/bcbooks/scriptures-json/master/book-of-mormon.json",
    "dc": "https://raw.githubusercontent.com/bcbooks/scriptures-json/master/doctrine-and-covenants.json",
    "nt_eng": "https://raw.githubusercontent.com/bcbooks/scriptures-json/master/new-testament.json",
    "ot_eng": "https://raw.githubusercontent.com/bcbooks/scriptures-json/master/old-testament.json",
}


def download_json(url, label):
    log.info(f"  Descargando {label}...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        log.info(f"    OK")
        return data
    except Exception as e:
        log.error(f"    FAILED: {e}")
        return None


def parse_bom(data):
    """Parse BOM JSON → list of (book, chapter, verse, text)."""
    verses = []
    if not data:
        return verses
    for book_obj in data.get("books", []):
        book_name = book_obj.get("book", "")
        for ch_obj in book_obj.get("chapters", []):
            ch = ch_obj.get("chapter", 0)
            for v_obj in ch_obj.get("verses", []):
                text = v_obj.get("text", "")
                verse_num = v_obj.get("verse", 0)
                verses.append((book_name, ch, verse_num, text))
    return verses


def parse_dc(data):
    """Parse D&C JSON → list of (section, verse, text)."""
    verses = []
    if not data:
        return verses
    for sec_obj in data.get("sections", []):
        sec = sec_obj.get("section", 0)
        for v_obj in sec_obj.get("verses", []):
            text = v_obj.get("text", "")
            verse_num = v_obj.get("verse", 0)
            verses.append((sec, verse_num, text))
    return verses


def parse_bible_eng(data):
    """Parse OT/NT English JSON → list of (book, chapter, verse, text)."""
    verses = []
    if not data:
        return verses
    for book_obj in data.get("books", []):
        book_name = book_obj.get("book", "")
        for ch_obj in book_obj.get("chapters", []):
            ch = ch_obj.get("chapter", 0)
            for v_obj in ch_obj.get("verses", []):
                text = v_obj.get("text", "")
                verse_num = v_obj.get("verse", 0)
                verses.append((book_name, ch, verse_num, text))
    return verses


def verse_lengths(verse_list, text_idx=-1):
    """Extract word counts from verse text."""
    lengths = []
    for v in verse_list:
        text = v[text_idx] if isinstance(v[text_idx], str) else ""
        words = text.split()
        if words:
            lengths.append(len(words))
    return np.array(lengths, dtype=float)


# ── POS entropy ───────────────────────────────────────────────────────

def pos_entropy_nltk(verse_list, text_idx=-1, sample_size=2000):
    """Compute POS entropy using NLTK on a sample of verses."""
    try:
        import nltk
        nltk.download('averaged_perceptron_tagger_eng', quiet=True)
        nltk.download('punkt_tab', quiet=True)
    except ImportError:
        log.warning("NLTK not available")
        return None

    rng = np.random.default_rng(42)
    indices = rng.choice(len(verse_list), size=min(sample_size, len(verse_list)), replace=False)
    pos_counts = Counter()
    n_words = 0

    for idx in indices:
        text = verse_list[idx][text_idx] if isinstance(verse_list[idx][text_idx], str) else ""
        words = text.split()
        if words:
            tagged = nltk.pos_tag(words)
            for _, tag in tagged:
                pos_counts[tag] += 1
                n_words += 1

    if n_words == 0:
        return None

    probs = np.array(list(pos_counts.values()), dtype=float) / n_words
    entropy = float(-np.sum(probs * np.log2(probs + 1e-15)))

    return {
        "pos_entropy": round(entropy, 4),
        "n_words_sampled": n_words,
        "n_verses_sampled": len(indices),
        "n_pos_tags": len(pos_counts),
        "top_5_pos": dict(pos_counts.most_common(5)),
        "note": "English NLTK POS tagger — NOT comparable with OSHB Hebrew POS",
    }


# ── KJV formula detection ────────────────────────────────────────────

KJV_FORMULAS = [
    "and it came to pass",
    "behold",
    "saith the lord",
    "thus saith",
    "the lord god",
    "and now",
    "yea",
    "wo unto",
    "i say unto you",
    "hearken",
]


def detect_kjv_formulas(verse_list, text_idx=-1):
    """Count KJV formulaic expressions per book in BOM."""
    book_formulas = defaultdict(lambda: defaultdict(int))
    book_n_words = defaultdict(int)
    book_n_verses = defaultdict(int)

    for v in verse_list:
        book = v[0]
        text = v[text_idx].lower() if isinstance(v[text_idx], str) else ""
        words = text.split()
        book_n_words[book] += len(words)
        book_n_verses[book] += 1
        for formula in KJV_FORMULAS:
            count = text.count(formula)
            if count > 0:
                book_formulas[book][formula] += count

    results = {}
    for book in sorted(book_n_verses.keys()):
        total_formulas = sum(book_formulas[book].values())
        density = total_formulas / book_n_words[book] if book_n_words[book] > 0 else 0
        results[book] = {
            "n_verses": book_n_verses[book],
            "n_words": book_n_words[book],
            "total_formula_count": total_formulas,
            "formula_density": round(density, 5),
            "formulas": dict(book_formulas[book]),
        }

    return results


# ── Main ──────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    log.info("=" * 70)
    log.info("FASE 16 — Script 2: book_of_mormon.py")
    log.info("Book of Mormon como control moderno")
    log.info("=" * 70)

    # 1. Download
    log.info("\n=== Section 1: Download ===")
    raw = {}
    for key, url in URLS.items():
        raw[key] = download_json(url, key)

    bom_verses = parse_bom(raw.get("bom"))
    dc_verses = parse_dc(raw.get("dc"))
    ot_eng_verses = parse_bible_eng(raw.get("ot_eng"))
    nt_eng_verses = parse_bible_eng(raw.get("nt_eng"))

    log.info(f"  BOM: {len(bom_verses)} verses")
    log.info(f"  D&C: {len(dc_verses)} verses")
    log.info(f"  OT_eng: {len(ot_eng_verses)} verses")
    log.info(f"  NT_eng: {len(nt_eng_verses)} verses")

    # 2. BOM metrics
    log.info("\n=== Section 2: BOM Metrics ===")
    bom_lens = verse_lengths(bom_verses, text_idx=3)
    if len(bom_lens) > 0:
        bom_metrics = compute_full_metrics(bom_lens, "Book of Mormon")
        log.info(f"  BOM: H={bom_metrics['H']}, AC1={bom_metrics['AC1']}, "
                 f"DFA={bom_metrics['DFA']}, CV={bom_metrics['CV']}")

        # Per-book metrics
        bom_by_book = defaultdict(list)
        for v in bom_verses:
            words = v[3].split() if isinstance(v[3], str) else []
            if words:
                bom_by_book[v[0]].append(len(words))

        per_book = {}
        for book, lens in bom_by_book.items():
            arr = np.array(lens, dtype=float)
            if len(arr) >= 20:
                per_book[book] = {
                    "n_verses": len(arr),
                    "H": round(hurst_exponent_rs(arr), 4),
                    "AC1": round(autocorr_lag1(arr), 4),
                    "mean": round(float(np.mean(arr)), 2),
                    "CV": round(float(np.std(arr) / np.mean(arr)), 4),
                }

        bom_metrics["per_book"] = per_book
        with open(RESULTS_DIR / "bom_metrics.json", "w") as f:
            json.dump(bom_metrics, f, indent=2, ensure_ascii=False)
    else:
        bom_metrics = {}
        log.warning("  BOM: no verses parsed!")

    # 3. D&C metrics
    log.info("\n=== Section 3: D&C Metrics ===")
    dc_lens = verse_lengths(dc_verses, text_idx=2)
    if len(dc_lens) > 0:
        dc_metrics = compute_full_metrics(dc_lens, "Doctrine & Covenants")
        log.info(f"  D&C: H={dc_metrics['H']}, AC1={dc_metrics['AC1']}, "
                 f"DFA={dc_metrics['DFA']}, CV={dc_metrics['CV']}")
        with open(RESULTS_DIR / "dc_metrics.json", "w") as f:
            json.dump(dc_metrics, f, indent=2, ensure_ascii=False)
    else:
        dc_metrics = {}
        log.warning("  D&C: no verses parsed!")

    # 4. English OT/NT comparison
    log.info("\n=== Section 4: English Bible Comparison ===")
    comparison = {}

    ot_eng_lens = verse_lengths(ot_eng_verses, text_idx=3)
    nt_eng_lens = verse_lengths(nt_eng_verses, text_idx=3)

    for label, lens in [("OT_eng", ot_eng_lens), ("NT_eng", nt_eng_lens)]:
        if len(lens) > 0:
            metrics = {
                "n_verses": len(lens),
                "H": round(hurst_exponent_rs(lens), 4),
                "DFA": round(dfa_exponent(lens), 4),
                "AC1": round(autocorr_lag1(lens), 4),
                "mean_verse_len": round(float(np.mean(lens)), 2),
                "std_verse_len": round(float(np.std(lens)), 2),
                "CV": round(float(np.std(lens) / np.mean(lens)), 4),
                "skewness": round(float(sp_stats.skew(lens)), 4),
            }
            comparison[label] = metrics
            log.info(f"  {label}: H={metrics['H']}, AC1={metrics['AC1']}, "
                     f"DFA={metrics['DFA']}")

    # Add BOM and D&C for comparison
    if bom_metrics:
        comparison["BOM"] = {k: bom_metrics[k] for k in
                             ["n_verses", "H", "DFA", "AC1", "mean_verse_len",
                              "std_verse_len", "CV", "skewness"]
                             if k in bom_metrics}
    if dc_metrics:
        comparison["DC"] = {k: dc_metrics[k] for k in
                            ["n_verses", "H", "DFA", "AC1", "mean_verse_len",
                             "std_verse_len", "CV", "skewness"]
                            if k in dc_metrics}

    with open(RESULTS_DIR / "english_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    # 5. POS entropy
    log.info("\n=== Section 5: POS Entropy (NLTK) ===")
    pos_results = {}
    for label, verses, tidx in [("BOM", bom_verses, 3), ("DC", dc_verses, 2),
                                  ("OT_eng", ot_eng_verses, 3), ("NT_eng", nt_eng_verses, 3)]:
        if verses:
            pe = pos_entropy_nltk(verses, text_idx=tidx)
            if pe:
                pos_results[label] = pe
                log.info(f"  {label}: pe={pe['pos_entropy']:.4f}")

    pos_results["caveat"] = (
        "POS entropy computed with English NLTK tagger. "
        "NOT comparable with OSHB Hebrew POS tags. "
        "Valid for BOM vs OT_eng/NT_eng comparison only."
    )

    with open(RESULTS_DIR / "pos_entropy_english.json", "w") as f:
        json.dump(pos_results, f, indent=2, ensure_ascii=False)

    # 6. KJV citation confound
    log.info("\n=== Section 6: KJV Citation Confound ===")
    if bom_verses:
        formula_results = detect_kjv_formulas(bom_verses, text_idx=3)

        total_formulas = sum(b["total_formula_count"] for b in formula_results.values())
        total_words = sum(b["n_words"] for b in formula_results.values())
        overall_density = total_formulas / total_words if total_words > 0 else 0

        citation_data = {
            "overall_formula_density": round(overall_density, 5),
            "total_formulas": total_formulas,
            "total_words": total_words,
            "formulas_tested": KJV_FORMULAS,
            "by_book": formula_results,
            "interpretation": (
                f"BOM has {total_formulas} KJV formulaic expressions "
                f"(density={overall_density:.5f}). "
                "These archaic formulas inflate AC1 by creating repetitive patterns."
            ),
        }

        log.info(f"  Total KJV formulas: {total_formulas}, density={overall_density:.5f}")

        with open(RESULTS_DIR / "at_citation_confound.json", "w") as f:
            json.dump(citation_data, f, indent=2, ensure_ascii=False)

    # 7. Classification
    log.info("\n=== Section 7: Classification ===")

    # Load Hebrew/Greek reference ranges from book_features
    features_file = BASE / "results" / "refined_classifier" / "book_features.json"
    if features_file.exists():
        with open(features_file, "r") as f:
            book_features = json.load(f)

        at_h = [b["H"] for b in book_features.values()
                if b.get("testament") == "AT" and b.get("H") is not None]
        nt_h = [b["H"] for b in book_features.values()
                if b.get("testament") == "NT" and b.get("H") is not None]
        at_ac1 = [b["AC1"] for b in book_features.values()
                  if b.get("testament") == "AT" and b.get("AC1") is not None]
        nt_ac1 = [b["AC1"] for b in book_features.values()
                  if b.get("testament") == "NT" and b.get("AC1") is not None]

        classifier = {
            "reference_ranges": {
                "AT_H": {"mean": round(np.mean(at_h), 4), "std": round(np.std(at_h), 4),
                          "range": [round(min(at_h), 4), round(max(at_h), 4)]},
                "NT_H": {"mean": round(np.mean(nt_h), 4), "std": round(np.std(nt_h), 4),
                          "range": [round(min(nt_h), 4), round(max(nt_h), 4)]},
                "AT_AC1": {"mean": round(np.mean(at_ac1), 4), "std": round(np.std(at_ac1), 4),
                           "range": [round(min(at_ac1), 4), round(max(at_ac1), 4)]},
                "NT_AC1": {"mean": round(np.mean(nt_ac1), 4), "std": round(np.std(nt_ac1), 4),
                           "range": [round(min(nt_ac1), 4), round(max(nt_ac1), 4)]},
            },
            "note": (
                "BOM metrics are in English → not directly comparable with "
                "Hebrew (AT) or Greek (NT) metrics. "
                "Valid comparison: BOM vs OT_eng and NT_eng."
            ),
        }

        # Compare BOM with English Bible ranges
        if comparison.get("OT_eng") and comparison.get("NT_eng") and bom_metrics:
            bom_h = bom_metrics.get("H")
            bom_ac1 = bom_metrics.get("AC1")
            ot_h = comparison["OT_eng"]["H"]
            nt_h = comparison["NT_eng"]["H"]
            ot_ac1 = comparison["OT_eng"]["AC1"]
            nt_ac1_eng = comparison["NT_eng"]["AC1"]

            classifier["english_comparison"] = {
                "BOM_H": bom_h,
                "OT_eng_H": ot_h,
                "NT_eng_H": nt_h,
                "BOM_AC1": bom_ac1,
                "OT_eng_AC1": ot_ac1,
                "NT_eng_AC1": nt_ac1_eng,
                "H_closer_to": "OT" if abs(bom_h - ot_h) < abs(bom_h - nt_h) else "NT",
                "AC1_closer_to": "OT" if abs(bom_ac1 - ot_ac1) < abs(bom_ac1 - nt_ac1_eng) else "NT",
            }

            log.info(f"  BOM H={bom_h} (OT_eng={ot_h}, NT_eng={nt_h})")
            log.info(f"  BOM AC1={bom_ac1} (OT_eng={ot_ac1}, NT_eng={nt_ac1_eng})")

        with open(RESULTS_DIR / "classifier_result.json", "w") as f:
            json.dump(classifier, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    log.info(f"\n{'=' * 70}")
    log.info(f"book_of_mormon.py completado en {elapsed:.1f}s")
    print(f"[book_of_mormon] DONE — {elapsed:.1f}s")


if __name__ == "__main__":
    main()
