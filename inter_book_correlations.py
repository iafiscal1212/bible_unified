#!/usr/bin/env python3
"""
Fase 20 — Script 2: inter_book_correlations.py

Inter-book structure analysis:
1. Permutation test: H(AT canon) vs random book orderings
2. ACF inter-book: autocorrelation of mean_verse_len per book
3. Test A: AC1 of [n_verses per book] in canonical order
4. Test B: genre-match probability in neighbors vs random
5. Test C: correlation book_num vs chronological period
6. NT decomposition (same tests)
7. Quran: H per sura vs corpus
"""

import json
import logging
import numpy as np
from pathlib import Path
from scipy import stats as sp_stats

BASE = Path(__file__).resolve().parent
RESULTS_DIR = BASE / "results" / "inter_book"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Metric functions (from gap_corpora_f19.py)
# ═══════════════════════════════════════════════════════════════

def hurst_exponent_rs(series):
    """R/S Hurst exponent with logspace sampling for large series."""
    series = np.asarray(series, dtype=float)
    n = len(series)
    if n < 20:
        return np.nan
    max_k, min_k = n // 2, 10
    if max_k <= min_k:
        return np.nan
    # Use logspace sampling (30 points) for efficiency on large series
    ks = np.unique(np.logspace(np.log10(min_k), np.log10(max_k), 30).astype(int))
    ns, rs = [], []
    for k in ks:
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


def dfa_exponent(series):
    s = np.asarray(series, dtype=float)
    n = len(s)
    if n < 20:
        return np.nan
    y = np.cumsum(s - np.mean(s))
    mn, mx = 4, n // 4
    if mx < mn + 2:
        return np.nan
    bsz = np.unique(np.logspace(np.log10(mn), np.log10(mx), 20).astype(int))
    fl, sz = [], []
    for bs in bsz:
        nb = n // bs
        if nb < 1:
            continue
        f2 = []
        for i in range(nb):
            seg = y[i*bs:(i+1)*bs]
            x = np.arange(bs)
            trend = np.polyval(np.polyfit(x, seg, 1), x)
            f2.append(np.mean((seg - trend) ** 2))
        if f2:
            fl.append(np.sqrt(np.mean(f2))); sz.append(bs)
    if len(sz) < 3:
        return np.nan
    slope, *_ = sp_stats.linregress(np.log(sz), np.log(fl))
    return round(slope, 4)


def autocorr_lag1(series):
    s = np.asarray(series, dtype=float)
    if len(s) < 3:
        return np.nan
    m, v = np.mean(s), np.var(s)
    if v == 0:
        return 0.0
    return round(float(np.sum((s[:-1] - m) * (s[1:] - m)) / (len(s) * v)), 4)


def autocorr_lag_k(series, k):
    s = np.asarray(series, dtype=float)
    n = len(s)
    if n < k + 3:
        return np.nan
    m, v = np.mean(s), np.var(s)
    if v == 0:
        return 0.0
    return round(float(np.sum((s[:n-k] - m) * (s[k:] - m)) / (n * v)), 4)


# ═══════════════════════════════════════════════════════════════
# Genre classification
# ═══════════════════════════════════════════════════════════════

GENRE_MAP = {
    # AT Poético
    "Psalms": "AT_poetry", "Job": "AT_poetry", "Proverbs": "AT_poetry",
    "Song of Songs": "AT_poetry", "Lamentations": "AT_poetry",
    "Ecclesiastes": "AT_poetry",
    # AT Legal
    "Leviticus": "AT_legal", "Numbers": "AT_legal", "Deuteronomy": "AT_legal",
    # AT Profético
    "Isaiah": "AT_prophetic", "Jeremiah": "AT_prophetic",
    "Ezekiel": "AT_prophetic", "Hosea": "AT_prophetic", "Joel": "AT_prophetic",
    "Amos": "AT_prophetic", "Obadiah": "AT_prophetic", "Jonah": "AT_prophetic",
    "Micah": "AT_prophetic", "Nahum": "AT_prophetic",
    "Habakkuk": "AT_prophetic", "Zephaniah": "AT_prophetic",
    "Haggai": "AT_prophetic", "Zechariah": "AT_prophetic",
    "Malachi": "AT_prophetic", "Daniel": "AT_prophetic",
    # AT Narrativo
    "Genesis": "AT_narrative", "Exodus": "AT_narrative",
    "Joshua": "AT_narrative", "Judges": "AT_narrative", "Ruth": "AT_narrative",
    "1 Samuel": "AT_narrative", "2 Samuel": "AT_narrative",
    "1 Kings": "AT_narrative", "2 Kings": "AT_narrative",
    "1 Chronicles": "AT_narrative", "2 Chronicles": "AT_narrative",
    "Ezra": "AT_narrative", "Nehemiah": "AT_narrative", "Esther": "AT_narrative",
    # NT Narrativo
    "Matthew": "NT_narrative", "Mark": "NT_narrative", "Luke": "NT_narrative",
    "John": "NT_narrative", "Acts": "NT_narrative",
    # NT Epistolar
    "Romans": "NT_epistolar", "1 Corinthians": "NT_epistolar",
    "2 Corinthians": "NT_epistolar", "Galatians": "NT_epistolar",
    "Ephesians": "NT_epistolar", "Philippians": "NT_epistolar",
    "Colossians": "NT_epistolar", "1 Thessalonians": "NT_epistolar",
    "2 Thessalonians": "NT_epistolar", "1 Timothy": "NT_epistolar",
    "2 Timothy": "NT_epistolar", "Titus": "NT_epistolar",
    "Philemon": "NT_epistolar", "Hebrews": "NT_epistolar",
    "James": "NT_epistolar", "1 Peter": "NT_epistolar",
    "2 Peter": "NT_epistolar", "1 John": "NT_epistolar",
    "2 John": "NT_epistolar", "3 John": "NT_epistolar", "Jude": "NT_epistolar",
    # NT Apocalíptico
    "Revelation": "NT_apocalyptic",
}

# Chronological periods (approximate, for Test C)
CHRONO_PERIOD = {
    "Genesis": 1, "Exodus": 1, "Leviticus": 1, "Numbers": 1,
    "Deuteronomy": 1, "Joshua": 2, "Judges": 2, "Ruth": 2,
    "1 Samuel": 3, "2 Samuel": 3, "1 Kings": 4, "2 Kings": 4,
    "1 Chronicles": 5, "2 Chronicles": 5, "Ezra": 5, "Nehemiah": 5,
    "Esther": 5, "Job": 3, "Psalms": 3, "Proverbs": 4,
    "Ecclesiastes": 4, "Song of Songs": 4,
    "Isaiah": 4, "Jeremiah": 4, "Lamentations": 4, "Ezekiel": 5,
    "Daniel": 5, "Hosea": 4, "Joel": 4, "Amos": 4,
    "Obadiah": 4, "Jonah": 4, "Micah": 4, "Nahum": 4,
    "Habakkuk": 4, "Zephaniah": 4, "Haggai": 5, "Zechariah": 5,
    "Malachi": 5,
    "Matthew": 6, "Mark": 6, "Luke": 6, "John": 6, "Acts": 6,
    "Romans": 6, "1 Corinthians": 6, "2 Corinthians": 6,
    "Galatians": 6, "Ephesians": 6, "Philippians": 6,
    "Colossians": 6, "1 Thessalonians": 6, "2 Thessalonians": 6,
    "1 Timothy": 7, "2 Timothy": 7, "Titus": 7, "Philemon": 6,
    "Hebrews": 7, "James": 6, "1 Peter": 7, "2 Peter": 7,
    "1 John": 7, "2 John": 7, "3 John": 7, "Jude": 7,
    "Revelation": 7,
}


# ═══════════════════════════════════════════════════════════════
# Build verse lengths per book from bible_unified.json
# ═══════════════════════════════════════════════════════════════

def build_book_verse_lengths(words):
    """Build {book_name: [verse_lengths]} in canonical order."""
    books = {}
    for w in words:
        book = w["book"]
        key = (w["book_num"], w["chapter"], w["verse"])
        if book not in books:
            books[book] = {"book_num": w["book_num"], "corpus": w["corpus"],
                           "verses": {}}
        books[book]["verses"][key] = books[book]["verses"].get(key, 0) + 1

    result = {}
    for book in sorted(books, key=lambda b: books[b]["book_num"]):
        info = books[book]
        vl = np.array([v for _, v in sorted(info["verses"].items())], dtype=float)
        result[book] = {
            "book_num": info["book_num"],
            "corpus": info["corpus"],
            "verse_lengths": vl,
            "n_verses": len(vl),
            "mean_verse_len": float(np.mean(vl)),
        }
    return result


def run_permutation_test(book_data, corpus_filter, n_perm=200):
    """Permutation test: H of canonical order vs shuffled book order.
    Uses n_perm=200 (not 1000) because H+DFA on 23K verses is O(n²) per call.
    DFA is computed only for canonical (not per-permutation) to save time.
    """
    filtered = {b: d for b, d in book_data.items() if d["corpus"] == corpus_filter}
    canonical_books = sorted(filtered.keys(), key=lambda b: filtered[b]["book_num"])

    canonical_series = np.concatenate(
        [filtered[b]["verse_lengths"] for b in canonical_books])
    h_canon = hurst_exponent_rs(canonical_series)
    dfa_canon = dfa_exponent(canonical_series)
    log.info(f"  {corpus_filter} canonical: H={h_canon}, DFA={dfa_canon}, "
             f"n={len(canonical_series)}")

    rng = np.random.RandomState(42)
    h_perms = []
    for i in range(n_perm):
        perm_order = rng.permutation(canonical_books)
        perm_series = np.concatenate(
            [filtered[b]["verse_lengths"] for b in perm_order])
        h_perms.append(hurst_exponent_rs(perm_series))
        if (i + 1) % 50 == 0:
            log.info(f"    permutation {i + 1}/{n_perm}...")

    h_perms = np.array([h for h in h_perms if not np.isnan(h)])
    h_p = float(np.mean(h_perms >= h_canon)) if len(h_perms) > 0 else np.nan

    return {
        "corpus": corpus_filter,
        "n_books": len(canonical_books),
        "n_verses": len(canonical_series),
        "H_canonical": float(h_canon) if not np.isnan(h_canon) else None,
        "H_perm_mean": round(float(np.mean(h_perms)), 4) if len(h_perms) > 0 else None,
        "H_perm_std": round(float(np.std(h_perms)), 4) if len(h_perms) > 0 else None,
        "H_p_value": round(h_p, 4) if not np.isnan(h_p) else None,
        "DFA_canonical": float(dfa_canon) if not np.isnan(dfa_canon) else None,
        "DFA_note": "DFA computed only for canonical order (permutation DFA too expensive on 23K+ verses)",
        "n_permutations": n_perm,
    }


def compute_acf_inter_book(book_data, corpus_filter, max_lag=10):
    """ACF of mean_verse_len series by book in canonical order."""
    filtered = {b: d for b, d in book_data.items() if d["corpus"] == corpus_filter}
    canonical_books = sorted(filtered.keys(), key=lambda b: filtered[b]["book_num"])
    series = np.array([filtered[b]["mean_verse_len"] for b in canonical_books])

    acf_values = {}
    for lag in range(1, min(max_lag + 1, len(series) // 2)):
        acf_values[str(lag)] = autocorr_lag_k(series, lag)

    return {
        "corpus": corpus_filter,
        "n_books": len(canonical_books),
        "books": canonical_books,
        "mean_verse_len_series": [round(float(v), 2) for v in series],
        "acf": acf_values,
    }


def test_a_nverses_ac1(book_data, corpus_filter):
    """Test A: AC1 of [n_verses per book] in canonical order."""
    filtered = {b: d for b, d in book_data.items() if d["corpus"] == corpus_filter}
    canonical_books = sorted(filtered.keys(), key=lambda b: filtered[b]["book_num"])
    series = np.array([filtered[b]["n_verses"] for b in canonical_books], dtype=float)
    ac1 = autocorr_lag1(series)
    return {
        "corpus": corpus_filter,
        "n_books": len(canonical_books),
        "n_verses_series": [int(v) for v in series],
        "AC1_n_verses": float(ac1) if not np.isnan(ac1) else None,
    }


def test_b_genre_neighbors(book_data, corpus_filter):
    """Test B: probability of genre-match in canonical neighbors vs random."""
    filtered = {b: d for b, d in book_data.items() if d["corpus"] == corpus_filter}
    canonical_books = sorted(filtered.keys(), key=lambda b: filtered[b]["book_num"])
    genres = [GENRE_MAP.get(b, "unknown") for b in canonical_books]
    n = len(genres)

    # Canonical neighbor matches
    canon_matches = sum(1 for i in range(n - 1) if genres[i] == genres[i + 1])
    canon_prob = canon_matches / (n - 1) if n > 1 else 0

    # Random expectation (Monte Carlo)
    rng = np.random.RandomState(42)
    rand_matches = []
    for _ in range(10000):
        perm = rng.permutation(genres)
        m = sum(1 for i in range(n - 1) if perm[i] == perm[i + 1])
        rand_matches.append(m)
    rand_mean = float(np.mean(rand_matches)) / (n - 1) if n > 1 else 0
    p_value = float(np.mean(np.array(rand_matches) >= canon_matches))

    return {
        "corpus": corpus_filter,
        "n_books": n,
        "canonical_genre_match_rate": round(float(canon_prob), 4),
        "random_genre_match_rate": round(rand_mean, 4),
        "enrichment_ratio": round(canon_prob / rand_mean, 2) if rand_mean > 0 else None,
        "p_value": round(p_value, 4),
    }


def test_c_booknum_chrono(book_data, corpus_filter):
    """Test C: correlation book_num vs chronological period."""
    filtered = {b: d for b, d in book_data.items() if d["corpus"] == corpus_filter}
    canonical_books = sorted(filtered.keys(), key=lambda b: filtered[b]["book_num"])

    book_nums = []
    periods = []
    for b in canonical_books:
        if b in CHRONO_PERIOD:
            book_nums.append(filtered[b]["book_num"])
            periods.append(CHRONO_PERIOD[b])

    if len(book_nums) < 5:
        return {"corpus": corpus_filter, "status": "insufficient_data"}

    rho, p = sp_stats.spearmanr(book_nums, periods)
    return {
        "corpus": corpus_filter,
        "n_books": len(book_nums),
        "spearman_rho": round(float(rho), 4),
        "spearman_p": round(float(p), 6),
    }


# ═══════════════════════════════════════════════════════════════
# Main analysis
# ═══════════════════════════════════════════════════════════════

def main():
    log.info("=" * 70)
    log.info("FASE 20 — Script 2: Inter-book Correlations")
    log.info("=" * 70)

    # Load bible_unified.json
    corpus_file = BASE / "bible_unified.json"
    log.info("  Loading bible_unified.json...")
    with open(corpus_file) as f:
        words = json.load(f)
    log.info(f"  Loaded {len(words)} words")

    # Build verse lengths per book
    book_data = build_book_verse_lengths(words)
    log.info(f"  Built data for {len(book_data)} books")

    # Free memory
    del words

    results = {}

    # 1. Permutation test — AT
    log.info("\n--- Permutation Test ---")
    perm_at = run_permutation_test(book_data, "OT", n_perm=1000)
    log.info(f"  AT: H_canon={perm_at['H_canonical']}, "
             f"H_perm_mean={perm_at['H_perm_mean']}, p={perm_at['H_p_value']}")

    perm_nt = run_permutation_test(book_data, "NT", n_perm=1000)
    log.info(f"  NT: H_canon={perm_nt['H_canonical']}, "
             f"H_perm_mean={perm_nt['H_perm_mean']}, p={perm_nt['H_p_value']}")

    results["permutation_test"] = {"AT": perm_at, "NT": perm_nt}
    with open(RESULTS_DIR / "permutation_test.json", "w") as f:
        json.dump(results["permutation_test"], f, indent=2, ensure_ascii=False)
    log.info("  Saved permutation_test.json")

    # 2. ACF inter-book
    log.info("\n--- ACF Inter-book ---")
    acf_at = compute_acf_inter_book(book_data, "OT")
    acf_nt = compute_acf_inter_book(book_data, "NT")
    log.info(f"  AT ACF(1)={acf_at['acf'].get('1')}")
    log.info(f"  NT ACF(1)={acf_nt['acf'].get('1')}")

    results["acf_by_book"] = {"AT": acf_at, "NT": acf_nt}
    with open(RESULTS_DIR / "acf_by_book.json", "w") as f:
        json.dump(results["acf_by_book"], f, indent=2, ensure_ascii=False)
    log.info("  Saved acf_by_book.json")

    # 3. Test A: AC1 of n_verses
    log.info("\n--- Test A: AC1 of n_verses ---")
    ta_at = test_a_nverses_ac1(book_data, "OT")
    ta_nt = test_a_nverses_ac1(book_data, "NT")
    log.info(f"  AT: AC1(n_verses)={ta_at['AC1_n_verses']}")
    log.info(f"  NT: AC1(n_verses)={ta_nt['AC1_n_verses']}")

    # 4. Test B: genre-match in neighbors
    log.info("\n--- Test B: Genre-match ---")
    tb_at = test_b_genre_neighbors(book_data, "OT")
    tb_nt = test_b_genre_neighbors(book_data, "NT")
    log.info(f"  AT: canon={tb_at['canonical_genre_match_rate']}, "
             f"random={tb_at['random_genre_match_rate']}, "
             f"enrichment={tb_at['enrichment_ratio']}x, p={tb_at['p_value']}")
    log.info(f"  NT: canon={tb_nt['canonical_genre_match_rate']}, "
             f"random={tb_nt['random_genre_match_rate']}, "
             f"enrichment={tb_nt['enrichment_ratio']}x, p={tb_nt['p_value']}")

    # 5. Test C: book_num vs chronological period
    log.info("\n--- Test C: Canon vs Chronology ---")
    tc_at = test_c_booknum_chrono(book_data, "OT")
    tc_nt = test_c_booknum_chrono(book_data, "NT")
    log.info(f"  AT: Spearman rho={tc_at.get('spearman_rho')}, p={tc_at.get('spearman_p')}")
    log.info(f"  NT: Spearman rho={tc_nt.get('spearman_rho')}, p={tc_nt.get('spearman_p')}")

    results["correlation_sources"] = {
        "test_A": {"AT": ta_at, "NT": ta_nt},
        "test_B": {"AT": tb_at, "NT": tb_nt},
        "test_C": {"AT": tc_at, "NT": tc_nt},
    }
    with open(RESULTS_DIR / "correlation_sources.json", "w") as f:
        json.dump(results["correlation_sources"], f, indent=2, ensure_ascii=False)
    log.info("  Saved correlation_sources.json")

    # 6. AT vs NT comparison
    log.info("\n--- AT vs NT Bonus ---")
    at_means = [book_data[b]["mean_verse_len"] for b in book_data
                if book_data[b]["corpus"] == "OT"]
    nt_means = [book_data[b]["mean_verse_len"] for b in book_data
                if book_data[b]["corpus"] == "NT"]
    mw_u, mw_p = sp_stats.mannwhitneyu(at_means, nt_means, alternative="two-sided")
    log.info(f"  AT vs NT mean_verse_len: U={mw_u:.1f}, p={mw_p:.6f}")

    # 7. Quran: H per sura
    log.info("\n--- Quran Sura Analysis ---")
    quran_file = BASE / "results" / "recitation" / "quran_sura_ac1.json"
    quran_result = {"status": "unavailable"}
    if quran_file.exists():
        with open(quran_file) as f:
            quran_data = json.load(f)
        quran_result = {
            "status": "available",
            "global_H": quran_data.get("global", {}).get("H_words"),
            "global_AC1": quran_data.get("global", {}).get("ac1_words"),
            "n_suras_analyzed": quran_data.get("per_sura", {}).get("n_suras_analyzed"),
            "mean_sura_AC1": quran_data.get("per_sura", {}).get("ac1_words_mean"),
            "note": "Quran DFA=0.9336 from data_matrix.json (corpus-level)",
        }
        log.info(f"  Quran: H={quran_result['global_H']}, "
                 f"AC1={quran_result['global_AC1']}, "
                 f"n_suras={quran_result['n_suras_analyzed']}")

    results["external_corpus_decomposition"] = {
        "quran": quran_result,
        "at_vs_nt_mean_verse_len": {
            "AT_mean": round(float(np.mean(at_means)), 2),
            "NT_mean": round(float(np.mean(nt_means)), 2),
            "mann_whitney_U": round(float(mw_u), 1),
            "mann_whitney_p": round(float(mw_p), 6),
        }
    }
    with open(RESULTS_DIR / "external_corpus_decomposition.json", "w") as f:
        json.dump(results["external_corpus_decomposition"], f, indent=2,
                  ensure_ascii=False)
    log.info("  Saved external_corpus_decomposition.json")

    # Final
    log.info(f"\n{'=' * 70}")
    log.info("Script 2 completado.")
    log.info(f"  AT permutation p(H)={perm_at['H_p_value']}")
    log.info(f"  NT permutation p(H)={perm_nt['H_p_value']}")


if __name__ == "__main__":
    main()
