#!/usr/bin/env python3
"""
Fase 20 — Script 3: genre_controlled_analysis.py

Genre-controlled analysis:
1. DFA by genre: mean, std, n → Mann-Whitney AT_narrative vs NT_narrative
2. H by genre with bootstrap CI (n=1000)
3. Sensitivity: without AT_poetry → ¿DFA AT > NT sigue sig?
4. External corpora → Euclidean distance in (H, DFA, AC1) to each genre
5. Genre-controlled H4': AT_narrative vs NT_narrative in DFA
"""

import json
import logging
import numpy as np
from pathlib import Path
from scipy import stats as sp_stats

BASE = Path(__file__).resolve().parent
RESULTS_DIR = BASE / "results" / "genre_controlled"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Genre classification (canonical, NOT from deep_bimodal)
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


def load_book_features():
    """Load book features with DFA, H, AC1, CV."""
    bf_file = BASE / "results" / "refined_classifier" / "book_features.json"
    with open(bf_file) as f:
        return json.load(f)


def load_external_corpora():
    """Load external corpus metrics (H, DFA, AC1)."""
    corpora = {}

    # data_matrix.json
    dm_file = BASE / "results" / "transmission_origin" / "data_matrix.json"
    if dm_file.exists():
        with open(dm_file) as f:
            dm = json.load(f)
        for entry in dm:
            name = entry["corpus"]
            corpora[name] = {
                "H": entry.get("H"),
                "DFA": entry.get("DFA"),
                "AC1": entry.get("AC1"),
            }

    # Individual files
    extras = {
        "Didache": "didache/didache_metrics.json",
        "1_Clemente": "gap_corpora/1_clemente_metrics.json",
        "Tosefta": "tosefta/tosefta_metrics.json",
    }
    for name, rel in extras.items():
        fpath = BASE / "results" / rel
        if fpath.exists():
            with open(fpath) as f:
                data = json.load(f)
            corpora[name] = {
                "H": data.get("H"),
                "DFA": data.get("DFA"),
                "AC1": data.get("AC1"),
            }

    return corpora


# ═══════════════════════════════════════════════════════════════
# Main analysis
# ═══════════════════════════════════════════════════════════════

def main():
    log.info("=" * 70)
    log.info("FASE 20 — Script 3: Genre-Controlled Analysis")
    log.info("=" * 70)

    all_books = load_book_features()

    # Group books by genre, only those with DFA
    genre_books = {}
    for name, info in all_books.items():
        genre = GENRE_MAP.get(name)
        if genre is None:
            continue
        if info.get("DFA") is None:
            continue
        if genre not in genre_books:
            genre_books[genre] = []
        genre_books[genre].append({
            "book": name,
            "DFA": float(info["DFA"]),
            "H": float(info["H"]) if info.get("H") is not None else None,
            "AC1": float(info["AC1"]) if info.get("AC1") is not None else None,
            "CV": float(info["CV"]) if info.get("CV") is not None else None,
        })

    # ──────────────────────────────────────────────────────────
    # 1. DFA by genre
    # ──────────────────────────────────────────────────────────
    log.info("\n--- DFA by Genre ---")
    dfa_by_genre = {}
    for genre in sorted(genre_books.keys()):
        books = genre_books[genre]
        dfa_vals = [b["DFA"] for b in books]
        dfa_by_genre[genre] = {
            "n": len(books),
            "mean": round(float(np.mean(dfa_vals)), 4),
            "std": round(float(np.std(dfa_vals, ddof=1)), 4) if len(dfa_vals) > 1 else 0.0,
            "min": round(float(np.min(dfa_vals)), 4),
            "max": round(float(np.max(dfa_vals)), 4),
            "books": [b["book"] for b in books],
            "values": [round(v, 4) for v in dfa_vals],
        }
        log.info(f"  {genre}: n={len(books)}, "
                 f"DFA={np.mean(dfa_vals):.4f}±{np.std(dfa_vals, ddof=1):.4f}"
                 if len(dfa_vals) > 1 else
                 f"  {genre}: n={len(books)}, DFA={np.mean(dfa_vals):.4f}")

    # Mann-Whitney AT_narrative vs NT_narrative
    at_narr = [b["DFA"] for b in genre_books.get("AT_narrative", [])]
    nt_narr = [b["DFA"] for b in genre_books.get("NT_narrative", [])]

    mw_narr = None
    if at_narr and nt_narr:
        u, p = sp_stats.mannwhitneyu(at_narr, nt_narr, alternative="two-sided")
        cohen_d = float((np.mean(at_narr) - np.mean(nt_narr)) / np.sqrt(
            (np.var(at_narr, ddof=1) + np.var(nt_narr, ddof=1)) / 2))
        mw_narr = {
            "AT_narrative_mean": round(float(np.mean(at_narr)), 4),
            "NT_narrative_mean": round(float(np.mean(nt_narr)), 4),
            "U": round(float(u), 1),
            "p": round(float(p), 6),
            "cohen_d": round(cohen_d, 2),
            "n_AT": len(at_narr),
            "n_NT": len(nt_narr),
        }
        log.info(f"  AT_narrative vs NT_narrative: U={u:.1f}, p={p:.6f}, d={cohen_d:.2f}")

    dfa_by_genre["_mann_whitney_narrative"] = mw_narr

    with open(RESULTS_DIR / "dfa_by_genre.json", "w") as f:
        json.dump(dfa_by_genre, f, indent=2, ensure_ascii=False)
    log.info("  Saved dfa_by_genre.json")

    # ──────────────────────────────────────────────────────────
    # 2. H by genre with bootstrap CI
    # ──────────────────────────────────────────────────────────
    log.info("\n--- H by Genre (Bootstrap CI) ---")
    rng = np.random.RandomState(42)
    h_by_genre = {}

    for genre in sorted(genre_books.keys()):
        books = genre_books[genre]
        h_vals = [b["H"] for b in books if b["H"] is not None]
        if not h_vals:
            continue

        entry = {
            "n": len(h_vals),
            "mean": round(float(np.mean(h_vals)), 4),
            "std": round(float(np.std(h_vals, ddof=1)), 4) if len(h_vals) > 1 else 0.0,
        }

        # Bootstrap CI
        if len(h_vals) >= 3:
            boot_means = []
            for _ in range(1000):
                sample = rng.choice(h_vals, size=len(h_vals), replace=True)
                boot_means.append(float(np.mean(sample)))
            entry["H_CI95"] = [
                round(float(np.percentile(boot_means, 2.5)), 4),
                round(float(np.percentile(boot_means, 97.5)), 4),
            ]
            entry["H_bootstrap_std"] = round(float(np.std(boot_means)), 4)

        h_by_genre[genre] = entry
        ci_str = f", CI={entry.get('H_CI95')}" if 'H_CI95' in entry else ""
        log.info(f"  {genre}: H={entry['mean']:.4f}±{entry['std']:.4f}{ci_str}")

    # Check overlap: AT_prophetic CI vs NT_epistolar CI
    at_proph = h_by_genre.get("AT_prophetic", {})
    nt_epist = h_by_genre.get("NT_epistolar", {})
    if at_proph.get("H_CI95") and nt_epist.get("H_CI95"):
        overlap = (at_proph["H_CI95"][0] <= nt_epist["H_CI95"][1] and
                   nt_epist["H_CI95"][0] <= at_proph["H_CI95"][1])
        h_by_genre["_AT_prophetic_vs_NT_epistolar_overlap"] = bool(overlap)
        log.info(f"  AT_prophetic vs NT_epistolar CI overlap: {overlap}")

    with open(RESULTS_DIR / "h_by_genre_bootstrap.json", "w") as f:
        json.dump(h_by_genre, f, indent=2, ensure_ascii=False)
    log.info("  Saved h_by_genre_bootstrap.json")

    # ──────────────────────────────────────────────────────────
    # 3. Sensitivity analysis
    # ──────────────────────────────────────────────────────────
    log.info("\n--- Sensitivity Analysis ---")
    sensitivity = {}

    # All AT vs NT (with DFA)
    all_at_dfa = [b["DFA"] for name, info in all_books.items()
                  if info.get("DFA") is not None and info["testament"] == "AT"
                  for b in [{"DFA": float(info["DFA"])}]]
    all_nt_dfa = [b["DFA"] for name, info in all_books.items()
                  if info.get("DFA") is not None and info["testament"] == "NT"
                  for b in [{"DFA": float(info["DFA"])}]]

    u_all, p_all = sp_stats.mannwhitneyu(all_at_dfa, all_nt_dfa, alternative="two-sided")
    sensitivity["all_AT_vs_NT"] = {
        "U": round(float(u_all), 1),
        "p": round(float(p_all), 6),
        "n_AT": len(all_at_dfa),
        "n_NT": len(all_nt_dfa),
        "AT_mean": round(float(np.mean(all_at_dfa)), 4),
        "NT_mean": round(float(np.mean(all_nt_dfa)), 4),
    }
    log.info(f"  All AT vs NT: p={p_all:.6f}")

    # Without AT_poetry
    at_no_poetry = [float(info["DFA"]) for name, info in all_books.items()
                    if info.get("DFA") is not None and info["testament"] == "AT"
                    and GENRE_MAP.get(name) != "AT_poetry"]
    if at_no_poetry and all_nt_dfa:
        u_np, p_np = sp_stats.mannwhitneyu(at_no_poetry, all_nt_dfa,
                                           alternative="two-sided")
        sensitivity["without_AT_poetry"] = {
            "U": round(float(u_np), 1),
            "p": round(float(p_np), 6),
            "n_AT": len(at_no_poetry),
            "n_NT": len(all_nt_dfa),
            "AT_mean": round(float(np.mean(at_no_poetry)), 4),
            "still_significant": bool(p_np < 0.05),
        }
        log.info(f"  Without AT_poetry: p={p_np:.6f}, sig={p_np < 0.05}")

    # Without Psalms
    at_no_psalms = [float(info["DFA"]) for name, info in all_books.items()
                    if info.get("DFA") is not None and info["testament"] == "AT"
                    and name != "Psalms"]
    if at_no_psalms:
        h_with = np.mean(all_at_dfa)
        h_without = np.mean(at_no_psalms)
        sensitivity["without_Psalms"] = {
            "DFA_AT_with_Psalms": round(float(h_with), 4),
            "DFA_AT_without_Psalms": round(float(h_without), 4),
            "change_pct": round(float((h_without - h_with) / h_with * 100), 2),
        }
        log.info(f"  Without Psalms: DFA AT {h_with:.4f} → {h_without:.4f}")

    # Without Lamentations (extreme DFA=1.35)
    at_no_lam = [float(info["DFA"]) for name, info in all_books.items()
                 if info.get("DFA") is not None and info["testament"] == "AT"
                 and name != "Lamentations"]
    if at_no_lam:
        u_nl, p_nl = sp_stats.mannwhitneyu(at_no_lam, all_nt_dfa,
                                           alternative="two-sided")
        sensitivity["without_Lamentations"] = {
            "U": round(float(u_nl), 1),
            "p": round(float(p_nl), 6),
            "still_significant": bool(p_nl < 0.05),
        }
        log.info(f"  Without Lamentations: p={p_nl:.6f}")

    with open(RESULTS_DIR / "genre_separated_tests.json", "w") as f:
        json.dump(sensitivity, f, indent=2, ensure_ascii=False)
    log.info("  Saved genre_separated_tests.json")

    # ──────────────────────────────────────────────────────────
    # 4. External corpus → distance to each genre
    # ──────────────────────────────────────────────────────────
    log.info("\n--- External Corpus Genre Matching ---")
    ext_corpora = load_external_corpora()

    # Compute genre centroids in (H, DFA, AC1) space
    genre_centroids = {}
    for genre in sorted(genre_books.keys()):
        books = genre_books[genre]
        h_vals = [b["H"] for b in books if b["H"] is not None]
        dfa_vals = [b["DFA"] for b in books]
        ac1_vals = [b["AC1"] for b in books if b["AC1"] is not None]
        if h_vals and dfa_vals and ac1_vals:
            genre_centroids[genre] = {
                "H": float(np.mean(h_vals)),
                "DFA": float(np.mean(dfa_vals)),
                "AC1": float(np.mean(ac1_vals)),
            }

    ext_matches = {}
    for name, metrics in ext_corpora.items():
        if metrics.get("DFA") is None:
            continue
        h = metrics.get("H")
        dfa = metrics.get("DFA")
        ac1 = metrics.get("AC1")

        distances = {}
        for genre, centroid in genre_centroids.items():
            d = 0
            n_dim = 0
            if h is not None and centroid.get("H") is not None:
                d += (h - centroid["H"]) ** 2
                n_dim += 1
            if dfa is not None and centroid.get("DFA") is not None:
                d += (dfa - centroid["DFA"]) ** 2
                n_dim += 1
            if ac1 is not None and centroid.get("AC1") is not None:
                d += (ac1 - centroid["AC1"]) ** 2
                n_dim += 1
            if n_dim > 0:
                distances[genre] = round(float(np.sqrt(d)), 4)

        closest = min(distances, key=distances.get) if distances else None
        ext_matches[name] = {
            "H": round(float(h), 4) if h is not None else None,
            "DFA": round(float(dfa), 4) if dfa is not None else None,
            "AC1": round(float(ac1), 4) if ac1 is not None else None,
            "distances": distances,
            "closest_genre": closest,
        }
        log.info(f"  {name}: closest={closest}, "
                 f"d={distances.get(closest, '?') if closest else '?'}")

    with open(RESULTS_DIR / "external_corpus_genre_match.json", "w") as f:
        json.dump(ext_matches, f, indent=2, ensure_ascii=False)
    log.info("  Saved external_corpus_genre_match.json")

    # ──────────────────────────────────────────────────────────
    # 5. Genre-controlled H4': AT_narrative vs NT_narrative DFA
    # ──────────────────────────────────────────────────────────
    log.info("\n--- Genre-Controlled H4' (narrative only) ---")
    gc_h4prime = {}

    at_narr_dfa = [b["DFA"] for b in genre_books.get("AT_narrative", [])]
    nt_narr_dfa = [b["DFA"] for b in genre_books.get("NT_narrative", [])]

    if at_narr_dfa and nt_narr_dfa:
        u, p = sp_stats.mannwhitneyu(at_narr_dfa, nt_narr_dfa,
                                     alternative="two-sided")
        cohen_d = float((np.mean(at_narr_dfa) - np.mean(nt_narr_dfa)) / np.sqrt(
            (np.var(at_narr_dfa, ddof=1) + np.var(nt_narr_dfa, ddof=1)) / 2))

        gc_h4prime = {
            "test": "Mann-Whitney AT_narrative vs NT_narrative (DFA)",
            "AT_narrative": {
                "n": len(at_narr_dfa),
                "mean": round(float(np.mean(at_narr_dfa)), 4),
                "std": round(float(np.std(at_narr_dfa, ddof=1)), 4),
                "books": [b["book"] for b in genre_books["AT_narrative"]],
            },
            "NT_narrative": {
                "n": len(nt_narr_dfa),
                "mean": round(float(np.mean(nt_narr_dfa)), 4),
                "std": round(float(np.std(nt_narr_dfa, ddof=1)), 4),
                "books": [b["book"] for b in genre_books["NT_narrative"]],
            },
            "mann_whitney_U": round(float(u), 1),
            "p_value": round(float(p), 6),
            "cohen_d": round(cohen_d, 2),
            "significant": bool(p < 0.05),
            "interpretation": (
                "DFA difference between AT and NT narrative books is "
                + ("statistically significant" if p < 0.05 else "NOT significant")
                + f" (p={p:.4f}, Cohen d={cohen_d:.2f})"
            ),
        }
        log.info(f"  AT_narrative DFA={np.mean(at_narr_dfa):.4f} vs "
                 f"NT_narrative DFA={np.mean(nt_narr_dfa):.4f}")
        log.info(f"  U={u:.1f}, p={p:.6f}, Cohen d={cohen_d:.2f}")

    with open(RESULTS_DIR / "genre_controlled_h4prime.json", "w") as f:
        json.dump(gc_h4prime, f, indent=2, ensure_ascii=False)
    log.info("  Saved genre_controlled_h4prime.json")

    # Final
    log.info(f"\n{'=' * 70}")
    log.info("Script 3 completado.")
    log.info(f"  Genres analyzed: {len(genre_books)}")
    if mw_narr:
        log.info(f"  AT_narrative vs NT_narrative: p={mw_narr['p']}")


if __name__ == "__main__":
    main()
