#!/usr/bin/env python3
"""
Fase 15 — Script 2: Daniel Analysis
¿Por qué Daniel es el único libro AT fronterizo (P_AT=0.567)?

1. Separar por lengua (hebreo vs arameo via morph prefix H/A)
2. Separar por contenido (narrativo cap 1-6 vs apocalíptico cap 7-12)
3. Comparar con textos similares (Ezequiel, Zacarías, Ester, Esdras)
4. Test de datación (sugerente, no demostrativo)
5. Factor arameo (comparar con arameo de Esdras)
"""

import json
import logging
import time
import numpy as np
from pathlib import Path
from scipy import stats as sp_stats
from collections import defaultdict, Counter

BASE = Path(__file__).resolve().parent
RESULTS_DIR = BASE / "results" / "daniel"
LOG_DIR = BASE / "logs"
BIBLE_JSON = BASE / "bible_unified.json"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "fase15_daniel.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


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


def shannon_entropy(counts):
    total = sum(counts.values())
    if total == 0:
        return 0.0
    ent = 0.0
    for c in counts.values():
        if c > 0:
            p = c / total
            ent -= p * np.log2(p)
    return ent


def compute_book_metrics(words):
    """Compute H, AC1, DFA, pos_entropy from word list."""
    if not words:
        return {}
    verse_lens = defaultdict(int)
    pos_counts = Counter()
    for w in words:
        key = (w.get("chapter", 0), w.get("verse", 0))
        verse_lens[key] += 1
        pos = w.get("pos", "")
        if pos:
            pos_counts[pos] += 1

    lens = np.array([verse_lens[k] for k in sorted(verse_lens.keys())], dtype=float)
    n = len(lens)
    h = hurst_exponent_rs(lens) if n >= 20 else float("nan")
    ac1 = autocorr_lag1(lens)
    dfa = dfa_exponent(lens) if n >= 50 else float("nan")
    pe = shannon_entropy(pos_counts)
    cv = float(lens.std() / lens.mean()) if lens.mean() > 0 else 0

    return {
        "n_words": len(words),
        "n_verses": n,
        "H": round(h, 4) if not np.isnan(h) else None,
        "AC1": round(ac1, 4) if not np.isnan(ac1) else None,
        "DFA": round(dfa, 4) if not np.isnan(dfa) else None,
        "pos_entropy": round(pe, 4),
        "mean_verse_len": round(float(lens.mean()), 2),
        "CV": round(cv, 4),
        "pos_distribution": {k: v for k, v in pos_counts.most_common()},
    }


def main():
    t0 = time.time()
    log.info("=" * 70)
    log.info("FASE 15 — Script 2: Daniel Analysis")
    log.info("=" * 70)

    log.info("\nCargando corpus...")
    with open(BIBLE_JSON, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    # Extract Daniel words
    daniel = [w for w in corpus if w.get("book") == "Daniel"]
    log.info(f"  Daniel: {len(daniel)} palabras")

    # ── 1. Separate by language ──
    log.info("\n=== 1. Daniel por lengua (H vs A morph prefix) ===")

    heb_words = [w for w in daniel if w.get("morph", "").startswith("H")]
    ara_words = [w for w in daniel if w.get("morph", "").startswith("A")]
    other_words = [w for w in daniel
                   if not w.get("morph", "").startswith("H")
                   and not w.get("morph", "").startswith("A")]

    log.info(f"  Hebrew: {len(heb_words)} words")
    log.info(f"  Aramaic: {len(ara_words)} words")
    log.info(f"  Other/unmarked: {len(other_words)} words")

    heb_metrics = compute_book_metrics(heb_words)
    ara_metrics = compute_book_metrics(ara_words)
    all_metrics = compute_book_metrics(daniel)

    heb_vs_ara = {
        "hebrew": heb_metrics,
        "aramaic": ara_metrics,
        "combined": all_metrics,
        "hebrew_chapters": "1, 8-12",
        "aramaic_chapters": "2-7",
    }

    # Which section produces the borderline P_AT?
    if heb_metrics.get("pos_entropy") and ara_metrics.get("pos_entropy"):
        heb_vs_ara["which_is_more_NT_like"] = (
            "hebrew" if heb_metrics["pos_entropy"] > ara_metrics["pos_entropy"] else "aramaic"
        )

    log.info(f"  Hebrew: H={heb_metrics.get('H')}, AC1={heb_metrics.get('AC1')}, "
             f"pe={heb_metrics.get('pos_entropy')}")
    log.info(f"  Aramaic: H={ara_metrics.get('H')}, AC1={ara_metrics.get('AC1')}, "
             f"pe={ara_metrics.get('pos_entropy')}")

    with open(RESULTS_DIR / "hebrew_vs_aramaic.json", "w") as f:
        json.dump(heb_vs_ara, f, indent=2, ensure_ascii=False)

    # ── 2. Separate by content ──
    log.info("\n=== 2. Daniel por contenido ===")

    narrative = [w for w in daniel if w.get("chapter", 0) in range(1, 7)]
    apocalyptic = [w for w in daniel if w.get("chapter", 0) in range(7, 13)]

    nar_metrics = compute_book_metrics(narrative)
    apo_metrics = compute_book_metrics(apocalyptic)

    nar_vs_apo = {
        "narrative_ch1_6": nar_metrics,
        "apocalyptic_ch7_12": apo_metrics,
        "combined": all_metrics,
    }

    log.info(f"  Narrative (1-6): H={nar_metrics.get('H')}, pe={nar_metrics.get('pos_entropy')}")
    log.info(f"  Apocalyptic (7-12): H={apo_metrics.get('H')}, pe={apo_metrics.get('pos_entropy')}")

    with open(RESULTS_DIR / "narrative_vs_apocalyptic.json", "w") as f:
        json.dump(nar_vs_apo, f, indent=2, ensure_ascii=False)

    # ── 3. Comparative analysis ──
    log.info("\n=== 3. Comparación con textos similares ===")

    comparison_books = {
        "Ezekiel": "prophetic+visionary",
        "Zechariah": "apocalyptic_late",
        "Esther": "narrative_postexilic",
        "Ezra": "narrative_postexilic+aramaic",
        "Nehemiah": "narrative_postexilic",
        "Isaiah": "prophetic_classic",
        "Jeremiah": "prophetic_babylonian",
        "Lamentations": "poetic_babylonian",
    }

    comparative = {"Daniel": all_metrics}
    for book, genre in comparison_books.items():
        book_words = [w for w in corpus if w.get("book") == book]
        if book_words:
            metrics = compute_book_metrics(book_words)
            metrics["genre_label"] = genre
            comparative[book] = metrics
            log.info(f"  {book} ({genre}): H={metrics.get('H')}, "
                     f"pe={metrics.get('pos_entropy')}")

    # Euclidean distance in (H, AC1, pe) space
    dan_vec = np.array([
        all_metrics.get("H", 0) or 0,
        all_metrics.get("AC1", 0) or 0,
        all_metrics.get("pos_entropy", 0) or 0,
    ])
    distances = {}
    for book, metrics in comparative.items():
        if book == "Daniel":
            continue
        vec = np.array([
            metrics.get("H", 0) or 0,
            metrics.get("AC1", 0) or 0,
            metrics.get("pos_entropy", 0) or 0,
        ])
        dist = float(np.linalg.norm(dan_vec - vec))
        distances[book] = round(dist, 4)

    closest = min(distances.items(), key=lambda x: x[1])
    comparative["distances_to_daniel"] = distances
    comparative["closest_book"] = closest[0]
    comparative["closest_distance"] = closest[1]
    log.info(f"  Closest to Daniel: {closest[0]} (dist={closest[1]:.3f})")

    with open(RESULTS_DIR / "comparative_analysis.json", "w") as f:
        json.dump(comparative, f, indent=2, ensure_ascii=False)

    # ── 4. Dating evidence ──
    log.info("\n=== 4. Evidencia sugerente de datación ===")

    # Group by period
    babylonian = ["Ezekiel", "Jeremiah", "Lamentations"]  # ~600 BCE
    postexilic = ["Ezra", "Nehemiah", "Esther", "Zechariah"]  # ~500-400 BCE

    bab_hs = [comparative[b].get("H") for b in babylonian
              if b in comparative and comparative[b].get("H") is not None]
    post_hs = [comparative[b].get("H") for b in postexilic
               if b in comparative and comparative[b].get("H") is not None]
    bab_pes = [comparative[b].get("pos_entropy") for b in babylonian
               if b in comparative]
    post_pes = [comparative[b].get("pos_entropy") for b in postexilic
                if b in comparative]

    dan_h = all_metrics.get("H")
    dan_pe = all_metrics.get("pos_entropy")

    dating = {
        "daniel_H": dan_h,
        "daniel_pos_entropy": dan_pe,
        "babylonian_period": {
            "books": babylonian,
            "H_mean": round(float(np.mean(bab_hs)), 4) if bab_hs else None,
            "H_std": round(float(np.std(bab_hs)), 4) if bab_hs else None,
            "pe_mean": round(float(np.mean(bab_pes)), 4) if bab_pes else None,
        },
        "postexilic_period": {
            "books": postexilic,
            "H_mean": round(float(np.mean(post_hs)), 4) if post_hs else None,
            "H_std": round(float(np.std(post_hs)), 4) if post_hs else None,
            "pe_mean": round(float(np.mean(post_pes)), 4) if post_pes else None,
        },
    }

    # Which period is Daniel closer to?
    if bab_hs and post_hs and dan_h is not None:
        dist_bab = abs(dan_h - np.mean(bab_hs))
        dist_post = abs(dan_h - np.mean(post_hs))
        dating["closer_by_H"] = "babylonian" if dist_bab < dist_post else "postexilic"
        dating["H_dist_babylonian"] = round(dist_bab, 4)
        dating["H_dist_postexilic"] = round(dist_post, 4)

    if bab_pes and post_pes and dan_pe is not None:
        dist_bab_pe = abs(dan_pe - np.mean(bab_pes))
        dist_post_pe = abs(dan_pe - np.mean(post_pes))
        dating["closer_by_pe"] = "babylonian" if dist_bab_pe < dist_post_pe else "postexilic"

    dating["caveat"] = (
        "IMPORTANT: This is suggestive evidence only. Statistical proximity "
        "in (H, pos_entropy) space cannot determine authorship dates. "
        "Many confounding factors (genre, topic, redaction) affect these metrics."
    )

    log.info(f"  Daniel H={dan_h}, closer to {dating.get('closer_by_H', 'N/A')} "
             f"(dist_bab={dating.get('H_dist_babylonian')}, "
             f"dist_post={dating.get('H_dist_postexilic')})")

    with open(RESULTS_DIR / "dating_evidence.json", "w") as f:
        json.dump(dating, f, indent=2, ensure_ascii=False)

    # ── 5. Aramaic factor ──
    log.info("\n=== 5. Factor arameo (Daniel vs Esdras) ===")

    ezra_words = [w for w in corpus if w.get("book") == "Ezra"]
    ezra_heb = [w for w in ezra_words if w.get("morph", "").startswith("H")]
    ezra_ara = [w for w in ezra_words if w.get("morph", "").startswith("A")]

    ezra_heb_metrics = compute_book_metrics(ezra_heb)
    ezra_ara_metrics = compute_book_metrics(ezra_ara)

    aramaic_factor = {
        "daniel_hebrew": heb_metrics,
        "daniel_aramaic": ara_metrics,
        "ezra_hebrew": ezra_heb_metrics,
        "ezra_aramaic": ezra_ara_metrics,
        "n_ezra_hebrew": len(ezra_heb),
        "n_ezra_aramaic": len(ezra_ara),
    }

    # Does Aramaic consistently produce different pos_entropy?
    dan_ara_pe = ara_metrics.get("pos_entropy", 0)
    dan_heb_pe = heb_metrics.get("pos_entropy", 0)
    ezra_ara_pe = ezra_ara_metrics.get("pos_entropy", 0)
    ezra_heb_pe = ezra_heb_metrics.get("pos_entropy", 0)

    aramaic_factor["aramaic_effect"] = {
        "daniel_delta_pe": round(dan_ara_pe - dan_heb_pe, 4) if dan_ara_pe and dan_heb_pe else None,
        "ezra_delta_pe": round(ezra_ara_pe - ezra_heb_pe, 4) if ezra_ara_pe and ezra_heb_pe else None,
        "aramaic_consistently_different": bool(
            dan_ara_pe and dan_heb_pe and ezra_ara_pe and ezra_heb_pe and
            (dan_ara_pe - dan_heb_pe) * (ezra_ara_pe - ezra_heb_pe) > 0
        ),
    }

    log.info(f"  Daniel: Heb pe={dan_heb_pe}, Ara pe={dan_ara_pe}, "
             f"Δ={dan_ara_pe - dan_heb_pe:.3f}")
    log.info(f"  Ezra: Heb pe={ezra_heb_pe}, Ara pe={ezra_ara_pe}, "
             f"Δ={ezra_ara_pe - ezra_heb_pe:.3f}")

    with open(RESULTS_DIR / "aramaic_factor.json", "w") as f:
        json.dump(aramaic_factor, f, indent=2, ensure_ascii=False)

    # ── VERDICT ──
    log.info("\n=== VEREDICTO ===")
    verdict = {
        "primary_factor": None,
        "secondary_factors": [],
        "summary": None,
    }

    # Determine primary factor
    factors = []
    if heb_metrics.get("pos_entropy") and ara_metrics.get("pos_entropy"):
        if abs(heb_metrics["pos_entropy"] - ara_metrics["pos_entropy"]) > 0.1:
            factors.append("language_bilingualism")
    if nar_metrics.get("pos_entropy") and apo_metrics.get("pos_entropy"):
        if abs(nar_metrics["pos_entropy"] - apo_metrics["pos_entropy"]) > 0.1:
            factors.append("genre_apocalyptic")
    if aramaic_factor["aramaic_effect"].get("aramaic_consistently_different"):
        factors.append("aramaic_grammar")

    verdict["factors_identified"] = factors
    verdict["primary_factor"] = factors[0] if factors else "combination"
    verdict["secondary_factors"] = factors[1:] if len(factors) > 1 else []
    verdict["summary"] = (
        f"Daniel's borderline P_AT=0.567 is explained by {len(factors)} factor(s): "
        f"{', '.join(factors) if factors else 'a combination of bilingualism, genre, and late dating'}. "
        f"Hebrew sections: pe={dan_heb_pe:.3f}; Aramaic sections: pe={dan_ara_pe:.3f}. "
        f"Narrative (1-6): pe={nar_metrics.get('pos_entropy')}; "
        f"Apocalyptic (7-12): pe={apo_metrics.get('pos_entropy')}."
    )
    log.info(f"  {verdict['summary']}")

    with open(RESULTS_DIR / "verdict.json", "w") as f:
        json.dump(verdict, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    log.info(f"\n{'=' * 70}")
    log.info(f"Script 2 completado en {elapsed:.1f}s")


if __name__ == "__main__":
    main()
