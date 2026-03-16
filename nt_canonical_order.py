#!/usr/bin/env python3
"""
Fase 11 — Script 2: NT Canonical Order
¿El orden canónico del NT que produce H=0.993 es único?

Compara órdenes históricos, permutaciones aleatorias, y búsqueda greedy.
"""

import json
import logging
import os
import sys
import time
import numpy as np
from pathlib import Path
from scipy import stats
from collections import defaultdict

BASE = Path(__file__).resolve().parent
RESULTS_DIR = BASE / "results" / "nt_order"
LOG_DIR = BASE / "logs"
BIBLE_JSON = BASE / "bible_unified.json"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "fase11_nt_order.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


def hurst_exponent_rs(series):
    series = np.asarray(series, dtype=float)
    n = len(series)
    if n < 20:
        return float("nan"), 0.0
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
        return float("nan"), 0.0
    slope, _, r, _, _ = stats.linregress(np.log(sizes), np.log(rs_values))
    return float(slope), float(r ** 2)


# ── Órdenes históricos del NT ────────────────────────────────────────────

# Orden canónico occidental (Vulgata/moderno)
CANONICAL_ORDER = [
    "Matthew", "Mark", "Luke", "John", "Acts",
    "Romans", "1 Corinthians", "2 Corinthians", "Galatians", "Ephesians",
    "Philippians", "Colossians", "1 Thessalonians", "2 Thessalonians",
    "1 Timothy", "2 Timothy", "Titus", "Philemon",
    "Hebrews", "James", "1 Peter", "2 Peter",
    "1 John", "2 John", "3 John", "Jude", "Revelation"
]

# Orden del Códice Vaticano (~350 d.C.) — evangelios + Hechos + católicas + paulinas + Apocalipsis
VATICANUS_ORDER = [
    "Matthew", "Mark", "Luke", "John", "Acts",
    "James", "1 Peter", "2 Peter", "1 John", "2 John", "3 John", "Jude",
    "Romans", "1 Corinthians", "2 Corinthians", "Galatians", "Ephesians",
    "Philippians", "Colossians", "1 Thessalonians", "2 Thessalonians",
    "Hebrews",  # Hebrews follows Thessalonians in Vaticanus
    "1 Timothy", "2 Timothy", "Titus", "Philemon",
    "Revelation"
]

# Canon de Marción (~140 d.C.) — solo paulinas + Lucas modificado (aproximación)
MARCION_ORDER = [
    "Galatians", "1 Corinthians", "2 Corinthians", "Romans",
    "1 Thessalonians", "2 Thessalonians",
    "Ephesians",  # "Laodiceans" in Marcion
    "Colossians", "Philippians", "Philemon",
    "Luke"  # Marcion's modified Luke — we use canonical Luke as approximation
]

# Orden cronológico aproximado de composición
CHRONOLOGICAL_ORDER = [
    "1 Thessalonians", "Galatians", "1 Corinthians", "2 Corinthians",
    "Romans", "Philippians", "Philemon",
    "Mark", "Colossians",
    "Matthew", "Luke", "Acts",
    "Ephesians", "Hebrews",
    "James", "1 Peter",
    "John", "1 John", "2 John", "3 John",
    "2 Thessalonians", "1 Timothy", "2 Timothy", "Titus",
    "Jude", "2 Peter", "Revelation"
]

# Canon siríaco Peshitta (sin 2Pe, 2Jn, 3Jn, Jud, Ap)
PESHITTA_ORDER = [
    "Matthew", "Mark", "Luke", "John", "Acts",
    "James", "1 Peter", "1 John",
    "Romans", "1 Corinthians", "2 Corinthians", "Galatians", "Ephesians",
    "Philippians", "Colossians", "1 Thessalonians", "2 Thessalonians",
    "1 Timothy", "2 Timothy", "Titus", "Philemon", "Hebrews"
]


HISTORICAL_ORDERS = {
    "canonical": {
        "name": "Occidental/Vulgata (moderno)",
        "order": CANONICAL_ORDER,
        "date": "~400 d.C.",
        "source": "Vulgata latina, estandarizado en Concilio de Trento (1546)",
    },
    "vaticanus": {
        "name": "Códice Vaticano",
        "order": VATICANUS_ORDER,
        "date": "~350 d.C.",
        "source": "Codex Vaticanus (B), uno de los manuscritos más antiguos",
    },
    "marcion": {
        "name": "Canon de Marción (aprox.)",
        "order": MARCION_ORDER,
        "date": "~140 d.C.",
        "source": "Reconstrucción del canon marcionita (10 epístolas + Evangelium)",
        "caveat": "Aproximación: usamos Lucas canónico, no el Evangelium editado de Marción",
    },
    "chronological": {
        "name": "Cronológico (consenso académico)",
        "order": CHRONOLOGICAL_ORDER,
        "date": "reconstrucción moderna",
        "source": "Datación consensuada: Brown (1997), Ehrman (2012)",
    },
    "peshitta": {
        "name": "Siríaco Peshitta",
        "order": PESHITTA_ORDER,
        "date": "~400 d.C.",
        "source": "Canon siríaco antiguo (22 libros, sin 2Pe, 2-3Jn, Jud, Ap)",
        "caveat": "Canon reducido: 22 de 27 libros",
    },
}


def load_nt_by_book():
    """Carga el NT de bible_unified.json agrupado por libro."""
    log.info("Cargando bible_unified.json (NT)...")
    with open(BIBLE_JSON, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    ot_books = {"Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy",
                "Joshua", "Judges", "Ruth", "1 Samuel", "2 Samuel",
                "1 Kings", "2 Kings", "1 Chronicles", "2 Chronicles",
                "Ezra", "Nehemiah", "Esther", "Job", "Psalms",
                "Proverbs", "Ecclesiastes", "Song of Solomon",
                "Isaiah", "Jeremiah", "Lamentations", "Ezekiel",
                "Daniel", "Hosea", "Joel", "Amos", "Obadiah",
                "Jonah", "Micah", "Nahum", "Habakkuk", "Zephaniah",
                "Haggai", "Zechariah", "Malachi"}

    books = defaultdict(lambda: defaultdict(list))
    for w in corpus:
        book = w.get("book", "")
        if book in ot_books:
            continue
        ch = w.get("chapter", 0)
        vs = w.get("verse", 0)
        books[book][(ch, vs)].append(w)

    log.info(f"  {len(books)} libros NT cargados")
    for book_name in CANONICAL_ORDER:
        if book_name in books:
            n_verses = len(books[book_name])
            log.info(f"    {book_name}: {n_verses} versículos")
        else:
            log.warning(f"    {book_name}: NO ENCONTRADO")
    return books


def book_verse_lengths(books, book_name):
    """Retorna lista de longitudes de versículo de un libro."""
    if book_name not in books:
        return []
    verses = books[book_name]
    sorted_keys = sorted(verses.keys())
    return [len(verses[k]) for k in sorted_keys]


def concatenated_h(books, book_order):
    """Calcula H de la concatenación de libros en el orden dado."""
    all_lens = []
    for book_name in book_order:
        all_lens.extend(book_verse_lengths(books, book_name))
    if len(all_lens) < 20:
        return float("nan"), 0.0, len(all_lens)
    h, r2 = hurst_exponent_rs(all_lens)
    return h, r2, len(all_lens)


def main():
    t0 = time.time()
    log.info("=" * 70)
    log.info("FASE 11 — Script 2: NT Canonical Order")
    log.info("=" * 70)

    books = load_nt_by_book()

    # 1. H para cada orden histórico
    log.info("\n── Órdenes históricos ──")
    order_results = {}
    for key, info in HISTORICAL_ORDERS.items():
        order = info["order"]
        # Filter to available books
        available = [b for b in order if b in books]
        h, r2, n_verses = concatenated_h(books, available)
        order_results[key] = {
            "name": info["name"],
            "date": info["date"],
            "source": info["source"],
            "n_books": len(available),
            "n_books_requested": len(order),
            "n_verses": n_verses,
            "H": float(h) if not np.isnan(h) else None,
            "H_R2": float(r2),
            "books_used": available,
        }
        if "caveat" in info:
            order_results[key]["caveat"] = info["caveat"]
        log.info(f"  {info['name']}: H={h:.4f}, {len(available)} libros, "
                 f"{n_verses} versículos")

    with open(RESULTS_DIR / "historical_orders.json", "w") as f:
        json.dump(order_results, f, indent=2, ensure_ascii=False)

    # 2. Permutaciones aleatorias (full 27 books)
    log.info("\n── Permutaciones aleatorias (27 libros) ──")
    rng = np.random.default_rng(42)
    n_perms = 10000

    # Pre-compute verse lengths per book
    book_lens = {}
    for book_name in CANONICAL_ORDER:
        book_lens[book_name] = book_verse_lengths(books, book_name)

    h_canonical = order_results["canonical"]["H"]

    perm_h_values = []
    for i in range(n_perms):
        perm = list(CANONICAL_ORDER)
        rng.shuffle(perm)
        all_lens = []
        for book_name in perm:
            all_lens.extend(book_lens.get(book_name, []))
        h, _ = hurst_exponent_rs(all_lens)
        if not np.isnan(h):
            perm_h_values.append(float(h))
        if (i + 1) % 2000 == 0:
            log.info(f"  {i + 1}/{n_perms} permutaciones completadas")

    perm_arr = np.array(perm_h_values)
    percentile_canonical = float(np.mean(perm_arr <= h_canonical) * 100) \
        if h_canonical is not None else None
    p_value = float(np.mean(perm_arr >= h_canonical)) if h_canonical else None

    log.info(f"  H canónico: {h_canonical:.4f}")
    log.info(f"  H random: {perm_arr.mean():.4f} ± {perm_arr.std():.4f}")
    log.info(f"  Percentil del canónico: {percentile_canonical:.2f}%")
    log.info(f"  p-value (H≥canonical): {p_value:.6f}")

    perm_results = {
        "n_permutations": n_perms,
        "H_canonical": h_canonical,
        "H_random_mean": float(perm_arr.mean()),
        "H_random_std": float(perm_arr.std()),
        "H_random_min": float(perm_arr.min()),
        "H_random_max": float(perm_arr.max()),
        "percentile_canonical": percentile_canonical,
        "p_value": p_value,
    }

    # Where do historical orders rank?
    for key, info in order_results.items():
        if info["H"] is not None:
            pct = float(np.mean(perm_arr <= info["H"]) * 100)
            order_results[key]["percentile_vs_random"] = pct
            log.info(f"  {info['name']}: percentil={pct:.1f}%")

    # 3. H por libro individual
    log.info("\n── H por libro individual ──")
    book_h_results = []
    for book_name in CANONICAL_ORDER:
        lens = book_lens.get(book_name, [])
        if len(lens) >= 20:
            h, r2 = hurst_exponent_rs(lens)
            book_h_results.append({
                "book": book_name,
                "n_verses": len(lens),
                "H": float(h) if not np.isnan(h) else None,
                "H_R2": float(r2),
            })
            log.info(f"  {book_name}: H={h:.4f}")

    # 4. Greedy search for H-maximizing order
    log.info("\n── Búsqueda greedy para orden óptimo ──")
    remaining = list(CANONICAL_ORDER)
    greedy_order = []

    for step in range(len(CANONICAL_ORDER)):
        best_h = -np.inf
        best_book = None
        current_lens = []
        for book_name in greedy_order:
            current_lens.extend(book_lens.get(book_name, []))

        for candidate in remaining:
            trial_lens = current_lens + book_lens.get(candidate, [])
            if len(trial_lens) < 20:
                continue
            h, _ = hurst_exponent_rs(trial_lens)
            if not np.isnan(h) and h > best_h:
                best_h = h
                best_book = candidate

        if best_book:
            greedy_order.append(best_book)
            remaining.remove(best_book)
            if step < 5 or step == len(CANONICAL_ORDER) - 1:
                log.info(f"  Step {step + 1}: +{best_book}, H={best_h:.4f}")

    h_greedy, _, n_greedy = concatenated_h(books, greedy_order)
    log.info(f"  Greedy final: H={h_greedy:.4f}")

    # Also try reverse greedy (minimize H)
    remaining_min = list(CANONICAL_ORDER)
    greedy_min_order = []
    for step in range(len(CANONICAL_ORDER)):
        best_h = np.inf
        best_book = None
        current_lens = []
        for book_name in greedy_min_order:
            current_lens.extend(book_lens.get(book_name, []))

        for candidate in remaining_min:
            trial_lens = current_lens + book_lens.get(candidate, [])
            if len(trial_lens) < 20:
                # Just pick it
                best_book = candidate
                best_h = 0
                continue
            h, _ = hurst_exponent_rs(trial_lens)
            if not np.isnan(h) and h < best_h:
                best_h = h
                best_book = candidate

        if best_book:
            greedy_min_order.append(best_book)
            remaining_min.remove(best_book)

    h_greedy_min, _, _ = concatenated_h(books, greedy_min_order)
    log.info(f"  Greedy MIN: H={h_greedy_min:.4f}")

    greedy_results = {
        "greedy_max_order": greedy_order,
        "greedy_max_H": float(h_greedy) if not np.isnan(h_greedy) else None,
        "greedy_min_order": greedy_min_order,
        "greedy_min_H": float(h_greedy_min) if not np.isnan(h_greedy_min) else None,
        "canonical_H": h_canonical,
        "canonical_is_optimal": h_canonical is not None and h_greedy is not None
                                 and abs(h_canonical - h_greedy) < 0.01,
    }

    # Check if greedy order resembles any historical order
    def order_similarity(o1, o2):
        """Kendall tau between two orderings of books."""
        common = [b for b in o1 if b in o2]
        if len(common) < 3:
            return 0.0, 1.0
        idx1 = [o1.index(b) for b in common]
        idx2 = [o2.index(b) for b in common]
        tau, p = stats.kendalltau(idx1, idx2)
        return float(tau), float(p)

    similarities = {}
    for key, info in HISTORICAL_ORDERS.items():
        tau, p = order_similarity(greedy_order, info["order"])
        similarities[key] = {"kendall_tau": tau, "p_value": p}
        log.info(f"  Greedy vs {info['name']}: τ={tau:.4f}, p={p:.4f}")
    greedy_results["similarity_to_historical"] = similarities

    # Save all results
    all_results = {
        "historical_orders": order_results,
        "permutation_test": perm_results,
        "book_h_individual": book_h_results,
        "greedy_search": greedy_results,
    }

    with open(RESULTS_DIR / "h_by_order.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    with open(RESULTS_DIR / "optimal_order_search.json", "w") as f:
        json.dump(greedy_results, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    log.info(f"\n{'=' * 70}")
    log.info(f"Script 2 completado en {elapsed:.1f}s")

    return all_results


if __name__ == "__main__":
    main()
