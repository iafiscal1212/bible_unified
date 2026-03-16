#!/usr/bin/env python3
"""
Fase 9 — Script 1: El NT como caso especial
¿Por qué H=0.993 pero MPS no significativo (p=0.223)?
Análisis por tradición textual (TAGNT) y por grupo cronológico.
"""

import json
import logging
import os
import sys
import time
import numpy as np
from pathlib import Path
from scipy import stats
from collections import Counter, defaultdict
from urllib.request import urlretrieve
from urllib.error import URLError

# ── Configuración ────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent
RESULTS_DIR = BASE / "results" / "nt_special"
LOG_DIR = BASE / "logs"
BIBLE_JSON = BASE / "bible_unified.json"
TAGNT_DIR = BASE / "results" / "comparison_corpora" / "tagnt"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
TAGNT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "fase9_nt_special.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ── Métricas (copiadas de analyze_dss.py) ────────────────────────────────

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
    log_n = np.log(sizes)
    log_rs = np.log(rs_values)
    slope, _, r, _, _ = stats.linregress(log_n, log_rs)
    return float(slope), float(r ** 2)


def dfa_exponent(series):
    series = np.asarray(series, dtype=float)
    n = len(series)
    if n < 20:
        return float("nan"), 0.0
    y = np.cumsum(series - series.mean())
    min_box, max_box = 4, n // 4
    sizes, flucts = [], []
    box = min_box
    while box <= max_box:
        sizes.append(box)
        n_boxes = n // box
        rms_list = []
        for i in range(n_boxes):
            seg = y[i * box:(i + 1) * box]
            coeffs = np.polyfit(np.arange(box), seg, 1)
            trend = np.polyval(coeffs, np.arange(box))
            rms_list.append(np.sqrt(np.mean((seg - trend) ** 2)))
        if rms_list:
            flucts.append(np.mean(rms_list))
        box = int(box * 1.5)
        if box == sizes[-1]:
            box += 1
    if len(sizes) < 3:
        return float("nan"), 0.0
    slope, _, r, _, _ = stats.linregress(np.log(sizes), np.log(flucts))
    return float(slope), float(r ** 2)


def compute_bond_dimension(series, max_lag=256, threshold=0.99):
    series = np.asarray(series, dtype=float)
    n = len(series)
    if n < max_lag * 2:
        max_lag = n // 4
    if max_lag < 4:
        return 1
    mean, var = series.mean(), series.var()
    if var == 0:
        return 1
    acf = np.correlate(series - mean, series - mean, mode="full")
    acf = acf[n - 1:n - 1 + max_lag] / (var * n)
    from scipy.linalg import toeplitz
    T = toeplitz(acf)
    try:
        _, sigma, _ = np.linalg.svd(T)
    except np.linalg.LinAlgError:
        return max_lag
    total = np.sum(sigma ** 2)
    if total == 0:
        return 1
    cumvar = np.cumsum(sigma ** 2) / total
    chi = int(np.searchsorted(cumvar, threshold) + 1)
    return min(chi, max_lag)


def permutation_test_chi(series, n_perm=1000, max_lag=64, threshold=0.99):
    series = np.asarray(series, dtype=float)
    chi_obs = compute_bond_dimension(series, max_lag=max_lag, threshold=threshold)
    chi_rand = []
    for i in range(n_perm):
        perm = np.random.permutation(series)
        chi_rand.append(compute_bond_dimension(perm, max_lag=max_lag, threshold=threshold))
    chi_rand = np.array(chi_rand)
    p_value = float(np.mean(chi_rand <= chi_obs))
    return {
        "chi_obs": int(chi_obs),
        "chi_rand_mean": float(chi_rand.mean()),
        "p_value": p_value,
        "significant": bool(p_value < 0.05),
    }


# ── Grupos cronológicos del NT ───────────────────────────────────────────

# Estimaciones académicas consensuadas (rangos)
NT_CHRONOLOGY = {
    # Cartas paulinas auténticas (~50-60 d.C.)
    "pauline_early": {
        "label": "Paulinas auténticas (~50-60 d.C.)",
        "books": ["Romans", "1 Corinthians", "2 Corinthians", "Galatians",
                  "Philippians", "1 Thessalonians", "Philemon"],
        "date_range": (50, 60),
        "authority_type": "testimonio_mediado",
    },
    # Evangelios sinópticos (~70-90 d.C.)
    "synoptic": {
        "label": "Sinópticos (~70-90 d.C.)",
        "books": ["Matthew", "Mark", "Luke"],
        "date_range": (70, 90),
        "authority_type": "testimonio_mediado",
    },
    # Escritos joánicos (~90-100 d.C.)
    "johannine": {
        "label": "Joánicos (~90-100 d.C.)",
        "books": ["John", "1 John", "2 John", "3 John", "Revelation"],
        "date_range": (90, 100),
        "authority_type": "testimonio_mediado",
    },
    # Escritos tardíos (~80-120 d.C.)
    "late": {
        "label": "Tardíos (~80-120 d.C.)",
        "books": ["Hebrews", "James", "1 Peter", "2 Peter", "Jude",
                  "1 Timothy", "2 Timothy", "Titus", "Acts",
                  "Ephesians", "Colossians", "2 Thessalonians"],
        "date_range": (80, 120),
        "authority_type": "testimonio_mediado",
    },
}


# ── Descarga y parseo TAGNT ──────────────────────────────────────────────

TAGNT_URLS = {
    "Mat-Jhn": "https://raw.githubusercontent.com/STEPBible/STEPBible-Data/master/Translators%20Amalgamated%20OT%2BNT/TAGNT%20Mat-Jhn%20-%20Translators%20Amalgamated%20Greek%20NT%20-%20STEPBible.org%20CC-BY.txt",
    "Act-Rev": "https://raw.githubusercontent.com/STEPBible/STEPBible-Data/master/Translators%20Amalgamated%20OT%2BNT/TAGNT%20Act-Rev%20-%20Translators%20Amalgamated%20Greek%20NT%20-%20STEPBible.org%20CC-BY.txt",
}

# Tradiciones textuales conocidas en TAGNT
TRADITIONS = ["NA28", "TR", "SBL", "TH", "Byz", "WH", "Treg"]


def download_tagnt():
    """Descarga archivos TAGNT si no existen."""
    downloaded = {}
    for name, url in TAGNT_URLS.items():
        fpath = TAGNT_DIR / f"TAGNT_{name}.txt"
        if fpath.exists() and fpath.stat().st_size > 100000:
            log.info(f"TAGNT {name} ya existe ({fpath.stat().st_size:,} bytes)")
            downloaded[name] = fpath
            continue
        try:
            log.info(f"Descargando TAGNT {name}...")
            urlretrieve(url, fpath)
            log.info(f"  → {fpath.stat().st_size:,} bytes")
            downloaded[name] = fpath
        except (URLError, Exception) as e:
            log.warning(f"  No se pudo descargar TAGNT {name}: {e}")
    return downloaded


def parse_tagnt(fpath):
    """
    Parsea un archivo TAGNT y extrae verse_lengths por tradición.
    Returns: dict[tradition] -> dict[(book,chap,verse)] -> word_count
    """
    verse_words = defaultdict(lambda: defaultdict(int))
    current_ref = None

    with open(fpath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("$"):
                continue

            parts = line.split("\t")
            if len(parts) < 2:
                continue

            # Referencia: parece formato "BookName.Chap.Verse"
            ref_part = parts[0].strip()
            if "." in ref_part and not ref_part.startswith("#"):
                # Es una línea de datos con referencia
                try:
                    # Extraer referencia bíblica
                    ref_clean = ref_part.lstrip("# ")
                    segments = ref_clean.split(".")
                    if len(segments) >= 3:
                        book = segments[0]
                        chap = int(segments[1])
                        verse = int(segments[2].split(" ")[0])
                        current_ref = (book, chap, verse)
                except (ValueError, IndexError):
                    pass

            if current_ref is None:
                continue

            # Buscar columna de tradiciones (contiene letras como N, K, O, etc.)
            # En TAGNT, cada línea de palabra tiene info de en qué tradiciones aparece
            # Asumimos que la presencia de la palabra cuenta para todas las tradiciones listadas
            word_found = False
            for part in parts:
                if "=" in part:
                    continue
                # Si tiene caracteres griegos, es una palabra
                if any("\u0370" <= c <= "\u03FF" or "\u1F00" <= c <= "\u1FFF" for c in part):
                    word_found = True
                    break

            if word_found:
                # Determinar en qué tradiciones aparece esta palabra
                # Buscamos marcadores de tradición en la línea
                line_str = "\t".join(parts)

                # Por defecto, contar para "all" (todo el NT combinado)
                verse_words["all"][current_ref] += 1

                # Buscar tradiciones específicas
                for trad in TRADITIONS:
                    # Si la tradición está mencionada en la línea
                    if trad in line_str:
                        verse_words[trad][current_ref] += 1

    return verse_words


def try_parse_tagnt_simple(fpath):
    """
    Parseo simplificado: cuenta palabras griegas por versículo.
    No intenta separar tradiciones (demasiado complejo sin documentación exacta).
    """
    verse_words = defaultdict(int)
    current_ref = None

    with open(fpath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("$"):
                continue

            parts = line.split("\t")
            if len(parts) < 2:
                continue

            # Buscar referencia
            for part in parts:
                part = part.strip()
                # Formato típico: "Mat.1.1" o similar
                segs = part.split(".")
                if len(segs) == 3:
                    try:
                        book = segs[0]
                        chap = int(segs[1])
                        verse = int(segs[2].split()[0].split("#")[0])
                        current_ref = (book, chap, verse)
                        break
                    except (ValueError, IndexError):
                        pass

            if current_ref is None:
                continue

            # Contar palabras griegas en esta línea
            for part in parts:
                part = part.strip()
                if any("\u0370" <= c <= "\u03FF" or "\u1F00" <= c <= "\u1FFF" for c in part):
                    verse_words[current_ref] += 1

    return verse_words


# ── Análisis principal ───────────────────────────────────────────────────

def load_nt_data():
    """Carga datos del NT de bible_unified.json."""
    log.info("Cargando bible_unified.json...")
    with open(BIBLE_JSON) as f:
        data = json.load(f)

    nt_words = [w for w in data if w.get("corpus") == "NT"]
    log.info(f"  NT: {len(nt_words):,} palabras")

    # Agrupar por (book, chapter, verse) → lista de POS
    verses = defaultdict(list)
    book_verses = defaultdict(lambda: defaultdict(list))
    for w in nt_words:
        key = (w["book"], w["chapter"], w["verse"])
        verses[key].append(w)
        book_verses[w["book"]][key].append(w)

    return nt_words, verses, book_verses


def compute_metrics_for_series(series, label="", n_perm=1000):
    """Calcula H, α, χ, MPS_p para una serie de longitudes de versículo."""
    series = np.array(series, dtype=float)
    if len(series) < 30:
        return {
            "n_verses": len(series),
            "H": None, "H_R2": None,
            "alpha": None, "alpha_R2": None,
            "chi": None, "mps_p": None, "mps_significant": None,
            "mean_verse_len": float(series.mean()) if len(series) > 0 else None,
            "note": f"Insuficiente (<30 versos) para {label}"
        }

    H, H_r2 = hurst_exponent_rs(series)
    alpha, alpha_r2 = dfa_exponent(series)
    chi = compute_bond_dimension(series)

    # Permutation test (reducido para velocidad)
    mps = permutation_test_chi(series, n_perm=n_perm, max_lag=64)

    return {
        "n_verses": len(series),
        "mean_verse_len": float(series.mean()),
        "std_verse_len": float(series.std()),
        "H": H, "H_R2": H_r2,
        "alpha": alpha, "alpha_R2": alpha_r2,
        "chi": mps["chi_obs"],
        "mps_p": mps["p_value"],
        "mps_significant": mps["significant"],
    }


def analyze_by_chronological_group(book_verses):
    """Analiza métricas por grupo cronológico del NT."""
    log.info("\n=== Análisis por grupo cronológico ===")
    results = {}

    for group_id, group_info in NT_CHRONOLOGY.items():
        log.info(f"\nGrupo: {group_info['label']}")
        group_books = group_info["books"]

        # Recoger todos los versículos del grupo en orden
        all_lens = []
        book_stats = {}
        for book_name in group_books:
            if book_name not in book_verses:
                log.info(f"  {book_name}: no encontrado en datos")
                continue
            bv = book_verses[book_name]
            # Ordenar por (chapter, verse)
            sorted_keys = sorted(bv.keys(), key=lambda k: (k[1], k[2]))
            lens = [len(bv[k]) for k in sorted_keys]
            all_lens.extend(lens)
            book_stats[book_name] = {
                "n_verses": len(lens),
                "mean_verse_len": float(np.mean(lens)) if lens else 0,
                "n_words": sum(lens),
            }
            log.info(f"  {book_name}: {len(lens)} versos, mean_len={np.mean(lens):.1f}")

        # VN ratio por grupo
        group_words = []
        for book_name in group_books:
            if book_name in book_verses:
                for key, words in book_verses[book_name].items():
                    group_words.extend(words)

        n_verbs = sum(1 for w in group_words if w.get("pos") == "verb")
        n_nouns = sum(1 for w in group_words if w.get("pos") == "noun")
        vn_ratio = n_verbs / n_nouns if n_nouns > 0 else float("nan")

        # Métricas de la serie concatenada
        metrics = compute_metrics_for_series(all_lens, label=group_info["label"])
        metrics["group_id"] = group_id
        metrics["label"] = group_info["label"]
        metrics["date_range"] = list(group_info["date_range"])
        metrics["books_found"] = list(book_stats.keys())
        metrics["book_details"] = book_stats
        metrics["vn_ratio"] = float(vn_ratio)
        metrics["n_total_words"] = sum(s["n_words"] for s in book_stats.values())

        results[group_id] = metrics
        log.info(f"  → H={metrics['H']}, α={metrics['alpha']}, "
                f"MPS_p={metrics['mps_p']}, V/N={vn_ratio:.3f}")

    return results


def analyze_per_book(book_verses):
    """Calcula métricas por libro individual del NT."""
    log.info("\n=== Análisis por libro individual del NT ===")
    results = {}

    for book_name in sorted(book_verses.keys()):
        bv = book_verses[book_name]
        sorted_keys = sorted(bv.keys(), key=lambda k: (k[1], k[2]))
        lens = [len(bv[k]) for k in sorted_keys]

        if len(lens) < 20:
            log.info(f"  {book_name}: {len(lens)} versos (muy pocos, skip)")
            continue

        H, H_r2 = hurst_exponent_rs(lens)
        alpha, alpha_r2 = dfa_exponent(lens)

        # VN ratio
        all_words = []
        for key in sorted_keys:
            all_words.extend(bv[key])
        n_verbs = sum(1 for w in all_words if w.get("pos") == "verb")
        n_nouns = sum(1 for w in all_words if w.get("pos") == "noun")
        vn_ratio = n_verbs / n_nouns if n_nouns > 0 else float("nan")

        # Determinar grupo cronológico
        chrono_group = "unknown"
        for gid, ginfo in NT_CHRONOLOGY.items():
            if book_name in ginfo["books"]:
                chrono_group = gid
                break

        results[book_name] = {
            "n_verses": len(lens),
            "n_words": sum(lens),
            "mean_verse_len": float(np.mean(lens)),
            "H": H, "H_R2": H_r2,
            "alpha": alpha, "alpha_R2": alpha_r2,
            "vn_ratio": float(vn_ratio),
            "chronological_group": chrono_group,
        }
        log.info(f"  {book_name}: {len(lens)} v, H={H:.3f}, α={alpha:.3f}, V/N={vn_ratio:.3f}")

    return results


def analyze_tagnt_traditions(tagnt_files):
    """
    Analiza TAGNT: si disponible, calcula métricas por tradición textual.
    Si el parseo es complejo, documenta la disponibilidad y usa verse-length simple.
    """
    log.info("\n=== Análisis de tradiciones TAGNT ===")

    if not tagnt_files:
        return {
            "status": "not_available",
            "note": "TAGNT files could not be downloaded. Tradition comparison not possible.",
        }

    # Intentar parsear
    all_verse_words = defaultdict(int)
    for name, fpath in tagnt_files.items():
        log.info(f"Parseando TAGNT {name}...")
        try:
            vw = try_parse_tagnt_simple(fpath)
            for ref, count in vw.items():
                all_verse_words[ref] += count
            log.info(f"  → {len(vw)} versículos detectados")
        except Exception as e:
            log.warning(f"  Error parseando {name}: {e}")

    if not all_verse_words:
        return {
            "status": "parse_failed",
            "note": "Could not parse TAGNT data. Format may have changed.",
        }

    # Construir serie de verse lengths
    sorted_refs = sorted(all_verse_words.keys())
    tagnt_lens = [all_verse_words[ref] for ref in sorted_refs]

    log.info(f"TAGNT total: {len(tagnt_lens)} versículos, {sum(tagnt_lens)} palabras")

    # Comparar con SBLGNT (nuestro NT base)
    metrics = compute_metrics_for_series(tagnt_lens, label="TAGNT_all")
    metrics["status"] = "available"
    metrics["n_total_verses"] = len(tagnt_lens)
    metrics["n_total_words"] = sum(tagnt_lens)
    metrics["note"] = ("TAGNT parsed as single combined text. "
                       "Individual tradition separation requires deeper format analysis. "
                       "Verse-level word counts may differ from SBLGNT due to variant inclusion.")

    return metrics


def analyze_nt_composition(book_verses):
    """Análisis de composición del NT: qué hace especial al NT respecto de H y MPS."""
    log.info("\n=== Análisis de composición del NT ===")

    # 1. Heterogeneidad entre libros: medir dispersión de mean_verse_len
    book_means = {}
    book_genres = {}
    genre_map = {
        "Matthew": "gospel", "Mark": "gospel", "Luke": "gospel", "John": "gospel",
        "Acts": "narrative",
        "Romans": "epistle", "1 Corinthians": "epistle", "2 Corinthians": "epistle",
        "Galatians": "epistle", "Ephesians": "epistle", "Philippians": "epistle",
        "Colossians": "epistle", "1 Thessalonians": "epistle", "2 Thessalonians": "epistle",
        "1 Timothy": "epistle", "2 Timothy": "epistle", "Titus": "epistle",
        "Philemon": "epistle", "Hebrews": "epistle", "James": "epistle",
        "1 Peter": "epistle", "2 Peter": "epistle", "1 John": "epistle",
        "2 John": "epistle", "3 John": "epistle", "Jude": "epistle",
        "Revelation": "apocalyptic",
    }

    for book_name, bv in book_verses.items():
        sorted_keys = sorted(bv.keys(), key=lambda k: (k[1], k[2]))
        lens = [len(bv[k]) for k in sorted_keys]
        if lens:
            book_means[book_name] = float(np.mean(lens))
            book_genres[book_name] = genre_map.get(book_name, "unknown")

    # Dispersión entre libros
    means_arr = np.array(list(book_means.values()))
    cv = float(means_arr.std() / means_arr.mean()) if means_arr.mean() > 0 else 0

    # Dispersión por género
    genre_stats = defaultdict(list)
    for book, mean_len in book_means.items():
        g = book_genres.get(book, "unknown")
        genre_stats[g].append(mean_len)

    genre_summary = {}
    for genre, vals in genre_stats.items():
        arr = np.array(vals)
        genre_summary[genre] = {
            "n_books": len(vals),
            "mean_verse_len": float(arr.mean()),
            "std_verse_len": float(arr.std()),
        }

    # 2. Transiciones de género: contar cuántas veces cambia el género
    # en la secuencia canónica del NT
    canonical_order = [
        "Matthew", "Mark", "Luke", "John", "Acts",
        "Romans", "1 Corinthians", "2 Corinthians", "Galatians",
        "Ephesians", "Philippians", "Colossians",
        "1 Thessalonians", "2 Thessalonians",
        "1 Timothy", "2 Timothy", "Titus", "Philemon",
        "Hebrews", "James", "1 Peter", "2 Peter",
        "1 John", "2 John", "3 John", "Jude", "Revelation"
    ]
    genre_transitions = 0
    prev_genre = None
    for book in canonical_order:
        g = genre_map.get(book, "unknown")
        if prev_genre is not None and g != prev_genre:
            genre_transitions += 1
        prev_genre = g

    return {
        "n_books": len(book_means),
        "coefficient_of_variation_verse_len": cv,
        "mean_of_means": float(means_arr.mean()),
        "std_of_means": float(means_arr.std()),
        "genre_summary": genre_summary,
        "n_genre_transitions_canonical": genre_transitions,
        "n_distinct_genres": len(genre_summary),
        "hypothesis": (
            "El H alto del NT (~0.993) se explica por cambios de régimen: "
            f"{len(book_means)} libros heterogéneos con CV={cv:.3f} entre medias de verso, "
            f"{genre_transitions} transiciones de género en el orden canónico, y "
            f"{len(genre_summary)} géneros distintos (evangelio, epístola, narrativa, apocalíptico). "
            "Cada transición introduce una correlación a larga distancia en la serie de "
            "longitudes de versículo, inflando H. Pero la autocorrelación resultante es "
            "COMPLEJA (χ=239, no compresible → MPS p=0.223), a diferencia del cluster AT-like "
            "donde la autocorrelación es SIMPLE (χ∈[49,122], compresible → MPS p=0.000)."
        ),
        "canonization_note": (
            "La canonización política del NT (Concilio de Nicea 325, Canon de Atanasio 367) "
            "NO es la causa del H alto: la heterogeneidad existía antes de la canonización "
            "(los 27 libros ya circulaban independientemente). Lo que la canonización hizo fue "
            "FIJAR el orden canónico, creando la secuencia específica de transiciones que genera "
            "el H observado. Pero el H dependería del orden de los libros, no de su contenido."
        ),
    }


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    log.info("=" * 60)
    log.info("FASE 9 — Script 1: NT como caso especial")
    log.info("=" * 60)

    # 1. Cargar datos NT
    nt_words, verses, book_verses = load_nt_data()

    # 2. Intentar descargar y analizar TAGNT
    tagnt_files = download_tagnt()
    tagnt_results = analyze_tagnt_traditions(tagnt_files)

    # 3. Análisis por grupo cronológico
    chrono_results = analyze_by_chronological_group(book_verses)

    # 4. Análisis por libro individual
    per_book = analyze_per_book(book_verses)

    # 5. Análisis de composición
    composition = analyze_nt_composition(book_verses)

    # 6. Test: ¿H depende del orden canónico?
    log.info("\n=== Test: ¿H depende del orden canónico? ===")
    # Serie canónica (orden real del NT)
    canonical_lens = []
    canonical_order = [
        "Matthew", "Mark", "Luke", "John", "Acts",
        "Romans", "1 Corinthians", "2 Corinthians", "Galatians",
        "Ephesians", "Philippians", "Colossians",
        "1 Thessalonians", "2 Thessalonians",
        "1 Timothy", "2 Timothy", "Titus", "Philemon",
        "Hebrews", "James", "1 Peter", "2 Peter",
        "1 John", "2 John", "3 John", "Jude", "Revelation"
    ]
    for book_name in canonical_order:
        if book_name in book_verses:
            bv = book_verses[book_name]
            sorted_keys = sorted(bv.keys(), key=lambda k: (k[1], k[2]))
            canonical_lens.extend([len(bv[k]) for k in sorted_keys])

    H_canonical, _ = hurst_exponent_rs(canonical_lens)
    log.info(f"H(canonical order) = {H_canonical:.4f}")

    # Probar 50 órdenes aleatorios de libros
    random_H_values = []
    all_book_names = list(book_verses.keys())
    for trial in range(50):
        np.random.seed(trial)
        shuffled = np.random.permutation(all_book_names)
        shuffled_lens = []
        for book_name in shuffled:
            bv = book_verses[book_name]
            sorted_keys = sorted(bv.keys(), key=lambda k: (k[1], k[2]))
            shuffled_lens.extend([len(bv[k]) for k in sorted_keys])
        h_shuf, _ = hurst_exponent_rs(shuffled_lens)
        random_H_values.append(h_shuf)

    random_H = np.array(random_H_values)
    order_test = {
        "H_canonical": float(H_canonical),
        "H_random_mean": float(random_H.mean()),
        "H_random_std": float(random_H.std()),
        "H_random_min": float(random_H.min()),
        "H_random_max": float(random_H.max()),
        "p_value_canonical_vs_random": float(stats.ttest_1samp(random_H_values, H_canonical).pvalue),
        "conclusion": ("" if abs(H_canonical - random_H.mean()) > 2 * random_H.std()
                       else "H del NT NO depende significativamente del orden canónico. "
                            "La heterogeneidad entre libros genera H alto independientemente del orden.")
    }
    log.info(f"H(random orders): mean={random_H.mean():.4f} ± {random_H.std():.4f}")
    log.info(f"  p(canonical vs random) = {order_test['p_value_canonical_vs_random']:.4f}")

    # ── Guardar resultados ───────────────────────────────────────────────

    # Tradiciones TAGNT
    with open(RESULTS_DIR / "tradition_comparison.json", "w") as f:
        json.dump(tagnt_results, f, indent=2, ensure_ascii=False)

    # Grupos cronológicos
    with open(RESULTS_DIR / "chronological_groups.json", "w") as f:
        json.dump(chrono_results, f, indent=2, ensure_ascii=False)

    # Composición
    composition["order_test"] = order_test
    with open(RESULTS_DIR / "nt_composition_analysis.json", "w") as f:
        json.dump(composition, f, indent=2, ensure_ascii=False)

    # Por libro
    with open(RESULTS_DIR / "nt_per_book.json", "w") as f:
        json.dump(per_book, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    log.info(f"\n{'=' * 60}")
    log.info(f"Script 1 completado en {elapsed:.1f}s")
    log.info(f"Resultados en {RESULTS_DIR}")


if __name__ == "__main__":
    main()
