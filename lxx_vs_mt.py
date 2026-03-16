#!/usr/bin/env python3
"""
Fase 9 — Script 2: LXX (Septuaginta) vs MT (Masorético)
¿Dos líneas de transmisión independientes del mismo texto producen el mismo H?
"""

import json
import logging
import os
import sys
import time
import re
import numpy as np
from pathlib import Path
from scipy import stats
from collections import defaultdict
from urllib.request import urlretrieve
from urllib.error import URLError
import subprocess

# ── Configuración ────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent
RESULTS_DIR = BASE / "results" / "lxx_mt"
LOG_DIR = BASE / "logs"
BIBLE_JSON = BASE / "bible_unified.json"
LXX_DIR = BASE / "results" / "comparison_corpora" / "lxx"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
LXX_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "fase9_lxx_mt.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ── Métricas ─────────────────────────────────────────────────────────────

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


# ── Búsqueda y descarga de LXX ──────────────────────────────────────────

LXX_SOURCES = [
    {
        "name": "LXX-Rahlfs-Hanhart (eliranwong)",
        "repo": "https://github.com/eliranwong/LXX-Rahlfs-Hanhart",
        "type": "github_clone",
    },
    {
        "name": "OpenScriptures LXX",
        "repo": "https://github.com/openscriptures/LXX",
        "type": "github_clone",
    },
    {
        "name": "CCAT LXX Morphology (jtauber)",
        "repo": "https://github.com/jtauber/lxx-swete",
        "type": "github_clone",
    },
    {
        "name": "Perseus Digital Library (tlg0527)",
        "repo": "https://github.com/PerseusDL/canonical-greekLit",
        "type": "github_clone",
        "subpath": "data/tlg0527",
    },
]


def try_clone_repo(repo_url, target_dir, depth=1):
    """Intenta clonar un repositorio Git."""
    if target_dir.exists() and any(target_dir.iterdir()):
        log.info(f"  Ya existe: {target_dir}")
        return True
    try:
        log.info(f"  Clonando {repo_url}...")
        result = subprocess.run(
            ["git", "clone", "--depth", str(depth), repo_url, str(target_dir)],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0:
            log.info(f"  → Clonado exitosamente")
            return True
        else:
            log.warning(f"  → Error: {result.stderr[:200]}")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        log.warning(f"  → Error: {e}")
        return False


def find_lxx_data():
    """Busca y descarga datos de la LXX de múltiples fuentes."""
    log.info("\n=== Buscando datos de la LXX ===")
    availability = []

    for source in LXX_SOURCES:
        target = LXX_DIR / source["name"].replace(" ", "_").replace("(", "").replace(")", "")
        log.info(f"\nFuente: {source['name']}")

        success = try_clone_repo(source["repo"], target)

        if success:
            # Buscar archivos útiles
            text_files = []
            for ext in ["*.txt", "*.xml", "*.csv", "*.tsv", "*.json"]:
                text_files.extend(list(target.rglob(ext)))

            # Buscar archivos del Pentateuco específicamente
            pentateuch_keywords = ["genesis", "gen", "exodus", "exod", "leviticus", "lev",
                                   "numbers", "num", "deuteronomy", "deut"]
            penta_files = []
            for tf in text_files:
                name_lower = tf.name.lower()
                if any(k in name_lower for k in pentateuch_keywords):
                    penta_files.append(str(tf))

            availability.append({
                "source": source["name"],
                "repo": source["repo"],
                "cloned": True,
                "total_text_files": len(text_files),
                "pentateuch_files": penta_files[:10],
                "sample_files": [str(f) for f in text_files[:10]],
                "path": str(target),
            })
        else:
            availability.append({
                "source": source["name"],
                "repo": source["repo"],
                "cloned": False,
                "error": "Could not clone repository",
            })

    return availability


def parse_ccat_lxx(fpath):
    """
    Parsea archivo LXX en formato CCAT (jtauber/lxx-swete).
    Formato: BBCCCVVV WORD_SURFACE WORD_NORMALIZED (una línea por palabra).
    Ref: 01001001 = libro 01, capítulo 001, versículo 001.
    Devuelve dict[(chapter, verse)] → word_count.
    """
    verses = defaultdict(int)
    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            ref = parts[0]
            if len(ref) != 8 or not ref.isdigit():
                continue
            try:
                chap = int(ref[2:5])
                verse = int(ref[5:8])
                verses[(chap, verse)] += 1
            except (ValueError, IndexError):
                continue
    return dict(verses) if verses else None


def parse_lxx_verse_lengths(lxx_path, book_name):
    """
    Extrae longitudes de versículo de la LXX para un libro dado.
    Busca archivos CCAT primero (formato conocido), luego intenta otros formatos.
    """
    # Mapeo de nombres a prefijos de archivo CCAT
    ccat_files = {
        "Genesis": "01.Gen.txt",
        "Exodus": "02.Exo.txt",
        "Leviticus": "03.Lev.txt",
        "Numbers": "04.Num.txt",
        "Deuteronomy": "05.Deu.txt",
    }

    book_aliases = {
        "Genesis": ["gen", "genesis", "01"],
        "Exodus": ["exod", "exodus", "02"],
        "Leviticus": ["lev", "leviticus", "03"],
        "Numbers": ["num", "numbers", "04"],
        "Deuteronomy": ["deut", "deuteronomy", "05"],
    }

    # 1. Intentar formato CCAT primero (más fiable)
    if book_name in ccat_files:
        ccat_name = ccat_files[book_name]
        for root, dirs, files in os.walk(lxx_path):
            if ccat_name in files:
                fpath = os.path.join(root, ccat_name)
                # Verificar que NO es el directorio 'norm/' (preferir raíz)
                result = parse_ccat_lxx(fpath)
                if result:
                    return result, fpath

    # 2. Buscar por alias en cualquier archivo
    aliases = book_aliases.get(book_name, [book_name.lower()])
    for root, dirs, files in os.walk(lxx_path):
        for fname in files:
            if not fname.endswith(".txt"):
                continue
            name_lower = fname.lower()
            if any(a in name_lower for a in aliases):
                fpath = os.path.join(root, fname)
                # Probar formato CCAT
                result = parse_ccat_lxx(fpath)
                if result:
                    return result, fpath

    # 3. Buscar en archivos XML (formato TEI/OSIS)
    for root, dirs, files in os.walk(lxx_path):
        for fname in files:
            if not fname.endswith(".xml"):
                continue
            name_lower = fname.lower()
            if any(a in name_lower for a in aliases):
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                    verse_pattern = re.compile(
                        r'<verse[^>]*osisID="[^"]*?\.(\d+)\.(\d+)"[^>]*>(.*?)</verse>',
                        re.DOTALL
                    )
                    matches = verse_pattern.findall(content)
                    if matches:
                        verses = defaultdict(int)
                        for chap, verse, text in matches:
                            clean = re.sub(r"<[^>]+>", " ", text)
                            words = clean.split()
                            if words:
                                verses[(int(chap), int(verse))] = len(words)
                        if verses:
                            return dict(verses), fpath
                except Exception as e:
                    log.warning(f"  Error leyendo {fpath}: {e}")

    return None, f"No files found for {book_name}"


# ── Extracción MT del bible_unified.json ─────────────────────────────────

def load_mt_verse_lengths():
    """Extrae longitudes de versículo del Pentateuco masorético (WLC)."""
    log.info("Cargando Pentateuco MT de bible_unified.json...")
    with open(BIBLE_JSON) as f:
        data = json.load(f)

    pentateuch = ["Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy"]
    mt_data = {}

    for book_name in pentateuch:
        verses = defaultdict(int)
        for w in data:
            if w.get("corpus") == "OT" and w.get("book") == book_name:
                key = (w["chapter"], w["verse"])
                verses[key] += 1
        if verses:
            mt_data[book_name] = dict(verses)
            sorted_keys = sorted(verses.keys())
            lens = [verses[k] for k in sorted_keys]
            log.info(f"  {book_name}: {len(lens)} versos, {sum(lens)} palabras, "
                    f"mean_len={np.mean(lens):.1f}")

    return mt_data


# ── Comparación LXX vs MT ───────────────────────────────────────────────

def compare_lxx_mt(lxx_data, mt_data, book_name):
    """Compara métricas entre LXX y MT para un libro."""
    if book_name not in lxx_data or lxx_data[book_name] is None:
        return None

    lxx_verses = lxx_data[book_name]
    mt_verses = mt_data.get(book_name, {})

    if not lxx_verses or not mt_verses:
        return None

    # Serie LXX (ordenada por capítulo:versículo)
    sorted_lxx = sorted(lxx_verses.keys())
    lxx_lens = [lxx_verses[k] for k in sorted_lxx]

    # Serie MT (ordenada)
    sorted_mt = sorted(mt_verses.keys())
    mt_lens = [mt_verses[k] for k in sorted_mt]

    if len(lxx_lens) < 30 or len(mt_lens) < 30:
        return {
            "book": book_name,
            "lxx_n_verses": len(lxx_lens),
            "mt_n_verses": len(mt_lens),
            "note": "Insufficient data (<30 verses)",
        }

    # Métricas LXX
    H_lxx, H_lxx_r2 = hurst_exponent_rs(lxx_lens)
    alpha_lxx, alpha_lxx_r2 = dfa_exponent(lxx_lens)
    mps_lxx = permutation_test_chi(lxx_lens, n_perm=1000)

    # Métricas MT
    H_mt, H_mt_r2 = hurst_exponent_rs(mt_lens)
    alpha_mt, alpha_mt_r2 = dfa_exponent(mt_lens)
    mps_mt = permutation_test_chi(mt_lens, n_perm=1000)

    # Mann-Whitney
    mw_stat, mw_p = stats.mannwhitneyu(lxx_lens, mt_lens, alternative="two-sided")

    return {
        "book": book_name,
        "lxx": {
            "n_verses": len(lxx_lens),
            "n_words": sum(lxx_lens),
            "mean_verse_len": float(np.mean(lxx_lens)),
            "H": H_lxx, "H_R2": H_lxx_r2,
            "alpha": alpha_lxx, "alpha_R2": alpha_lxx_r2,
            "chi": mps_lxx["chi_obs"],
            "mps_p": mps_lxx["p_value"],
            "mps_significant": mps_lxx["significant"],
        },
        "mt": {
            "n_verses": len(mt_lens),
            "n_words": sum(mt_lens),
            "mean_verse_len": float(np.mean(mt_lens)),
            "H": H_mt, "H_R2": H_mt_r2,
            "alpha": alpha_mt, "alpha_R2": alpha_mt_r2,
            "chi": mps_mt["chi_obs"],
            "mps_p": mps_mt["p_value"],
            "mps_significant": mps_mt["significant"],
        },
        "delta_H": float(H_lxx - H_mt) if not (np.isnan(H_lxx) or np.isnan(H_mt)) else None,
        "delta_alpha": float(alpha_lxx - alpha_mt) if not (np.isnan(alpha_lxx) or np.isnan(alpha_mt)) else None,
        "mann_whitney_p": float(mw_p),
        "verse_len_ratio": float(np.mean(lxx_lens) / np.mean(mt_lens)) if np.mean(mt_lens) > 0 else None,
    }


# ── Documentación Pentateuco Samaritano ──────────────────────────────────

def document_sp_gap():
    """Documenta la ausencia del Pentateuco Samaritano como gap."""
    return {
        "corpus": "Samaritan Pentateuch",
        "status": "NOT_AVAILABLE_DIGITALLY_WITH_MORPHOLOGY",
        "significance": (
            "El SP es el candidato ideal para resolver H5a vs H5b: "
            "misma lengua (hebreo), mismo texto base, dos líneas de transmisión "
            "independientes desde ~400 a.C. Si H(SP) ≈ H(MT), H5a se confirma "
            "(estructura predates la divergencia). Si H(MT) > H(SP), H5b gana apoyo."
        ),
        "known_variants": (
            "~6,000 diferencias con MT, mayoritariamente ortográficas. "
            "~1,900 acuerdos SP+LXX contra MT, sugiriendo un Vorlage común."
        ),
        "available_resources": [
            "August von Gall (1918): edición crítica impresa",
            "Abraham Tal & Moshe Florentin (2010): transcripción",
            "Stefan Schorch (2021): Samaritan Pentateuch Commentary Series",
        ],
        "what_is_needed": (
            "Un corpus digital etiquetado morfológicamente del SP, con segmentación "
            "por versículo compatible con el esquema BHSA/WLC, para aplicar las "
            "mismas 6 métricas (H, α, D_f, χ, ΔS, Zipf s)."
        ),
        "alternative_used": "LXX (Septuaginta) como segunda línea de transmisión",
        "lxx_vs_sp_tradeoff": (
            "La LXX es una TRADUCCIÓN al griego (cambia la lengua), "
            "mientras el SP es el MISMO texto en hebreo (solo cambian variantes). "
            "Por eso el SP sería más informativo: controla por lengua. "
            "Pero la LXX está disponible digitalmente y el SP no."
        ),
    }


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    log.info("=" * 60)
    log.info("FASE 9 — Script 2: LXX vs MT")
    log.info("=" * 60)

    # 1. Buscar datos LXX
    availability = find_lxx_data()

    # Guardar disponibilidad
    with open(RESULTS_DIR / "lxx_availability.json", "w") as f:
        json.dump(availability, f, indent=2, ensure_ascii=False)

    # 2. Cargar MT
    mt_data = load_mt_verse_lengths()

    # 3. Intentar parsear LXX y comparar
    pentateuch = ["Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy"]
    lxx_parsed = {}
    parse_log = []

    for source_info in availability:
        if not source_info.get("cloned"):
            continue
        lxx_path = source_info["path"]
        log.info(f"\nIntentando parsear LXX de: {source_info['source']}")

        for book in pentateuch:
            if book in lxx_parsed and lxx_parsed[book] is not None:
                continue
            verses, info = parse_lxx_verse_lengths(lxx_path, book)
            if verses:
                lxx_parsed[book] = verses
                sorted_keys = sorted(verses.keys())
                lens = [verses[k] for k in sorted_keys]
                log.info(f"  {book}: {len(lens)} versos, {sum(lens)} palabras "
                        f"(de {source_info['source']})")
                parse_log.append({
                    "book": book,
                    "source": source_info["source"],
                    "n_verses": len(lens),
                    "n_words": sum(lens),
                    "file": info,
                })
            else:
                log.info(f"  {book}: no encontrado ({info})")

    # 4. Comparar LXX vs MT
    comparisons = []
    for book in pentateuch:
        if book in lxx_parsed and lxx_parsed[book]:
            log.info(f"\nComparando LXX vs MT para {book}...")
            result = compare_lxx_mt(lxx_parsed, mt_data, book)
            if result:
                comparisons.append(result)
                if result.get("delta_H") is not None:
                    log.info(f"  ΔH = {result['delta_H']:.4f}, "
                            f"MW p = {result['mann_whitney_p']:.4f}")
        else:
            log.info(f"\n{book}: no hay datos LXX disponibles")
            comparisons.append({
                "book": book,
                "status": "lxx_not_available",
                "mt_n_verses": len(mt_data.get(book, {})),
            })

    # Si no se pudo parsear ningún libro de la LXX, documentar
    n_compared = sum(1 for c in comparisons if "delta_H" in c and c["delta_H"] is not None)
    if n_compared == 0:
        log.warning("\n⚠ No se pudo parsear ningún libro de la LXX con formato utilizable.")
        log.warning("  Documentando la situación y métricas disponibles del MT solo.")

        # Calcular métricas MT para el Pentateuco completo
        all_mt_lens = []
        mt_book_metrics = {}
        for book in pentateuch:
            if book in mt_data:
                sorted_keys = sorted(mt_data[book].keys())
                lens = [mt_data[book][k] for k in sorted_keys]
                all_mt_lens.extend(lens)
                H, H_r2 = hurst_exponent_rs(lens)
                alpha, alpha_r2 = dfa_exponent(lens)
                mt_book_metrics[book] = {
                    "n_verses": len(lens),
                    "n_words": sum(lens),
                    "mean_verse_len": float(np.mean(lens)),
                    "H": H, "H_R2": H_r2,
                    "alpha": alpha, "alpha_R2": alpha_r2,
                }
                log.info(f"  MT {book}: H={H:.4f}, α={alpha:.4f}")

        # Pentateuco MT completo
        if all_mt_lens:
            H_pent, H_pent_r2 = hurst_exponent_rs(all_mt_lens)
            alpha_pent, alpha_pent_r2 = dfa_exponent(all_mt_lens)
            mps_pent = permutation_test_chi(all_mt_lens, n_perm=1000)
            mt_pentateuch_complete = {
                "n_verses": len(all_mt_lens),
                "n_words": sum(all_mt_lens),
                "mean_verse_len": float(np.mean(all_mt_lens)),
                "H": H_pent, "H_R2": H_pent_r2,
                "alpha": alpha_pent, "alpha_R2": alpha_pent_r2,
                "chi": mps_pent["chi_obs"],
                "mps_p": mps_pent["p_value"],
                "mps_significant": mps_pent["significant"],
            }
            log.info(f"\n  MT Pentateuco completo: H={H_pent:.4f}, α={alpha_pent:.4f}, "
                    f"MPS_p={mps_pent['p_value']:.4f}")
        else:
            mt_pentateuch_complete = None

        comparisons = {
            "status": "lxx_not_parseable",
            "note": ("LXX repositories were cloned but verse-level data could not be "
                     "extracted in a usable format. The available LXX digital texts lack "
                     "standardized verse segmentation compatible with MT/WLC. "
                     "Manual preprocessing or a dedicated LXX corpus (e.g., CATSS) "
                     "would be needed for this comparison."),
            "mt_pentateuch_metrics": mt_book_metrics,
            "mt_pentateuch_complete": mt_pentateuch_complete,
            "parse_attempts": parse_log,
        }

    # 5. Guardar resultados
    with open(RESULTS_DIR / "lxx_mt_comparison.json", "w") as f:
        json.dump(comparisons, f, indent=2, ensure_ascii=False, default=str)

    # 6. Documentar gap del SP
    sp_doc = document_sp_gap()
    with open(RESULTS_DIR / "sp_gap_documentation.json", "w") as f:
        json.dump(sp_doc, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    log.info(f"\n{'=' * 60}")
    log.info(f"Script 2 completado en {elapsed:.1f}s")
    log.info(f"Resultados en {RESULTS_DIR}")


if __name__ == "__main__":
    main()
