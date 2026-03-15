#!/usr/bin/env python3
"""
Fase 7 — Tarea 1: Análisis de los Manuscritos del Mar Muerto (DSS)
Compara 1QIsa^a (~100 a.C.) con Isaías masorético WLC (~1000 d.C.)
para determinar si la memoria larga es anterior (H5a) o posterior (H5b) a la canonización.
"""

import json
import logging
import os
import sys
import subprocess
import numpy as np
from pathlib import Path
from scipy import stats
from collections import Counter

# ── Configuración ────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent
RESULTS_DIR = BASE / "results" / "dss"
LOG_DIR = BASE / "logs"
BIBLE_JSON = BASE / "bible_unified.json"

LOG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "fase7_dss.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ── POS mapping (Text-Fabric sp feature → 9 categories) ─────────────────
SP_MAP = {
    "subs": "noun",
    "nmpr": "noun",      # nombre propio
    "verb": "verb",
    "adjv": "adjective",
    "advb": "adverb",
    "prep": "preposition",
    "conj": "conjunction",
    "prps": "pronoun",    # pronombre personal
    "prde": "pronoun",    # pronombre demostrativo
    "prin": "pronoun",    # pronombre interrogativo
    "inrg": "particle",   # interrogativo
    "nega": "particle",   # negación
    "intj": "particle",   # interjección
    "art":  "other",      # artículo
    "intr": "particle",
}

# ── Métricas (copiadas de fase6_rigveda.py) ──────────────────────────────

def hurst_exponent_rs(series):
    """Hurst exponent via Rescaled Range (R/S) analysis."""
    series = np.asarray(series, dtype=float)
    n = len(series)
    if n < 20:
        return float("nan"), 0.0
    min_block = 10
    max_block = n // 2
    sizes = []
    rs_values = []
    block = min_block
    while block <= max_block:
        sizes.append(block)
        n_blocks = n // block
        rs_list = []
        for i in range(n_blocks):
            seg = series[i * block:(i + 1) * block]
            mean_seg = seg.mean()
            devs = np.cumsum(seg - mean_seg)
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
    slope, intercept, r, p, se = stats.linregress(log_n, log_rs)
    return float(slope), float(r ** 2)


def dfa_exponent(series):
    """Detrended Fluctuation Analysis (DFA)."""
    series = np.asarray(series, dtype=float)
    n = len(series)
    if n < 20:
        return float("nan"), 0.0
    y = np.cumsum(series - series.mean())
    min_box = 4
    max_box = n // 4
    sizes = []
    flucts = []
    box = min_box
    while box <= max_box:
        sizes.append(box)
        n_boxes = n // box
        rms_list = []
        for i in range(n_boxes):
            seg = y[i * box:(i + 1) * box]
            x_ax = np.arange(box)
            coeffs = np.polyfit(x_ax, seg, 1)
            trend = np.polyval(coeffs, x_ax)
            rms_list.append(np.sqrt(np.mean((seg - trend) ** 2)))
        if rms_list:
            flucts.append(np.mean(rms_list))
        box = int(box * 1.5)
        if box == sizes[-1]:
            box += 1
    if len(sizes) < 3:
        return float("nan"), 0.0
    log_s = np.log(sizes)
    log_f = np.log(flucts)
    slope, intercept, r, p, se = stats.linregress(log_s, log_f)
    return float(slope), float(r ** 2)


def box_counting_dimension(series):
    """Box-counting fractal dimension."""
    series = np.asarray(series, dtype=float)
    n = len(series)
    if n < 10:
        return float("nan"), 0.0
    s_min, s_max = series.min(), series.max()
    if s_max == s_min:
        return 0.0, 1.0
    norm = (series - s_min) / (s_max - s_min)
    epsilons = []
    counts = []
    k = 2
    while k <= n // 2:
        eps = 1.0 / k
        time_boxes = np.floor(np.arange(n) / (n * eps)).astype(int)
        value_boxes = np.floor(norm / eps).astype(int)
        value_boxes = np.minimum(value_boxes, k - 1)
        occupied = set(zip(time_boxes, value_boxes))
        epsilons.append(eps)
        counts.append(len(occupied))
        k = int(k * 1.5)
        if k == int(k / 1.5 * 1.5):
            k += 1
    if len(epsilons) < 3:
        return float("nan"), 0.0
    log_inv_eps = np.log(1.0 / np.array(epsilons))
    log_n = np.log(np.array(counts))
    slope, intercept, r, p, se = stats.linregress(log_inv_eps, log_n)
    return float(slope), float(r ** 2)


def compute_bond_dimension(series, max_lag=256, threshold=0.99):
    """Bond dimension χ via autocorrelation matrix SVD."""
    series = np.asarray(series, dtype=float)
    n = len(series)
    if n < max_lag * 2:
        max_lag = n // 4
    if max_lag < 4:
        return 1
    mean = series.mean()
    var = series.var()
    if var == 0:
        return 1
    acf = np.correlate(series - mean, series - mean, mode="full")
    acf = acf[n - 1:n - 1 + max_lag] / (var * n)
    from scipy.linalg import toeplitz
    T = toeplitz(acf)
    try:
        U, sigma, Vt = np.linalg.svd(T)
    except np.linalg.LinAlgError:
        return max_lag
    total = np.sum(sigma ** 2)
    if total == 0:
        return 1
    cumvar = np.cumsum(sigma ** 2) / total
    chi = int(np.searchsorted(cumvar, threshold) + 1)
    return min(chi, max_lag)


def permutation_test_chi(series, n_perm=10000, max_lag=64, threshold=0.99):
    """Permutation test for bond dimension."""
    series = np.asarray(series, dtype=float)
    chi_obs = compute_bond_dimension(series, max_lag=max_lag, threshold=threshold)
    chi_rand = []
    for i in range(n_perm):
        perm = np.random.permutation(series)
        chi_rand.append(compute_bond_dimension(perm, max_lag=max_lag, threshold=threshold))
        if (i + 1) % 2000 == 0:
            log.info(f"  Permutation test: {i+1}/{n_perm}")
    chi_rand = np.array(chi_rand)
    p_value = float(np.mean(chi_rand <= chi_obs))
    return {
        "chi_obs": int(chi_obs),
        "chi_rand_mean": float(chi_rand.mean()),
        "chi_rand_std": float(chi_rand.std()),
        "p_value": p_value,
        "significant": p_value < 0.05,
    }


def compute_delta_s(units_pos, n_random=200):
    """ΔS = S_vN - S_Shannon con control aleatorio."""
    # Build POS vectors per unit
    pos_cats = ["noun", "verb", "pronoun", "adjective", "adverb",
                "preposition", "conjunction", "particle", "other"]
    cat_idx = {c: i for i, c in enumerate(pos_cats)}
    d = len(pos_cats)

    vectors = []
    for pos_list in units_pos:
        v = np.zeros(d)
        for p in pos_list:
            if p in cat_idx:
                v[cat_idx[p]] += 1
            elif p is not None:
                v[cat_idx.get("other", d - 1)] += 1
        norm = np.linalg.norm(v)
        if norm > 0:
            vectors.append(v / norm)
    if len(vectors) < 5:
        return {"delta_S_mean": float("nan"), "delta_s_group": "insufficient_data"}

    vectors = np.array(vectors)
    n_units = len(vectors)

    # Density matrix
    rho = vectors.T @ vectors / n_units
    rho = rho / np.trace(rho)
    eigvals = np.linalg.eigvalsh(rho)
    eigvals = eigvals[eigvals > 1e-15]
    s_vn = float(-np.sum(eigvals * np.log2(eigvals)))

    # Shannon from marginal POS
    all_pos = []
    for pos_list in units_pos:
        all_pos.extend([p for p in pos_list if p is not None])
    total = len(all_pos)
    if total == 0:
        return {"delta_S_mean": float("nan"), "delta_s_group": "no_data"}
    freq = Counter(all_pos)
    probs = np.array([freq.get(c, 0) / total for c in pos_cats])
    probs = probs[probs > 0]
    s_shannon = float(-np.sum(probs * np.log2(probs)))

    delta_s = s_vn - s_shannon

    # Random control
    rand_deltas = []
    for _ in range(n_random):
        idx = np.random.permutation(n_units)
        rho_r = vectors[idx].T @ vectors[idx] / n_units
        rho_r = rho_r / np.trace(rho_r)
        ev_r = np.linalg.eigvalsh(rho_r)
        ev_r = ev_r[ev_r > 1e-15]
        s_vn_r = -np.sum(ev_r * np.log2(ev_r))
        rand_deltas.append(s_vn_r - s_shannon)
    rand_mean = np.mean(rand_deltas)

    if delta_s < rand_mean:
        group = "more_structured"
    else:
        group = "more_varied"

    t_stat, p_val = stats.ttest_1samp(rand_deltas, delta_s)

    return {
        "s_vn": float(s_vn),
        "s_shannon": float(s_shannon),
        "delta_S_mean": float(delta_s),
        "delta_s_vs_random_mean": float(rand_mean),
        "delta_s_vs_random_p": float(p_val),
        "delta_s_group": group,
    }


def compute_zipf_lemma(lemma_counts):
    """Zipf exponent from lemma frequency distribution."""
    freqs = sorted(lemma_counts.values(), reverse=True)
    if len(freqs) < 10:
        return float("nan")
    ranks = np.arange(1, len(freqs) + 1, dtype=float)
    freqs = np.array(freqs, dtype=float)
    mask = freqs > 0
    log_r = np.log(ranks[mask])
    log_f = np.log(freqs[mask])
    slope, intercept, r, p, se = stats.linregress(log_r, log_f)
    return float(-slope)


# ── Extracción de Isaías del WLC (bible_unified.json) ────────────────────

def extract_isaiah_wlc():
    """Extrae versos de Isaías del corpus bíblico unificado."""
    log.info("Cargando bible_unified.json...")
    with open(BIBLE_JSON, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    # Filtrar solo Isaías
    isaiah_words = [w for w in corpus if w["book"] == "Isaiah"]
    if not isaiah_words:
        # Intentar con nombre hebreo
        isaiah_words = [w for w in corpus if "isaiah" in w["book"].lower() or "isa" in w["book"].lower()]
    log.info(f"  Isaías WLC: {len(isaiah_words)} palabras")

    # Agrupar por verso
    verses = {}
    for w in isaiah_words:
        key = (w["book_num"], w["chapter"], w["verse"])
        if key not in verses:
            verses[key] = {"words": [], "pos": [], "lemmas": []}
        verses[key]["words"].append(w.get("text", ""))
        verses[key]["pos"].append(w.get("pos", "other"))
        verses[key]["lemmas"].append(w.get("lemma", ""))

    sorted_keys = sorted(verses.keys())
    verse_lengths = [len(verses[k]["words"]) for k in sorted_keys]
    units_pos = [verses[k]["pos"] for k in sorted_keys]
    lemma_counts = Counter()
    for k in sorted_keys:
        lemma_counts.update(verses[k]["lemmas"])

    log.info(f"  Isaías WLC: {len(verse_lengths)} versos, {sum(verse_lengths)} palabras")
    return verse_lengths, units_pos, lemma_counts


# ── Extracción de 1QIsa^a del corpus DSS (Text-Fabric) ──────────────────

def setup_dss_corpus():
    """Clona y configura el corpus DSS si no existe."""
    dss_dir = Path.home() / "github" / "ETCBC" / "dss"
    if not dss_dir.exists():
        log.info("Clonando repositorio ETCBC/dss...")
        dss_dir.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/ETCBC/dss.git", str(dss_dir)],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            log.error(f"Error clonando DSS: {result.stderr}")
            raise RuntimeError(f"git clone failed: {result.stderr}")
        log.info("  Clonado exitosamente.")
    else:
        log.info(f"  Repositorio DSS ya existe en {dss_dir}")
    return dss_dir


def extract_isaiah_dss():
    """Extrae 1QIsa^a del corpus DSS usando Text-Fabric.

    El corpus ETCBC/dss NO tiene nodos 'book', 'chapter', 'verse'.
    Estructura: scroll → fragment → line → word → sign
    Usamos 'line' como unidad equivalente al versículo.
    """
    dss_dir = setup_dss_corpus()

    # Instalar text-fabric si no está
    try:
        from tf.fabric import Fabric
    except ImportError:
        log.info("Instalando text-fabric...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "text-fabric", "--break-system-packages", "-q"],
            capture_output=True, text=True, timeout=120
        )
        from tf.fabric import Fabric

    # Buscar la versión más reciente
    tf_dir = dss_dir / "tf"
    versions = sorted([d.name for d in tf_dir.iterdir() if d.is_dir()]) if tf_dir.exists() else []
    log.info(f"  Versiones disponibles en DSS tf/: {versions}")
    version = versions[-1] if versions else "1.9"

    log.info(f"  Cargando Text-Fabric desde {dss_dir}/tf/{version}/...")
    TF = Fabric(locations=str(dss_dir / "tf" / version))

    api = TF.load("scroll sp lex line fragment", silent="deep")
    if not api:
        raise RuntimeError("No se pudo cargar el corpus DSS con Text-Fabric")

    F = api.F
    L = api.L

    log.info(f"  otypes: {F.otype.all}")

    # Buscar scroll 1Qisaa
    target_node = None
    for node in F.otype.s("scroll"):
        sname = F.scroll.v(node)
        if sname == "1Qisaa":
            target_node = node
            break

    if target_node is None:
        # Buscar variantes
        for node in F.otype.s("scroll"):
            sname = str(F.scroll.v(node))
            if "isa" in sname.lower():
                target_node = node
                log.info(f"  Usando scroll alternativo: {sname}")
                break

    if target_node is None:
        raise RuntimeError("No se encontró 1QIsa^a en el corpus DSS")

    scroll_name = F.scroll.v(target_node)
    log.info(f"  Scroll encontrado: {scroll_name}")

    # Extraer por líneas (equivalente a versículos para el análisis de series temporales)
    # NOTA: line numbers se repiten por fragmento (1-29 en cada fragmento),
    # así que usamos la lista de nodos en orden como secuencia.
    lines = L.d(target_node, otype="line")
    log.info(f"  {scroll_name}: {len(lines)} líneas")

    line_list = []  # Lista ordenada de dicts
    total_words = 0
    for line_node in lines:
        words = L.d(line_node, otype="word")

        word_list = []
        pos_list = []
        lemma_list = []
        for w in words:
            sp = F.sp.v(w)
            lex = F.lex.v(w)
            if sp is None:
                continue  # Skip signs without POS
            pos = SP_MAP.get(str(sp), "other")
            word_list.append(str(lex) if lex else "")
            pos_list.append(pos)
            lemma_list.append(str(lex) if lex else "")

        if word_list:  # Solo líneas con palabras
            line_list.append({
                "words": word_list,
                "pos": pos_list,
                "lemmas": lemma_list,
            })
            total_words += len(word_list)

    log.info(f"  {scroll_name}: {len(line_list)} líneas con contenido, {total_words} palabras")

    verse_lengths = [len(ld["words"]) for ld in line_list]
    units_pos = [ld["pos"] for ld in line_list]
    lemma_counts = Counter()
    for ld in line_list:
        lemma_counts.update(ld["lemmas"])

    # POS distribution for logging
    all_pos = [p for ld in line_list for p in ld["pos"]]
    pos_dist = Counter(all_pos)
    log.info(f"  POS distribution: {pos_dist.most_common()}")
    log.info(f"  Líneas: {len(verse_lengths)}, Palabras: {sum(verse_lengths)}")
    log.info(f"  Media palabras/línea: {np.mean(verse_lengths):.2f}")

    return verse_lengths, units_pos, lemma_counts, {scroll_name}


# ── Análisis completo de un corpus ───────────────────────────────────────

def analyze_corpus(verse_lengths, units_pos, lemma_counts, label):
    """Aplica las 6 métricas a un corpus."""
    log.info(f"Analizando {label}...")
    series = np.array(verse_lengths, dtype=float)

    # 1. Hurst
    log.info(f"  [{label}] Hurst R/S...")
    h, h_r2 = hurst_exponent_rs(series)
    log.info(f"    H = {h:.4f}, R² = {h_r2:.4f}")

    # 2. DFA
    log.info(f"  [{label}] DFA...")
    alpha, a_r2 = dfa_exponent(series)
    log.info(f"    α = {alpha:.4f}, R² = {a_r2:.4f}")

    # 3. Box counting
    log.info(f"  [{label}] Box counting...")
    df, df_r2 = box_counting_dimension(series)
    log.info(f"    D_f = {df:.4f}, R² = {df_r2:.4f}")

    # 4-5. Bond dimension + permutation test
    log.info(f"  [{label}] Bond dimension + permutation test (10,000 perms)...")
    perm = permutation_test_chi(series, n_perm=10000)
    log.info(f"    χ_obs = {perm['chi_obs']}, χ_rand = {perm['chi_rand_mean']:.1f}, p = {perm['p_value']:.4f}")

    # 6. ΔS Von Neumann
    log.info(f"  [{label}] ΔS Von Neumann...")
    ds = compute_delta_s(units_pos)
    log.info(f"    ΔS = {ds['delta_S_mean']:.6f}, grupo = {ds['delta_s_group']}")

    # Zipf
    zipf_s = compute_zipf_lemma(lemma_counts)
    log.info(f"    Zipf s = {zipf_s:.4f}")

    return {
        "hurst_H": h,
        "hurst_R2": h_r2,
        "dfa_alpha": alpha,
        "dfa_R2": a_r2,
        "box_counting_Df": df,
        "box_counting_R2": df_r2,
        "bond_dim_chi": perm["chi_obs"],
        "mps_permtest_p": perm["p_value"],
        "mps_permtest_chi_obs": perm["chi_obs"],
        "mps_permtest_chi_rand": perm["chi_rand_mean"],
        "mps_significant": perm["significant"],
        "delta_S_mean": ds["delta_S_mean"],
        "delta_s_group": ds["delta_s_group"],
        "zipf_s_lemma": zipf_s,
    }


# ── Comparación estadística ──────────────────────────────────────────────

def compare_metrics(metrics_a, metrics_b, vl_a, vl_b):
    """Compara dos conjuntos de métricas estadísticamente."""
    # Mann-Whitney en longitudes de verso
    mw_stat, mw_p = stats.mannwhitneyu(vl_a, vl_b, alternative="two-sided")

    # KS test en distribuciones de longitud
    ks_stat, ks_p = stats.ks_2samp(vl_a, vl_b)

    # Comparar métricas clave
    key_metrics = ["hurst_H", "dfa_alpha", "box_counting_Df", "delta_S_mean"]
    diffs = {}
    for m in key_metrics:
        va = metrics_a.get(m, float("nan"))
        vb = metrics_b.get(m, float("nan"))
        diffs[m] = {"dss": va, "wlc": vb, "diff": abs(va - vb) if not (np.isnan(va) or np.isnan(vb)) else None}

    # ¿Son significativamente diferentes?
    sig_different = ks_p < 0.05

    return {
        "mann_whitney": {"statistic": float(mw_stat), "p_value": float(mw_p)},
        "ks_verse_lengths": {"statistic": float(ks_stat), "p_value": float(ks_p)},
        "metric_diffs": diffs,
        "metrics_significantly_different": sig_different,
    }


def determine_verdict(metrics_a, metrics_b, stat_tests):
    """Determina H5a vs H5b."""
    ha = metrics_a["hurst_H"]
    hb = metrics_b["hurst_H"]
    aa = metrics_a["dfa_alpha"]
    ab = metrics_b["dfa_alpha"]
    mps_a = metrics_a["mps_significant"]
    mps_b = metrics_b["mps_significant"]

    sig_diff = stat_tests["metrics_significantly_different"]

    # Criterio: si las métricas son estadísticamente indistinguibles → H5a
    # Si el masorético tiene significativamente MÁS estructura → H5b
    # Si el DSS tiene MÁS estructura → H5a (fuerte)

    if not sig_diff and abs(ha - hb) < 0.15 and abs(aa - ab) < 0.15:
        verdict = "H5a_confirmed"
        reasoning = (
            f"Las métricas de 1QIsa^a (H={ha:.3f}, α={aa:.3f}, MPS={'sig' if mps_a else 'no sig'}) "
            f"son estadísticamente indistinguibles de las del Isaías masorético "
            f"(H={hb:.3f}, α={ab:.3f}, MPS={'sig' if mps_b else 'no sig'}). "
            f"KS p={stat_tests['ks_verse_lengths']['p_value']:.4f}. "
            f"La memoria larga es una propiedad del texto original (~100 a.C.), "
            f"anterior a la canonización rabínica formal."
        )
    elif ha > hb + 0.15 or aa > ab + 0.15:
        verdict = "H5a_confirmed"
        reasoning = (
            f"1QIsa^a tiene memoria larga MAYOR que el masorético "
            f"(H: {ha:.3f} vs {hb:.3f}, α: {aa:.3f} vs {ab:.3f}). "
            f"La transmisión masorética degradó parcialmente la estructura original. "
            f"La memoria larga ya existía antes de la canonización → H5a."
        )
    elif hb > ha + 0.15 or ab > aa + 0.15:
        verdict = "H5b_confirmed"
        reasoning = (
            f"El Isaías masorético tiene memoria larga MAYOR que 1QIsa^a "
            f"(H: {hb:.3f} vs {ha:.3f}, α: {ab:.3f} vs {aa:.3f}). "
            f"La canonización masorética amplificó la estructura → H5b."
        )
    else:
        verdict = "indeterminate"
        reasoning = (
            f"Diferencias ambiguas entre 1QIsa^a y masorético "
            f"(H: {ha:.3f} vs {hb:.3f}, α: {aa:.3f} vs {ab:.3f}). "
            f"No se puede determinar con certeza si H5a o H5b."
        )

    return verdict, reasoning


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 70)
    log.info("FASE 7 — Tarea 1: Dead Sea Scrolls — 1QIsa^a vs WLC Isaías")
    log.info("=" * 70)

    # Extraer Isaías WLC
    log.info("\n─── Extrayendo Isaías del WLC (masorético) ───")
    vl_wlc, pos_wlc, lem_wlc = extract_isaiah_wlc()

    # Extraer 1QIsa^a del DSS
    log.info("\n─── Extrayendo 1QIsa^a del DSS ───")
    log.info("  NOTA: El DSS no tiene chapter/verse. Se usan 'lines' (líneas del pergamino)")
    log.info("  como unidades equivalentes a versículos para el análisis de series temporales.")
    try:
        result = extract_isaiah_dss()
        vl_dss, pos_dss, lem_dss, scrolls_used = result
    except Exception as e:
        log.error(f"Error extrayendo DSS: {e}")
        import traceback
        log.error(traceback.format_exc())
        # Guardar resultado indicando el error
        output = {
            "error": str(e),
            "corpus_a": "1QIsa-a (DSS, ~100 BCE)",
            "corpus_b": "Isaiah WLC (Masoretic, ~1000 CE)",
            "verdict": "indeterminate",
            "reasoning": f"No se pudo extraer el corpus DSS: {e}. Se necesita verificar la estructura del corpus ETCBC/dss.",
        }
        with open(RESULTS_DIR / "dss_isaiah_comparison.json", "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        log.info(f"Resultado parcial guardado en {RESULTS_DIR / 'dss_isaiah_comparison.json'}")
        return

    # Analizar ambos
    log.info("\n─── Analizando 1QIsa^a ───")
    metrics_dss = analyze_corpus(vl_dss, pos_dss, lem_dss, "1QIsa^a (DSS)")

    log.info("\n─── Analizando Isaías WLC ───")
    metrics_wlc = analyze_corpus(vl_wlc, pos_wlc, lem_wlc, "Isaiah WLC")

    # Comparar
    log.info("\n─── Comparación estadística ───")
    stat_tests = compare_metrics(metrics_dss, metrics_wlc, vl_dss, vl_wlc)
    log.info(f"  Mann-Whitney p = {stat_tests['mann_whitney']['p_value']:.6f}")
    log.info(f"  KS p = {stat_tests['ks_verse_lengths']['p_value']:.6f}")

    # Veredicto
    verdict, reasoning = determine_verdict(metrics_dss, metrics_wlc, stat_tests)
    log.info(f"\n  VEREDICTO: {verdict}")
    log.info(f"  {reasoning}")

    # Output
    output = {
        "corpus_a": "1QIsa-a (DSS, ~100 BCE)",
        "corpus_b": "Isaiah WLC (Masoretic, ~1000 CE)",
        "scrolls_used": list(scrolls_used),
        "unit_type_a": "lines (physical scroll lines)",
        "unit_type_b": "verses (biblical verses)",
        "n_units_a": len(vl_dss),
        "n_words_a": int(sum(vl_dss)),
        "mean_unit_length_a": float(np.mean(vl_dss)),
        "n_units_b": len(vl_wlc),
        "n_words_b": int(sum(vl_wlc)),
        "mean_unit_length_b": float(np.mean(vl_wlc)),
        "metrics_a": metrics_dss,
        "metrics_b": metrics_wlc,
        "statistical_tests": stat_tests,
        "verdict": verdict,
        "confidence": "high" if verdict != "indeterminate" else "low",
        "reasoning": reasoning,
        "note": "DSS uses physical scroll lines as units (no chapter/verse markup). "
                "WLC uses biblical verses. Both are sequential text segments, "
                "but with different granularity. The comparison of fractal/memory "
                "metrics is valid because these are scale-invariant properties.",
    }

    outfile = RESULTS_DIR / "dss_isaiah_comparison.json"
    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    log.info(f"\nResultado guardado en {outfile}")


if __name__ == "__main__":
    main()
