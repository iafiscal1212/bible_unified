#!/usr/bin/env python3
"""
Fase 10 — Análisis Comparativo Palabra-por-Palabra: 1QIsa^a (DSS) vs Isaías (WLC)

Niveles:
  0. Feature discovery en ETCBC/dss
  1. Perfil de memoria por fragmento (ventanas deslizantes)
  2. Divergencia POS n-gramas (Jensen-Shannon)
  3. Alineamiento por lemma (difflib.SequenceMatcher)
  4. Estudio de ablación (sensibilidad de H a tipos de variantes)
  5. Modelo de decaimiento H(t) con bootstrap

Contexto actualizado (post-Fase 8):
  Fase 7 reportó ΔH=0.197 (DSS lines vs WLC verses), pero Fase 8 mostró
  que con segmentación alineada por versículos, ΔH=0.008 (p=0.71).
  Este análisis word-level mapea la ESTRUCTURA de las variantes textuales
  y mide la sensibilidad de H a cada tipo de cambio.

Requiere: Text-Fabric, ETCBC/dss, bible_unified.json
Ejecutar en servidor con: python3 analyze_dss_wordlevel.py
"""

import json
import logging
import os
import sys
import time
import re
import subprocess
import difflib
import numpy as np
from pathlib import Path
from scipy import stats
from collections import Counter, defaultdict

# ── Configuración ────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent
RESULTS_DIR = BASE / "results" / "dss_wordlevel"
LOG_DIR = BASE / "logs"
BIBLE_JSON = BASE / "bible_unified.json"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "fase10_dss_wordlevel.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ── POS mapping (DSS ETCBC sp → unified categories) ──────────────────────
# DSS uses simplified tags: ptcl covers prep/conj/art/etc.
DSS_SP_MAP = {
    "subs": "noun", "nmpr": "noun", "verb": "verb", "adjv": "adjective",
    "advb": "adverb", "prep": "particle", "conj": "particle",
    "prps": "pronoun", "prde": "pronoun", "prin": "pronoun",
    "inrg": "particle", "nega": "particle", "intj": "particle",
    "art": "particle", "intr": "particle",
    "ptcl": "particle",  # DSS-specific: all function words
    "suff": "suffix",    # DSS-specific: pronominal suffixes
    "pron": "pronoun",   # DSS-specific
    "numr": "numeral",   # DSS-specific
}

# WLC POS mapping: normalize to same categories as DSS
# WLC parser.py already produces: noun, verb, suffix, particle,
# adjective, preposition, pronoun, conjunction, adverb
WLC_POS_NORMALIZE = {
    "noun": "noun", "verb": "verb", "adjective": "adjective",
    "adverb": "adverb", "pronoun": "pronoun", "suffix": "suffix",
    "preposition": "particle", "conjunction": "particle",
    "particle": "particle",  # merge function words
    "numeral": "numeral",
}


# ══════════════════════════════════════════════════════════════════════════
# Métricas (copiadas de analyze_dss.py)
# ══════════════════════════════════════════════════════════════════════════

def hurst_exponent_rs(series):
    """Hurst exponent via R/S analysis."""
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
    """Detrended Fluctuation Analysis."""
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


# ══════════════════════════════════════════════════════════════════════════
# Carga de datos
# ══════════════════════════════════════════════════════════════════════════

def setup_dss_corpus():
    """Clona y configura el corpus DSS si no existe."""
    dss_dir = Path.home() / "github" / "ETCBC" / "dss"
    if not dss_dir.exists():
        log.info("Clonando repositorio ETCBC/dss...")
        dss_dir.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/ETCBC/dss.git", str(dss_dir)],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            raise RuntimeError(f"git clone failed: {result.stderr}")
        log.info("  Clonado exitosamente.")
    else:
        log.info(f"  DSS ya existe en {dss_dir}")
    return dss_dir


def load_dss_data(dss_dir):
    """Extrae 1QIsa^a del corpus DSS con Text-Fabric.

    Carga features extendidas: scroll sp lex glex line fragment
    book chapter verse biblical g_cons.
    Agrupa por línea Y por versículo para diferentes análisis.
    """
    try:
        from tf.fabric import Fabric
    except ImportError:
        log.info("Instalando text-fabric...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "text-fabric",
             "--break-system-packages", "-q"],
            capture_output=True, text=True, timeout=120
        )
        from tf.fabric import Fabric

    tf_dir = dss_dir / "tf"
    versions = sorted([d.name for d in tf_dir.iterdir() if d.is_dir()])
    version = versions[-1] if versions else "1.9"
    log.info(f"  Cargando TF v{version} (features extendidas)...")

    TF = Fabric(locations=str(dss_dir / "tf" / version))
    api = TF.load("scroll sp lex glex line fragment "
                   "book chapter verse biblical g_cons", silent="deep")
    if not api:
        raise RuntimeError("No se pudo cargar corpus DSS")

    F, L = api.F, api.L

    # Buscar 1QIsa^a
    target = None
    for node in F.otype.s("scroll"):
        if F.scroll.v(node) == "1Qisaa":
            target = node
            break
    if target is None:
        for node in F.otype.s("scroll"):
            if "isa" in str(F.scroll.v(node)).lower():
                target = node
                break
    if target is None:
        raise RuntimeError("1QIsa^a no encontrado")

    scroll_name = F.scroll.v(target)
    log.info(f"  Scroll: {scroll_name}")

    # Extraer TODOS los datos (por línea y por versículo)
    lines_data = []
    all_words = []  # (glex_cons, pos, chapter, verse)
    verses_dict = defaultdict(lambda: {"words": [], "pos": [], "glex": [],
                                        "glex_cons": [], "n_words": 0})
    fragments = L.d(target, otype="fragment")

    for frag_node in fragments:
        frag_lines = L.d(frag_node, otype="line")
        for line_node in frag_lines:
            words_in_line = L.d(line_node, otype="word")
            line_words, line_pos = [], []
            for w in words_in_line:
                sp = F.sp.v(w)
                glex = F.glex.v(w)
                ch = F.chapter.v(w)
                vs = F.verse.v(w)
                if sp is None and glex is None:
                    continue
                pos = DSS_SP_MAP.get(str(sp), "other") if sp else "other"
                glex_str = str(glex) if glex else ""
                glex_cons = strip_hebrew_vowels(glex_str)
                line_words.append(glex_str)
                line_pos.append(pos)
                all_words.append((glex_cons, pos, ch, vs))
                # Agrupar por versículo
                if ch and vs:
                    vkey = (str(ch), str(vs))
                    verses_dict[vkey]["words"].append(glex_str)
                    verses_dict[vkey]["pos"].append(pos)
                    verses_dict[vkey]["glex"].append(glex_str)
                    verses_dict[vkey]["glex_cons"].append(glex_cons)
            if line_words:
                lines_data.append({
                    "words": line_words, "pos": line_pos,
                    "fragment": frag_node, "n_words": len(line_words),
                })

    # Ordenar versículos
    sorted_vkeys = sorted(verses_dict.keys(), key=lambda k: (int(k[0]), int(k[1])))
    verses_data = []
    for vk in sorted_vkeys:
        v = verses_dict[vk]
        v["chapter"] = int(vk[0])
        v["verse"] = int(vk[1])
        v["n_words"] = len(v["words"])
        verses_data.append(v)

    log.info(f"  {scroll_name}: {len(lines_data)} líneas, "
             f"{len(all_words)} palabras, {len(fragments)} fragmentos, "
             f"{len(verses_data)} versículos")

    # POS distribution
    pos_dist = Counter(w[1] for w in all_words)
    log.info(f"  POS dist: {pos_dist.most_common()}")

    return {
        "scroll_name": scroll_name,
        "lines": lines_data,
        "verses": verses_data,
        "all_words": all_words,
        "fragments": fragments,
        "n_fragments": len(fragments),
        "api": api,
    }


def load_wlc_data():
    """Extrae Isaías del WLC (bible_unified.json).

    Usa el campo 'text' (hebreo con vocales) y lo convierte a consonántico
    con strip_hebrew_vowels() para comparación con DSS glex.
    POS se normaliza con WLC_POS_NORMALIZE.
    """
    log.info("Cargando bible_unified.json...")
    with open(BIBLE_JSON, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    isaiah_words = [w for w in corpus
                    if w.get("book", "").lower() in ("isaiah", "isa")]
    if not isaiah_words:
        isaiah_words = [w for w in corpus
                        if "isai" in w.get("book", "").lower()]
    log.info(f"  Isaías WLC: {len(isaiah_words)} palabras")

    # Agrupar por verso
    verses_dict = defaultdict(lambda: {"words": [], "pos": [], "text": [],
                                        "text_cons": [], "n_words": 0})
    for w in isaiah_words:
        key = (w["chapter"], w["verse"])
        text = w.get("text", "")
        text_cons = strip_hebrew_vowels(text)
        raw_pos = w.get("pos", "other")
        pos = WLC_POS_NORMALIZE.get(raw_pos, raw_pos)
        verses_dict[key]["words"].append(text)
        verses_dict[key]["text"].append(text)
        verses_dict[key]["text_cons"].append(text_cons)
        verses_dict[key]["pos"].append(pos)

    sorted_keys = sorted(verses_dict.keys())
    verses_data = []
    all_words = []
    for k in sorted_keys:
        v = verses_dict[k]
        v["chapter"] = k[0]
        v["verse"] = k[1]
        v["n_words"] = len(v["words"])
        verses_data.append(v)
        for i in range(v["n_words"]):
            all_words.append((v["text_cons"][i], v["pos"][i]))

    log.info(f"  Isaías WLC: {len(verses_data)} versos, {len(all_words)} palabras")
    # Sample
    if verses_data:
        v0 = verses_data[0]
        log.info(f"  Muestra v1: text={v0['text'][:3]}, cons={v0['text_cons'][:3]}, pos={v0['pos'][:3]}")
    return {
        "verses": verses_data,
        "all_words": all_words,
    }


# ══════════════════════════════════════════════════════════════════════════
# Nivel 0: Feature Discovery
# ══════════════════════════════════════════════════════════════════════════

def level0_feature_discovery(dss_dir):
    """Lista TODAS las features disponibles en ETCBC/dss."""
    log.info("\n" + "=" * 60)
    log.info("NIVEL 0: Feature Discovery")
    log.info("=" * 60)

    tf_dir = dss_dir / "tf"
    versions = sorted([d.name for d in tf_dir.iterdir() if d.is_dir()])
    version = versions[-1] if versions else "1.9"
    feat_dir = tf_dir / version

    features = {}
    for f in sorted(feat_dir.iterdir()):
        if f.is_file() and f.suffix == ".tf":
            fname = f.stem
            with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                header_lines = []
                for line in fh:
                    if line.startswith("@"):
                        header_lines.append(line.strip())
                    else:
                        break
            meta = {}
            for hl in header_lines:
                if "=" in hl:
                    k, v = hl[1:].split("=", 1)
                    meta[k.strip()] = v.strip()
            features[fname] = {
                "file": str(f),
                "size_bytes": f.stat().st_size,
                "metadata": meta,
            }

    log.info(f"  {len(features)} features encontradas en v{version}")
    for name, info in features.items():
        desc = info["metadata"].get("description", "")
        log.info(f"    {name}: {desc[:80]}")

    key_features = ["g_cons", "g_word", "biblical", "rec", "book",
                     "chapter", "verse", "lex", "sp", "scroll",
                     "line", "fragment", "sign"]
    found = {k: k in features for k in key_features}
    log.info(f"\n  Features clave: {found}")

    result = {
        "version": version,
        "n_features": len(features),
        "features": {k: {kk: vv for kk, vv in v.items() if kk != "file"}
                     for k, v in features.items()},
        "key_features_available": found,
    }

    with open(RESULTS_DIR / "feature_discovery.json", "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result


# ══════════════════════════════════════════════════════════════════════════
# Nivel 1: Perfil de Memoria por Fragmento
# ══════════════════════════════════════════════════════════════════════════

def level1_memory_profile(dss_data, wlc_data, window=100, stride=25):
    """Perfil de H/α en ventanas deslizantes a lo largo de ambos textos."""
    log.info("\n" + "=" * 60)
    log.info("NIVEL 1: Perfil de Memoria por Fragmento")
    log.info("=" * 60)

    dss_lens = [ld["n_words"] for ld in dss_data["lines"]]
    wlc_lens = [v["n_words"] for v in wlc_data["verses"]]

    def sliding_window_metrics(series, window, stride, label):
        n = len(series)
        positions, h_vals, alpha_vals = [], [], []
        i = 0
        while i + window <= n:
            seg = series[i:i + window]
            pos_rel = (i + window / 2) / n
            h, _ = hurst_exponent_rs(seg)
            a, _ = dfa_exponent(seg)
            positions.append(float(pos_rel))
            h_vals.append(float(h))
            alpha_vals.append(float(a))
            i += stride
        log.info(f"  {label}: {len(positions)} ventanas")
        return positions, h_vals, alpha_vals

    dss_pos, dss_h, dss_a = sliding_window_metrics(
        dss_lens, window, stride, "DSS (lines)")
    wlc_pos, wlc_h, wlc_a = sliding_window_metrics(
        wlc_lens, window, stride, "WLC (verses)")

    # Correlación posición-H
    for label, pos_list, h_list in [("DSS", dss_pos, dss_h), ("WLC", wlc_pos, wlc_h)]:
        clean = [(p, h) for p, h in zip(pos_list, h_list) if not np.isnan(h)]
        if len(clean) > 5:
            ps, hs = zip(*clean)
            r, p = stats.pearsonr(ps, hs)
            log.info(f"  {label}: correlación posición-H: r={r:.3f}, p={p:.4f}")

    dss_h_valid = [h for h in dss_h if not np.isnan(h)]
    wlc_h_valid = [h for h in wlc_h if not np.isnan(h)]

    result = {
        "window_size": window, "stride": stride,
        "dss": {
            "n_windows": len(dss_pos),
            "positions": dss_pos, "H_values": dss_h, "alpha_values": dss_a,
            "H_mean": float(np.nanmean(dss_h)),
            "H_std": float(np.nanstd(dss_h)),
        },
        "wlc": {
            "n_windows": len(wlc_pos),
            "positions": wlc_pos, "H_values": wlc_h, "alpha_values": wlc_a,
            "H_mean": float(np.nanmean(wlc_h)),
            "H_std": float(np.nanstd(wlc_h)),
        },
        "delta_H_mean": float(np.nanmean(dss_h_valid) - np.nanmean(wlc_h_valid))
        if dss_h_valid and wlc_h_valid else None,
    }

    with open(RESULTS_DIR / "memory_profile.json", "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    log.info(f"  DSS H_mean={result['dss']['H_mean']:.4f} ± {result['dss']['H_std']:.4f}")
    log.info(f"  WLC H_mean={result['wlc']['H_mean']:.4f} ± {result['wlc']['H_std']:.4f}")
    return result


# ══════════════════════════════════════════════════════════════════════════
# Nivel 2: Divergencia POS n-gramas
# ══════════════════════════════════════════════════════════════════════════

def jensen_shannon_divergence(p, q):
    """Jensen-Shannon divergence entre dos distribuciones."""
    all_keys = set(p.keys()) | set(q.keys())
    p_vec = np.array([p.get(k, 0) for k in all_keys], dtype=float)
    q_vec = np.array([q.get(k, 0) for k in all_keys], dtype=float)
    p_vec = p_vec / p_vec.sum() if p_vec.sum() > 0 else p_vec
    q_vec = q_vec / q_vec.sum() if q_vec.sum() > 0 else q_vec
    m = (p_vec + q_vec) / 2
    eps = 1e-15
    kl_pm = np.sum(p_vec * np.log2((p_vec + eps) / (m + eps)))
    kl_qm = np.sum(q_vec * np.log2((q_vec + eps) / (m + eps)))
    return float((kl_pm + kl_qm) / 2)


def level2_pos_divergence(dss_data, wlc_data):
    """Divergencia POS n-gramas entre DSS y WLC."""
    log.info("\n" + "=" * 60)
    log.info("NIVEL 2: Divergencia POS n-gramas")
    log.info("=" * 60)

    dss_pos = [w[1] for w in dss_data["all_words"]]
    wlc_pos = [w[1] for w in wlc_data["all_words"]]

    log.info(f"  DSS: {len(dss_pos)} tokens POS")
    log.info(f"  WLC: {len(wlc_pos)} tokens POS")

    dss_uni = Counter(dss_pos)
    wlc_uni = Counter(wlc_pos)
    log.info(f"  DSS POS dist: {dss_uni.most_common(5)}")
    log.info(f"  WLC POS dist: {wlc_uni.most_common(5)}")

    results = {"n_gram_divergences": {}}

    for n in [1, 2, 3, 4]:
        dss_ngrams = Counter()
        for i in range(len(dss_pos) - n + 1):
            dss_ngrams[tuple(dss_pos[i:i + n])] += 1
        wlc_ngrams = Counter()
        for i in range(len(wlc_pos) - n + 1):
            wlc_ngrams[tuple(wlc_pos[i:i + n])] += 1

        jsd = jensen_shannon_divergence(dss_ngrams, wlc_ngrams)
        log.info(f"  JSD(n={n}): {jsd:.6f}")

        all_ngrams = set(dss_ngrams.keys()) | set(wlc_ngrams.keys())
        total_dss = sum(dss_ngrams.values())
        total_wlc = sum(wlc_ngrams.values())
        changes = []
        for ng in all_ngrams:
            freq_dss = dss_ngrams.get(ng, 0) / total_dss
            freq_wlc = wlc_ngrams.get(ng, 0) / total_wlc
            delta = freq_dss - freq_wlc
            changes.append({
                "ngram": " ".join(ng), "freq_dss": round(freq_dss, 6),
                "freq_wlc": round(freq_wlc, 6), "delta": round(delta, 6),
                "abs_delta": round(abs(delta), 6),
            })
        changes.sort(key=lambda x: x["abs_delta"], reverse=True)

        results["n_gram_divergences"][f"n={n}"] = {
            "jsd": jsd,
            "n_unique_dss": len(dss_ngrams),
            "n_unique_wlc": len(wlc_ngrams),
            "top_50_changes": changes[:50],
        }

    # Chi-cuadrado contingencia 2×K (DSS vs WLC POS counts)
    pos_cats = sorted(set(dss_uni.keys()) | set(wlc_uni.keys()))
    obs_dss = np.array([dss_uni.get(c, 0) for c in pos_cats])
    obs_wlc = np.array([wlc_uni.get(c, 0) for c in pos_cats])
    mask = (obs_dss + obs_wlc) > 10  # filter rare categories
    if mask.sum() > 1:
        contingency = np.array([obs_dss[mask], obs_wlc[mask]])
        chi2, p_chi, dof, _ = stats.chi2_contingency(contingency)
        log.info(f"  Chi² contingencia POS: χ²={chi2:.1f}, dof={dof}, p={p_chi:.6f}")
        results["chi2_pos_contingency"] = {
            "chi2": float(chi2), "p_value": float(p_chi), "dof": int(dof),
            "categories": [pos_cats[i] for i in range(len(pos_cats)) if mask[i]],
            "conclusion": "different" if p_chi < 0.05 else "compatible",
        }

    with open(RESULTS_DIR / "pos_divergence.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return results


# ══════════════════════════════════════════════════════════════════════════
# Nivel 3: Alineamiento por Lemma
# ══════════════════════════════════════════════════════════════════════════

def strip_hebrew_vowels(text):
    """Elimina vocales y cantilación del texto hebreo."""
    return re.sub(r'[\u0591-\u05C7]', '', text)


def levenshtein_distance(s1, s2):
    """Distancia de Levenshtein entre dos strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
        prev = curr
    return prev[-1]


def level3_alignment(dss_data, wlc_data):
    """Alineamiento word-level versículo-por-versículo usando texto consonántico.

    Estrategia: Para cada versículo con match (chapter, verse) en ambos textos,
    alinear las listas de palabras consonánticas con SequenceMatcher.
    DSS usa glex_cons (glex sin vocales), WLC usa text_cons (text sin vocales).
    """
    log.info("\n" + "=" * 60)
    log.info("NIVEL 3: Alineamiento Consonántico por Versículo")
    log.info("=" * 60)

    # Indexar DSS por (chapter, verse)
    dss_by_verse = {}
    for v in dss_data["verses"]:
        key = (v["chapter"], v["verse"])
        dss_by_verse[key] = v

    # Indexar WLC por (chapter, verse)
    wlc_by_verse = {}
    for v in wlc_data["verses"]:
        key = (v["chapter"], v["verse"])
        wlc_by_verse[key] = v

    common_keys = sorted(set(dss_by_verse.keys()) & set(wlc_by_verse.keys()))
    dss_only = set(dss_by_verse.keys()) - set(wlc_by_verse.keys())
    wlc_only = set(wlc_by_verse.keys()) - set(dss_by_verse.keys())

    log.info(f"  DSS versículos: {len(dss_by_verse)}")
    log.info(f"  WLC versículos: {len(wlc_by_verse)}")
    log.info(f"  En común: {len(common_keys)}")
    log.info(f"  Solo DSS: {len(dss_only)}, Solo WLC: {len(wlc_only)}")

    all_ops = []
    global_dss_idx = 0  # running index into DSS words for ablation mapping

    # Construir mapa global DSS word → verse for ablation
    dss_word_verse_map = []
    for v in dss_data["verses"]:
        for _ in range(v["n_words"]):
            dss_word_verse_map.append((v["chapter"], v["verse"]))

    for vkey in common_keys:
        dv = dss_by_verse[vkey]
        wv = wlc_by_verse[vkey]
        ch, vs = vkey

        dss_cons = dv.get("glex_cons", [])
        wlc_cons = wv.get("text_cons", [])
        dss_pos_list = dv.get("pos", [])
        wlc_pos_list = wv.get("pos", [])
        dss_text = dv.get("glex", dv.get("words", []))
        wlc_text = wv.get("text", wv.get("words", []))

        matcher = difflib.SequenceMatcher(None, dss_cons, wlc_cons, autojunk=False)

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                for k in range(i2 - i1):
                    all_ops.append({
                        "type": "identical",
                        "chapter": ch, "verse": vs,
                        "dss_word": dss_text[i1 + k] if i1 + k < len(dss_text) else "",
                        "wlc_word": wlc_text[j1 + k] if j1 + k < len(wlc_text) else "",
                        "dss_cons": dss_cons[i1 + k],
                        "wlc_cons": wlc_cons[j1 + k],
                        "dss_pos": dss_pos_list[i1 + k] if i1 + k < len(dss_pos_list) else "?",
                        "wlc_pos": wlc_pos_list[j1 + k] if j1 + k < len(wlc_pos_list) else "?",
                    })
            elif tag == "replace":
                n_d, n_w = i2 - i1, j2 - j1
                n_pairs = min(n_d, n_w)
                for k in range(n_pairs):
                    di, wi = i1 + k, j1 + k
                    d_cons = dss_cons[di] if di < len(dss_cons) else ""
                    w_cons = wlc_cons[wi] if wi < len(wlc_cons) else ""
                    lev = levenshtein_distance(d_cons, w_cons)
                    d_pos = dss_pos_list[di] if di < len(dss_pos_list) else "?"
                    w_pos = wlc_pos_list[wi] if wi < len(wlc_pos_list) else "?"
                    if lev <= 1:
                        rtype = "orthographic"
                    elif d_pos == w_pos:
                        rtype = "substitution_same_pos"
                    else:
                        rtype = "substitution_diff_pos"
                    all_ops.append({
                        "type": rtype,
                        "chapter": ch, "verse": vs,
                        "dss_word": dss_text[di] if di < len(dss_text) else "",
                        "wlc_word": wlc_text[wi] if wi < len(wlc_text) else "",
                        "dss_cons": d_cons, "wlc_cons": w_cons,
                        "dss_pos": d_pos, "wlc_pos": w_pos,
                        "levenshtein": lev,
                    })
                for k in range(n_pairs, n_d):
                    di = i1 + k
                    all_ops.append({
                        "type": "deletion",
                        "chapter": ch, "verse": vs,
                        "dss_word": dss_text[di] if di < len(dss_text) else "",
                        "dss_cons": dss_cons[di] if di < len(dss_cons) else "",
                        "dss_pos": dss_pos_list[di] if di < len(dss_pos_list) else "?",
                    })
                for k in range(n_pairs, n_w):
                    wi = j1 + k
                    all_ops.append({
                        "type": "insertion",
                        "chapter": ch, "verse": vs,
                        "wlc_word": wlc_text[wi] if wi < len(wlc_text) else "",
                        "wlc_cons": wlc_cons[wi] if wi < len(wlc_cons) else "",
                        "wlc_pos": wlc_pos_list[wi] if wi < len(wlc_pos_list) else "?",
                    })
            elif tag == "delete":
                for k in range(i2 - i1):
                    di = i1 + k
                    all_ops.append({
                        "type": "deletion",
                        "chapter": ch, "verse": vs,
                        "dss_word": dss_text[di] if di < len(dss_text) else "",
                        "dss_cons": dss_cons[di] if di < len(dss_cons) else "",
                        "dss_pos": dss_pos_list[di] if di < len(dss_pos_list) else "?",
                    })
            elif tag == "insert":
                for k in range(j2 - j1):
                    wi = j1 + k
                    all_ops.append({
                        "type": "insertion",
                        "chapter": ch, "verse": vs,
                        "wlc_word": wlc_text[wi] if wi < len(wlc_text) else "",
                        "wlc_cons": wlc_cons[wi] if wi < len(wlc_cons) else "",
                        "wlc_pos": wlc_pos_list[wi] if wi < len(wlc_pos_list) else "?",
                    })

    # Verses only in DSS → all deletions
    for vkey in sorted(dss_only):
        dv = dss_by_verse[vkey]
        for i, w in enumerate(dv.get("glex", dv.get("words", []))):
            all_ops.append({
                "type": "deletion",
                "chapter": vkey[0], "verse": vkey[1],
                "dss_word": w,
                "dss_pos": dv["pos"][i] if i < len(dv["pos"]) else "?",
            })

    # Verses only in WLC → all insertions
    for vkey in sorted(wlc_only):
        wv = wlc_by_verse[vkey]
        for i, w in enumerate(wv.get("text", wv.get("words", []))):
            all_ops.append({
                "type": "insertion",
                "chapter": vkey[0], "verse": vkey[1],
                "wlc_word": w,
                "wlc_pos": wv["pos"][i] if i < len(wv["pos"]) else "?",
            })

    type_counts = Counter(op["type"] for op in all_ops)
    total = len(all_ops)
    n_variants = total - type_counts.get("identical", 0)

    log.info(f"\n  Alineamiento completado:")
    log.info(f"    Total operaciones: {total}")
    for t, c in type_counts.most_common():
        log.info(f"    {t}: {c} ({100*c/total:.1f}%)")
    log.info(f"    Variantes: {n_variants} ({100*n_variants/total:.1f}%)")

    # Densidad por capítulo
    ch_variants = defaultdict(lambda: Counter())
    for op in all_ops:
        ch_variants[op.get("chapter", 0)][op["type"]] += 1

    variant_density = []
    for ch in sorted(ch_variants.keys()):
        ch_total = sum(ch_variants[ch].values())
        ch_var = ch_total - ch_variants[ch].get("identical", 0)
        variant_density.append({
            "chapter": ch, "total": ch_total, "variants": ch_var,
            "density": round(ch_var / ch_total, 4) if ch_total > 0 else 0,
            "breakdown": dict(ch_variants[ch]),
        })

    result = {
        "total_operations": total,
        "type_counts": dict(type_counts),
        "n_variants": n_variants,
        "variant_rate": round(n_variants / total, 4) if total > 0 else 0,
        "n_verses_common": len(common_keys),
        "n_verses_dss_only": len(dss_only),
        "n_verses_wlc_only": len(wlc_only),
        "variant_density_by_chapter": variant_density,
        "sample_orthographic": [op for op in all_ops if op["type"] == "orthographic"][:20],
        "sample_substitution_same_pos": [op for op in all_ops if op["type"] == "substitution_same_pos"][:20],
        "sample_substitution_diff_pos": [op for op in all_ops if op["type"] == "substitution_diff_pos"][:20],
        "sample_insertion": [op for op in all_ops if op["type"] == "insertion"][:20],
        "sample_deletion": [op for op in all_ops if op["type"] == "deletion"][:20],
    }

    with open(RESULTS_DIR / "alignment_variants.json", "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)

    return result, all_ops


# ══════════════════════════════════════════════════════════════════════════
# Nivel 4: Estudio de Ablación
# ══════════════════════════════════════════════════════════════════════════

def level4_ablation(dss_data, all_ops, n_permutations=100):
    """Sensibilidad de H a cada tipo de variante textual.

    Usa verse-lengths (no line-lengths) como serie temporal base,
    consistente con Fase 8 que demostró verse-alignment es correcto.
    """
    log.info("\n" + "=" * 60)
    log.info("NIVEL 4: Estudio de Ablación (verse-based)")
    log.info("=" * 60)

    # Serie temporal = palabras por versículo (verse-aligned con Fase 8)
    dss_verse_lens = np.array([v["n_words"] for v in dss_data["verses"]], dtype=float)
    h_baseline, _ = hurst_exponent_rs(dss_verse_lens)
    a_baseline, _ = dfa_exponent(dss_verse_lens)
    log.info(f"  Baseline DSS (verses): H={h_baseline:.4f}, α={a_baseline:.4f}")
    log.info(f"  {len(dss_verse_lens)} versículos")

    # Mapear (chapter, verse) → índice en dss_verse_lens
    verse_to_idx = {}
    for idx, v in enumerate(dss_data["verses"]):
        verse_to_idx[(v["chapter"], v["verse"])] = idx

    # Clasificar operaciones por tipo
    variant_types = {
        "orthographic": [],
        "substitution_same_pos": [],
        "substitution_diff_pos": [],
        "insertion": [],
        "deletion": [],
    }
    for op in all_ops:
        t = op["type"]
        if t in variant_types:
            variant_types[t].append(op)

    n_verses = len(dss_verse_lens)
    experiments = {}

    for vtype, ops in variant_types.items():
        if not ops:
            log.info(f"  E_{vtype}: sin operaciones")
            experiments[vtype] = {"n_ops": 0, "H_modified": None, "delta_H": None}
            continue

        def apply_ops(base_lens, ops_to_apply):
            mod = base_lens.copy()
            for op in ops_to_apply:
                ch = op.get("chapter")
                vs = op.get("verse")
                vidx = verse_to_idx.get((ch, vs))
                if vidx is None:
                    continue
                if op["type"] == "deletion":
                    mod[vidx] = max(1, mod[vidx] - 1)
                elif op["type"] == "insertion":
                    mod[vidx] += 1
                else:
                    # Sustituciones: cambio de longitud menor
                    mod[vidx] += np.random.choice([-1, 0, 1], p=[0.1, 0.8, 0.1])
                    mod[vidx] = max(1, mod[vidx])
            return mod

        modified = apply_ops(dss_verse_lens, ops)
        h_mod, _ = hurst_exponent_rs(modified)
        a_mod, _ = dfa_exponent(modified)
        delta_h = h_mod - h_baseline if not np.isnan(h_mod) else None

        # Bootstrap CI
        h_boot = []
        for _ in range(n_permutations):
            boot_idx = np.random.choice(len(ops), size=len(ops), replace=True)
            boot_ops = [ops[i] for i in boot_idx]
            boot_mod = apply_ops(dss_verse_lens, boot_ops)
            hb, _ = hurst_exponent_rs(boot_mod)
            if not np.isnan(hb):
                h_boot.append(hb)

        ci = (float(np.percentile(h_boot, 2.5)),
              float(np.percentile(h_boot, 97.5))) if h_boot else (None, None)

        log.info(f"  E_{vtype}: {len(ops)} ops, H={h_mod:.4f}, "
                 f"ΔH={delta_h:.4f}" if delta_h else f"  E_{vtype}: {len(ops)} ops, H=NaN")
        if ci[0] is not None:
            log.info(f"    CI95: [{ci[0]:.4f}, {ci[1]:.4f}]")

        experiments[vtype] = {
            "n_ops": len(ops),
            "H_modified": float(h_mod) if not np.isnan(h_mod) else None,
            "alpha_modified": float(a_mod) if not np.isnan(a_mod) else None,
            "delta_H": float(delta_h) if delta_h is not None else None,
            "ci_95": ci,
        }

    # E_ALL: aplicar todas las variantes
    log.info("\n  E_ALL: todas las variantes...")
    all_mod = dss_verse_lens.copy()
    for op in all_ops:
        t = op["type"]
        if t == "identical":
            continue
        ch = op.get("chapter")
        vs = op.get("verse")
        vidx = verse_to_idx.get((ch, vs))
        if vidx is None:
            continue
        if t == "deletion":
            all_mod[vidx] = max(1, all_mod[vidx] - 1)
        elif t == "insertion":
            all_mod[vidx] += 1
        else:
            all_mod[vidx] += np.random.choice([-1, 0, 1], p=[0.1, 0.8, 0.1])
            all_mod[vidx] = max(1, all_mod[vidx])

    h_all, _ = hurst_exponent_rs(all_mod)
    a_all, _ = dfa_exponent(all_mod)
    log.info(f"  E_ALL: H={h_all:.4f}, ΔH={h_all - h_baseline:.4f}")

    experiments["all_variants"] = {
        "n_ops": sum(len(v) for v in variant_types.values()),
        "H_modified": float(h_all),
        "alpha_modified": float(a_all),
        "delta_H": float(h_all - h_baseline),
    }

    ranking = sorted(
        [(k, v) for k, v in experiments.items()
         if v.get("delta_H") is not None and k != "all_variants"],
        key=lambda x: abs(x[1]["delta_H"]), reverse=True
    )

    result = {
        "H_baseline": float(h_baseline),
        "alpha_baseline": float(a_baseline),
        "n_permutations": n_permutations,
        "experiments": experiments,
        "impact_ranking": [
            {"type": k, "n_ops": v["n_ops"],
             "delta_H": v["delta_H"], "abs_delta_H": abs(v["delta_H"])}
            for k, v in ranking
        ],
    }

    with open(RESULTS_DIR / "ablation_study.json", "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result


# ══════════════════════════════════════════════════════════════════════════
# Nivel 5: Modelo de Decaimiento
# ══════════════════════════════════════════════════════════════════════════

def level5_decay_model(n_bootstrap=1000):
    """Modelos de decaimiento H(t) con bootstrap.

    Datos verse-aligned de Fase 8:
    - DSS (~100 a.C.): H = 0.681
    - WLC (~1000 d.C.): H = 0.672
    """
    log.info("\n" + "=" * 60)
    log.info("NIVEL 5: Modelo de Decaimiento")
    log.info("=" * 60)

    t_dss, t_wlc = -100, 1000
    h_dss, h_wlc = 0.681, 0.672
    sigma_h = 0.05
    t_comp = -700  # composición de Isaías

    log.info(f"  DSS: t={t_dss}, H={h_dss}")
    log.info(f"  WLC: t={t_wlc}, H={h_wlc}")
    log.info(f"  ΔH={h_dss - h_wlc:.4f} sobre {t_wlc - t_dss} años")

    # Lineal: H(t) = a + b*t
    b = (h_wlc - h_dss) / (t_wlc - t_dss)
    a = h_dss - b * t_dss
    h_comp_lin = a + b * t_comp

    # Exponencial: H(t) = H∞ + (H₀-H∞)*exp(-λ*(t-t₀))
    h_inf = 0.5
    ratio = (h_wlc - h_inf) / (h_dss - h_inf)
    if ratio > 0:
        lam = -np.log(ratio) / (t_wlc - t_dss)
        h_comp_exp = h_inf + (h_dss - h_inf) * np.exp(-lam * (t_comp - t_dss))
    else:
        lam, h_comp_exp = 0.0, h_dss

    log.info(f"  Lineal: H({t_comp})={h_comp_lin:.4f}")
    log.info(f"  Exponencial: λ={lam:.8f}, H({t_comp})={h_comp_exp:.4f}")

    # Bootstrap
    rng = np.random.default_rng(42)
    boot_lin, boot_exp = [], []
    for _ in range(n_bootstrap):
        hd = h_dss + rng.normal(0, sigma_h)
        hw = h_wlc + rng.normal(0, sigma_h)
        bb = (hw - hd) / (t_wlc - t_dss)
        aa = hd - bb * t_dss
        boot_lin.append(aa + bb * t_comp)
        r = (hw - h_inf) / (hd - h_inf)
        if r > 0 and hd > h_inf:
            ll = -np.log(r) / (t_wlc - t_dss)
            boot_exp.append(h_inf + (hd - h_inf) * np.exp(-ll * (t_comp - t_dss)))
        else:
            boot_exp.append(hd)

    ci_lin = (float(np.percentile(boot_lin, 2.5)), float(np.percentile(boot_lin, 97.5)))
    ci_exp = (float(np.percentile(boot_exp, 2.5)), float(np.percentile(boot_exp, 97.5)))
    decay_rate = abs(h_dss - h_wlc) / (t_wlc - t_dss) * 100

    log.info(f"  Tasa: {decay_rate:.4f} H/siglo")
    log.info(f"  CI95 lin: {ci_lin}, CI95 exp: {ci_exp}")

    result = {
        "data_points": {
            "dss": {"year": t_dss, "H": h_dss},
            "wlc": {"year": t_wlc, "H": h_wlc},
        },
        "sigma_H": sigma_h,
        "decay_rate_per_century": float(decay_rate),
        "linear_model": {"a": float(a), "b": float(b),
                         "H_at_composition": float(h_comp_lin), "ci_95": ci_lin},
        "exponential_model": {"H_inf": h_inf, "lambda": float(lam),
                              "H_at_composition": float(h_comp_exp), "ci_95": ci_exp},
        "composition_year": t_comp,
        "n_bootstrap": n_bootstrap,
        "conclusion": (
            f"Tasa: {decay_rate:.4f} H/siglo (efectivamente cero). "
            f"H({t_comp}) ≈ {h_comp_lin:.3f}. La estructura es invariante."
        ),
    }

    with open(RESULTS_DIR / "decay_model.json", "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result


# ══════════════════════════════════════════════════════════════════════════
# Consolidación
# ══════════════════════════════════════════════════════════════════════════

def generate_summary(results):
    """Resumen consolidado de Fase 10."""
    summary = {
        "phase": 10,
        "title": "Análisis Palabra-por-Palabra: 1QIsa^a vs WLC Isaiah",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    if "n0" in results:
        summary["feature_discovery"] = {
            "n_features": results["n0"]["n_features"],
            "key_available": results["n0"]["key_features_available"],
        }
    if "n1" in results:
        summary["memory_profile"] = {
            "dss_H_mean": results["n1"]["dss"]["H_mean"],
            "wlc_H_mean": results["n1"]["wlc"]["H_mean"],
            "delta_H_mean": results["n1"]["delta_H_mean"],
        }
    if "n2" in results:
        summary["pos_divergence"] = {
            "jsd_unigram": results["n2"]["n_gram_divergences"].get("n=1", {}).get("jsd"),
            "jsd_4gram": results["n2"]["n_gram_divergences"].get("n=4", {}).get("jsd"),
        }
    if "n3" in results:
        summary["alignment"] = {
            "total_operations": results["n3"]["total_operations"],
            "n_variants": results["n3"]["n_variants"],
            "variant_rate": results["n3"]["variant_rate"],
            "type_counts": results["n3"]["type_counts"],
        }
    if "n4" in results:
        summary["ablation"] = {
            "H_baseline": results["n4"]["H_baseline"],
            "impact_ranking": results["n4"]["impact_ranking"],
            "E_all_delta_H": results["n4"]["experiments"].get(
                "all_variants", {}).get("delta_H"),
        }
    if "n5" in results:
        summary["decay_model"] = {
            "decay_rate_per_century": results["n5"]["decay_rate_per_century"],
            "H_at_composition": results["n5"]["linear_model"]["H_at_composition"],
        }

    with open(RESULTS_DIR / "dss_wordlevel_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    log.info("=" * 70)
    log.info("FASE 10 — Análisis Palabra-por-Palabra")
    log.info("1QIsa^a (DSS, ~100 a.C.) vs Isaías (WLC, ~1000 d.C.)")
    log.info("=" * 70)

    results = {}

    # N0: Feature Discovery
    try:
        dss_dir = setup_dss_corpus()
        results["n0"] = level0_feature_discovery(dss_dir)
    except Exception as e:
        log.error(f"N0 falló: {e}")
        import traceback; log.error(traceback.format_exc())
        return

    # Cargar datos
    log.info("\n── Cargando datos ──")
    try:
        dss_data = load_dss_data(dss_dir)
    except Exception as e:
        log.error(f"Error cargando DSS: {e}")
        import traceback; log.error(traceback.format_exc())
        return
    try:
        wlc_data = load_wlc_data()
    except Exception as e:
        log.error(f"Error cargando WLC: {e}")
        import traceback; log.error(traceback.format_exc())
        return

    # N1, N2, N5 (paralelos, no dependen de N3)
    for level_name, level_func, level_args in [
        ("n1", level1_memory_profile, (dss_data, wlc_data)),
        ("n2", level2_pos_divergence, (dss_data, wlc_data)),
        ("n5", level5_decay_model, ()),
    ]:
        try:
            results[level_name] = level_func(*level_args)
        except Exception as e:
            log.error(f"{level_name.upper()} falló: {e}")
            import traceback; log.error(traceback.format_exc())

    # N3: Alineamiento
    all_ops = None
    try:
        results["n3"], all_ops = level3_alignment(dss_data, wlc_data)
    except Exception as e:
        log.error(f"N3 falló: {e}")
        import traceback; log.error(traceback.format_exc())

    # N4: Ablación (depende de N3)
    if all_ops is not None:
        try:
            results["n4"] = level4_ablation(dss_data, all_ops)
        except Exception as e:
            log.error(f"N4 falló: {e}")
            import traceback; log.error(traceback.format_exc())
    else:
        log.warning("N4 saltado: N3 no completó")

    # Consolidar
    summary = generate_summary(results)

    elapsed = time.time() - t0
    log.info(f"\n{'=' * 70}")
    log.info(f"FASE 10 completada en {elapsed:.1f}s")
    log.info(f"Niveles: {list(results.keys())}")
    log.info(f"Resultados en {RESULTS_DIR}")

    for key, val in summary.items():
        if isinstance(val, dict):
            log.info(f"  {key}: {json.dumps(val, default=str)[:200]}")
        else:
            log.info(f"  {key}: {val}")


if __name__ == "__main__":
    main()
