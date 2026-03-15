#!/usr/bin/env python3
"""
orchestrator_fase5.py — Fase 5: Corpus de Comparación
¿Es el AT único o cualquier texto antiguo extenso tiene las mismas propiedades?

Descarga y analiza 4 corpus externos con las mismas 6 métricas:
1. Hurst H (R/S analysis)
2. DFA α (Detrended Fluctuation Analysis)
3. Box-counting D_f
4. Bond dimension χ (SVD correlaciones, 99% varianza)
5. Permutation test MPS (n=10,000)
6. ΔS Von Neumann (donde hay morfología POS)

Corpus: Corán (árabe), Homero (griego), Heródoto (griego), Mishnah (hebreo)
"""
import json, logging, math, os, time, re, sys
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from scipy import linalg as la
from scipy import stats as sp_stats

BASE = Path(__file__).parent
CORPUS_DIR = BASE / "results" / "comparison_corpora"
CORPUS_DIR.mkdir(parents=True, exist_ok=True)
OUT = BASE / "results"
OUT.mkdir(parents=True, exist_ok=True)
LOG_DIR = BASE / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=LOG_DIR / "orchestrator_fase5.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("fase5")

# Our 9 standard POS categories
POS_CATEGORIES = [
    "noun", "verb", "pronoun", "adjective", "adverb",
    "preposition", "conjunction", "particle", "other",
]
N_POS = len(POS_CATEGORIES)

# ═══════════════════════════════════════════════════════════════════════
# METRIC FUNCTIONS (identical to our AT/NT pipeline)
# ═══════════════════════════════════════════════════════════════════════

def hurst_exponent_rs(series):
    """Hurst exponent via Rescaled Range (R/S) analysis."""
    n = len(series)
    if n < 20:
        return None, None
    min_len = 10
    max_divs = min(50, n // min_len)
    log_ns, log_rs = [], []
    for div_count in range(2, max_divs + 1):
        sub_len = n // div_count
        if sub_len < min_len:
            break
        rs_values = []
        for i in range(div_count):
            sub = series[i * sub_len:(i + 1) * sub_len]
            mean = np.mean(sub)
            deviations = sub - mean
            cumulative = np.cumsum(deviations)
            R = np.max(cumulative) - np.min(cumulative)
            S = np.std(sub, ddof=1)
            if S > 0:
                rs_values.append(R / S)
        if rs_values:
            mean_rs = np.mean(rs_values)
            if mean_rs > 0:
                log_ns.append(math.log(sub_len))
                log_rs.append(math.log(mean_rs))
    if len(log_ns) < 3:
        return None, None
    slope, _, r_val, _, _ = sp_stats.linregress(log_ns, log_rs)
    return round(float(slope), 4), round(float(r_val ** 2), 4)


def dfa_exponent(series, min_box=4, max_box=None):
    """Detrended Fluctuation Analysis."""
    n = len(series)
    if n < 20:
        return None, None
    if max_box is None:
        max_box = n // 4
    mean = np.mean(series)
    y = np.cumsum(series - mean)
    box_sizes = []
    s = min_box
    while s <= max_box:
        box_sizes.append(int(s))
        s *= 1.5
    box_sizes = sorted(set(box_sizes))
    if len(box_sizes) < 3:
        return None, None
    log_s, log_f = [], []
    for s in box_sizes:
        n_boxes = n // s
        if n_boxes < 1:
            continue
        fluctuations = []
        for i in range(n_boxes):
            segment = y[i * s:(i + 1) * s]
            x_range = np.arange(len(segment), dtype=float)
            if len(segment) < 2:
                continue
            slope, intercept = np.polyfit(x_range, segment, 1)
            trend = slope * x_range + intercept
            residual = segment - trend
            rms = np.sqrt(np.mean(residual ** 2))
            fluctuations.append(rms)
        if fluctuations:
            mean_f = np.mean(fluctuations)
            if mean_f > 0:
                log_s.append(math.log(s))
                log_f.append(math.log(mean_f))
    if len(log_s) < 3:
        return None, None
    slope, _, r_val, _, _ = sp_stats.linregress(log_s, log_f)
    return round(float(slope), 4), round(float(r_val ** 2), 4)


def box_counting_dimension(series):
    """Box-counting dimension of a 1D signal."""
    if len(series) < 10:
        return None, None
    n = len(series)
    scales = [2 ** i for i in range(0, int(math.log2(n))) if 2 ** i < n]
    if not scales:
        scales = [1, 2, 4, 8]
    smin, smax = series.min(), series.max()
    if smax == smin:
        return None, None
    normalized = (series - smin) / (smax - smin)
    log_eps, log_n = [], []
    for eps in scales:
        if eps >= n:
            continue
        n_time_boxes = int(math.ceil(n / eps))
        n_val_boxes = max(1, int(math.ceil(1.0 / (eps / n))))
        occupied = set()
        for i, val in enumerate(normalized):
            t_box = i // eps
            v_box = min(int(val * n_val_boxes), n_val_boxes - 1)
            occupied.add((t_box, v_box))
        count = len(occupied)
        if count > 0:
            log_eps.append(math.log(1.0 / eps))
            log_n.append(math.log(count))
    if len(log_eps) < 3:
        return None, None
    slope, _, r_val, _, _ = sp_stats.linregress(log_eps, log_n)
    return round(float(slope), 4), round(float(r_val ** 2), 4)


def compute_bond_dimension(series, max_lag=256, threshold=0.99):
    """Bond dimension χ from autocorrelation matrix SVD."""
    n = min(max_lag, len(series) // 4)
    if n < 2:
        return 1, np.array([1.0])
    mean = np.mean(series)
    centered = series - mean
    acf = np.zeros(n)
    for lag in range(n):
        if lag >= len(centered):
            break
        acf[lag] = np.mean(centered[: len(centered) - lag] * centered[lag:])
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


def permutation_test_chi(series, n_perm=10000, max_lag=64, threshold=0.99):
    """Permutation test: is χ_obs significantly lower than random?"""
    chi_real, _ = compute_bond_dimension(series, max_lag=max_lag, threshold=threshold)
    np.random.seed(42)
    chi_perms = []
    last_log = time.time()
    for i in range(n_perm):
        shuffled = np.random.permutation(series)
        chi_s, _ = compute_bond_dimension(shuffled, max_lag=max_lag, threshold=threshold)
        chi_perms.append(chi_s)
        now = time.time()
        if now - last_log >= 30:
            log.info(f"    Permutation {i + 1}/{n_perm}")
            last_log = now
    chi_arr = np.array(chi_perms)
    p_value = float(np.mean(chi_arr <= chi_real))
    return {
        "chi_observed": chi_real,
        "chi_perm_mean": round(float(np.mean(chi_arr)), 2),
        "chi_perm_std": round(float(np.std(chi_arr)), 2),
        "p_value": round(p_value, 6),
        "significant": bool(p_value < 0.05),
    }


def build_density_matrix(verse_pos_vectors):
    """Build density matrix ρ from POS frequency vectors of each verse/unit."""
    d = verse_pos_vectors.shape[1]
    rho = np.zeros((d, d))
    for vec in verse_pos_vectors:
        norm = np.linalg.norm(vec)
        if norm > 0:
            v = vec / norm
            rho += np.outer(v, v)
    tr = np.trace(rho)
    if tr > 0:
        rho = rho / tr
    return rho


def von_neumann_entropy(rho):
    """S_vN = -Tr(ρ log₂ ρ)"""
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = eigenvalues[eigenvalues > 1e-15]
    if len(eigenvalues) == 0:
        return 0.0
    return float(-np.sum(eigenvalues * np.log2(eigenvalues)))


def shannon_entropy(probs):
    """Shannon entropy in bits."""
    probs = probs[probs > 0]
    if len(probs) == 0:
        return 0.0
    return float(-np.sum(probs * np.log2(probs)))


def compute_delta_s(units_pos):
    """
    Compute ΔS = S_vN - S_Shannon for a list of units.
    units_pos: list of lists of POS strings per unit.
    Returns mean ΔS and delta_s_group.
    """
    # Build per-unit POS vectors
    n_units = len(units_pos)
    vectors = np.zeros((n_units, N_POS))
    total_counts = np.zeros(N_POS)
    for i, unit in enumerate(units_pos):
        for pos in unit:
            try:
                idx = POS_CATEGORIES.index(pos)
            except ValueError:
                idx = POS_CATEGORIES.index("other")
            vectors[i, idx] += 1
            total_counts[idx] += 1

    # S_vN from density matrix
    rho = build_density_matrix(vectors)
    s_vn = von_neumann_entropy(rho)

    # S_Shannon from marginal POS
    total = np.sum(total_counts)
    if total > 0:
        probs = total_counts / total
    else:
        probs = np.ones(N_POS) / N_POS
    s_sh = shannon_entropy(probs)

    delta_s = s_vn - s_sh

    # Random control (100 simulations for speed)
    np.random.seed(42)
    sizes = [len(u) for u in units_pos]
    marginal = probs.copy()
    marginal = marginal / marginal.sum() if marginal.sum() > 0 else np.ones(N_POS) / N_POS
    ds_sims = []
    for _ in range(100):
        syn_vecs = np.zeros((n_units, N_POS))
        for vi, sz in enumerate(sizes):
            if sz > 0:
                draws = np.random.choice(N_POS, size=sz, p=marginal)
                for p in draws:
                    syn_vecs[vi, p] += 1
        rho_syn = build_density_matrix(syn_vecs)
        s_vn_syn = von_neumann_entropy(rho_syn)
        syn_total = syn_vecs.sum(axis=0)
        st = syn_total.sum()
        syn_probs = syn_total / st if st > 0 else marginal
        s_sh_syn = shannon_entropy(syn_probs)
        ds_sims.append(s_vn_syn - s_sh_syn)

    ds_rand_mean = np.mean(ds_sims)
    if delta_s < ds_rand_mean:
        group = "more_structured"
    else:
        group = "more_varied"

    return round(delta_s, 6), group


def compute_zipf_lemma(lemma_counts):
    """Zipf exponent from lemma frequency distribution."""
    if not lemma_counts:
        return None
    freqs = sorted(lemma_counts.values(), reverse=True)
    if len(freqs) < 10:
        return None
    ranks = np.arange(1, len(freqs) + 1, dtype=float)
    freqs_arr = np.array(freqs, dtype=float)
    # log-log linear regression
    mask = freqs_arr > 0
    log_r = np.log(ranks[mask])
    log_f = np.log(freqs_arr[mask])
    if len(log_r) < 5:
        return None
    slope, _, _, _, _ = sp_stats.linregress(log_r, log_f)
    return round(float(-slope), 4)


# ═══════════════════════════════════════════════════════════════════════
# FULL ANALYSIS PIPELINE for a parsed corpus
# ═══════════════════════════════════════════════════════════════════════

def analyze_corpus(name, unit_lengths, units_pos=None, lemma_counts=None):
    """
    Run the full 6-metric pipeline on a corpus.
    unit_lengths: np.array of unit lengths (words per verse/sentence/mishnah)
    units_pos: list of lists of POS strings per unit (None if no morphology)
    lemma_counts: Counter of lemma frequencies
    """
    log.info(f"=== Analizando {name} ({len(unit_lengths)} unidades) ===")
    series = unit_lengths.astype(float)
    result = {
        "corpus": name,
        "n_units": len(series),
        "n_words": int(np.sum(series)),
        "mean_unit_length": round(float(np.mean(series)), 4),
    }

    # 1. Hurst H
    log.info(f"  [{name}] Hurst H...")
    H, H_r2 = hurst_exponent_rs(series)
    result["hurst_H"] = H
    result["hurst_R2"] = H_r2
    log.info(f"  [{name}] H={H}")

    # 2. DFA α
    log.info(f"  [{name}] DFA α...")
    alpha, alpha_r2 = dfa_exponent(series)
    result["dfa_alpha"] = alpha
    result["dfa_R2"] = alpha_r2
    log.info(f"  [{name}] α={alpha}")

    # 3. Box-counting D_f
    log.info(f"  [{name}] Box-counting D_f...")
    D_f, D_r2 = box_counting_dimension(series)
    result["box_counting_Df"] = D_f
    result["box_counting_R2"] = D_r2
    log.info(f"  [{name}] D_f={D_f}")

    # 4. Bond dimension χ
    log.info(f"  [{name}] Bond dimension χ...")
    chi, sigma = compute_bond_dimension(series, max_lag=256, threshold=0.99)
    result["bond_dim_chi"] = chi
    log.info(f"  [{name}] χ={chi}")

    # 5. Permutation test MPS (n=10,000)
    log.info(f"  [{name}] Permutation test MPS (n=10000)...")
    perm = permutation_test_chi(series, n_perm=10000, max_lag=64, threshold=0.99)
    result["mps_permtest_p"] = perm["p_value"]
    result["mps_permtest_chi_obs"] = perm["chi_observed"]
    result["mps_permtest_chi_rand"] = perm["chi_perm_mean"]
    result["mps_significant"] = perm["significant"]
    log.info(f"  [{name}] χ_obs={perm['chi_observed']}, χ_rand={perm['chi_perm_mean']}, p={perm['p_value']}")

    # 6. ΔS Von Neumann (only if POS available)
    if units_pos is not None:
        log.info(f"  [{name}] ΔS Von Neumann...")
        delta_s_mean, delta_s_group = compute_delta_s(units_pos)
        result["delta_S_mean"] = delta_s_mean
        result["delta_s_group"] = delta_s_group
        log.info(f"  [{name}] ΔS={delta_s_mean}, group={delta_s_group}")
    else:
        result["delta_S_mean"] = None
        result["delta_s_group"] = "na"

    # Zipf
    if lemma_counts:
        log.info(f"  [{name}] Zipf lemas...")
        result["zipf_s_lemma"] = compute_zipf_lemma(lemma_counts)
        log.info(f"  [{name}] zipf_s={result['zipf_s_lemma']}")
    else:
        result["zipf_s_lemma"] = None

    log.info(f"  [{name}] DONE")
    return result


# ═══════════════════════════════════════════════════════════════════════
# DOWNLOADERS
# ═══════════════════════════════════════════════════════════════════════

def download_file(url, dest):
    """Download a file if not already cached."""
    if dest.exists() and dest.stat().st_size > 100:
        log.info(f"  Cached: {dest}")
        return True
    log.info(f"  Downloading {url} -> {dest}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = resp.read()
        dest.write_bytes(data)
        log.info(f"  Downloaded {len(data)} bytes -> {dest}")
        return True
    except Exception as e:
        log.error(f"  FAILED to download {url}: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════
# PARSERS
# ═══════════════════════════════════════════════════════════════════════

# --- Quran Arabic ---
QURAN_POS_MAP = {
    "N": "noun", "PN": "noun", "ADJ": "adjective",
    "V": "verb", "IMPV": "verb", "IV": "verb", "PV": "verb",
    "PRON": "pronoun", "DEM": "pronoun", "REL": "pronoun",
    "P": "preposition",
    "CONJ": "conjunction",
    "DET": "particle", "EMPH": "particle", "NEG": "particle",
    "INTG": "particle", "SUP": "particle", "VOC": "particle",
    "CIRC": "particle", "RES": "particle", "EXP": "particle",
    "COND": "particle", "AMD": "particle", "ANS": "particle",
    "AVR": "particle", "CERT": "particle", "INC": "particle",
    "INT": "particle", "PREV": "particle", "PRO": "particle",
    "RET": "particle", "SUR": "particle", "REM": "particle",
    "COM": "particle", "EXL": "particle", "FUT": "particle",
    "ACC": "particle",
    "INL": "particle", "T": "noun", "LOC": "noun",
    "IMPN": "noun",
}


def parse_quran(filepath):
    """Parse Quranic Arabic Corpus TSV. Returns unit_lengths, units_pos, lemma_counts."""
    log.info("  Parsing Quran...")
    # Format: LOCATION FORM TAG FEATURES
    # LOCATION = (sura:aya:word:segment)
    ayas = {}  # (sura, aya) -> list of (word_pos, form, pos, lemma)
    lemma_counts = Counter()

    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("LOCATION"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            loc = parts[0].strip("()")
            loc_parts = loc.split(":")
            if len(loc_parts) < 4:
                continue
            sura = int(loc_parts[0])
            aya = int(loc_parts[1])
            word_pos = int(loc_parts[2])

            tag = parts[2].strip()
            features = parts[3] if len(parts) > 3 else ""

            # Extract lemma from features
            lemma = None
            for feat in features.split("|"):
                if feat.startswith("LEM:"):
                    lemma = feat[4:]
                    break
            if lemma is None:
                lemma = parts[1].strip() if len(parts) > 1 else "UNK"

            # Map POS
            pos = QURAN_POS_MAP.get(tag, "other")

            key = (sura, aya)
            if key not in ayas:
                ayas[key] = []
            ayas[key].append((word_pos, pos, lemma))
            lemma_counts[lemma] += 1

    # Build outputs ordered by (sura, aya)
    ordered_keys = sorted(ayas.keys())
    unit_lengths = []
    units_pos = []
    for key in ordered_keys:
        words = ayas[key]
        # Deduplicate by word_pos (segments within same word)
        word_set = {}
        for wp, pos, lem in words:
            if wp not in word_set:
                word_set[wp] = (pos, lem)
        unit_lengths.append(len(word_set))
        units_pos.append([pos for pos, _ in word_set.values()])

    log.info(f"  Quran: {len(ordered_keys)} ayas, {sum(unit_lengths)} words, "
             f"{len(lemma_counts)} unique lemmas")
    return np.array(unit_lengths, dtype=float), units_pos, lemma_counts


# --- AGDT Greek (Homer, Herodotus) ---
AGDT_POS_MAP = {
    "n": "noun", "v": "verb", "a": "adjective", "d": "adverb",
    "l": "particle",  # article
    "g": "particle",  # particle
    "c": "conjunction", "r": "preposition", "p": "pronoun",
    "m": "noun",  # numeral -> noun
    "i": "particle",  # interjection
    "u": None,  # punctuation, skip
    "x": "other",
    "-": None,
}


def parse_agdt_xml(filepath, corpus_name):
    """Parse AGDT treebank XML. Returns unit_lengths, units_pos, lemma_counts."""
    log.info(f"  Parsing AGDT XML: {corpus_name}...")
    tree = ET.parse(filepath)
    root = tree.getroot()

    unit_lengths = []
    units_pos = []
    lemma_counts = Counter()

    # Handle different XML namespaces
    sentences = root.findall(".//sentence")
    if not sentences:
        # Try with namespace
        for child in root:
            if "sentence" in child.tag.lower() or child.tag == "body":
                sentences = child.findall("sentence")
                break
    if not sentences:
        sentences = list(root.iter())
        sentences = [s for s in sentences if s.tag == "sentence"]

    for sent in sentences:
        words = sent.findall("word")
        if not words:
            continue
        pos_list = []
        n_words = 0
        for w in words:
            postag = w.get("postag", "")
            lemma = w.get("lemma", "")
            form = w.get("form", "")

            if not postag or not form:
                continue

            # First character of postag is the POS
            p = postag[0].lower() if postag else "x"
            mapped = AGDT_POS_MAP.get(p, "other")

            if mapped is None:  # punctuation
                continue

            pos_list.append(mapped)
            n_words += 1
            if lemma:
                lemma_counts[lemma] += 1

        if n_words > 0:
            unit_lengths.append(n_words)
            units_pos.append(pos_list)

    log.info(f"  {corpus_name}: {len(unit_lengths)} sentences, {sum(unit_lengths)} words, "
             f"{len(lemma_counts)} unique lemmas")
    return np.array(unit_lengths, dtype=float), units_pos, lemma_counts


# --- Mishnah Hebrew ---
HEBREW_PREFIX_POS = {
    "ב": "preposition",
    "ה": "particle",
    "ו": "conjunction",
    "ל": "preposition",
    "מ": "preposition",
    "כ": "preposition",
    "ש": "conjunction",
}


def parse_mishnah_texts(mishnah_dir):
    """
    Parse Mishnah plain text files.
    Returns unit_lengths for structural metrics.
    Also attempts Hebrew prefix heuristic for POS approximation.
    """
    log.info("  Parsing Mishnah texts...")
    all_units = []
    units_pos_heuristic = []
    word_counts = Counter()

    txt_files = sorted(mishnah_dir.rglob("*.txt"))
    if not txt_files:
        # Try alternate structure
        txt_files = sorted(mishnah_dir.rglob("*"))
        txt_files = [f for f in txt_files if f.is_file() and f.suffix in (".txt", "")]

    log.info(f"  Found {len(txt_files)} Mishnah text files")

    for tf in txt_files:
        try:
            text = tf.read_text(encoding="utf-8")
        except Exception:
            continue

        # Split into mishnaiot (paragraphs or numbered sections)
        # Sefaria format: each line is a segment
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        for line in lines:
            # Remove HTML tags if any
            line = re.sub(r"<[^>]+>", "", line)
            line = line.strip()
            if not line:
                continue

            words = line.split()
            if len(words) < 2:
                continue

            all_units.append(len(words))

            # Hebrew prefix heuristic for POS approximation
            pos_list = []
            for w in words:
                word_counts[w] += 1
                if not w:
                    continue
                first_char = w[0] if w else ""
                # Simple heuristic based on first character
                if first_char in HEBREW_PREFIX_POS and len(w) > 2:
                    pos_list.append(HEBREW_PREFIX_POS[first_char])
                elif w in ("את", "של", "על", "אל", "מן", "עם", "בין", "תחת", "אחר", "לפני", "אחרי"):
                    pos_list.append("preposition")
                elif w in ("הוא", "היא", "הם", "הן", "אני", "אנחנו", "אתה", "את", "אתם", "אתן",
                           "זה", "זו", "זאת", "אלה", "אלו"):
                    pos_list.append("pronoun")
                elif w in ("לא", "אין", "אם", "כי", "גם", "רק", "עוד", "כל", "כן", "אף"):
                    pos_list.append("particle")
                elif w in ("או", "אבל", "אלא", "אך"):
                    pos_list.append("conjunction")
                else:
                    # Default: noun (most common in Mishnah legal text)
                    pos_list.append("noun")

            if pos_list:
                units_pos_heuristic.append(pos_list)

    if not all_units:
        log.warning("  No Mishnah units found!")
        return None, None, None, "none"

    log.info(f"  Mishnah: {len(all_units)} units, {sum(all_units)} words")
    morph_level = "heuristic_prefix"
    return (
        np.array(all_units, dtype=float),
        units_pos_heuristic,
        word_counts,
        morph_level,
    )


def download_mishnah_sefaria():
    """Download Mishnah texts from Sefaria API."""
    mishnah_dir = CORPUS_DIR / "mishnah"
    mishnah_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    existing = list(mishnah_dir.rglob("*.txt"))
    if len(existing) > 50:
        log.info(f"  Mishnah already downloaded ({len(existing)} files)")
        return mishnah_dir

    # Mishnah tractates
    tractates = [
        "Berakhot", "Peah", "Demai", "Kilayim", "Sheviit",
        "Terumot", "Maasrot", "Maaser_Sheni", "Challah", "Orlah", "Bikkurim",
        "Shabbat", "Eruvin", "Pesachim", "Shekalim", "Yoma",
        "Sukkah", "Beitzah", "Rosh_Hashanah", "Taanit", "Megillah",
        "Moed_Katan", "Chagigah",
        "Yevamot", "Ketubot", "Nedarim", "Nazir", "Sotah",
        "Gittin", "Kiddushin",
        "Bava_Kamma", "Bava_Metzia", "Bava_Batra", "Sanhedrin",
        "Makkot", "Shevuot", "Eduyot", "Avodah_Zarah",
        "Avot", "Horayot",
        "Zevachim", "Menachot", "Chullin", "Bekhorot",
        "Arakhin", "Temurah", "Keritot", "Meilah",
        "Tamid", "Middot", "Kinnim",
        "Kelim", "Oholot", "Negaim", "Parah", "Tahorot",
        "Mikvaot", "Niddah", "Makhshirin", "Zavim",
        "Tevul_Yom", "Yadayim", "Oktzin",
    ]

    log.info(f"  Downloading {len(tractates)} Mishnah tractates from Sefaria API...")
    downloaded = 0
    for tract in tractates:
        dest = mishnah_dir / f"{tract}.txt"
        if dest.exists() and dest.stat().st_size > 10:
            downloaded += 1
            continue

        # Sefaria API
        api_name = f"Mishnah_{tract}" if not tract.startswith("Mishnah") else tract
        url = f"https://www.sefaria.org/api/texts/{api_name}?lang=he"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            # Extract Hebrew text
            he_text = data.get("he", [])
            lines = []
            if isinstance(he_text, list):
                for chapter in he_text:
                    if isinstance(chapter, list):
                        for mishnah in chapter:
                            if isinstance(mishnah, str):
                                clean = re.sub(r"<[^>]+>", "", mishnah).strip()
                                if clean:
                                    lines.append(clean)
                    elif isinstance(chapter, str):
                        clean = re.sub(r"<[^>]+>", "", chapter).strip()
                        if clean:
                            lines.append(clean)

            if lines:
                dest.write_text("\n".join(lines), encoding="utf-8")
                downloaded += 1
                log.info(f"    {tract}: {len(lines)} mishnaiot")
            else:
                log.warning(f"    {tract}: no text found")

        except Exception as e:
            log.warning(f"    {tract}: API error: {e}")

        # Rate limiting
        time.sleep(0.5)

    log.info(f"  Mishnah: {downloaded}/{len(tractates)} tractates downloaded")
    return mishnah_dir


# ═══════════════════════════════════════════════════════════════════════
# LOAD OUR AT/NT RESULTS
# ═══════════════════════════════════════════════════════════════════════

def load_our_results():
    """Load existing AT/NT results from previous phases."""
    results = []

    # Fractal results
    try:
        with open(BASE / "results" / "deep_fractal" / "fractal_by_corpus.json") as f:
            fractal = json.load(f)
    except Exception:
        fractal = {}

    # MPS permutation test
    try:
        with open(BASE / "results" / "mps_compression" / "permutation_test_chi.json") as f:
            mps_perm = json.load(f)
    except Exception:
        mps_perm = {}

    # Von Neumann
    try:
        with open(BASE / "results" / "von_neumann" / "entropy_comparison.json") as f:
            vn = json.load(f)
    except Exception:
        vn = {}

    # MPS bond dimension
    try:
        with open(BASE / "results" / "mps" / "bond_dimension.json") as f:
            mps_bond = json.load(f)
    except Exception:
        mps_bond = {}

    # Zipf
    try:
        with open(BASE / "results" / "deep_zipf_semantic" / "zipf_summary.json") as f:
            zipf_data = json.load(f)
    except Exception:
        zipf_data = {}

    for corpus_name, lang, ctype in [("AT (Hebreo)", "heb", "religioso"), ("NT (Griego)", "grc", "religioso")]:
        key = "OT" if "AT" in corpus_name else "NT"
        frac = fractal.get(key, {})
        perm = mps_perm.get(key, {})
        vn_data = vn.get("ot_vs_nt", {})
        bond = mps_bond.get(key, {})

        r = {
            "corpus": corpus_name,
            "lang": lang,
            "type": ctype,
            "n_units": frac.get("n_verses"),
            "n_words": None,  # filled later if available
            "hurst_H": frac.get("hurst", {}).get("H"),
            "dfa_alpha": frac.get("dfa", {}).get("alpha"),
            "box_counting_Df": frac.get("box_counting", {}).get("D_f"),
            "bond_dim_chi": bond.get("chi_990", {}).get("bond_dimension") if bond else None,
            "mps_permtest_p": perm.get("p_value"),
            "mps_significant": perm.get("significant"),
            "delta_S_mean": vn_data.get(f"{key.lower()}_mean_delta_s") if key == "OT" else vn_data.get("nt_mean_delta_s"),
            "zipf_s_lemma": None,
            "delta_s_group": "more_varied" if key == "OT" else "more_structured",
        }
        # Fix delta_S key
        if key == "OT":
            r["delta_S_mean"] = vn_data.get("ot_mean_delta_s")
        else:
            r["delta_S_mean"] = vn_data.get("nt_mean_delta_s")

        # Get word counts from corpus
        try:
            with open(BASE / "bible_unified.json") as f:
                all_words = json.load(f)
            r["n_words"] = sum(1 for w in all_words if w["corpus"] == key)
        except Exception:
            pass

        results.append(r)

    return results


# ═══════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════

def process_quran():
    """Download + parse + analyze Quran."""
    log.info("=== CORPUS 1: CORÁN ÁRABE ===")
    url = "https://raw.githubusercontent.com/cltk/arabic_morphology_quranic-corpus/master/quranic-corpus-morphology-0.4.txt"
    dest = CORPUS_DIR / "quran_morphology.txt"

    if not download_file(url, dest):
        return None

    unit_lengths, units_pos, lemma_counts = parse_quran(dest)
    result = analyze_corpus("Corán (Árabe)", unit_lengths, units_pos, lemma_counts)
    result["lang"] = "ara"
    result["type"] = "religioso"
    return result


def process_homer():
    """Download + parse + analyze Homer (Iliad + Odyssey combined)."""
    log.info("=== CORPUS 2: HOMERO GRIEGO ===")
    urls = {
        "iliad": "https://raw.githubusercontent.com/PerseusDL/treebank_data/master/v2.1/Greek/texts/tlg0012.tlg001.perseus-grc1.tb.xml",
        "odyssey": "https://raw.githubusercontent.com/PerseusDL/treebank_data/master/v2.1/Greek/texts/tlg0012.tlg002.perseus-grc1.tb.xml",
    }

    all_lengths = []
    all_pos = []
    all_lemmas = Counter()

    for name, url in urls.items():
        dest = CORPUS_DIR / f"homer_{name}.xml"
        if not download_file(url, dest):
            continue
        lengths, pos, lemmas = parse_agdt_xml(dest, f"Homer-{name}")
        all_lengths.extend(lengths.tolist())
        all_pos.extend(pos)
        all_lemmas.update(lemmas)

    if not all_lengths:
        return None

    unit_lengths = np.array(all_lengths, dtype=float)
    result = analyze_corpus("Homero (Griego)", unit_lengths, all_pos, all_lemmas)
    result["lang"] = "grc"
    result["type"] = "literario"
    return result


def process_herodotus():
    """Download + parse + analyze Herodotus."""
    log.info("=== CORPUS 3: HERÓDOTO GRIEGO ===")
    # Try multiple possible URLs
    urls_to_try = [
        "https://raw.githubusercontent.com/PerseusDL/treebank_data/master/v2.1/Greek/texts/tlg0016.tlg001.perseus-grc1.1.tb.xml",
        "https://raw.githubusercontent.com/PerseusDL/treebank_data/master/v2.1/Greek/texts/tlg0016.tlg001.perseus-grc1.tb.xml",
    ]

    dest = CORPUS_DIR / "herodotus.xml"
    downloaded = False
    for url in urls_to_try:
        if download_file(url, dest):
            downloaded = True
            break

    if not downloaded:
        log.error("  Failed to download Herodotus from all URLs")
        return None

    unit_lengths, units_pos, lemma_counts = parse_agdt_xml(dest, "Herodotus")

    if len(unit_lengths) == 0:
        log.error("  No sentences parsed from Herodotus")
        return None

    result = analyze_corpus("Heródoto (Griego)", unit_lengths, units_pos, lemma_counts)
    result["lang"] = "grc"
    result["type"] = "historico"
    return result


def process_mishnah():
    """Download + parse + analyze Mishnah."""
    log.info("=== CORPUS 4: MISHNAH HEBREA ===")

    mishnah_dir = download_mishnah_sefaria()
    unit_lengths, units_pos, word_counts, morph_level = parse_mishnah_texts(mishnah_dir)

    if unit_lengths is None:
        return None

    log.info(f"  Mishnah morphology level: {morph_level}")

    # Use heuristic POS if available
    if morph_level == "heuristic_prefix" and units_pos:
        result = analyze_corpus(
            "Mishnah (Hebreo)", unit_lengths, units_pos, word_counts
        )
        result["morph_level"] = morph_level
        result["morph_note"] = (
            "APROXIMACIÓN: POS obtenida mediante heurística de prefijos hebreos. "
            "NO es morfología verificada. Los valores de ΔS y distribución POS "
            "son indicativos, no definitivos."
        )
    else:
        result = analyze_corpus("Mishnah (Hebreo)", unit_lengths, None, word_counts)
        result["morph_level"] = "none"
        result["morph_note"] = "Sin morfología. Solo métricas estructurales."

    result["lang"] = "heb"
    result["type"] = "religioso"
    return result


def generate_verdict(comparison_results):
    """Generate the verdict comparing all corpora."""
    log.info("=== Generando veredicto ===")

    # Find AT and NT
    at = next((r for r in comparison_results if "AT" in r.get("corpus", "")), None)
    nt = next((r for r in comparison_results if "NT" in r.get("corpus", "")), None)
    externals = [r for r in comparison_results if r not in [at, nt]]

    verdict = {
        "question": "¿Es el AT estadísticamente único respecto a los corpus externos?",
        "corpora_analyzed": [r["corpus"] for r in comparison_results],
        "n_external_corpora": len(externals),
    }

    # Compare each external corpus to AT and NT
    comparisons = []
    for ext in externals:
        comp = {"corpus": ext["corpus"], "lang": ext.get("lang"), "type": ext.get("type")}

        # Metric-by-metric comparison
        metrics_like_at = 0
        metrics_like_nt = 0
        metrics_total = 0
        details = {}

        if at and ext.get("hurst_H") is not None and at.get("hurst_H") is not None:
            # AT has higher Hurst (more persistent)
            at_h = at["hurst_H"]
            nt_h = nt["hurst_H"] if nt else 0.5
            ext_h = ext["hurst_H"]
            diff_at = abs(ext_h - at_h)
            diff_nt = abs(ext_h - nt_h)
            closer_to = "AT" if diff_at < diff_nt else "NT"
            if closer_to == "AT":
                metrics_like_at += 1
            else:
                metrics_like_nt += 1
            metrics_total += 1
            details["hurst_H"] = {
                "ext": ext_h, "AT": at_h, "NT": nt_h,
                "closer_to": closer_to,
            }

        if at and ext.get("dfa_alpha") is not None and at.get("dfa_alpha") is not None:
            at_a = at["dfa_alpha"]
            nt_a = nt["dfa_alpha"] if nt else 0.5
            ext_a = ext["dfa_alpha"]
            diff_at = abs(ext_a - at_a)
            diff_nt = abs(ext_a - nt_a)
            closer_to = "AT" if diff_at < diff_nt else "NT"
            if closer_to == "AT":
                metrics_like_at += 1
            else:
                metrics_like_nt += 1
            metrics_total += 1
            details["dfa_alpha"] = {
                "ext": ext_a, "AT": at_a, "NT": nt_a,
                "closer_to": closer_to,
            }

        if at and ext.get("box_counting_Df") is not None and at.get("box_counting_Df") is not None:
            at_d = at["box_counting_Df"]
            nt_d = nt["box_counting_Df"] if nt else 1.0
            ext_d = ext["box_counting_Df"]
            diff_at = abs(ext_d - at_d)
            diff_nt = abs(ext_d - nt_d)
            closer_to = "AT" if diff_at < diff_nt else "NT"
            if closer_to == "AT":
                metrics_like_at += 1
            else:
                metrics_like_nt += 1
            metrics_total += 1
            details["box_counting_Df"] = {
                "ext": ext_d, "AT": at_d, "NT": nt_d,
                "closer_to": closer_to,
            }

        if at and ext.get("mps_permtest_p") is not None and at.get("mps_permtest_p") is not None:
            # AT has significant MPS compressibility (p≈0), NT does not (p=0.223)
            ext_sig = ext.get("mps_significant", False)
            at_sig = at.get("mps_significant", True)
            nt_sig = nt.get("mps_significant", False) if nt else False
            if ext_sig == at_sig:
                closer_to = "AT"
                metrics_like_at += 1
            elif ext_sig == nt_sig:
                closer_to = "NT"
                metrics_like_nt += 1
            else:
                closer_to = "neither"
            metrics_total += 1
            details["mps_significant"] = {
                "ext": ext_sig, "AT": at_sig, "NT": nt_sig,
                "closer_to": closer_to,
            }

        if (ext.get("delta_S_mean") is not None and at and at.get("delta_S_mean") is not None):
            at_ds = at["delta_S_mean"]
            nt_ds = nt["delta_S_mean"] if nt else -1.1
            ext_ds = ext["delta_S_mean"]
            diff_at = abs(ext_ds - at_ds)
            diff_nt = abs(ext_ds - nt_ds)
            closer_to = "AT" if diff_at < diff_nt else "NT"
            if closer_to == "AT":
                metrics_like_at += 1
            else:
                metrics_like_nt += 1
            metrics_total += 1
            details["delta_S"] = {
                "ext": ext_ds, "AT": at_ds, "NT": nt_ds,
                "closer_to": closer_to,
            }

        comp["metrics_like_AT"] = metrics_like_at
        comp["metrics_like_NT"] = metrics_like_nt
        comp["metrics_total"] = metrics_total
        comp["overall_profile"] = (
            "AT-like" if metrics_like_at > metrics_like_nt
            else "NT-like" if metrics_like_nt > metrics_like_at
            else "mixed"
        )
        comp["details"] = details
        comparisons.append(comp)

    verdict["comparisons"] = comparisons

    # Hypothesis testing
    hypotheses = {}

    # H1: Language hypothesis (semitic languages behave differently)
    semitic = [c for c in comparisons if c.get("lang") in ("heb", "ara")]
    non_semitic = [c for c in comparisons if c.get("lang") not in ("heb", "ara")]
    semitic_at_like = sum(1 for c in semitic if c["overall_profile"] == "AT-like")
    non_semitic_at_like = sum(1 for c in non_semitic if c["overall_profile"] == "AT-like")

    hypotheses["H1_language"] = {
        "hypothesis": "Las propiedades del AT se deben a la lengua (hebreo/semítico)",
        "semitic_corpora": [c["corpus"] for c in semitic],
        "semitic_AT_like": semitic_at_like,
        "non_semitic_AT_like": non_semitic_at_like,
        "supported": semitic_at_like > 0 and non_semitic_at_like == 0,
        "evidence": (
            f"{semitic_at_like}/{len(semitic)} corpus semíticos son AT-like, "
            f"{non_semitic_at_like}/{len(non_semitic)} no-semíticos son AT-like."
        ),
    }

    # H2: Composition hypothesis (multi-author/oral texts)
    oral_multi = [c for c in comparisons if c.get("type") in ("literario",)]  # Homer
    single_author = [c for c in comparisons if c.get("type") in ("historico",)]  # Herodotus
    oral_at_like = sum(1 for c in oral_multi if c["overall_profile"] == "AT-like")
    single_at_like = sum(1 for c in single_author if c["overall_profile"] == "AT-like")

    hypotheses["H2_composition"] = {
        "hypothesis": "Las propiedades del AT se deben a composición multi-autor/oral",
        "oral_multi_author": [c["corpus"] for c in oral_multi],
        "oral_AT_like": oral_at_like,
        "single_author_AT_like": single_at_like,
        "supported": oral_at_like > 0 and single_at_like == 0,
        "evidence": (
            f"{oral_at_like}/{len(oral_multi)} corpus orales/multi-autor son AT-like, "
            f"{single_at_like}/{len(single_author)} de autor único son AT-like."
        ),
    }

    # H3: Communicative function (religious texts)
    religious = [c for c in comparisons if c.get("type") == "religioso"]
    non_religious = [c for c in comparisons if c.get("type") != "religioso"]
    religious_at_like = sum(1 for c in religious if c["overall_profile"] == "AT-like")
    non_religious_at_like = sum(1 for c in non_religious if c["overall_profile"] == "AT-like")

    hypotheses["H3_communicative"] = {
        "hypothesis": "Las propiedades del AT se deben a su función comunicativa (religioso/ritual)",
        "religious_corpora": [c["corpus"] for c in religious],
        "religious_AT_like": religious_at_like,
        "non_religious_AT_like": non_religious_at_like,
        "supported": religious_at_like > 0 and non_religious_at_like == 0,
        "evidence": (
            f"{religious_at_like}/{len(religious)} corpus religiosos son AT-like, "
            f"{non_religious_at_like}/{len(non_religious)} no-religiosos son AT-like."
        ),
    }

    verdict["hypotheses"] = hypotheses

    # Overall conclusion
    any_at_like = any(c["overall_profile"] == "AT-like" for c in comparisons)
    all_at_like = all(c["overall_profile"] == "AT-like" for c in comparisons)
    n_at_like = sum(1 for c in comparisons if c["overall_profile"] == "AT-like")

    if not any_at_like:
        verdict["conclusion"] = (
            f"NINGUNO de los {len(comparisons)} corpus externos reproduce el perfil del AT. "
            f"El AT hebreo es estadísticamente ÚNICO en las 6 métricas analizadas."
        )
        verdict["at_unique"] = True
    elif all_at_like:
        verdict["conclusion"] = (
            f"TODOS los corpus externos muestran un perfil similar al AT. "
            f"Las propiedades del AT NO son únicas — son comunes a textos antiguos extensos."
        )
        verdict["at_unique"] = False
    else:
        at_like_names = [c["corpus"] for c in comparisons if c["overall_profile"] == "AT-like"]
        verdict["conclusion"] = (
            f"{n_at_like}/{len(comparisons)} corpus externos muestran un perfil AT-like: "
            f"{', '.join(at_like_names)}. El AT comparte algunas propiedades con estos corpus "
            f"pero la combinación completa puede ser parcialmente única."
        )
        verdict["at_unique"] = "partial"

    # Determine which hypothesis is best supported
    supported = [h for h, v in hypotheses.items() if v["supported"]]
    if supported:
        verdict["best_hypothesis"] = supported
    else:
        verdict["best_hypothesis"] = ["none_clearly_supported"]
        verdict["note"] = (
            "Ninguna hipótesis individual explica completamente los datos. "
            "Las propiedades del AT pueden deberse a una combinación de factores."
        )

    return verdict


def main():
    log.info("════════════════════════════════════════════════════════")
    log.info("  FASE 5 — Corpus de Comparación: ¿Es el AT único?")
    log.info("════════════════════════════════════════════════════════")
    t0 = time.time()

    # Load our AT/NT results
    log.info("Cargando resultados AT/NT existentes...")
    our_results = load_our_results()
    log.info(f"  AT/NT: {len(our_results)} corpus cargados")

    # Process each corpus sequentially (each has internal parallelism potential)
    # Sequential is safer for memory and network
    external_results = []

    # 1. Quran
    try:
        r = process_quran()
        if r:
            external_results.append(r)
    except Exception as e:
        log.error(f"QURAN FAILED: {e}")

    # 2. Homer
    try:
        r = process_homer()
        if r:
            external_results.append(r)
    except Exception as e:
        log.error(f"HOMER FAILED: {e}")

    # 3. Herodotus
    try:
        r = process_herodotus()
        if r:
            external_results.append(r)
    except Exception as e:
        log.error(f"HERODOTUS FAILED: {e}")

    # 4. Mishnah
    try:
        r = process_mishnah()
        if r:
            external_results.append(r)
    except Exception as e:
        log.error(f"MISHNAH FAILED: {e}")

    log.info(f"\n=== {len(external_results)}/4 corpus externos procesados ===")

    # Combine all results
    all_results = our_results + external_results

    # Save comparison table
    log.info("Guardando fase5_comparison.json...")
    with open(OUT / "fase5_comparison.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # Generate verdict
    verdict = generate_verdict(all_results)

    log.info("Guardando fase5_verdict.json...")
    with open(OUT / "fase5_verdict.json", "w", encoding="utf-8") as f:
        json.dump(verdict, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t0
    log.info(f"════════════════════════════════════════════════════════")
    log.info(f"  FASE 5 COMPLETA — {elapsed:.1f}s ({elapsed/60:.1f} min)")
    log.info(f"  Corpus analizados: {len(all_results)} ({len(external_results)} externos)")
    log.info(f"  Veredicto: {verdict.get('conclusion', 'N/A')[:200]}")
    log.info(f"════════════════════════════════════════════════════════")
    print(f"[orchestrator_fase5] DONE — {elapsed:.1f}s ({len(external_results)}/4 corpus externos)")
    print(f"  Resultado: {verdict.get('conclusion', 'N/A')[:200]}")


if __name__ == "__main__":
    main()
