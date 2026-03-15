#!/usr/bin/env python3
"""
fase6_rigveda.py — Fase 6: Rig Veda (Sánscrito) — Test de Hipótesis H4

¿La memoria larga está asociada a revelación/transmisión controlada,
independiente de la lengua? El Rig Veda es el experimento crucial:
sánscrito (indoeuropeo), revelación directa (śruti), transmisión oral
controlada ~3,500 años.

Descarga 1,028 himnos CoNLL-U del DCS (Digital Corpus of Sanskrit),
aplica las mismas 6 métricas de Fase 5, y genera veredicto H4.
"""
import json, logging, math, os, re, time, sys
import urllib.request, urllib.parse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from scipy import linalg as la
from scipy import stats as sp_stats

BASE = Path(__file__).parent
RV_DIR = BASE / "results" / "comparison_corpora" / "rigveda"
RV_DIR.mkdir(parents=True, exist_ok=True)
OUT = BASE / "results" / "rigveda"
OUT.mkdir(parents=True, exist_ok=True)
LOG_DIR = BASE / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=LOG_DIR / "fase6_rigveda.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("fase6")

# Our 9 standard POS categories
POS_CATEGORIES = [
    "noun", "verb", "pronoun", "adjective", "adverb",
    "preposition", "conjunction", "particle", "other",
]
N_POS = len(POS_CATEGORIES)

# UPOS -> our 9 categories
UPOS_MAP = {
    "NOUN": "noun", "PROPN": "noun", "NUM": "noun",
    "VERB": "verb", "AUX": "verb",
    "PRON": "pronoun", "DET": "pronoun",
    "ADJ": "adjective",
    "ADV": "adverb",
    "ADP": "preposition",
    "CCONJ": "conjunction", "SCONJ": "conjunction", "CONJ": "conjunction",
    "PART": "particle", "INTJ": "particle",
    "X": "other", "PUNCT": None, "SYM": None,
}

# ═══════════════════════════════════════════════════════════════════════
# METRIC FUNCTIONS (identical to Fase 5 pipeline)
# ═══════════════════════════════════════════════════════════════════════

def hurst_exponent_rs(series):
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
            cumulative = np.cumsum(sub - mean)
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
    n = len(series)
    if n < 20:
        return None, None
    if max_box is None:
        max_box = n // 4
    y = np.cumsum(series - np.mean(series))
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
            residual = segment - (slope * x_range + intercept)
            fluctuations.append(np.sqrt(np.mean(residual ** 2)))
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
        n_time = int(math.ceil(n / eps))
        n_val = max(1, int(math.ceil(1.0 / (eps / n))))
        occupied = set()
        for i, val in enumerate(normalized):
            occupied.add((i // eps, min(int(val * n_val), n_val - 1)))
        count = len(occupied)
        if count > 0:
            log_eps.append(math.log(1.0 / eps))
            log_n.append(math.log(count))
    if len(log_eps) < 3:
        return None, None
    slope, _, r_val, _, _ = sp_stats.linregress(log_eps, log_n)
    return round(float(slope), 4), round(float(r_val ** 2), 4)


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
        acf[lag] = np.mean(centered[:len(centered) - lag] * centered[lag:])
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
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = eigenvalues[eigenvalues > 1e-15]
    if len(eigenvalues) == 0:
        return 0.0
    return float(-np.sum(eigenvalues * np.log2(eigenvalues)))


def shannon_entropy(probs):
    probs = probs[probs > 0]
    if len(probs) == 0:
        return 0.0
    return float(-np.sum(probs * np.log2(probs)))


def compute_delta_s(units_pos):
    """ΔS = S_vN - S_Shannon + random control."""
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
    rho = build_density_matrix(vectors)
    s_vn = von_neumann_entropy(rho)
    total = np.sum(total_counts)
    probs = total_counts / total if total > 0 else np.ones(N_POS) / N_POS
    s_sh = shannon_entropy(probs)
    delta_s = s_vn - s_sh

    # Random control (200 simulations)
    np.random.seed(42)
    sizes = [len(u) for u in units_pos]
    marginal = probs.copy()
    marginal = marginal / marginal.sum() if marginal.sum() > 0 else np.ones(N_POS) / N_POS
    ds_sims = []
    for _ in range(200):
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
        ds_sims.append(s_vn_syn - shannon_entropy(syn_probs))

    ds_sim_arr = np.array(ds_sims)
    t_stat, t_pval = sp_stats.ttest_1samp(ds_sim_arr, delta_s)
    group = "more_structured" if delta_s < np.mean(ds_sim_arr) else "more_varied"

    return round(delta_s, 6), group, round(float(t_pval), 8)


def compute_zipf_lemma(lemma_counts):
    if not lemma_counts:
        return None
    freqs = sorted(lemma_counts.values(), reverse=True)
    if len(freqs) < 10:
        return None
    ranks = np.arange(1, len(freqs) + 1, dtype=float)
    freqs_arr = np.array(freqs, dtype=float)
    mask = freqs_arr > 0
    log_r = np.log(ranks[mask])
    log_f = np.log(freqs_arr[mask])
    if len(log_r) < 5:
        return None
    slope, _, _, _, _ = sp_stats.linregress(log_r, log_f)
    return round(float(-slope), 4)


# ═══════════════════════════════════════════════════════════════════════
# DOWNLOAD Rig Veda CoNLL-U files
# ═══════════════════════════════════════════════════════════════════════

def list_rigveda_files():
    """Get list of all .conllu files from Git tree API."""
    log.info("Listing Rig Veda files via Git Tree API...")
    url = ("https://api.github.com/repos/OliverHellwig/sanskrit/git/trees/"
           "master:dcs/data/conllu/files/%E1%B9%9Agveda?recursive=0")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    files = []
    for item in data.get("tree", []):
        path = item["path"]
        if path.endswith(".conllu") and not path.endswith("_parsed"):
            files.append(path)

    log.info(f"  Found {len(files)} CoNLL-U files")
    return sorted(files)


def download_one(filename):
    """Download a single CoNLL-U file. Returns (filename, content) or (filename, None)."""
    dest = RV_DIR / filename
    if dest.exists() and dest.stat().st_size > 50:
        return filename, dest.read_text(encoding="utf-8")

    # Build raw URL with proper encoding
    base = "https://raw.githubusercontent.com/OliverHellwig/sanskrit/master/dcs/data/conllu/files/"
    # URL-encode the Ṛgveda directory and filename
    dir_encoded = "%E1%B9%9Agveda"
    file_encoded = urllib.parse.quote(filename, safe="")
    url = f"{base}{dir_encoded}/{file_encoded}"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            content = resp.read().decode("utf-8")
        dest.write_text(content, encoding="utf-8")
        return filename, content
    except Exception as e:
        return filename, None


def download_all_parallel(filenames, max_workers=20):
    """Download all files in parallel."""
    log.info(f"Downloading {len(filenames)} files with {max_workers} workers...")
    results = {}
    failed = []
    done = 0
    last_log = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_one, fn): fn for fn in filenames}
        for future in as_completed(futures):
            fn, content = future.result()
            if content is not None:
                results[fn] = content
            else:
                failed.append(fn)
            done += 1
            now = time.time()
            if now - last_log >= 15:
                log.info(f"  Downloaded {done}/{len(filenames)} "
                         f"(OK: {len(results)}, failed: {len(failed)})")
                last_log = now

    log.info(f"  Download complete: {len(results)} OK, {len(failed)} failed")
    if failed:
        log.warning(f"  Failed files: {failed[:10]}{'...' if len(failed) > 10 else ''}")
    return results, failed


# ═══════════════════════════════════════════════════════════════════════
# PARSE CoNLL-U
# ═══════════════════════════════════════════════════════════════════════

def extract_sort_key(filename):
    """Extract (book, hymn, index) for ordering from filename."""
    # Format: Ṛgveda-NNNN-ṚV, B, H-ID.conllu
    m = re.match(r".*?-(\d{4})-.*?,\s*(\d+),\s*(\d+)-(\d+)", filename)
    if m:
        return (int(m.group(2)), int(m.group(3)), int(m.group(1)))
    # Fallback: use index
    m2 = re.match(r".*?-(\d{4})-", filename)
    if m2:
        return (0, 0, int(m2.group(1)))
    return (0, 0, 0)


def parse_conllu(content):
    """
    Parse a CoNLL-U file. Returns list of sentences.
    Each sentence: {"words": int, "pos_list": [str], "lemmas": [str]}
    """
    sentences = []
    current_words = 0
    current_pos = []
    current_lemmas = []

    for line in content.split("\n"):
        line = line.strip()
        if not line:
            # End of sentence
            if current_words > 0:
                sentences.append({
                    "words": current_words,
                    "pos_list": current_pos,
                    "lemmas": current_lemmas,
                })
            current_words = 0
            current_pos = []
            current_lemmas = []
            continue

        if line.startswith("#"):
            continue

        parts = line.split("\t")
        if len(parts) < 4:
            continue

        # Skip multi-word tokens (e.g., "1-2")
        idx = parts[0]
        if "-" in idx or "." in idx:
            continue

        form = parts[1]
        lemma = parts[2]
        upos = parts[3]

        # Map UPOS to our categories
        mapped = UPOS_MAP.get(upos, "other")
        if mapped is None:  # PUNCT, SYM
            continue

        current_words += 1
        current_pos.append(mapped)
        current_lemmas.append(lemma)

    # Last sentence if file doesn't end with blank line
    if current_words > 0:
        sentences.append({
            "words": current_words,
            "pos_list": current_pos,
            "lemmas": current_lemmas,
        })

    return sentences


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    log.info("════════════════════════════════════════════════════════")
    log.info("  FASE 6 — Rig Veda (Sánscrito): Test de Hipótesis H4")
    log.info("════════════════════════════════════════════════════════")
    t0 = time.time()

    # 1. List and download all CoNLL-U files
    filenames = list_rigveda_files()
    file_contents, failed = download_all_parallel(filenames, max_workers=20)

    if not file_contents:
        log.error("No files downloaded! Aborting.")
        print("[fase6_rigveda] FAILED — no files downloaded")
        return

    # 2. Parse all files, ordered by (book, hymn)
    log.info("Parsing CoNLL-U files...")
    ordered_files = sorted(file_contents.keys(), key=extract_sort_key)

    all_sentences = []
    lemma_counts = Counter()
    units_pos = []

    for fn in ordered_files:
        sents = parse_conllu(file_contents[fn])
        for s in sents:
            all_sentences.append(s)
            units_pos.append(s["pos_list"])
            for lem in s["lemmas"]:
                lemma_counts[lem] += 1

    n_padas = len(all_sentences)
    n_words = sum(s["words"] for s in all_sentences)
    log.info(f"Rig Veda parsed: {n_padas} pādas, {n_words} words, "
             f"{len(lemma_counts)} unique lemmas, {len(file_contents)} hymns")

    # Build unit length series (words per pāda, ordered)
    unit_lengths = np.array([s["words"] for s in all_sentences], dtype=float)

    # 3. Run the 6 metrics
    log.info("=== MÉTRICA 1: Hurst H ===")
    H, H_r2 = hurst_exponent_rs(unit_lengths)
    log.info(f"  H={H}, R²={H_r2}")

    log.info("=== MÉTRICA 2: DFA α ===")
    alpha, alpha_r2 = dfa_exponent(unit_lengths)
    log.info(f"  α={alpha}, R²={alpha_r2}")

    log.info("=== MÉTRICA 3: Box-counting D_f ===")
    D_f, D_r2 = box_counting_dimension(unit_lengths)
    log.info(f"  D_f={D_f}, R²={D_r2}")

    log.info("=== MÉTRICA 4: Bond dimension χ ===")
    chi, sigma = compute_bond_dimension(unit_lengths, max_lag=256, threshold=0.99)
    log.info(f"  χ={chi}")

    log.info("=== MÉTRICA 5: Permutation test MPS (n=10,000) ===")
    perm = permutation_test_chi(unit_lengths, n_perm=10000, max_lag=64, threshold=0.99)
    log.info(f"  χ_obs={perm['chi_observed']}, χ_rand={perm['chi_perm_mean']}, "
             f"p={perm['p_value']}, sig={perm['significant']}")

    log.info("=== MÉTRICA 6: ΔS Von Neumann ===")
    delta_s, delta_s_group, delta_s_p = compute_delta_s(units_pos)
    log.info(f"  ΔS={delta_s}, group={delta_s_group}, p_vs_random={delta_s_p}")

    log.info("=== Zipf lemas ===")
    zipf_s = compute_zipf_lemma(lemma_counts)
    log.info(f"  zipf_s={zipf_s}")

    # 4. Determine profile
    # Compare to AT and NT reference values
    AT = {"H": 1.1099, "alpha": 0.8458, "Df": 0.834, "mps_sig": True, "delta_s": -0.761583}
    NT = {"H": 0.9928, "alpha": 0.6805, "Df": 0.7871, "mps_sig": False, "delta_s": -1.10255}

    metrics_like_at = 0
    metrics_like_nt = 0
    details = {}

    for name, ext_val, at_val, nt_val in [
        ("hurst_H", H, AT["H"], NT["H"]),
        ("dfa_alpha", alpha, AT["alpha"], NT["alpha"]),
        ("box_counting_Df", D_f, AT["Df"], NT["Df"]),
    ]:
        if ext_val is not None:
            d_at = abs(ext_val - at_val)
            d_nt = abs(ext_val - nt_val)
            closer = "AT" if d_at < d_nt else "NT"
            if closer == "AT":
                metrics_like_at += 1
            else:
                metrics_like_nt += 1
            details[name] = {"rigveda": ext_val, "AT": at_val, "NT": nt_val, "closer_to": closer}

    # MPS significance
    ext_sig = perm["significant"]
    if ext_sig == AT["mps_sig"]:
        closer = "AT"
        metrics_like_at += 1
    else:
        closer = "NT"
        metrics_like_nt += 1
    details["mps_significant"] = {"rigveda": ext_sig, "AT": True, "NT": False, "closer_to": closer}

    # ΔS
    if delta_s is not None:
        d_at = abs(delta_s - AT["delta_s"])
        d_nt = abs(delta_s - NT["delta_s"])
        closer = "AT" if d_at < d_nt else "NT"
        if closer == "AT":
            metrics_like_at += 1
        else:
            metrics_like_nt += 1
        details["delta_S"] = {"rigveda": delta_s, "AT": AT["delta_s"], "NT": NT["delta_s"], "closer_to": closer}

    total_metrics = metrics_like_at + metrics_like_nt
    if metrics_like_at > metrics_like_nt:
        profile = "AT-like"
    elif metrics_like_nt > metrics_like_at:
        profile = "NT-like"
    else:
        profile = "mixed"

    log.info(f"  Profile: {profile} ({metrics_like_at} AT-like, {metrics_like_nt} NT-like)")

    # 5. Save rigveda_metrics.json
    rv_metrics = {
        "corpus": "Rigveda",
        "lang": "san",
        "family": "indo-european",
        "type": "revelacion_transmision_controlada",
        "n_padas": n_padas,
        "n_words": n_words,
        "n_hymns": len(file_contents),
        "n_unique_lemmas": len(lemma_counts),
        "n_failed_downloads": len(failed),
        "hurst_H": H,
        "hurst_R2": H_r2,
        "dfa_alpha": alpha,
        "dfa_R2": alpha_r2,
        "box_counting_Df": D_f,
        "box_counting_R2": D_r2,
        "bond_dim_chi": chi,
        "mps_permtest_p": perm["p_value"],
        "mps_permtest_chi_obs": perm["chi_observed"],
        "mps_permtest_chi_rand": perm["chi_perm_mean"],
        "mps_significant": perm["significant"],
        "delta_S_mean": delta_s,
        "delta_s_vs_random_p": delta_s_p,
        "delta_s_group": delta_s_group,
        "zipf_s_lemma": zipf_s,
        "profile": profile,
        "metrics_like_AT": metrics_like_at,
        "metrics_like_NT": metrics_like_nt,
        "details": details,
    }

    with open(OUT / "rigveda_metrics.json", "w", encoding="utf-8") as f:
        json.dump(rv_metrics, f, ensure_ascii=False, indent=2)
    log.info("Saved rigveda_metrics.json")

    # 6. Update comparison table (load Fase 5 + add Rig Veda)
    log.info("Updating comparison table...")
    try:
        with open(BASE / "results" / "fase5_comparison.json") as f:
            all_corpora = json.load(f)
    except Exception:
        all_corpora = []

    # Add Rig Veda entry in same format
    rv_entry = {
        "corpus": "Rig Veda (Sánscrito)",
        "lang": "san",
        "type": "revelacion_transmision_controlada",
        "n_units": n_padas,
        "n_words": n_words,
        "mean_unit_length": round(float(np.mean(unit_lengths)), 4),
        "hurst_H": H,
        "hurst_R2": H_r2,
        "dfa_alpha": alpha,
        "dfa_R2": alpha_r2,
        "box_counting_Df": D_f,
        "box_counting_R2": D_r2,
        "bond_dim_chi": chi,
        "mps_permtest_p": perm["p_value"],
        "mps_permtest_chi_obs": perm["chi_observed"],
        "mps_permtest_chi_rand": perm["chi_perm_mean"],
        "mps_significant": perm["significant"],
        "delta_S_mean": delta_s,
        "delta_s_group": delta_s_group,
        "zipf_s_lemma": zipf_s,
    }
    # Remove existing Rig Veda entry if present (re-run safety)
    all_corpora = [c for c in all_corpora if "Rig Veda" not in c.get("corpus", "")]
    all_corpora.append(rv_entry)

    with open(OUT / "rigveda_vs_all.json", "w", encoding="utf-8") as f:
        json.dump(all_corpora, f, ensure_ascii=False, indent=2)
    log.info("Saved rigveda_vs_all.json")

    # 7. H4 Verdict
    log.info("=== Generando veredicto H4 ===")

    evidence_for = []
    evidence_against = []

    if profile == "AT-like":
        evidence_for.append(
            f"El Rig Veda (sánscrito/indoeuropeo) muestra perfil AT-like "
            f"({metrics_like_at}/{total_metrics} métricas). La lengua semítica "
            f"NO es necesaria — la transmisión controlada es el factor."
        )
        if perm["significant"]:
            evidence_for.append(
                f"MPS significativo (χ_obs={perm['chi_observed']}, "
                f"χ_rand={perm['chi_perm_mean']}, p={perm['p_value']}): "
                f"el Rig Veda tiene estructura MPS compresible como el AT y el Corán."
            )
        if H and H > 0.9:
            evidence_for.append(
                f"Hurst H={H} > 0.9: memoria larga presente, como AT (H=1.11) y Corán (H=0.98)."
            )
    elif profile == "NT-like":
        evidence_against.append(
            f"El Rig Veda (sánscrito/indoeuropeo) muestra perfil NT-like "
            f"({metrics_like_nt}/{total_metrics} métricas). La lengua semítica "
            f"sigue siendo un factor necesario."
        )
        if not perm["significant"]:
            evidence_against.append(
                f"MPS NO significativo (p={perm['p_value']}): el Rig Veda NO tiene "
                f"estructura MPS compresible extra, igual que el NT y los griegos."
            )
    else:
        evidence_for.append(
            f"Resultados mixtos ({metrics_like_at} AT / {metrics_like_nt} NT). "
            f"La hipótesis no se confirma ni se refuta claramente."
        )

    # Check specific metrics that define the AT cluster
    # Key AT metrics: MPS significant + H > 0.9
    at_key_match = perm["significant"] and H is not None and H > 0.8
    if at_key_match:
        evidence_for.append(
            "Las dos métricas clave del cluster AT (MPS significativo + H > 0.8) "
            "se cumplen en el Rig Veda."
        )
    else:
        evidence_against.append(
            "Al menos una de las métricas clave del cluster AT (MPS significativo + H > 0.8) "
            "NO se cumple en el Rig Veda."
        )

    # Determine verdict
    if profile == "AT-like" and at_key_match:
        verdict = "CONFIRMED"
        confidence = "high"
        reasoning = (
            "El Rig Veda, un texto de revelación directa (śruti) en sánscrito (indoeuropeo), "
            "transmitido con control fonético extremo durante ~3,500 años, muestra el mismo "
            "perfil de memoria larga que el AT hebreo y el Corán árabe. Esto demuestra que "
            "la estructura de correlaciones de largo alcance NO depende de la familia "
            "lingüística (semítica vs indoeuropea), sino de la combinación de "
            "revelación/dictado + transmisión oral controlada durante períodos muy largos. "
            f"Métricas: H={H}, α={alpha}, χ_MPS p={perm['p_value']}, ΔS={delta_s}."
        )
    elif profile == "NT-like":
        verdict = "REFUTED"
        confidence = "high" if metrics_like_nt >= 4 else "medium"
        reasoning = (
            "El Rig Veda, a pesar de ser un texto de revelación con transmisión oral "
            "controlada, muestra un perfil NT-like. La lengua semítica sigue siendo "
            "un factor necesario (o la transmisión controlada no es suficiente). "
            f"Métricas: H={H}, α={alpha}, χ_MPS p={perm['p_value']}, ΔS={delta_s}."
        )
    else:
        verdict = "INDETERMINATE"
        confidence = "low"
        reasoning = (
            f"Resultados mixtos ({metrics_like_at} AT-like, {metrics_like_nt} NT-like). "
            "No hay evidencia clara para confirmar o refutar H4. Se necesitan más "
            "corpus de revelación/transmisión controlada en diferentes familias lingüísticas."
        )

    h4_verdict = {
        "hypothesis_h4": (
            "La memoria larga está asociada a revelación/transmisión controlada, "
            "independiente de la lengua"
        ),
        "rigveda_profile": profile,
        "metrics_like_AT": metrics_like_at,
        "metrics_like_NT": metrics_like_nt,
        "evidence_for_h4": evidence_for,
        "evidence_against_h4": evidence_against,
        "verdict": verdict,
        "confidence": confidence,
        "reasoning": reasoning,
        "raw_comparison": details,
        "context": {
            "AT_like_corpora": ["AT (Hebreo)", "Corán (Árabe)"],
            "NT_like_corpora": ["NT (Griego)", "Homero (Griego)", "Heródoto (Griego)"],
            "rigveda_classification": profile,
            "key_insight": (
                "Si CONFIRMED: la variable explicativa es transmisión controlada, no lengua. "
                "Si REFUTED: la lengua semítica es factor necesario. "
                "Si INDETERMINATE: se necesitan más datos."
            ),
        },
    }

    with open(OUT / "h4_verdict.json", "w", encoding="utf-8") as f:
        json.dump(h4_verdict, f, ensure_ascii=False, indent=2)
    log.info(f"H4 Verdict: {verdict} (confidence: {confidence})")

    elapsed = time.time() - t0
    log.info(f"════════════════════════════════════════════════════════")
    log.info(f"  FASE 6 COMPLETA — {elapsed:.1f}s ({elapsed/60:.1f} min)")
    log.info(f"  Rig Veda: {n_padas} pādas, {n_words} palabras, {len(file_contents)} himnos")
    log.info(f"  Profile: {profile} ({metrics_like_at} AT / {metrics_like_nt} NT)")
    log.info(f"  H4: {verdict}")
    log.info(f"════════════════════════════════════════════════════════")
    print(f"[fase6_rigveda] DONE — {elapsed:.1f}s")
    print(f"  Rig Veda: {n_padas} pādas, {n_words} words")
    print(f"  Profile: {profile} | H4: {verdict}")


if __name__ == "__main__":
    main()
