#!/usr/bin/env python3
"""
Fase 18 — Script 1: mishnah_sefaria.py
Descargar Mishnah de Sefaria API, computar métricas con misma metodología.

La Mishnah tiene 63 tractados en 6 órdenes. Usamos la API v3 de Sefaria.
Unidad de análisis: mishnah (= versículo/segmento).
Longitud: número de palabras por mishnah.
"""

import json
import logging
import time
import urllib.request
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from scipy import stats as sp_stats
from scipy import linalg as la
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent
RESULTS_DIR = BASE / "results" / "mishnah"
LOG_DIR = BASE / "logs"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "fase18_mishnah_sefaria.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


# ── Reusable functions ────────────────────────────────────────────────

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


def autocorr_lag1(series):
    arr = np.asarray(series, dtype=float)
    if len(arr) < 10:
        return float("nan")
    return float(np.corrcoef(arr[:-1], arr[1:])[0, 1])


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


def compute_bond_dimension(series, max_lag=256, threshold=0.99):
    n = min(max_lag, len(series) // 4)
    if n < 2:
        return 1, np.array([1.0])
    mean_val = np.mean(series)
    centered = series - mean_val
    acf = np.zeros(n)
    for lag in range(n):
        if lag >= len(centered):
            break
        acf[lag] = np.mean(centered[:len(centered)-lag] * centered[lag:])
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


# ── Mishnah tractates ─────────────────────────────────────────────────

# All 63 tractates with expected chapter counts
TRACTATES = {
    # Seder Zeraim
    "Berakhot": 9, "Peah": 8, "Demai": 7, "Kilayim": 9, "Sheviit": 10,
    "Terumot": 11, "Maaserot": 5, "Maaser_Sheni": 5, "Challah": 4,
    "Orlah": 3, "Bikkurim": 4,
    # Seder Moed
    "Shabbat": 24, "Eruvin": 10, "Pesachim": 10, "Shekalim": 8,
    "Yoma": 8, "Sukkah": 5, "Beitzah": 5, "Rosh_Hashanah": 4,
    "Taanit": 4, "Megillah": 4, "Moed_Katan": 3, "Chagigah": 3,
    # Seder Nashim
    "Yevamot": 16, "Ketubot": 13, "Nedarim": 11, "Nazir": 9,
    "Sotah": 9, "Gittin": 9, "Kiddushin": 4,
    # Seder Nezikin
    "Bava_Kamma": 10, "Bava_Metzia": 10, "Bava_Batra": 10,
    "Sanhedrin": 11, "Makkot": 3, "Shevuot": 8, "Eduyot": 8,
    "Avodah_Zarah": 5, "Avot": 6, "Horayot": 3,
    # Seder Kodashim
    "Zevachim": 14, "Menachot": 13, "Chullin": 12, "Bekhorot": 9,
    "Arakhin": 9, "Temurah": 7, "Keritot": 6, "Meilah": 6,
    "Tamid": 7, "Middot": 5, "Kinnim": 3,
    # Seder Tohorot
    "Kelim": 30, "Oholot": 18, "Negaim": 14, "Parah": 12,
    "Tohorot": 10, "Mikvaot": 10, "Niddah": 10, "Makhshirin": 6,
    "Zavim": 5, "Tevul_Yom": 4, "Yadayim": 4, "Oktzin": 3,
}

SEDER_MAP = {
    "Zeraim": ["Berakhot", "Peah", "Demai", "Kilayim", "Sheviit",
               "Terumot", "Maaserot", "Maaser_Sheni", "Challah", "Orlah", "Bikkurim"],
    "Moed": ["Shabbat", "Eruvin", "Pesachim", "Shekalim", "Yoma",
             "Sukkah", "Beitzah", "Rosh_Hashanah", "Taanit", "Megillah",
             "Moed_Katan", "Chagigah"],
    "Nashim": ["Yevamot", "Ketubot", "Nedarim", "Nazir", "Sotah",
               "Gittin", "Kiddushin"],
    "Nezikin": ["Bava_Kamma", "Bava_Metzia", "Bava_Batra", "Sanhedrin",
                "Makkot", "Shevuot", "Eduyot", "Avodah_Zarah", "Avot", "Horayot"],
    "Kodashim": ["Zevachim", "Menachot", "Chullin", "Bekhorot", "Arakhin",
                 "Temurah", "Keritot", "Meilah", "Tamid", "Middot", "Kinnim"],
    "Tohorot": ["Kelim", "Oholot", "Negaim", "Parah", "Tohorot",
                "Mikvaot", "Niddah", "Makhshirin", "Zavim", "Tevul_Yom",
                "Yadayim", "Oktzin"],
}


def download_chapter(tractate, chapter):
    """Download a single chapter from Sefaria API v3."""
    ref = f"Mishnah_{tractate}.{chapter}"
    url = f"https://www.sefaria.org/api/v3/texts/{ref}?version=hebrew"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def extract_mishnayot(data):
    """Extract individual mishnayot (text segments) from API response."""
    mishnayot = []
    if not data:
        return mishnayot

    versions = data.get("versions", [])
    if not versions:
        return mishnayot

    text = versions[0].get("text", [])
    if not text:
        return mishnayot

    # text is a list of strings (each mishnah in the chapter)
    if isinstance(text, list):
        for item in text:
            if isinstance(item, str) and item.strip():
                # Hebrew text — count words by whitespace
                words = item.strip().split()
                if words:
                    mishnayot.append(len(words))
            elif isinstance(item, list):
                # Nested list (some tractates have deeper structure)
                for subitem in item:
                    if isinstance(subitem, str) and subitem.strip():
                        words = subitem.strip().split()
                        if words:
                            mishnayot.append(len(words))

    return mishnayot


# ── Main pipeline ─────────────────────────────────────────────────────

def main():
    t0 = time.time()
    log.info("=" * 70)
    log.info("FASE 18 — Script 1: mishnah_sefaria.py")
    log.info("Mishnah from Sefaria API")
    log.info("=" * 70)

    # ── Download all tractates ────────────────────────────────────────
    log.info("\n=== Section 1: Download ===")

    all_lengths = []
    tractate_stats = {}
    seder_lengths = defaultdict(list)
    last_log = time.time()
    total_chapters = sum(TRACTATES.values())
    chapters_done = 0

    for tractate, n_chapters in TRACTATES.items():
        tract_lengths = []
        for ch in range(1, n_chapters + 1):
            data = download_chapter(tractate, ch)
            mishnayot = extract_mishnayot(data)
            tract_lengths.extend(mishnayot)
            chapters_done += 1

            now = time.time()
            if now - last_log >= 30:
                log.info(f"    [{chapters_done}/{total_chapters}] "
                         f"{tractate}.{ch}, total mishnayot so far: {len(all_lengths) + len(tract_lengths)}")
                last_log = now

            # Rate limiting: small delay between requests
            time.sleep(0.1)

        if tract_lengths:
            tractate_stats[tractate] = {
                "n_mishnayot": len(tract_lengths),
                "n_words": sum(tract_lengths),
                "mean_len": round(float(np.mean(tract_lengths)), 2),
            }
            all_lengths.extend(tract_lengths)

            # Map to seder
            for seder, tracts in SEDER_MAP.items():
                if tractate in tracts:
                    seder_lengths[seder].extend(tract_lengths)
                    break

        log.info(f"  {tractate}: {len(tract_lengths)} mishnayot")

    log.info(f"\n  Total: {len(all_lengths)} mishnayot, {sum(all_lengths)} words")

    if len(all_lengths) < 50:
        log.error("  INSUFFICIENT DATA")
        with open(RESULTS_DIR / "mishnah_metrics.json", "w") as f:
            json.dump({"error": "insufficient_data",
                        "n_mishnayot": len(all_lengths)}, f, indent=2)
        print(f"[mishnah_sefaria] DONE (insufficient data) — {time.time()-t0:.1f}s")
        return

    series = np.array(all_lengths, dtype=float)

    # ── Section 2: Corpus metrics ─────────────────────────────────────
    log.info("\n=== Section 2: Corpus Metrics ===")

    H = hurst_exponent_rs(series)
    dfa = dfa_exponent(series)
    ac1 = autocorr_lag1(series)
    mean_len = float(np.mean(series))
    std_len = float(np.std(series))
    cv = std_len / mean_len if mean_len > 0 else 0
    skew = float(sp_stats.skew(series))

    # MPS bond dimension with permutation test
    chi, _ = compute_bond_dimension(series, max_lag=min(256, len(series) // 4),
                                     threshold=0.99)
    rng = np.random.default_rng(42)
    n_perm = 10000
    chi_perms = []
    last_log = time.time()
    for i in range(n_perm):
        shuffled = rng.permutation(series)
        chi_s, _ = compute_bond_dimension(shuffled, max_lag=min(64, len(series) // 4),
                                           threshold=0.99)
        chi_perms.append(chi_s)
        now = time.time()
        if now - last_log >= 30:
            log.info(f"    MPS permutation {i+1}/{n_perm}")
            last_log = now

    chi_real, _ = compute_bond_dimension(series, max_lag=min(64, len(series) // 4),
                                          threshold=0.99)
    p_mps = float(np.mean(np.array(chi_perms) <= chi_real))

    mishnah_metrics = {
        "corpus": "Mishnah (Sefaria, Hebrew)",
        "language": "Mishnaic Hebrew",
        "n_tractates": len(tractate_stats),
        "n_mishnayot": len(series),
        "n_words": int(np.sum(series)),
        "H": round(H, 4) if not np.isnan(H) else None,
        "DFA": round(dfa, 4) if not np.isnan(dfa) else None,
        "AC1": round(ac1, 4) if not np.isnan(ac1) else None,
        "mean_verse_len": round(mean_len, 2),
        "std_verse_len": round(std_len, 2),
        "CV": round(cv, 4),
        "skewness": round(skew, 4),
        "bond_dimension": chi,
        "bond_dimension_64": chi_real,
        "mps_perm_p": round(p_mps, 4),
        "tractate_stats": tractate_stats,
    }

    log.info(f"  H={H:.4f}, DFA={dfa:.4f}, AC1={ac1:.4f}")
    log.info(f"  mean={mean_len:.2f}, std={std_len:.2f}, CV={cv:.4f}, skew={skew:.4f}")
    log.info(f"  MPS: chi={chi_real}, perm_mean={np.mean(chi_perms):.1f}, p={p_mps:.4f}")

    with open(RESULTS_DIR / "mishnah_metrics.json", "w") as f:
        json.dump(mishnah_metrics, f, indent=2, ensure_ascii=False)

    # ── Section 3: Per-seder metrics ──────────────────────────────────
    log.info("\n=== Section 3: Per-Seder Metrics ===")

    seder_metrics = {}
    for seder, lens in seder_lengths.items():
        arr = np.array(lens, dtype=float)
        if len(arr) >= 20:
            seder_metrics[seder] = {
                "n_mishnayot": len(arr),
                "H": round(hurst_exponent_rs(arr), 4),
                "AC1": round(autocorr_lag1(arr), 4),
                "DFA": round(dfa_exponent(arr), 4) if len(arr) >= 50 else None,
                "mean_len": round(float(np.mean(arr)), 2),
                "CV": round(float(np.std(arr) / np.mean(arr)), 4),
            }
            log.info(f"  {seder}: H={seder_metrics[seder]['H']}, "
                     f"AC1={seder_metrics[seder]['AC1']}, n={len(arr)}")

    with open(RESULTS_DIR / "seder_metrics.json", "w") as f:
        json.dump(seder_metrics, f, indent=2, ensure_ascii=False)

    # ── Section 4: Classifier ─────────────────────────────────────────
    log.info("\n=== Section 4: Classifier ===")

    features_file = BASE / "results" / "refined_classifier" / "book_features.json"
    with open(features_file, "r") as f:
        book_features = json.load(f)

    feature_names = ["H", "DFA", "AC1", "mean_verse_len", "std_verse_len",
                     "CV", "skewness", "pos_entropy"]
    X_train = []
    y_train = []
    for book, feats in book_features.items():
        row = []
        valid = True
        for fn in feature_names:
            val = feats.get(fn)
            if val is None:
                valid = False
                break
            row.append(val)
        if valid and feats.get("testament") in ("AT", "NT"):
            X_train.append(row)
            y_train.append(0 if feats["testament"] == "AT" else 1)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_scaled, y_train)

    # Mishnah is Hebrew — no POS tagger for Mishnaic Hebrew available
    # Use median AT pos_entropy as stand-in, clearly documented
    at_pe = [f["pos_entropy"] for f in book_features.values()
             if f.get("testament") == "AT" and f.get("pos_entropy") is not None]
    pe_standin = float(np.median(at_pe)) if at_pe else 2.4

    mish_features = [
        mishnah_metrics.get("H") or 0.5,
        mishnah_metrics.get("DFA") or 0.5,
        mishnah_metrics.get("AC1") or 0.0,
        mishnah_metrics.get("mean_verse_len") or 10.0,
        mishnah_metrics.get("std_verse_len") or 5.0,
        mishnah_metrics.get("CV") or 0.4,
        mishnah_metrics.get("skewness") or 0.5,
        pe_standin,
    ]

    mish_scaled = scaler.transform([mish_features])
    pred = clf.predict(mish_scaled)[0]
    proba = clf.predict_proba(mish_scaled)[0]

    classifier_result = {
        "predicted_class": "AT" if pred == 0 else "NT",
        "P_AT": round(float(proba[0]), 4),
        "P_NT": round(float(proba[1]), 4),
        "features_used": dict(zip(feature_names, [round(f, 4) for f in mish_features])),
        "pos_entropy_note": (
            f"No POS tagger for Mishnaic Hebrew available. "
            f"Used median AT pos_entropy ({pe_standin:.4f}) as stand-in. "
            f"Classification relies primarily on H, DFA, AC1, CV, mean_verse_len."
        ),
    }

    log.info(f"  Prediction: {'AT' if pred == 0 else 'NT'}-like "
             f"(P_AT={proba[0]:.4f}, P_NT={proba[1]:.4f})")

    with open(RESULTS_DIR / "classifier_result.json", "w") as f:
        json.dump(classifier_result, f, indent=2, ensure_ascii=False)

    # ── Section 5: φ, d placement ─────────────────────────────────────
    log.info("\n=== Section 5: φ, d Placement ===")

    fitted_file = BASE / "results" / "unified_model" / "fitted_params.json"
    with open(fitted_file, "r") as f:
        fitted_params = json.load(f)

    ref_points = {}
    for corpus, data in fitted_params.items():
        p = data.get("params", {})
        t = data.get("target", {})
        ref_points[corpus] = {
            "phi": p.get("phi"), "d": p.get("d"),
            "H": t.get("H"), "AC1": t.get("AC1"),
        }

    mish_ac1 = mishnah_metrics.get("AC1") or 0.0
    mish_h = mishnah_metrics.get("H") or 0.5
    mish_d_approx = mish_h - 0.5

    distances = {}
    for corpus, pt in ref_points.items():
        ref_ac1 = pt.get("AC1") or 0
        ref_d = pt.get("d") or 0
        dist = np.sqrt((mish_ac1 - ref_ac1)**2 + (mish_d_approx - ref_d)**2)
        distances[corpus] = round(float(dist), 4)

    nearest = min(distances, key=distances.get)

    placement = {
        "AC1": round(mish_ac1, 4),
        "d_approx": round(mish_d_approx, 4),
        "H": round(mish_h, 4),
        "distances": distances,
        "nearest_corpus": nearest,
        "transmission_context": {
            "type": "delayed_control",
            "oral_free_period": "~200 BCE to ~200 CE (400 years of free oral debate)",
            "written_fixation": "~200 CE by Rabbi Yehuda ha-Nasi",
            "control_from_origin": False,
            "note": (
                "Mishnah content evolved through centuries of rabbinic debate "
                "before being codified. Unlike AT/Corán/BoD/Pali, control did NOT "
                "start from the moment of composition."
            ),
        },
    }

    log.info(f"  AC1={mish_ac1:.4f}, d_approx={mish_d_approx:.4f}")
    log.info(f"  Nearest corpus: {nearest}")

    with open(RESULTS_DIR / "phi_d_placement.json", "w") as f:
        json.dump(placement, f, indent=2, ensure_ascii=False)

    # ── Section 6: Comparison with AT legal genre ─────────────────────
    log.info("\n=== Section 6: Comparison with AT Legal ===")

    legal_books = ["Leviticus", "Numbers", "Deuteronomy"]
    legal_feats = {b: book_features[b] for b in legal_books if b in book_features}

    if legal_feats:
        legal_h = [f["H"] for f in legal_feats.values() if f.get("H") is not None]
        legal_ac1 = [f["AC1"] for f in legal_feats.values() if f.get("AC1") is not None]
        legal_cv = [f["CV"] for f in legal_feats.values() if f.get("CV") is not None]

        comparison = {
            "mishnah": {
                "H": mishnah_metrics.get("H"),
                "AC1": mishnah_metrics.get("AC1"),
                "CV": mishnah_metrics.get("CV"),
                "mean_verse_len": mishnah_metrics.get("mean_verse_len"),
            },
            "at_legal_mean": {
                "H": round(float(np.mean(legal_h)), 4),
                "AC1": round(float(np.mean(legal_ac1)), 4),
                "CV": round(float(np.mean(legal_cv)), 4),
            },
            "same_genre_different_transmission": True,
            "note": (
                "Mishnah and AT legal (Lev/Num/Deut) share the SAME genre (legal/halakhic) "
                "and the SAME language family (Hebrew). "
                "The key difference is transmission: AT legal was controlled from composition; "
                "Mishnah was freely debated for centuries before codification."
            ),
        }

        log.info(f"  Mishnah H={mishnah_metrics.get('H')}, AT legal H={np.mean(legal_h):.4f}")
        log.info(f"  Mishnah AC1={mishnah_metrics.get('AC1')}, AT legal AC1={np.mean(legal_ac1):.4f}")
    else:
        comparison = {"error": "legal book features not available"}

    with open(RESULTS_DIR / "comparison_at_legal.json", "w") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    log.info(f"\n{'=' * 70}")
    log.info(f"mishnah_sefaria.py completado en {elapsed:.1f}s")
    print(f"[mishnah_sefaria] DONE — {elapsed:.1f}s")


if __name__ == "__main__":
    main()
