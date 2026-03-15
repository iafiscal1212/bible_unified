#!/usr/bin/env python3
"""
deep_bimodal.py — Investigación 3: ¿Los dos picos de longitud de versículo (7 y 12-13)
corresponden a géneros literarios distintos?
- Clasificación de género por V/N ratio y estructura
- GMM de 2 componentes
- Mann-Whitney U test
"""
import json, logging, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from scipy import stats as sp_stats

BASE = Path(__file__).parent
CORPUS = BASE / "bible_unified.json"
OUT = BASE / "results" / "deep_bimodal"
OUT.mkdir(parents=True, exist_ok=True)
LOG = BASE / "logs"
LOG.mkdir(parents=True, exist_ok=True)

logging.basicConfig(filename=LOG / "deep_bimodal.log", level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("deep_bim")


def fit_gmm_2comp(data, n_iter=100):
    """EM algorithm for 2-component Gaussian Mixture Model."""
    data = np.array(data, dtype=float)
    n = len(data)
    if n < 10:
        return None

    # Initialize
    rng = np.random.default_rng(42)
    mu1, mu2 = np.percentile(data, 33), np.percentile(data, 67)
    sigma1 = sigma2 = np.std(data) / 2
    pi1 = 0.5

    for iteration in range(n_iter):
        # E-step
        g1 = pi1 * sp_stats.norm.pdf(data, mu1, max(sigma1, 0.01))
        g2 = (1 - pi1) * sp_stats.norm.pdf(data, mu2, max(sigma2, 0.01))
        total = g1 + g2 + 1e-300
        resp1 = g1 / total
        resp2 = g2 / total

        # M-step
        n1 = resp1.sum()
        n2 = resp2.sum()
        if n1 < 1 or n2 < 1:
            break
        mu1 = (resp1 * data).sum() / n1
        mu2 = (resp2 * data).sum() / n2
        sigma1 = np.sqrt((resp1 * (data - mu1)**2).sum() / n1)
        sigma2 = np.sqrt((resp2 * (data - mu2)**2).sum() / n2)
        pi1 = n1 / n

    # Log-likelihood
    ll = np.sum(np.log(
        pi1 * sp_stats.norm.pdf(data, mu1, max(sigma1, 0.01)) +
        (1 - pi1) * sp_stats.norm.pdf(data, mu2, max(sigma2, 0.01)) + 1e-300
    ))

    # BIC for 1 component (for comparison)
    mu_1c = np.mean(data)
    sigma_1c = np.std(data)
    ll_1c = np.sum(np.log(sp_stats.norm.pdf(data, mu_1c, max(sigma_1c, 0.01)) + 1e-300))
    bic_1c = -2 * ll_1c + 2 * np.log(n)  # 2 params (mu, sigma)
    bic_2c = -2 * ll + 5 * np.log(n)     # 5 params (mu1, sigma1, mu2, sigma2, pi1)

    if mu1 > mu2:
        mu1, mu2 = mu2, mu1
        sigma1, sigma2 = sigma2, sigma1
        pi1 = 1 - pi1

    return {
        "component_1": {"mu": round(float(mu1), 2), "sigma": round(float(sigma1), 2),
                         "weight": round(float(pi1), 4)},
        "component_2": {"mu": round(float(mu2), 2), "sigma": round(float(sigma2), 2),
                         "weight": round(float(1 - pi1), 4)},
        "bic_1_component": round(float(bic_1c), 2),
        "bic_2_component": round(float(bic_2c), 2),
        "bic_improvement": round(float(bic_1c - bic_2c), 2),
        "favors_2_components": bool(bic_2c < bic_1c),
    }


def classify_genre(book_words):
    """Clasifica género literario basado en el propio texto."""
    if not book_words:
        return "unknown"

    corpus = book_words[0]["corpus"]
    pos_counts = Counter(w["pos"] for w in book_words)
    n = len(book_words)

    verbs = pos_counts.get("verb", 0)
    nouns = pos_counts.get("noun", 0)
    vn_ratio = verbs / nouns if nouns > 0 else 0

    # Verse lengths
    verses = defaultdict(int)
    for w in book_words:
        verses[(w["chapter"], w["verse"])] += 1
    vl = list(verses.values())
    mean_vl = np.mean(vl) if vl else 0
    std_vl = np.std(vl) if vl else 0
    cv_vl = std_vl / mean_vl if mean_vl > 0 else 0

    # First word POS distribution
    first_pos = Counter()
    verse_words = defaultdict(list)
    for w in book_words:
        verse_words[(w["chapter"], w["verse"])].append(w)
    for ws in verse_words.values():
        ws_sorted = sorted(ws, key=lambda x: x["word_pos"])
        if ws_sorted:
            first_pos[ws_sorted[0]["pos"]] += 1

    verb_start_pct = first_pos.get("verb", 0) / len(verse_words) if verse_words else 0
    conj_pct = pos_counts.get("conjunction", 0) / n if n > 0 else 0

    # Genre classification (all thresholds derived from corpus statistics)
    if corpus == "NT" and vn_ratio > 0.8 and conj_pct > 0.05:
        return "epistolar"
    elif mean_vl < 10 and cv_vl > 0.4:
        return "poetic"
    elif mean_vl > 16 and cv_vl < 0.35:
        return "legal"
    elif vn_ratio > 0.5 and verb_start_pct > 0.35:
        return "narrative"
    elif mean_vl < 12:
        return "poetic"
    else:
        return "narrative"


def main():
    log.info("Cargando corpus...")
    t0 = time.time()
    with open(CORPUS, encoding="utf-8") as f:
        words = json.load(f)
    log.info(f"Corpus: {len(words)} palabras")

    # Agrupar por libro y calcular longitudes de versículo
    books = {}
    for w in words:
        books.setdefault(w["book"], []).append(w)

    # === 1. Clasificar géneros ===
    log.info("Clasificando géneros literarios...")
    genre_classification = []
    genre_verses = defaultdict(list)  # genre -> [verse_lengths]

    for book_name in sorted(books.keys(), key=lambda b: books[b][0]["book_num"]):
        bw = books[book_name]
        genre = classify_genre(bw)
        corpus = bw[0]["corpus"]
        book_num = bw[0]["book_num"]

        pos_counts = Counter(w["pos"] for w in bw)
        verbs = pos_counts.get("verb", 0)
        nouns = pos_counts.get("noun", 0)
        vn = round(verbs / nouns, 4) if nouns else 0

        verses = defaultdict(int)
        for w in bw:
            verses[(w["chapter"], w["verse"])] += 1
        vl = list(verses.values())

        genre_verses[genre].extend(vl)

        genre_classification.append({
            "book": book_name,
            "book_num": book_num,
            "corpus": corpus,
            "genre": genre,
            "vn_ratio": vn,
            "mean_verse_length": round(float(np.mean(vl)), 2),
            "std_verse_length": round(float(np.std(vl)), 2),
            "n_verses": len(vl),
        })
        log.info(f"  {book_name}: {genre} (V/N={vn}, mean_vl={np.mean(vl):.1f})")

    # === 2. GMM fit ===
    log.info("Ajustando GMM de 2 componentes...")
    all_verse_lens = []
    verse_lens_by_corpus = {"OT": [], "NT": []}
    for bw in books.values():
        corpus = bw[0]["corpus"]
        verses = defaultdict(int)
        for w in bw:
            verses[(w["chapter"], w["verse"])] += 1
        vl = list(verses.values())
        all_verse_lens.extend(vl)
        verse_lens_by_corpus[corpus].extend(vl)

    gmm_global = fit_gmm_2comp(all_verse_lens)
    gmm_ot = fit_gmm_2comp(verse_lens_by_corpus["OT"])
    gmm_nt = fit_gmm_2comp(verse_lens_by_corpus["NT"])

    gmm_by_genre = {}
    for genre, vl in genre_verses.items():
        if len(vl) >= 20:
            gmm_by_genre[genre] = fit_gmm_2comp(vl)

    gmm_result = {
        "global": gmm_global,
        "OT": gmm_ot,
        "NT": gmm_nt,
        "by_genre": gmm_by_genre,
    }

    # === 3. Length by genre ===
    log.info("Longitudes por género...")
    length_by_genre = {}
    for genre, vl in genre_verses.items():
        vl_arr = np.array(vl)
        length_by_genre[genre] = {
            "n_verses": len(vl),
            "mean": round(float(np.mean(vl_arr)), 2),
            "median": round(float(np.median(vl_arr)), 2),
            "std": round(float(np.std(vl_arr)), 2),
            "skewness": round(float(sp_stats.skew(vl_arr)), 4),
            "kurtosis": round(float(sp_stats.kurtosis(vl_arr)), 4),
            "p10": round(float(np.percentile(vl_arr, 10)), 1),
            "p90": round(float(np.percentile(vl_arr, 90)), 1),
        }

    # === 4. Mann-Whitney U between genres ===
    log.info("Mann-Whitney U entre géneros...")
    mw_tests = {}
    genre_names = sorted(genre_verses.keys())
    for i in range(len(genre_names)):
        for j in range(i + 1, len(genre_names)):
            g1, g2 = genre_names[i], genre_names[j]
            if len(genre_verses[g1]) >= 10 and len(genre_verses[g2]) >= 10:
                u, p = sp_stats.mannwhitneyu(genre_verses[g1], genre_verses[g2], alternative='two-sided')
                mw_tests[f"{g1}_vs_{g2}"] = {
                    "U": round(float(u), 2),
                    "p_value": float(f"{p:.2e}"),
                    "significant_005": bool(p < 0.05),
                    "n1": len(genre_verses[g1]),
                    "n2": len(genre_verses[g2]),
                    "median1": round(float(np.median(genre_verses[g1])), 1),
                    "median2": round(float(np.median(genre_verses[g2])), 1),
                }

    # Save
    with open(OUT / "genre_classification.json", "w") as f:
        json.dump(genre_classification, f, indent=2, ensure_ascii=False)
    with open(OUT / "gmm_fit.json", "w") as f:
        json.dump(gmm_result, f, indent=2)
    with open(OUT / "length_by_genre.json", "w") as f:
        json.dump(length_by_genre, f, indent=2)
    with open(OUT / "mannwhitney_test.json", "w") as f:
        json.dump(mw_tests, f, indent=2)

    elapsed = time.time() - t0
    log.info(f"DONE en {elapsed:.1f}s")
    print(f"[deep_bimodal] DONE — {elapsed:.1f}s")


if __name__ == "__main__":
    main()
