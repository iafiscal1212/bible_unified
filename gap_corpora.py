#!/usr/bin/env python3
"""
Fase 18 — gap_corpora.py
Download and analyze 3 corpora to fill the 20-200 year delay gap:

1. Tosefta (Sefaria) — delay ~400, confirms Mishnah pattern
2. Didache (GitHub, jtauber/apostolic-fathers) — delay ~50, IN the gap
3. Yasna/Gathas (TITUS) — delay ~0, confirms pattern

Then update data_matrix and re-run threshold robustness sweep.

Delay definition (consistent across all corpora):
  = time from origin of tradition to formal standardization/codification.
"""

import json
import logging
import re
import time
import urllib.request
import numpy as np
from pathlib import Path
from html.parser import HTMLParser
from scipy import stats as sp_stats
from scipy.linalg import toeplitz
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent
LOG_DIR = BASE / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "gap_corpora.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Reusable metric functions
# ═══════════════════════════════════════════════════════════════

def hurst_exponent_rs(series):
    series = np.asarray(series, dtype=float)
    n = len(series)
    if n < 20:
        return np.nan
    max_k = n // 2
    min_k = 10
    ns, rs = [], []
    for k in range(min_k, max_k + 1):
        nc = n // k
        if nc < 1:
            continue
        rv = []
        for i in range(nc):
            chunk = series[i * k:(i + 1) * k]
            m = np.mean(chunk)
            cum = np.cumsum(chunk - m)
            R = np.max(cum) - np.min(cum)
            S = np.std(chunk, ddof=1)
            if S > 0:
                rv.append(R / S)
        if rv:
            ns.append(k)
            rs.append(np.mean(rv))
    if len(ns) < 3:
        return np.nan
    slope, *_ = sp_stats.linregress(np.log(ns), np.log(rs))
    return round(slope, 4)


def autocorr_lag1(series):
    series = np.asarray(series, dtype=float)
    if len(series) < 3:
        return np.nan
    n = len(series)
    m = np.mean(series)
    v = np.var(series)
    if v == 0:
        return 0.0
    return round(float(np.sum((series[:-1] - m) * (series[1:] - m)) / (n * v)), 4)


def dfa_exponent(series):
    series = np.asarray(series, dtype=float)
    n = len(series)
    if n < 20:
        return np.nan
    y = np.cumsum(series - np.mean(series))
    min_box, max_box = 4, n // 4
    if max_box < min_box + 2:
        return np.nan
    box_sizes = np.unique(np.logspace(np.log10(min_box), np.log10(max_box), 20).astype(int))
    fluct, sizes = [], []
    for bs in box_sizes:
        nb = n // bs
        if nb < 1:
            continue
        f2 = []
        for i in range(nb):
            seg = y[i * bs:(i + 1) * bs]
            x = np.arange(bs)
            trend = np.polyval(np.polyfit(x, seg, 1), x)
            f2.append(np.mean((seg - trend) ** 2))
        if f2:
            fluct.append(np.sqrt(np.mean(f2)))
            sizes.append(bs)
    if len(sizes) < 3:
        return np.nan
    slope, *_ = sp_stats.linregress(np.log(sizes), np.log(fluct))
    return round(slope, 4)


def compute_bond_dimension(series, max_lag=512):
    series = np.asarray(series, dtype=float)
    n = len(series)
    L = min(max_lag, n // 3)
    if L < 10:
        return np.nan
    m, v = np.mean(series), np.var(series)
    if v == 0:
        return 1
    acf = np.array([np.mean((series[:n - k] - m) * (series[k:] - m)) / v for k in range(L)])
    T = toeplitz(acf)
    sv = np.linalg.svd(T, compute_uv=False)
    return int(np.sum(sv / sv[0] > 0.01))


def classify_corpus(features, features_file):
    with open(features_file) as f:
        book_features = json.load(f)
    # book_features may be dict {name: {...}} or list [{...}]
    if isinstance(book_features, dict):
        book_features = list(book_features.values())
    feat_names = ["H", "DFA", "AC1", "mean_verse_len", "std_verse_len",
                  "CV", "skewness", "pos_entropy"]
    X, y = [], []
    for b in book_features:
        row = [b.get(fn, 0) for fn in feat_names]
        if any(v is None for v in row):
            continue
        X.append(row)
        y.append(0 if b["testament"] == "AT" else 1)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_s, y)
    X_new = scaler.transform([list(features[fn] for fn in feat_names)])
    pred = clf.predict(X_new)[0]
    proba = clf.predict_proba(X_new)[0]
    return {
        "predicted_class": "AT" if pred == 0 else "NT",
        "P_AT": round(float(proba[0]), 4),
        "P_NT": round(float(proba[1]), 4),
        "features_used": {fn: features[fn] for fn in feat_names},
    }


def get_pos_entropy_standin(features_file, testament="AT"):
    """Get median pos_entropy from training data as stand-in."""
    with open(features_file) as f:
        bf = json.load(f)
    if isinstance(bf, dict):
        bf = list(bf.values())
    vals = [b["pos_entropy"] for b in bf
            if b["testament"] == testament and b.get("pos_entropy")]
    return float(np.median(vals)) if vals else 2.44


def compute_all_metrics(series, label):
    """Compute all metrics for a word-length series."""
    s = np.array(series, dtype=float)
    return {
        "H": hurst_exponent_rs(s),
        "DFA": dfa_exponent(s),
        "AC1": autocorr_lag1(s),
        "mean_verse_len": round(float(np.mean(s)), 2),
        "std_verse_len": round(float(np.std(s)), 2),
        "CV": round(float(np.std(s) / np.mean(s)), 4) if np.mean(s) > 0 else 0,
        "skewness": round(float(sp_stats.skew(s)), 4),
        "bond_dimension": compute_bond_dimension(s),
    }


# ═══════════════════════════════════════════════════════════════
# Section 1: Tosefta from Sefaria
# ═══════════════════════════════════════════════════════════════

TOSEFTA_TRACTATES = [
    "Berakhot", "Peah", "Demai", "Kilayim", "Sheviit",
    "Terumot", "Maaserot", "Maaser_Sheni", "Challah", "Orlah", "Bikkurim",
    "Shabbat", "Eruvin", "Pesachim", "Shekalim", "Yoma",
    "Sukkah", "Beitzah", "Rosh_Hashanah", "Taanit", "Megillah",
    "Moed_Katan", "Chagigah",
    "Yevamot", "Ketubot", "Nedarim", "Nazir", "Sotah", "Gittin", "Kiddushin",
    "Bava_Kamma", "Bava_Metzia", "Bava_Batra", "Sanhedrin", "Makkot",
    "Shevuot", "Eduyot", "Avodah_Zarah", "Horayot",
    "Zevachim", "Menachot", "Chullin", "Bekhorot", "Arakhin",
    "Temurah", "Keritot", "Meilah",
    "Kelim_Bava_Kamma", "Kelim_Bava_Metzia", "Kelim_Bava_Batra",
    "Oholot", "Negaim", "Parah", "Mikvaot", "Niddah",
    "Makhshirin", "Zavim", "Tevul_Yom", "Yadayim", "Oktzin",
]


def download_tosefta():
    log.info("=" * 60)
    log.info("SECTION 1: Tosefta from Sefaria")
    log.info("=" * 60)

    rdir = BASE / "results" / "tosefta"
    rdir.mkdir(parents=True, exist_ok=True)

    all_lengths = []
    tract_stats = {}
    n_ok = 0

    for tidx, tname in enumerate(TOSEFTA_TRACTATES):
        chapter = 1
        t_lengths = []
        fails = 0

        while fails < 2:
            url = f"https://www.sefaria.org/api/v3/texts/Tosefta_{tname}.{chapter}"
            try:
                req = urllib.request.Request(url)
                req.add_header("User-Agent", "Bible-Unified-Research/1.0")
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode("utf-8"))

                he_text = None
                for v in data.get("versions", []):
                    if v.get("language") == "he" and v.get("text"):
                        he_text = v["text"]
                        break

                if not he_text:
                    fails += 1
                    chapter += 1
                    time.sleep(0.1)
                    continue

                fails = 0
                if isinstance(he_text, list):
                    for p in he_text:
                        if isinstance(p, str) and p.strip():
                            clean = re.sub(r'<[^>]+>', '', p).strip()
                            if clean:
                                t_lengths.append(len(clean.split()))
                chapter += 1
                time.sleep(0.1)

            except urllib.error.HTTPError:
                fails += 1
                chapter += 1
                time.sleep(0.1)
            except Exception as e:
                log.warning(f"  Error Tosefta_{tname}.{chapter}: {e}")
                fails += 1
                chapter += 1
                time.sleep(0.1)

        if t_lengths:
            tract_stats[tname] = {
                "n": len(t_lengths), "words": sum(t_lengths),
                "mean": round(np.mean(t_lengths), 1),
            }
            all_lengths.extend(t_lengths)
            n_ok += 1

        if (tidx + 1) % 15 == 0:
            log.info(f"  Progress: {tidx + 1}/{len(TOSEFTA_TRACTATES)} tractates, "
                     f"{len(all_lengths)} passages so far")

    log.info(f"\n  Tosefta: {n_ok} tractates, {len(all_lengths)} passages, "
             f"{sum(all_lengths)} words")

    if len(all_lengths) < 50:
        log.warning("  Insufficient data")
        with open(rdir / "tosefta_metrics.json", "w") as f:
            json.dump({"error": "insufficient_data", "n": len(all_lengths)}, f)
        return None

    metrics = compute_all_metrics(all_lengths, "Tosefta")
    metrics.update({
        "corpus": "Tosefta (Sefaria, Hebrew)",
        "language": "Mishnaic Hebrew",
        "n_tractates": n_ok,
        "n_passages": len(all_lengths),
        "n_words": sum(all_lengths),
    })

    with open(rdir / "tosefta_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    log.info(f"  H={metrics['H']}, AC1={metrics['AC1']}, DFA={metrics['DFA']}")

    # Classify
    ff = BASE / "results" / "refined_classifier" / "book_features.json"
    if ff.exists():
        pe = get_pos_entropy_standin(ff, "AT")
        feat = {**{k: metrics[k] for k in ["H", "DFA", "AC1", "mean_verse_len",
                "std_verse_len", "CV", "skewness"]}, "pos_entropy": pe}
        cr = classify_corpus(feat, ff)
        cr["pos_entropy_note"] = "Median AT pos_entropy (Mishnaic Hebrew proxy)"
        with open(rdir / "classifier_result.json", "w") as f:
            json.dump(cr, f, indent=2, ensure_ascii=False)
        log.info(f"  Classification: {cr['predicted_class']}-like (P_AT={cr['P_AT']})")
        metrics["predicted"] = cr["predicted_class"]
        metrics["P_AT"] = cr["P_AT"]

    return metrics


# ═══════════════════════════════════════════════════════════════
# Section 2: Didache from GitHub
# ═══════════════════════════════════════════════════════════════

def download_didache():
    log.info("\n" + "=" * 60)
    log.info("SECTION 2: Didache (jtauber/apostolic-fathers)")
    log.info("=" * 60)

    rdir = BASE / "results" / "didache"
    rdir.mkdir(parents=True, exist_ok=True)

    url = ("https://raw.githubusercontent.com/jtauber/apostolic-fathers/"
           "master/texts/011-didache.txt")
    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "Bible-Unified-Research/1.0")
        with urllib.request.urlopen(req, timeout=30) as resp:
            text = resp.read().decode("utf-8")
    except Exception as e:
        log.error(f"  Download failed: {e}")
        with open(rdir / "didache_metrics.json", "w") as f:
            json.dump({"error": str(e)}, f)
        return None

    log.info(f"  Downloaded {len(text)} chars")

    # Save raw text for reference
    with open(rdir / "didache_raw.txt", "w") as f:
        f.write(text)

    # Parse: jtauber format uses "chapter.verse<tab>text"
    # or just verse markers at line starts
    lines = text.strip().split("\n")
    segments = []

    # Try tab-separated format first (jtauber standard)
    verse_tab = re.compile(r'^(\d+\.\d+)\t(.+)')
    verse_space = re.compile(r'^(\d+\.\d+)\s+(.+)')

    for line in lines:
        line = line.strip()
        if not line:
            continue
        m = verse_tab.match(line) or verse_space.match(line)
        if m:
            vtext = m.group(2).strip()
            if vtext:
                segments.append(len(vtext.split()))

    if len(segments) < 10:
        # Fallback: treat each non-empty line as segment
        log.info("  Verse markers not found, using line-based segmentation")
        segments = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith("#") and len(line.split()) >= 3:
                segments.append(len(line.split()))

    if len(segments) < 10:
        # Fallback 2: sentence split on Greek punctuation
        log.info("  Line-based insufficient, using sentence split")
        all_text = " ".join(l.strip() for l in lines if l.strip())
        sents = re.split(r'[.;·]+', all_text)
        segments = [len(s.split()) for s in sents if len(s.split()) >= 2]

    log.info(f"  Parsed {len(segments)} segments, {sum(segments)} words")

    if len(segments) < 20:
        log.error("  Insufficient data")
        with open(rdir / "didache_metrics.json", "w") as f:
            json.dump({"error": "insufficient_data", "n": len(segments)}, f)
        return None

    metrics = compute_all_metrics(segments, "Didache")
    metrics.update({
        "corpus": "Didache (Apostolic Fathers, Greek)",
        "language": "Koine Greek",
        "n_segments": len(segments),
        "n_words": sum(segments),
        "small_corpus_warning": len(segments) < 100,
    })

    with open(rdir / "didache_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    log.info(f"  H={metrics['H']}, AC1={metrics['AC1']}, DFA={metrics['DFA']}")

    ff = BASE / "results" / "refined_classifier" / "book_features.json"
    if ff.exists():
        # Greek like NT → use NT pos_entropy median
        pe = get_pos_entropy_standin(ff, "NT")
        feat = {**{k: metrics[k] for k in ["H", "DFA", "AC1", "mean_verse_len",
                "std_verse_len", "CV", "skewness"]}, "pos_entropy": pe}
        cr = classify_corpus(feat, ff)
        cr["pos_entropy_note"] = "Median NT pos_entropy (Koine Greek proxy)"
        with open(rdir / "classifier_result.json", "w") as f:
            json.dump(cr, f, indent=2, ensure_ascii=False)
        log.info(f"  Classification: {cr['predicted_class']}-like (P_AT={cr['P_AT']})")
        metrics["predicted"] = cr["predicted_class"]
        metrics["P_AT"] = cr["P_AT"]

    return metrics


# ═══════════════════════════════════════════════════════════════
# Section 3: Yasna/Gathas from TITUS
# ═══════════════════════════════════════════════════════════════

class SimpleHTMLTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip = True

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._skip = False
        if tag in ("br", "p", "div", "tr", "li"):
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)

    def get_text(self):
        return "".join(self.parts)


def download_yasna():
    log.info("\n" + "=" * 60)
    log.info("SECTION 3: Yasna/Gathas from TITUS")
    log.info("=" * 60)

    rdir = BASE / "results" / "yasna"
    rdir.mkdir(parents=True, exist_ok=True)

    all_verses = []

    # TITUS Geldner edition: Yasna 28-54 (Gathas + Yasna Haptanghaiti)
    base = ("https://titus.fkidg1.uni-frankfurt.de/texte/etcs/iran/"
            "airan/avesta/yasna/yasng/yasng{:03d}.htm")

    for ch in range(28, 55):
        url = base.format(ch)
        try:
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "Bible-Unified-Research/1.0")
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read()
                # Try different encodings
                for enc in ["utf-8", "latin-1", "iso-8859-1", "cp1252"]:
                    try:
                        html = raw.decode(enc)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    html = raw.decode("latin-1", errors="replace")

            parser = SimpleHTMLTextExtractor()
            parser.feed(html)
            text = parser.get_text()

            # TITUS format: verse numbers as "01", "02" etc. at start of lines
            # or as standalone numbers
            ch_verses = 0
            for line in text.split("\n"):
                line = line.strip()
                # Remove verse numbers, annotations
                line = re.sub(r'^\d+[.\s]*', '', line)
                line = re.sub(r'\([^)]*\)', '', line)
                line = re.sub(r'\[[^\]]*\]', '', line)
                line = re.sub(r'\{[^}]*\}', '', line)
                # Remove edition markers, manuscript sigla
                line = re.sub(r'\b(Ms\.|Pd|C1|Geldner|Insler)\b', '', line)
                words = line.split()
                # Filter: Avestan words typically have dots/special chars
                if len(words) >= 3:
                    all_verses.append(len(words))
                    ch_verses += 1

            if ch_verses > 0:
                log.info(f"  Yasna {ch}: {ch_verses} verse segments")
            time.sleep(0.2)

        except Exception as e:
            log.warning(f"  Yasna {ch} failed: {e}")
            time.sleep(0.2)

    log.info(f"\n  Yasna total: {len(all_verses)} verses, "
             f"{sum(all_verses) if all_verses else 0} words")

    if len(all_verses) < 20:
        log.warning("  Insufficient Yasna data — documenting as unavailable")
        with open(rdir / "yasna_metrics.json", "w") as f:
            json.dump({
                "error": "insufficient_data",
                "n_verses": len(all_verses),
                "note": ("TITUS HTML parsing yielded insufficient clean data. "
                         "Avestan texts require specialized parsing. "
                         "Future work: manual digitization from Geldner edition."),
            }, f, indent=2)
        return None

    metrics = compute_all_metrics(all_verses, "Yasna")
    metrics.update({
        "corpus": "Yasna/Gathas (TITUS Geldner, Avestan transliteration)",
        "language": "Avestan",
        "n_verses": len(all_verses),
        "n_words": sum(all_verses),
        "chapters": "28-54 (Gathas + Yasna Haptanghaiti)",
    })

    with open(rdir / "yasna_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    log.info(f"  H={metrics['H']}, AC1={metrics['AC1']}, DFA={metrics['DFA']}")

    ff = BASE / "results" / "refined_classifier" / "book_features.json"
    if ff.exists():
        pe = get_pos_entropy_standin(ff, "AT")
        feat = {**{k: metrics[k] for k in ["H", "DFA", "AC1", "mean_verse_len",
                "std_verse_len", "CV", "skewness"]}, "pos_entropy": pe}
        cr = classify_corpus(feat, ff)
        cr["pos_entropy_note"] = "Median AT pos_entropy (Avestan — no tagger available)"
        with open(rdir / "classifier_result.json", "w") as f:
            json.dump(cr, f, indent=2, ensure_ascii=False)
        log.info(f"  Classification: {cr['predicted_class']}-like (P_AT={cr['P_AT']})")
        metrics["predicted"] = cr["predicted_class"]
        metrics["P_AT"] = cr["P_AT"]

    return metrics


# ═══════════════════════════════════════════════════════════════
# Section 4: Update data matrix & re-run robustness
# ═══════════════════════════════════════════════════════════════

TRANSMISSION_META = {
    "Tosefta": {
        "control_delay_years": 400,
        "revelation_claim": False,
        "language_family": "Afroasiatic",
        "transmission_type": "free oral debate → codified ~300 CE (like Mishnah)",
    },
    "Didache": {
        "control_delay_years": 50,
        "revelation_claim": False,
        "language_family": "Indo-European",
        "transmission_type": "community manual, composed ~80 CE from traditions originating ~30 CE",
    },
    "Yasna": {
        "control_delay_years": 0,
        "revelation_claim": True,
        "language_family": "Indo-European",
        "transmission_type": "oral controlled by priestly class (Magi) from composition",
    },
}


def update_and_retest(new_corpora):
    log.info("\n" + "=" * 60)
    log.info("SECTION 4: Update matrix & robustness re-test")
    log.info("=" * 60)

    # Load existing matrix
    mf = BASE / "results" / "transmission_origin" / "data_matrix.json"
    with open(mf) as f:
        matrix = json.load(f)

    for name, met in new_corpora.items():
        if met is None:
            continue
        meta = TRANSMISSION_META[name]
        entry = {
            "corpus": name,
            "H": met.get("H"),
            "AC1": met.get("AC1"),
            "DFA": met.get("DFA"),
            "P_AT": met.get("P_AT"),
            "predicted": met.get("predicted"),
            "control_delay_years": meta["control_delay_years"],
            "revelation_claim": meta["revelation_claim"],
            "language_family": meta["language_family"],
            "transmission_type": meta["transmission_type"],
        }
        matrix.append(entry)
        log.info(f"  Added {name}: H={entry['H']}, pred={entry['predicted']}, "
                 f"delay={entry['control_delay_years']}")

    # Save expanded matrix
    rdir = BASE / "results" / "robustness"
    rdir.mkdir(parents=True, exist_ok=True)
    with open(rdir / "data_matrix_expanded.json", "w") as f:
        json.dump(matrix, f, indent=2, ensure_ascii=False)

    # ── Robustness sweep ──
    classified = [c for c in matrix if c.get("predicted") is not None]
    n_classified = len(classified)
    n_total = len(matrix)

    log.info(f"\n  Total corpora: {n_total}, classified: {n_classified}")
    log.info(f"  Delays: {sorted(set(c['control_delay_years'] for c in matrix))}")

    sweep = []
    for t in range(1, 501):
        ea = en = la = ln = 0
        for c in classified:
            at_like = c["predicted"] == "AT"
            if c["control_delay_years"] < t:
                if at_like: ea += 1
                else: en += 1
            else:
                if at_like: la += 1
                else: ln += 1

        total = ea + en + la + ln
        if total == 0:
            continue
        acc = (ea + ln) / total

        if (ea + en) > 0 and (la + ln) > 0:
            _, p = sp_stats.fisher_exact([[ea, en], [la, ln]])
        else:
            p = 1.0

        sweep.append({
            "threshold": t, "early_at": ea, "early_nt": en,
            "late_at": la, "late_nt": ln,
            "accuracy": round(acc, 4), "fisher_p": round(p, 6),
        })

    perfect = [r["threshold"] for r in sweep if r["accuracy"] == 1.0]
    sig = [r["threshold"] for r in sweep if r["fisher_p"] < 0.05]

    perf_range = {"min": min(perfect), "max": max(perfect),
                  "width": max(perfect) - min(perfect) + 1} if perfect else {"width": 0}
    sig_range = {"min": min(sig), "max": max(sig),
                 "width": max(sig) - min(sig) + 1} if sig else {"width": 0}

    log.info(f"  Perfect accuracy range: {perf_range}")
    log.info(f"  Significant (p<0.05) range: {sig_range}")

    # Spearman correlation (all with H)
    dh = [(c["control_delay_years"], c["H"]) for c in matrix if c.get("H") is not None]
    d_arr = np.array([x[0] for x in dh])
    h_arr = np.array([x[1] for x in dh])
    rho, p_sp = sp_stats.spearmanr(d_arr, h_arr) if len(dh) >= 3 else (np.nan, np.nan)

    # Compare with original (9-corpus) results
    original_perf_width = 280  # from first robustness test
    new_width = perf_range.get("width", 0)
    robustness_change = "IMPROVED" if new_width > original_perf_width else \
                        "MAINTAINED" if new_width >= original_perf_width * 0.8 else \
                        "DEGRADED"

    # Unique delays in data
    unique_delays = sorted(set(c["control_delay_years"] for c in matrix))
    gaps = []
    for i in range(len(unique_delays) - 1):
        g = unique_delays[i + 1] - unique_delays[i]
        if g > 50:
            gaps.append({"from": unique_delays[i], "to": unique_delays[i + 1], "width": g})

    analysis = {
        "n_corpora_total": n_total,
        "n_classified": n_classified,
        "unique_delays": unique_delays,
        "gaps_over_50": gaps,
        "original_perfect_width": original_perf_width,
        "expanded_perfect_range": perf_range,
        "expanded_significant_range": sig_range,
        "robustness_change": robustness_change,
        "spearman_delay_H": {"rho": round(rho, 4), "p": round(p_sp, 4)} if not np.isnan(rho) else None,
        "sample_thresholds": [r for r in sweep if r["threshold"] in
                              [1, 10, 20, 21, 50, 51, 100, 150, 200, 250, 300, 350, 400, 450]],
        "corpus_table": [
            {"corpus": c["corpus"], "delay": c["control_delay_years"],
             "H": c.get("H"), "predicted": c.get("predicted")}
            for c in sorted(matrix, key=lambda x: x["control_delay_years"])
        ],
    }

    # The KEY question: did the Didache (delay=50) break the pattern?
    didache_entry = next((c for c in matrix if c["corpus"] == "Didache"), None)
    if didache_entry and didache_entry.get("predicted"):
        dp = didache_entry["predicted"]
        analysis["didache_verdict"] = {
            "predicted": dp,
            "delay": 50,
            "interpretation": (
                f"Didache (delay=50) is {dp}-like. "
                + ("This means delay=50 is still 'early enough' for AT-like signatures. "
                   "Threshold is >50 years."
                   if dp == "AT" else
                   "This means delay=50 is already 'too late'. "
                   "Threshold is <50 years — very close to composition.")
            ),
        }

    with open(rdir / "robustness_expanded.json", "w") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)

    return analysis


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    log.info("=" * 70)
    log.info("GAP CORPORA — Tosefta + Didache + Yasna")
    log.info("=" * 70)

    new = {}
    new["Tosefta"] = download_tosefta()
    new["Didache"] = download_didache()
    new["Yasna"] = download_yasna()

    valid = {k: v for k, v in new.items() if v is not None}
    if valid:
        analysis = update_and_retest(valid)
    else:
        log.error("No valid corpora — cannot update matrix")
        analysis = None

    elapsed = time.time() - t0
    log.info(f"\n{'=' * 70}")
    log.info(f"COMPLETADO en {elapsed:.1f}s")
    log.info(f"{'=' * 70}")

    for name, m in new.items():
        meta = TRANSMISSION_META[name]
        if m:
            log.info(f"  {name}: H={m.get('H')}, AC1={m.get('AC1')}, "
                     f"pred={m.get('predicted')}, delay={meta['control_delay_years']}")
        else:
            log.info(f"  {name}: FAILED")

    if analysis:
        pr = analysis.get("expanded_perfect_range", {})
        log.info(f"\n  Robustness: perfect range = {pr.get('min', '?')}-{pr.get('max', '?')} "
                 f"(width={pr.get('width', 0)}, was {analysis.get('original_perfect_width')})")
        log.info(f"  Change: {analysis.get('robustness_change')}")

        dv = analysis.get("didache_verdict")
        if dv:
            log.info(f"\n  DIDACHE (delay=50): {dv['predicted']}-like")
            log.info(f"  {dv['interpretation']}")


if __name__ == "__main__":
    main()
