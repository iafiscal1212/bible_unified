#!/usr/bin/env python3
"""
Fase 12 — Script 1: Recitation Hypothesis
¿Las restricciones métricas de recitación oral producen AC(1) alto?

Tests:
1. AT: poetry vs prose AC(1)
2. Quran: syllable AC(1) vs word AC(1)
3. Homer: syllable AC(1) vs word AC(1)
4. Rig Veda: meter-based analysis (if DCS data available)

Syllable approximations documented:
- Hebrew: count Unicode vowel points (niqqud) \\u05B0-\\u05BB per word
- Arabic (Buckwalter): count vowel chars (a,i,u,A,I,U) per word
- Greek: count vowel character groups (α,ε,η,ι,ο,υ,ω + diphthongs)
"""

import json
import logging
import re
import time
import numpy as np
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).resolve().parent
RESULTS_DIR = BASE / "results" / "recitation"
LOG_DIR = BASE / "logs"
BIBLE_JSON = BASE / "bible_unified.json"
QURAN_FILE = BASE / "results" / "comparison_corpora" / "quran_morphology.txt"
HOMER_FILES = [
    BASE / "results" / "comparison_corpora" / "homer_iliad.xml",
    BASE / "results" / "comparison_corpora" / "homer_odyssey.xml",
]

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "fase12_recitation.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# AT poetry books (standard classification)
POETRY_BOOKS = {"Job", "Psalms", "Proverbs", "Ecclesiastes", "Song of Solomon",
                "Lamentations"}
# AT narrative/prose books
OT_BOOKS = {"Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy",
            "Joshua", "Judges", "Ruth", "1 Samuel", "2 Samuel",
            "1 Kings", "2 Kings", "1 Chronicles", "2 Chronicles",
            "Ezra", "Nehemiah", "Esther", "Job", "Psalms",
            "Proverbs", "Ecclesiastes", "Song of Solomon",
            "Isaiah", "Jeremiah", "Lamentations", "Ezekiel",
            "Daniel", "Hosea", "Joel", "Amos", "Obadiah",
            "Jonah", "Micah", "Nahum", "Habakkuk", "Zephaniah",
            "Haggai", "Zechariah", "Malachi"}
PROPHETIC_BOOKS = {"Isaiah", "Jeremiah", "Ezekiel", "Daniel",
                   "Hosea", "Joel", "Amos", "Obadiah", "Jonah", "Micah",
                   "Nahum", "Habakkuk", "Zephaniah", "Haggai",
                   "Zechariah", "Malachi"}


def autocorr_lag1(series):
    """AC(1) of a numeric series."""
    arr = np.asarray(series, dtype=float)
    if len(arr) < 10:
        return float("nan")
    return float(np.corrcoef(arr[:-1], arr[1:])[0, 1])


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
    from scipy import stats as sp_stats
    slope, _, r, _, _ = sp_stats.linregress(np.log(sizes), np.log(rs_values))
    return float(slope)


# ── Syllable approximations ─────────────────────────────────────────────

def count_hebrew_syllables(word_text):
    """Approximate syllable count from Hebrew text with niqqud.
    Counts Unicode vowel points (\\u05B0-\\u05BB).
    \\u05BC (dagesh) is excluded as it's not a vowel.
    Minimum 1 syllable per word."""
    vowels = len(re.findall(r'[\u05B0-\u05BB]', word_text))
    return max(1, vowels)


def count_arabic_syllables_buckwalter(form):
    """Approximate syllable count from Buckwalter transliteration.
    Vowels: a, i, u (short), A (alif), I (ya-vowel), U (waw-vowel).
    Minimum 1 syllable per word."""
    vowels = len(re.findall(r'[aiuAIU]', form))
    return max(1, vowels)


def count_greek_syllables(word_form):
    """Approximate syllable count from Greek text.
    Count vowel groups: α,ε,η,ι,ο,υ,ω and diphthongs.
    Diphthongs (αι,αυ,ει,ευ,οι,ου,υι) count as 1 syllable.
    Minimum 1 syllable per word."""
    # Normalize: lowercase
    w = word_form.lower()
    # Replace diphthongs with single marker
    for diph in ['αι', 'αυ', 'ει', 'ευ', 'οι', 'ου', 'υι', 'ηι', 'ωι']:
        w = w.replace(diph, 'V')
    # Count remaining vowels + markers
    vowels = len(re.findall(r'[αεηιουωV]', w))
    return max(1, vowels)


# ── Corpus loading ───────────────────────────────────────────────────────

def load_at_verse_data():
    """Load AT verse lengths in words and syllables, split by poetry/prose."""
    log.info("Cargando AT de bible_unified.json...")
    with open(BIBLE_JSON, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    # Group by (book, chapter, verse) → word count and syllable count
    verses = defaultdict(lambda: {"words": 0, "syllables": 0, "book": ""})

    for w in corpus:
        book = w.get("book", "")
        if book not in OT_BOOKS:
            continue
        key = (book, w.get("chapter", 0), w.get("verse", 0))
        verses[key]["words"] += 1
        verses[key]["book"] = book
        text = w.get("text", "")
        verses[key]["syllables"] += count_hebrew_syllables(text)

    # Split into poetry and prose
    poetry_word_lens = []
    poetry_syl_lens = []
    prose_word_lens = []
    prose_syl_lens = []
    prophetic_word_lens = []
    prophetic_syl_lens = []

    for key in sorted(verses.keys()):
        v = verses[key]
        book = v["book"]
        if book in POETRY_BOOKS:
            poetry_word_lens.append(v["words"])
            poetry_syl_lens.append(v["syllables"])
        elif book in PROPHETIC_BOOKS:
            prophetic_word_lens.append(v["words"])
            prophetic_syl_lens.append(v["syllables"])
        else:
            prose_word_lens.append(v["words"])
            prose_syl_lens.append(v["syllables"])

    # All AT
    all_word = [verses[k]["words"] for k in sorted(verses.keys())]
    all_syl = [verses[k]["syllables"] for k in sorted(verses.keys())]

    log.info(f"  AT total: {len(all_word)} versículos")
    log.info(f"  Poesía: {len(poetry_word_lens)}, Profetas: {len(prophetic_word_lens)}, "
             f"Prosa: {len(prose_word_lens)}")

    return {
        "all": {"words": all_word, "syllables": all_syl},
        "poetry": {"words": poetry_word_lens, "syllables": poetry_syl_lens},
        "prophetic": {"words": prophetic_word_lens, "syllables": prophetic_syl_lens},
        "prose": {"words": prose_word_lens, "syllables": prose_syl_lens},
    }


def load_quran_data():
    """Load Quran aya lengths in words and approximate syllables."""
    if not QURAN_FILE.exists():
        log.warning("Corán no encontrado")
        return None

    log.info("Cargando Corán...")
    # (sura:aya:word:part) → collect words per aya, syllables per aya
    aya_words = defaultdict(set)  # unique word positions
    aya_syllables = defaultdict(int)
    sura_lens = defaultdict(list)  # for per-sura analysis

    with open(QURAN_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("LOCATION"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            loc_match = re.match(r'\((\d+):(\d+):(\d+)(?::\d+)?\)', parts[0])
            if not loc_match:
                continue
            sura = int(loc_match.group(1))
            aya = int(loc_match.group(2))
            word_pos = int(loc_match.group(3))
            form = parts[1] if len(parts) > 1 else ""

            aya_words[(sura, aya)].add(word_pos)
            aya_syllables[(sura, aya)] += count_arabic_syllables_buckwalter(form)

    # Build series
    word_lens = []
    syl_lens = []
    for key in sorted(aya_words.keys()):
        wl = len(aya_words[key])
        sl = aya_syllables[key]
        word_lens.append(wl)
        syl_lens.append(sl)
        sura_lens[key[0]].append({"words": wl, "syllables": sl})

    log.info(f"  Corán: {len(word_lens)} aleyas, {len(sura_lens)} suras")

    # Per-sura AC(1)
    sura_ac1 = {}
    for sura, verses in sorted(sura_lens.items()):
        if len(verses) >= 10:
            w_series = [v["words"] for v in verses]
            s_series = [v["syllables"] for v in verses]
            sura_ac1[sura] = {
                "n_ayas": len(verses),
                "ac1_words": round(autocorr_lag1(w_series), 4),
                "ac1_syllables": round(autocorr_lag1(s_series), 4),
            }

    return {
        "words": word_lens,
        "syllables": syl_lens,
        "sura_ac1": sura_ac1,
    }


def load_homer_data():
    """Load Homer verse lengths in words and approximate syllables."""
    import xml.etree.ElementTree as ET

    all_word_lens = []
    all_syl_lens = []

    for hf in HOMER_FILES:
        if not hf.exists():
            continue
        log.info(f"  Cargando {hf.name}...")
        with open(hf, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        try:
            root = ET.fromstring(content)
            for sent in root.iter("sentence"):
                n_words = 0
                n_syllables = 0
                for w in sent.iter("word"):
                    form = w.get("form", "").strip()
                    if form:
                        n_words += 1
                        n_syllables += count_greek_syllables(form)
                if n_words > 0:
                    all_word_lens.append(n_words)
                    all_syl_lens.append(n_syllables)
        except ET.ParseError:
            sentences = re.findall(r'<sentence[^>]*>(.*?)</sentence>',
                                   content, re.DOTALL)
            for sent in sentences:
                words = re.findall(r'<word[^>]*form="([^"]*)"', sent)
                if words:
                    all_word_lens.append(len(words))
                    all_syl_lens.append(sum(count_greek_syllables(w) for w in words))

    log.info(f"  Homero: {len(all_word_lens)} sentencias")
    return {"words": all_word_lens, "syllables": all_syl_lens}


def load_rigveda_pada_data():
    """Try to load Rig Veda pada data from DCS CoNLL-U or metrics.
    Returns word lengths and syllable estimates per pada."""
    rv_metrics = BASE / "results" / "rigveda" / "rigveda_metrics.json"

    log.info("Intentando cargar Rig Veda...")

    # Try DCS CoNLL-U files
    dcs_dirs = [
        Path("/root/dcs/data/conllu/files"),
        BASE / "dcs_rigveda",
    ]
    conllu_files = []
    for d in dcs_dirs:
        if d.exists():
            conllu_files = sorted(d.glob("Rg*.conllu")) + sorted(d.glob("rg*.conllu"))
            if conllu_files:
                break

    if conllu_files:
        log.info(f"  Encontrados {len(conllu_files)} archivos CoNLL-U")
        # Parse CoNLL-U: each sentence = 1 pada
        word_lens = []
        syl_lens = []
        hymn_padas = defaultdict(list)

        for cf in conllu_files:
            hymn_id = cf.stem  # e.g. "Rg01001"
            with open(cf, "r", encoding="utf-8") as f:
                n_words = 0
                n_syl = 0
                for line in f:
                    line = line.strip()
                    if not line:
                        if n_words > 0:
                            word_lens.append(n_words)
                            syl_lens.append(n_syl)
                            hymn_padas[hymn_id].append({"words": n_words, "syl": n_syl})
                            n_words = 0
                            n_syl = 0
                        continue
                    if line.startswith("#"):
                        continue
                    cols = line.split("\t")
                    if len(cols) >= 2 and not cols[0].startswith("-"):
                        form = cols[1]
                        n_words += 1
                        # Sanskrit syllables: count vowels in IAST
                        syl = len(re.findall(
                            r'[aāiīuūṛṝḷeēoōAĀIĪUŪṚṜḶEĒOŌ]', form))
                        n_syl += max(1, syl)
                if n_words > 0:
                    word_lens.append(n_words)
                    syl_lens.append(n_syl)
                    hymn_padas[hymn_id].append({"words": n_words, "syl": n_syl})

        log.info(f"  Rig Veda: {len(word_lens)} pādas, {len(hymn_padas)} himnos")
        return {
            "words": word_lens,
            "syllables": syl_lens,
            "hymn_padas": dict(hymn_padas),
            "source": "DCS_CoNLL-U",
        }

    # Fallback: use metrics to generate calibrated data
    if rv_metrics.exists():
        log.info("  DCS CoNLL-U no disponible — usando métricas para análisis limitado")
        with open(rv_metrics, "r") as f:
            metrics = json.load(f)
        n_padas = metrics.get("n_padas", 21253)
        n_words = metrics.get("n_words", 169972)
        mean_words_per_pada = n_words / n_padas  # ~8.0

        # Known Rig Veda meters: Gayatri=8 syl, Tristubh=11, Jagati=12
        # ~40% Tristubh, ~25% Gayatri, ~10% Jagati, ~25% mixed
        # Generate calibrated series
        rng = np.random.default_rng(42)
        meter_syl = {
            "tristubh": 11, "gayatri": 8, "jagati": 12, "anustubh": 8,
        }
        # Assign meters to groups of padas (hymn-level consistency)
        n_hymns = metrics.get("n_hymns", 1028)
        padas_per_hymn = n_padas // n_hymns  # ~20
        word_lens = []
        syl_lens = []
        hymn_padas = {}

        for h in range(n_hymns):
            meter = rng.choice(["tristubh", "gayatri", "jagati", "anustubh"],
                               p=[0.40, 0.25, 0.10, 0.25])
            target_syl = meter_syl[meter]
            # Words per pada varies with meter
            mean_w = target_syl / 1.5  # ~1.5 syllables per word in Sanskrit
            n_p = rng.integers(3, 8) * 4 if meter != "gayatri" else rng.integers(3, 8) * 3
            hp = []
            for _ in range(n_p):
                syl = max(1, int(rng.normal(target_syl, 1.5)))
                words = max(1, int(rng.normal(mean_w, 1.5)))
                word_lens.append(words)
                syl_lens.append(syl)
                hp.append({"words": words, "syl": syl})
            hymn_padas[f"hymn_{h:04d}"] = hp

        log.info(f"  Rig Veda (calibrado): {len(word_lens)} pādas, {n_hymns} himnos")
        return {
            "words": word_lens,
            "syllables": syl_lens,
            "hymn_padas": hymn_padas,
            "source": "calibrated_synthetic",
            "note": ("Datos sintéticos calibrados con metros védicos conocidos. "
                     "No es texto real — solo estructura métrica."),
        }

    log.warning("  Rig Veda no disponible")
    return None


# ── Analysis ─────────────────────────────────────────────────────────────

def analyze_at_poetry_prose(at_data):
    """Compare AC(1) between poetry, prophetic, and prose books."""
    log.info("\n── AT: Poesía vs Profetas vs Prosa ──")

    results = {}
    for genre, data in at_data.items():
        if not data["words"] or len(data["words"]) < 20:
            continue
        ac1_w = autocorr_lag1(data["words"])
        ac1_s = autocorr_lag1(data["syllables"])
        h_w = hurst_exponent_rs(data["words"])
        h_s = hurst_exponent_rs(data["syllables"])
        results[genre] = {
            "n_verses": len(data["words"]),
            "ac1_words": round(ac1_w, 4),
            "ac1_syllables": round(ac1_s, 4),
            "ac1_syl_vs_word_ratio": round(ac1_s / ac1_w, 4) if ac1_w > 0.01 else None,
            "H_words": round(h_w, 4) if not np.isnan(h_w) else None,
            "H_syllables": round(h_s, 4) if not np.isnan(h_s) else None,
            "mean_words_per_verse": round(float(np.mean(data["words"])), 2),
            "mean_syl_per_verse": round(float(np.mean(data["syllables"])), 2),
            "syl_per_word": round(float(np.mean(data["syllables"])) /
                                  float(np.mean(data["words"])), 3),
        }
        log.info(f"  {genre}: AC1_w={ac1_w:.4f}, AC1_s={ac1_s:.4f}, "
                 f"H_w={h_w:.4f}, H_s={h_s:.4f}")

    # Conclusion
    poetry_ac1 = results.get("poetry", {}).get("ac1_words")
    prose_ac1 = results.get("prose", {}).get("ac1_words")
    prophetic_ac1 = results.get("prophetic", {}).get("ac1_words")

    if poetry_ac1 is not None and prose_ac1 is not None:
        results["conclusion"] = {
            "poetry_higher_than_prose": bool(poetry_ac1 > prose_ac1),
            "prophetic_higher_than_prose": bool(prophetic_ac1 > prose_ac1)
                if prophetic_ac1 is not None else None,
            "metric_constraint_contributes": bool(poetry_ac1 > prose_ac1 * 1.2),
            "note": ("Si poesía AC(1) > prosa AC(1), las restricciones métricas "
                     "contribuyen al AC(1) alto del AT")
        }

    return results


def analyze_quran(quran_data):
    """Analyze Quran AC(1) in words vs syllables, per-sura patterns."""
    log.info("\n── Corán: Palabras vs Sílabas ──")

    if quran_data is None:
        return {"error": "no data"}

    ac1_w = autocorr_lag1(quran_data["words"])
    ac1_s = autocorr_lag1(quran_data["syllables"])
    h_w = hurst_exponent_rs(quran_data["words"])
    h_s = hurst_exponent_rs(quran_data["syllables"])

    log.info(f"  Global: AC1_w={ac1_w:.4f}, AC1_s={ac1_s:.4f}, "
             f"H_w={h_w:.4f}, H_s={h_s:.4f}")

    # Per-sura AC(1) statistics
    sura_ac1 = quran_data.get("sura_ac1", {})
    if sura_ac1:
        ac1_word_vals = [v["ac1_words"] for v in sura_ac1.values()
                         if not np.isnan(v["ac1_words"])]
        ac1_syl_vals = [v["ac1_syllables"] for v in sura_ac1.values()
                        if not np.isnan(v["ac1_syllables"])]

        # Classify suras by rhythmic intensity (high AC1 = more rhythmic)
        median_ac1 = np.median(ac1_word_vals) if ac1_word_vals else 0
        rhythmic = {k: v for k, v in sura_ac1.items()
                    if v["ac1_words"] > median_ac1}
        non_rhythmic = {k: v for k, v in sura_ac1.items()
                        if v["ac1_words"] <= median_ac1}

        result = {
            "global": {
                "ac1_words": round(ac1_w, 4),
                "ac1_syllables": round(ac1_s, 4),
                "H_words": round(h_w, 4) if not np.isnan(h_w) else None,
                "H_syllables": round(h_s, 4) if not np.isnan(h_s) else None,
            },
            "per_sura": {
                "n_suras_analyzed": len(sura_ac1),
                "ac1_words_mean": round(float(np.mean(ac1_word_vals)), 4)
                    if ac1_word_vals else None,
                "ac1_syllables_mean": round(float(np.mean(ac1_syl_vals)), 4)
                    if ac1_syl_vals else None,
                "ac1_words_std": round(float(np.std(ac1_word_vals)), 4)
                    if ac1_word_vals else None,
            },
            "rhythmic_vs_non": {
                "n_rhythmic": len(rhythmic),
                "n_non_rhythmic": len(non_rhythmic),
                "rhythmic_mean_ac1_w": round(float(np.mean(
                    [v["ac1_words"] for v in rhythmic.values()])), 4)
                    if rhythmic else None,
                "non_rhythmic_mean_ac1_w": round(float(np.mean(
                    [v["ac1_words"] for v in non_rhythmic.values()])), 4)
                    if non_rhythmic else None,
            },
        }
    else:
        result = {
            "global": {
                "ac1_words": round(ac1_w, 4),
                "ac1_syllables": round(ac1_s, 4),
            }
        }

    return result


def analyze_homer(homer_data):
    """Analyze Homer AC(1) in words vs syllables."""
    log.info("\n── Homero: Palabras vs Sílabas ──")

    if not homer_data or not homer_data["words"]:
        return {"error": "no data"}

    ac1_w = autocorr_lag1(homer_data["words"])
    ac1_s = autocorr_lag1(homer_data["syllables"])
    h_w = hurst_exponent_rs(homer_data["words"])
    h_s = hurst_exponent_rs(homer_data["syllables"])

    log.info(f"  AC1_w={ac1_w:.4f}, AC1_s={ac1_s:.4f}, "
             f"H_w={h_w:.4f}, H_s={h_s:.4f}")

    return {
        "n_sentences": len(homer_data["words"]),
        "ac1_words": round(ac1_w, 4),
        "ac1_syllables": round(ac1_s, 4),
        "H_words": round(h_w, 4) if not np.isnan(h_w) else None,
        "H_syllables": round(h_s, 4) if not np.isnan(h_s) else None,
        "mean_words": round(float(np.mean(homer_data["words"])), 2),
        "mean_syl": round(float(np.mean(homer_data["syllables"])), 2),
        "syl_per_word": round(float(np.mean(homer_data["syllables"])) /
                              float(np.mean(homer_data["words"])), 3),
        "note": ("Homero tiene hexámetro dactílico (metro estricto) "
                 "pero es NT-like en H. ¿AC(1) en sílabas difiere?"),
    }


def analyze_rigveda(rv_data):
    """Analyze Rig Veda AC(1) with meter-level analysis."""
    log.info("\n── Rig Veda: Análisis métrico ──")

    if rv_data is None:
        return {"error": "no data"}

    ac1_w = autocorr_lag1(rv_data["words"])
    ac1_s = autocorr_lag1(rv_data["syllables"])
    h_w = hurst_exponent_rs(rv_data["words"])
    h_s = hurst_exponent_rs(rv_data["syllables"])

    log.info(f"  AC1_w={ac1_w:.4f}, AC1_s={ac1_s:.4f}, "
             f"H_w={h_w:.4f}, H_s={h_s:.4f}")

    result = {
        "source": rv_data.get("source", "unknown"),
        "n_padas": len(rv_data["words"]),
        "ac1_words": round(ac1_w, 4),
        "ac1_syllables": round(ac1_s, 4),
        "H_words": round(h_w, 4) if not np.isnan(h_w) else None,
        "H_syllables": round(h_s, 4) if not np.isnan(h_s) else None,
    }

    # Per-hymn analysis
    hymn_padas = rv_data.get("hymn_padas", {})
    if hymn_padas:
        # Calculate AC(1) within each hymn (same-meter hymns vs mixed)
        hymn_ac1s = []
        for hymn_id, padas in hymn_padas.items():
            if len(padas) >= 6:
                syl_series = [p["syl"] for p in padas]
                word_series = [p["words"] for p in padas]
                syl_cv = float(np.std(syl_series) / (np.mean(syl_series) + 1e-10))
                ac1 = autocorr_lag1(word_series)
                hymn_ac1s.append({
                    "hymn": hymn_id,
                    "n_padas": len(padas),
                    "syl_cv": round(syl_cv, 4),
                    "ac1_words": round(ac1, 4),
                })

        if hymn_ac1s:
            syl_cvs = [h["syl_cv"] for h in hymn_ac1s]
            median_cv = np.median(syl_cvs)
            # Low CV = consistent meter, high CV = mixed meter
            consistent = [h for h in hymn_ac1s if h["syl_cv"] <= median_cv]
            mixed = [h for h in hymn_ac1s if h["syl_cv"] > median_cv]

            result["meter_analysis"] = {
                "n_hymns_analyzed": len(hymn_ac1s),
                "consistent_meter": {
                    "n": len(consistent),
                    "mean_ac1": round(float(np.mean([h["ac1_words"]
                                     for h in consistent])), 4),
                    "mean_syl_cv": round(float(np.mean([h["syl_cv"]
                                         for h in consistent])), 4),
                },
                "mixed_meter": {
                    "n": len(mixed),
                    "mean_ac1": round(float(np.mean([h["ac1_words"]
                                     for h in mixed])), 4),
                    "mean_syl_cv": round(float(np.mean([h["syl_cv"]
                                         for h in mixed])), 4),
                },
            }

    if "note" in rv_data:
        result["note"] = rv_data["note"]

    return result


def generate_verdict(at_result, quran_result, homer_result, rv_result):
    """Synthesize findings into a verdict on recitation hypothesis."""
    log.info("\n── Veredicto ──")

    evidence = []

    # Test 1: AT poetry vs prose
    at_poetry = at_result.get("poetry", {})
    at_prose = at_result.get("prose", {})
    if at_poetry.get("ac1_words") is not None and at_prose.get("ac1_words") is not None:
        if at_poetry["ac1_words"] > at_prose["ac1_words"] * 1.2:
            evidence.append({
                "test": "AT_poetry_vs_prose",
                "direction": "SUPPORTS",
                "detail": (f"Poesía AC(1)={at_poetry['ac1_words']} > "
                          f"Prosa AC(1)={at_prose['ac1_words']}")
            })
        elif at_prose["ac1_words"] > at_poetry["ac1_words"] * 1.2:
            evidence.append({
                "test": "AT_poetry_vs_prose",
                "direction": "REFUTES",
                "detail": (f"Prosa AC(1)={at_prose['ac1_words']} > "
                          f"Poesía AC(1)={at_poetry['ac1_words']}")
            })
        else:
            evidence.append({
                "test": "AT_poetry_vs_prose",
                "direction": "NEUTRAL",
                "detail": (f"Poesía AC(1)={at_poetry['ac1_words']} ≈ "
                          f"Prosa AC(1)={at_prose['ac1_words']}")
            })

    # Test 2: Syllables vs words correlation
    for name, result in [("Corán", quran_result), ("Homero", homer_result)]:
        g = result.get("global", result) if isinstance(result, dict) else {}
        ac1_w = g.get("ac1_words")
        ac1_s = g.get("ac1_syllables")
        if ac1_w is not None and ac1_s is not None:
            if ac1_s > ac1_w * 1.3:
                evidence.append({
                    "test": f"{name}_syl_vs_word",
                    "direction": "SUPPORTS",
                    "detail": f"AC(1)_syl={ac1_s} > AC(1)_word={ac1_w}"
                })
            elif abs(ac1_s - ac1_w) < 0.05:
                evidence.append({
                    "test": f"{name}_syl_vs_word",
                    "direction": "NEUTRAL",
                    "detail": f"AC(1)_syl={ac1_s} ≈ AC(1)_word={ac1_w}"
                })
            else:
                evidence.append({
                    "test": f"{name}_syl_vs_word",
                    "direction": "WEAK",
                    "detail": f"AC(1)_syl={ac1_s} vs AC(1)_word={ac1_w}"
                })

    # Test 3: Homer contrast
    homer_ac1_w = homer_result.get("ac1_words", homer_result.get("global", {}).get("ac1_words"))
    at_all_ac1 = at_result.get("all", {}).get("ac1_words")
    if homer_ac1_w is not None and at_all_ac1 is not None:
        if homer_ac1_w < 0.05 and at_all_ac1 > 0.2:
            evidence.append({
                "test": "Homer_contrast",
                "direction": "KEY_FINDING",
                "detail": (f"Homero AC(1)={homer_ac1_w} ≈ 0 pese a metro estricto. "
                          f"AT AC(1)={at_all_ac1}. El metro homérico NO produce "
                          f"el mismo efecto que las restricciones AT-like.")
            })

    supports = sum(1 for e in evidence if e["direction"] == "SUPPORTS")
    refutes = sum(1 for e in evidence if e["direction"] == "REFUTES")

    verdict = {
        "evidence": evidence,
        "n_supports": supports,
        "n_refutes": refutes,
        "n_total": len(evidence),
        "conclusion": "",
        "mechanism_detail": "",
    }

    if supports > refutes:
        verdict["conclusion"] = ("PARTIALLY_SUPPORTED: Las restricciones métricas "
                                 "contribuyen al AC(1) alto, pero no son el único mecanismo.")
    elif refutes > supports:
        verdict["conclusion"] = ("REFUTED: Las restricciones métricas no explican "
                                 "el AC(1) alto. Otro mecanismo (temático, narrativo) opera.")
    else:
        verdict["conclusion"] = ("INCONCLUSIVE: La evidencia es mixta. "
                                 "Posiblemente AC(1) alto tiene múltiples fuentes.")

    for e in evidence:
        log.info(f"  {e['test']}: {e['direction']} — {e['detail']}")
    log.info(f"  Veredicto: {verdict['conclusion']}")

    return verdict


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    log.info("=" * 70)
    log.info("FASE 12 — Script 1: Recitation Hypothesis")
    log.info("=" * 70)

    # Load data
    at_data = load_at_verse_data()
    quran_data = load_quran_data()
    homer_data = load_homer_data()
    rv_data = load_rigveda_pada_data()

    # Analysis
    at_result = analyze_at_poetry_prose(at_data)
    with open(RESULTS_DIR / "ot_poetry_vs_prose_ac1.json", "w") as f:
        json.dump(at_result, f, indent=2, ensure_ascii=False)

    quran_result = analyze_quran(quran_data)
    with open(RESULTS_DIR / "quran_sura_ac1.json", "w") as f:
        json.dump(quran_result, f, indent=2, ensure_ascii=False)

    homer_result = analyze_homer(homer_data)
    with open(RESULTS_DIR / "homer_syllable_ac1.json", "w") as f:
        json.dump(homer_result, f, indent=2, ensure_ascii=False)

    rv_result = analyze_rigveda(rv_data)
    with open(RESULTS_DIR / "rigveda_meter_ac1.json", "w") as f:
        json.dump(rv_result, f, indent=2, ensure_ascii=False)

    # Verdict
    verdict = generate_verdict(at_result, quran_result, homer_result, rv_result)
    with open(RESULTS_DIR / "recitation_verdict.json", "w") as f:
        json.dump(verdict, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    log.info(f"\n{'=' * 70}")
    log.info(f"Script 1 completado en {elapsed:.1f}s")


if __name__ == "__main__":
    main()
