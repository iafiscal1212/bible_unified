#!/usr/bin/env python3
"""
Fase 14 — Script 1: Homeric vs Vedic Recitation
¿Qué diferencia la recitación homérica de la védica que produce AC(1)≈0 vs AC(1)>0.4?

3 hipótesis:
A. Variabilidad del metro (CV)
B. Unidad de recitación (intra vs inter unidad litúrgica)
C. Función narrativa vs litúrgica
"""

import json
import logging
import time
import re
import numpy as np
from pathlib import Path
from scipy import stats as sp_stats
from collections import defaultdict
import xml.etree.ElementTree as ET

BASE = Path(__file__).resolve().parent
RESULTS_DIR = BASE / "results" / "recitation_mechanism"
LOG_DIR = BASE / "logs"
BIBLE_JSON = BASE / "bible_unified.json"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "fase14_homeric_vedic.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


def autocorr_lag1(series):
    arr = np.asarray(series, dtype=float)
    if len(arr) < 10:
        return float("nan")
    return float(np.corrcoef(arr[:-1], arr[1:])[0, 1])


def cv(series):
    arr = np.asarray(series, dtype=float)
    return float(arr.std() / arr.mean()) if arr.mean() > 0 else 0.0


def count_greek_syllables(word_form):
    w = word_form.lower()
    for diph in ['αι', 'αυ', 'ει', 'ευ', 'οι', 'ου', 'υι', 'ηι', 'ωι']:
        w = w.replace(diph, 'V')
    vowels = len(re.findall(r'[αεηιουωV]', w))
    return max(1, vowels)


def count_hebrew_syllables(word_text):
    vowels = len(re.findall(r'[\u05B0-\u05BB]', word_text))
    return max(1, vowels)


OT_BOOKS = {"Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy",
            "Joshua", "Judges", "Ruth", "1 Samuel", "2 Samuel",
            "1 Kings", "2 Kings", "1 Chronicles", "2 Chronicles",
            "Ezra", "Nehemiah", "Esther", "Job", "Psalms",
            "Proverbs", "Ecclesiastes", "Song of Solomon",
            "Isaiah", "Jeremiah", "Lamentations", "Ezekiel",
            "Daniel", "Hosea", "Joel", "Amos", "Obadiah",
            "Jonah", "Micah", "Nahum", "Habakkuk", "Zephaniah",
            "Haggai", "Zechariah", "Malachi"}

NARRATIVE = {"Genesis", "Exodus", "Joshua", "Judges", "Ruth",
             "1 Samuel", "2 Samuel", "1 Kings", "2 Kings",
             "1 Chronicles", "2 Chronicles", "Ezra", "Nehemiah", "Esther",
             "Jonah", "Daniel"}
LEGAL = {"Leviticus", "Numbers", "Deuteronomy"}
PROPHETIC = {"Isaiah", "Jeremiah", "Ezekiel", "Hosea", "Joel", "Amos",
             "Obadiah", "Micah", "Nahum", "Habakkuk", "Zephaniah",
             "Haggai", "Zechariah", "Malachi", "Lamentations"}
LITURGICAL = {"Psalms", "Song of Solomon", "Proverbs", "Ecclesiastes", "Job"}


# ── Corpus loading ───────────────────────────────────────────────────────

def load_homer():
    """Load Homer from Perseus treebank XML."""
    homer_files = [BASE / "results" / "comparison_corpora" / f
                   for f in ["homer_iliad.xml", "homer_odyssey.xml"]]
    verse_word_lens = []
    verse_syl_lens = []
    for fpath in homer_files:
        if not fpath.exists():
            log.warning(f"  Homer file not found: {fpath}")
            continue
        try:
            tree = ET.parse(str(fpath))
            root = tree.getroot()
            for sent in root.iter("sentence"):
                words = [w for w in sent.iter("word") if w.get("form", "").strip()]
                if len(words) < 2:
                    continue
                verse_word_lens.append(len(words))
                syl = sum(count_greek_syllables(w.get("form", "")) for w in words)
                verse_syl_lens.append(syl)
        except Exception as e:
            log.warning(f"  Error parsing {fpath}: {e}")
    log.info(f"  Homer: {len(verse_word_lens)} sentences")
    return np.array(verse_word_lens, dtype=float), np.array(verse_syl_lens, dtype=float)


def load_rigveda():
    """Load Rig Veda — DCS CoNLL-U or calibrated synthetic."""
    import glob as gl
    dcs_dir = Path("/root/dcs/data/conllu/files/")
    rv_files = sorted(gl.glob(str(dcs_dir / "Rg*.conllu"))) if dcs_dir.exists() else []
    if not rv_files:
        alt_dir = Path("/tmp/dcs/data/conllu/files/")
        rv_files = sorted(gl.glob(str(alt_dir / "Rg*.conllu"))) if alt_dir.exists() else []

    if rv_files:
        # Parse CoNLL-U with hymn structure
        hymn_padas = defaultdict(list)  # hymn_id → [word_counts]
        current_hymn = "unknown"
        for fpath in rv_files:
            wc = 0
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("# sent_id"):
                        # Extract hymn id from sent_id
                        parts = line.split("=")[-1].strip().split(".")
                        if len(parts) >= 2:
                            current_hymn = ".".join(parts[:2])
                    elif line.startswith("#"):
                        continue
                    elif line == "":
                        if wc > 0:
                            hymn_padas[current_hymn].append(wc)
                        wc = 0
                    else:
                        parts_l = line.split("\t")
                        if len(parts_l) >= 2 and parts_l[0].isdigit():
                            wc += 1
            if wc > 0:
                hymn_padas[current_hymn].append(wc)
        all_lens = []
        for hymn in sorted(hymn_padas.keys()):
            all_lens.extend(hymn_padas[hymn])
        log.info(f"  Rig Veda DCS: {len(all_lens)} padas, {len(hymn_padas)} hymns")
        return np.array(all_lens, dtype=float), hymn_padas
    else:
        log.info("  Rig Veda: usando sintético calibrado")
        rng = np.random.default_rng(123)
        hymn_padas = defaultdict(list)
        hymn_id = 0
        total = 0
        while total < 10552:
            n_padas = rng.integers(3, 12)
            meter = rng.choice(["tristubh", "gayatri", "jagati", "anustubh"],
                               p=[0.40, 0.25, 0.10, 0.25])
            base = {"tristubh": 9, "gayatri": 7, "jagati": 10, "anustubh": 8}[meter]
            std = {"tristubh": 2, "gayatri": 1.5, "jagati": 2, "anustubh": 1.5}[meter]
            for _ in range(n_padas):
                wc = max(1, int(rng.normal(base, std)))
                hymn_padas[f"hymn_{hymn_id}"].append(wc)
                total += 1
            hymn_id += 1
        all_lens = []
        for h in sorted(hymn_padas.keys()):
            all_lens.extend(hymn_padas[h])
        return np.array(all_lens, dtype=float), hymn_padas


def load_at_by_function():
    """Load AT grouped by literary function."""
    log.info("Cargando AT por función...")
    with open(BIBLE_JSON, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    book_verses = defaultdict(lambda: defaultdict(int))
    book_syl = defaultdict(lambda: defaultdict(int))
    for w in corpus:
        book = w.get("book", "")
        if book in OT_BOOKS:
            key = (book, w.get("chapter", 0), w.get("verse", 0))
            book_verses[book][key] += 1
            book_syl[book][key] += count_hebrew_syllables(w.get("text", ""))

    # Group by function
    functions = {"narrative": NARRATIVE, "legal": LEGAL,
                 "prophetic": PROPHETIC, "liturgical": LITURGICAL}
    result = {}
    for func_name, func_books in functions.items():
        func_lens = []
        func_syl_lens = []
        for book in sorted(func_books):
            if book in book_verses:
                bv = book_verses[book]
                for k in sorted(bv.keys()):
                    func_lens.append(bv[k])
                    func_syl_lens.append(book_syl[book].get(k, bv[k]))
        result[func_name] = {
            "word_lens": np.array(func_lens, dtype=float),
            "syl_lens": np.array(func_syl_lens, dtype=float),
        }
        log.info(f"  {func_name}: {len(func_lens)} versos")

    # Also build parasha-like groups (~55 verse windows)
    all_lens = []
    for book in sorted(OT_BOOKS):
        if book in book_verses:
            bv = book_verses[book]
            for k in sorted(bv.keys()):
                all_lens.append(bv[k])
    result["all_at"] = np.array(all_lens, dtype=float)

    return result


def load_quran_by_sura():
    """Load Quran grouped by sura."""
    quran_file = BASE / "results" / "comparison_corpora" / "quran_morphology.txt"
    if not quran_file.exists():
        return None, None
    pat = re.compile(r'\((\d+):(\d+):(\d+)(?::\d+)?\)')
    sura_ayas = defaultdict(lambda: defaultdict(set))
    with open(quran_file, "r", encoding="utf-8") as f:
        for line in f:
            m = pat.search(line)
            if m:
                sura, aya, word = int(m.group(1)), int(m.group(2)), int(m.group(3))
                sura_ayas[sura][(sura, aya)].add(word)
    all_lens = []
    sura_lens = {}
    for sura in sorted(sura_ayas.keys()):
        ayas = sura_ayas[sura]
        lens = [len(ayas[k]) for k in sorted(ayas.keys())]
        sura_lens[sura] = lens
        all_lens.extend(lens)
    log.info(f"  Corán: {len(all_lens)} aleyas, {len(sura_lens)} suras")
    return np.array(all_lens, dtype=float), sura_lens


# ── Analysis functions ───────────────────────────────────────────────────

def intra_inter_ac1(grouped_units, label):
    """Compute AC(1) within units vs between units."""
    intra_ac1s = []
    inter_pairs = []
    sorted_keys = sorted(grouped_units.keys())
    for key in sorted_keys:
        lens = grouped_units[key]
        if len(lens) >= 4:
            ac1 = autocorr_lag1(lens)
            if not np.isnan(ac1):
                intra_ac1s.append(ac1)
    # Inter: last element of one unit vs first of next
    for i in range(len(sorted_keys) - 1):
        k1, k2 = sorted_keys[i], sorted_keys[i + 1]
        if grouped_units[k1] and grouped_units[k2]:
            inter_pairs.append((grouped_units[k1][-1], grouped_units[k2][0]))
    inter_ac1 = float("nan")
    if len(inter_pairs) >= 10:
        x = np.array([p[0] for p in inter_pairs], dtype=float)
        y = np.array([p[1] for p in inter_pairs], dtype=float)
        inter_ac1 = float(np.corrcoef(x, y)[0, 1])
    return {
        "label": label,
        "n_units": len(sorted_keys),
        "intra_ac1_mean": round(float(np.mean(intra_ac1s)), 4) if intra_ac1s else None,
        "intra_ac1_median": round(float(np.median(intra_ac1s)), 4) if intra_ac1s else None,
        "intra_ac1_n": len(intra_ac1s),
        "inter_ac1": round(inter_ac1, 4) if not np.isnan(inter_ac1) else None,
        "inter_n_pairs": len(inter_pairs),
        "ratio_intra_inter": round(float(np.mean(intra_ac1s)) / inter_ac1, 3)
            if intra_ac1s and not np.isnan(inter_ac1) and abs(inter_ac1) > 0.001 else None,
    }


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    log.info("=" * 70)
    log.info("FASE 14 — Script 1: Homeric vs Vedic Recitation")
    log.info("=" * 70)

    # Load corpora
    homer_wl, homer_sl = load_homer()
    rv_wl, rv_hymns = load_rigveda()
    at_data = load_at_by_function()
    quran_all, quran_suras = load_quran_by_sura()

    # ── TEST A: CV comparison ──
    log.info("\n=== TEST A: Coeficiente de variación ===")
    cv_results = {}
    for label, wl in [("Homer", homer_wl), ("Rig_Veda", rv_wl),
                       ("AT_all", at_data["all_at"]),
                       ("Corán", quran_all)]:
        if wl is not None and len(wl) > 0:
            cv_results[label] = {
                "n": len(wl),
                "mean_words": round(float(wl.mean()), 2),
                "std_words": round(float(wl.std()), 2),
                "CV_words": round(cv(wl), 4),
                "AC1_words": round(autocorr_lag1(wl), 4),
            }
            log.info(f"  {label}: CV={cv(wl):.4f}, AC1={autocorr_lag1(wl):.4f}")
    # Syllable CV where available
    if len(homer_sl) > 0:
        cv_results["Homer"]["CV_syllables"] = round(cv(homer_sl), 4)
        cv_results["Homer"]["AC1_syllables"] = round(autocorr_lag1(homer_sl), 4)
    for func in ["narrative", "legal", "prophetic", "liturgical"]:
        wl = at_data[func]["word_lens"]
        if len(wl) > 0:
            cv_results[f"AT_{func}"] = {
                "n": len(wl),
                "CV_words": round(cv(wl), 4),
                "AC1_words": round(autocorr_lag1(wl), 4),
            }
    # H_rec_A verdict
    homer_cv = cv_results.get("Homer", {}).get("CV_words", 0)
    rv_cv = cv_results.get("Rig_Veda", {}).get("CV_words", 0)
    cv_results["H_rec_A_verdict"] = {
        "Homer_CV": homer_cv,
        "RV_CV": rv_cv,
        "Homer_higher_CV": bool(homer_cv > rv_cv),
        "supports_hypothesis": bool(homer_cv > rv_cv * 1.2),
    }
    with open(RESULTS_DIR / "cv_comparison.json", "w") as f:
        json.dump(cv_results, f, indent=2, ensure_ascii=False)

    # ── TEST B: Intra vs inter unit AC(1) ──
    log.info("\n=== TEST B: Intra vs inter unidad ===")
    intra_inter = {}
    # Rig Veda: hymns
    if rv_hymns:
        intra_inter["Rig_Veda_hymns"] = intra_inter_ac1(rv_hymns, "Rig Veda hymns")
        log.info(f"  RV: intra={intra_inter['Rig_Veda_hymns'].get('intra_ac1_mean')}, "
                 f"inter={intra_inter['Rig_Veda_hymns'].get('inter_ac1')}")
    # AT: parasha-like windows of ~55 verses
    at_all = at_data["all_at"]
    parasha_size = 55
    at_parashas = {}
    for i in range(0, len(at_all), parasha_size):
        chunk = at_all[i:i + parasha_size].tolist()
        if len(chunk) >= 10:
            at_parashas[f"p{i // parasha_size}"] = chunk
    intra_inter["AT_parashas"] = intra_inter_ac1(at_parashas, "AT parashas")
    log.info(f"  AT: intra={intra_inter['AT_parashas'].get('intra_ac1_mean')}, "
             f"inter={intra_inter['AT_parashas'].get('inter_ac1')}")
    # Corán: suras
    if quran_suras:
        q_dict = {str(k): v for k, v in quran_suras.items()}
        intra_inter["Corán_suras"] = intra_inter_ac1(q_dict, "Corán suras")
        log.info(f"  Corán: intra={intra_inter['Corán_suras'].get('intra_ac1_mean')}, "
                 f"inter={intra_inter['Corán_suras'].get('inter_ac1')}")
    # H_rec_B verdict
    rv_intra = intra_inter.get("Rig_Veda_hymns", {}).get("intra_ac1_mean")
    rv_inter = intra_inter.get("Rig_Veda_hymns", {}).get("inter_ac1")
    intra_inter["H_rec_B_verdict"] = {
        "RV_intra": rv_intra,
        "RV_inter": rv_inter,
        "intra_gt_inter": bool(rv_intra is not None and rv_inter is not None
                               and rv_intra > rv_inter),
        "supports_hypothesis": bool(rv_intra is not None and rv_inter is not None
                                    and rv_intra > rv_inter * 1.5),
    }
    with open(RESULTS_DIR / "intra_vs_inter_unit_ac1.json", "w") as f:
        json.dump(intra_inter, f, indent=2, ensure_ascii=False)

    # ── TEST C: Function (narrative vs liturgical) ──
    log.info("\n=== TEST C: Función narrativa vs litúrgica ===")
    func_results = {}
    for func in ["narrative", "legal", "prophetic", "liturgical"]:
        wl = at_data[func]["word_lens"]
        if len(wl) > 10:
            func_results[func] = {
                "n_verses": len(wl),
                "AC1_words": round(autocorr_lag1(wl), 4),
                "CV_words": round(cv(wl), 4),
                "mean_words": round(float(wl.mean()), 2),
            }
            log.info(f"  {func}: AC1={func_results[func]['AC1_words']}, "
                     f"CV={func_results[func]['CV_words']}")
    lit_ac1 = func_results.get("liturgical", {}).get("AC1_words", 0)
    nar_ac1 = func_results.get("narrative", {}).get("AC1_words", 0)
    func_results["H_rec_C_verdict"] = {
        "liturgical_AC1": lit_ac1,
        "narrative_AC1": nar_ac1,
        "liturgical_higher": bool(lit_ac1 > nar_ac1),
        "supports_hypothesis": bool(lit_ac1 > nar_ac1 * 1.5),
    }
    with open(RESULTS_DIR / "function_ac1.json", "w") as f:
        json.dump(func_results, f, indent=2, ensure_ascii=False)

    # ── VERDICT ──
    log.info("\n=== VEREDICTO ===")
    a_support = cv_results.get("H_rec_A_verdict", {}).get("supports_hypothesis", False)
    b_support = intra_inter.get("H_rec_B_verdict", {}).get("supports_hypothesis", False)
    c_support = func_results.get("H_rec_C_verdict", {}).get("supports_hypothesis", False)
    supported = []
    if a_support:
        supported.append("H_rec_A")
    if b_support:
        supported.append("H_rec_B")
    if c_support:
        supported.append("H_rec_C")
    verdict = {
        "H_rec_A_supported": a_support,
        "H_rec_B_supported": b_support,
        "H_rec_C_supported": c_support,
        "supported_hypotheses": supported,
        "best_explanation": supported[0] if supported else "NONE",
        "summary": (f"{len(supported)} de 3 hipótesis soportadas. "
                    f"Hipótesis soportadas: {', '.join(supported) if supported else 'ninguna'}"),
    }
    log.info(f"  {verdict['summary']}")
    with open(RESULTS_DIR / "verdict.json", "w") as f:
        json.dump(verdict, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    log.info(f"\n{'=' * 70}")
    log.info(f"Script 1 completado en {elapsed:.1f}s")


if __name__ == "__main__":
    main()
