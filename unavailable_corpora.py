#!/usr/bin/env python3
"""
Fase 16 — Script 3: unavailable_corpora.py
Documentación sistemática de corpora no disponibles.

Genera JSON con 4 entradas: Avesta, Pentateuco Samaritano,
Nag Hammadi, Mishnah verificada.
"""

import json
import logging
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
RESULTS_DIR = BASE / "results" / "unavailable_corpora"
LOG_DIR = BASE / "logs"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "fase16_unavailable_corpora.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


CORPORA = [
    {
        "name": "Avesta (Zoroastrian scriptures)",
        "date_range": "1500-500 BCE (Gathas oldest, Vendidad youngest)",
        "language": "Avestan (Old Iranian)",
        "type": "liturgical/hymnic + legal",
        "searches_performed": [
            "TITUS Project (Thesaurus Indogermanischer Text- und Sprachmaterialien) — partial transcription exists but no word-level morphological annotation",
            "Geldner critical edition (1886-1896) — scanned but not digitized to word level",
            "Avestan Digital Archive (University of Vienna) — in development, not publicly released as of 2025",
            "GitHub/HuggingFace — no structured Avestan corpus found",
        ],
        "where_exists": "TITUS has chapter/verse divisions; no public word-level dataset with morphological tagging",
        "impact_if_available": (
            "HIGH: Avesta is the closest typological parallel to Rig Veda (same Indo-Iranian family, "
            "similar liturgical function). Would test whether metrical constraints produce similar "
            "φ/AC1 patterns across cognate traditions. Gathas (hymnic) vs Vendidad (legal) would "
            "parallel AT poetry vs law comparison."
        ),
        "priority": 1,
        "minimum_requirement": "Word-segmented text with verse boundaries, ideally with POS tags",
    },
    {
        "name": "Samaritan Pentateuch",
        "date_range": "~400 BCE divergence from MT (Masoretic Text)",
        "language": "Samaritan Hebrew (variant of Biblical Hebrew)",
        "type": "legal/narrative (same as Torah)",
        "searches_performed": [
            "August Freiherr von Gall critical edition (1918) — scanned, no digital text",
            "Samaritan Pentateuch Project (Tel Aviv University) — no public API or download",
            "ETANA (Electronic Tools and Ancient Near Eastern Archives) — reference only",
            "GitHub/academic repos — no structured word-level dataset found",
            "Tal & Florentin edition (2010) — print only, no digital version",
        ],
        "where_exists": "Critical editions exist in print; some verse-level transcriptions in academic databases, but no public morphologically tagged corpus",
        "impact_if_available": (
            "VERY HIGH: Direct textual control for Pentateuch. Same content, different transmission "
            "tradition (~2,400 years of independent copying). ~6,000 textual differences from MT. "
            "Would isolate the effect of scribal transmission on H, AC1, DFA — the most precise "
            "natural experiment possible for our methodology."
        ),
        "priority": 1,
        "minimum_requirement": "Word-segmented text aligned with MT verse boundaries",
    },
    {
        "name": "Nag Hammadi Library (Gnostic texts)",
        "date_range": "1st-4th century CE (composition), ~350 CE (codices)",
        "language": "Coptic (Sahidic dialect), some with Greek precursors",
        "type": "religious/philosophical — gospels, apocalypses, treatises",
        "searches_performed": [
            "The Nag Hammadi Library (James M. Robinson ed.) — English translations only",
            "Coptic SCRIPTORIUM (Georgetown University) — has some Coptic texts with POS annotation",
            "Marcion.de — Coptic texts but no structured morphological data",
            "TLG (Thesaurus Linguae Graecae) — Greek precursors only for some texts",
            "Digital Coptic projects — focus on Sahidic Bible, not Nag Hammadi specifically",
        ],
        "where_exists": "Coptic SCRIPTORIUM has SOME annotated Coptic; Nag Hammadi texts specifically are partially available but not systematically morphologically tagged",
        "impact_if_available": (
            "MODERATE: Tests whether Gnostic religious texts (heterodox Christianity) show "
            "different statistical signatures from canonical NT. Problem: Coptic word segmentation "
            "differs fundamentally from Hebrew/Greek, making direct H/AC1 comparison difficult. "
            "Best as a within-Coptic comparison (canonical Coptic NT vs Nag Hammadi)."
        ),
        "priority": 3,
        "minimum_requirement": "Word-segmented Coptic text with verse/section boundaries and POS tags",
    },
    {
        "name": "Mishnah (verified word-level)",
        "date_range": "~200 CE (codification by Rabbi Yehuda ha-Nasi)",
        "language": "Mishnaic Hebrew",
        "type": "legal/halakhic",
        "searches_performed": [
            "Sefaria API — has full Mishnah text but no morphological tagging",
            "Academy of the Hebrew Language Historical Dictionary — lemmatized but not public API",
            "Digital Mishnah Project (University of Maryland) — manuscript comparison, not morphological",
            "MILA (Israeli NLP Center) — modern Hebrew tools, limited Mishnaic coverage",
            "Dicta (Israel) — developing historical Hebrew NLP, partial Mishnaic coverage",
        ],
        "where_exists": "Full text available via Sefaria; Kaufmann manuscript digitized. No public word-level morphological annotation comparable to OSHB quality",
        "impact_if_available": (
            "HIGH: Mishnah is the earliest post-biblical Hebrew text corpus of significant size "
            "(~300,000 words). Written in a later Hebrew stratum than AT. Would test: "
            "(1) whether legal genre maintains similar AC1 across centuries, "
            "(2) whether Mishnaic Hebrew shows different H from Biblical Hebrew despite same language family, "
            "(3) temporal evolution of φ and d parameters. "
            "CAVEAT: Sefaria text is available but WITHOUT morphological tagging — "
            "we could compute verse-length metrics but NOT pos_entropy."
        ),
        "priority": 2,
        "minimum_requirement": "Word-segmented text with halakhah boundaries; ideally morphological POS tags",
        "partial_workaround": (
            "Sefaria text + simple whitespace tokenization → H, AC1, DFA computable. "
            "POS entropy requires morphological tagger for Mishnaic Hebrew (not yet available at OSHB quality)."
        ),
    },
]


def main():
    t0 = time.time()
    log.info("=" * 70)
    log.info("FASE 16 — Script 3: unavailable_corpora.py")
    log.info("Documentación de corpora no disponibles")
    log.info("=" * 70)

    result = {
        "title": "Systematic Gap Analysis: Unavailable Corpora",
        "date": "2026-03-17",
        "n_corpora": len(CORPORA),
        "corpora": CORPORA,
        "summary": {
            "priority_1": [c["name"] for c in CORPORA if c["priority"] == 1],
            "priority_2": [c["name"] for c in CORPORA if c["priority"] == 2],
            "priority_3": [c["name"] for c in CORPORA if c["priority"] == 3],
            "key_gap": (
                "Samaritan Pentateuch is the single most valuable missing corpus: "
                "same text, independent transmission, would provide the cleanest "
                "test of scribal transmission effects on statistical signatures."
            ),
            "actionable_next_step": (
                "Mishnah from Sefaria API can be partially analyzed (H, AC1, DFA) "
                "without morphological tagging. This is the lowest-hanging fruit."
            ),
        },
    }

    with open(RESULTS_DIR / "systematic_gap_analysis.json", "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    for c in CORPORA:
        log.info(f"\n  [{c['priority']}] {c['name']}")
        log.info(f"      Language: {c['language']}")
        log.info(f"      Impact: {c['impact_if_available'][:80]}...")

    elapsed = time.time() - t0
    log.info(f"\n{'=' * 70}")
    log.info(f"unavailable_corpora.py completado en {elapsed:.1f}s")
    print(f"[unavailable_corpora] DONE — {elapsed:.1f}s")


if __name__ == "__main__":
    main()
