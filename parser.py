#!/usr/bin/env python3
"""
Bible Research — Unified Corpus Parser
=======================================
AT Hebreo: WLC (Westminster Leningrad Codex) — OSHB XML con morfología
NT Griego: SBLGNT (morphgnt) — TXT con morfología MorphGNT

Genera bible_unified.json con estructura normalizada por palabra.
"""
import json
import xml.etree.ElementTree as ET
from pathlib import Path

# ── Rutas (servidor Hetzner) ────────────────────────────────────────────
BASE    = Path(__file__).parent
OT_DIR  = BASE / "sources" / "morphhb" / "wlc"
NT_DIR  = BASE / "sources" / "sblgnt"
OUT     = BASE / "bible_unified.json"

# ── Tablas canónicas ────────────────────────────────────────────────────
WLC_BOOKS = {
    "Gen.xml":   (1,  "Genesis"),      "Exod.xml":  (2,  "Exodus"),
    "Lev.xml":   (3,  "Leviticus"),    "Num.xml":   (4,  "Numbers"),
    "Deut.xml":  (5,  "Deuteronomy"),  "Josh.xml":  (6,  "Joshua"),
    "Judg.xml":  (7,  "Judges"),       "Ruth.xml":  (8,  "Ruth"),
    "1Sam.xml":  (9,  "1 Samuel"),     "2Sam.xml":  (10, "2 Samuel"),
    "1Kgs.xml":  (11, "1 Kings"),      "2Kgs.xml":  (12, "2 Kings"),
    "1Chr.xml":  (13, "1 Chronicles"), "2Chr.xml":  (14, "2 Chronicles"),
    "Ezra.xml":  (15, "Ezra"),         "Neh.xml":   (16, "Nehemiah"),
    "Esth.xml":  (17, "Esther"),       "Job.xml":   (18, "Job"),
    "Ps.xml":    (19, "Psalms"),       "Prov.xml":  (20, "Proverbs"),
    "Eccl.xml":  (21, "Ecclesiastes"), "Song.xml":  (22, "Song of Songs"),
    "Isa.xml":   (23, "Isaiah"),       "Jer.xml":   (24, "Jeremiah"),
    "Lam.xml":   (25, "Lamentations"), "Ezek.xml":  (26, "Ezekiel"),
    "Dan.xml":   (27, "Daniel"),       "Hos.xml":   (28, "Hosea"),
    "Joel.xml":  (29, "Joel"),         "Amos.xml":  (30, "Amos"),
    "Obad.xml":  (31, "Obadiah"),      "Jonah.xml": (32, "Jonah"),
    "Mic.xml":   (33, "Micah"),        "Nah.xml":   (34, "Nahum"),
    "Hab.xml":   (35, "Habakkuk"),     "Zeph.xml":  (36, "Zephaniah"),
    "Hag.xml":   (37, "Haggai"),       "Zech.xml":  (38, "Zechariah"),
    "Mal.xml":   (39, "Malachi"),
}

SBLGNT_BOOKS = {
    "61-Mt-morphgnt.txt":  (40, "Matthew"),       "62-Mk-morphgnt.txt":  (41, "Mark"),
    "63-Lk-morphgnt.txt":  (42, "Luke"),           "64-Jn-morphgnt.txt":  (43, "John"),
    "65-Ac-morphgnt.txt":  (44, "Acts"),            "66-Ro-morphgnt.txt":  (45, "Romans"),
    "67-1Co-morphgnt.txt": (46, "1 Corinthians"),  "68-2Co-morphgnt.txt": (47, "2 Corinthians"),
    "69-Ga-morphgnt.txt":  (48, "Galatians"),       "70-Eph-morphgnt.txt": (49, "Ephesians"),
    "71-Php-morphgnt.txt": (50, "Philippians"),     "72-Col-morphgnt.txt": (51, "Colossians"),
    "73-1Th-morphgnt.txt": (52, "1 Thessalonians"),"74-2Th-morphgnt.txt": (53, "2 Thessalonians"),
    "75-1Ti-morphgnt.txt": (54, "1 Timothy"),       "76-2Ti-morphgnt.txt": (55, "2 Timothy"),
    "77-Tit-morphgnt.txt": (56, "Titus"),           "78-Phm-morphgnt.txt": (57, "Philemon"),
    "79-Heb-morphgnt.txt": (58, "Hebrews"),         "80-Jas-morphgnt.txt": (59, "James"),
    "81-1Pe-morphgnt.txt": (60, "1 Peter"),         "82-2Pe-morphgnt.txt": (61, "2 Peter"),
    "83-1Jn-morphgnt.txt": (62, "1 John"),          "84-2Jn-morphgnt.txt": (63, "2 John"),
    "85-3Jn-morphgnt.txt": (64, "3 John"),          "86-Jud-morphgnt.txt": (65, "Jude"),
    "87-Re-morphgnt.txt":  (66, "Revelation"),
}

# POS normalización
# WLC: morph puede tener prefijos separados por / (ej: HR/Ncfsa = prep prefix + noun)
# El POS principal es el ÚLTIMO segmento
WLC_POS = {
    "N": "noun", "V": "verb", "A": "adjective", "D": "adverb",
    "P": "pronoun", "R": "preposition", "C": "conjunction",
    "T": "particle", "I": "interjection", "S": "suffix",
}

SBLGNT_POS = {
    "N-": "noun", "V-": "verb", "RA": "article", "RD": "pronoun",
    "RI": "pronoun", "RP": "pronoun", "RR": "pronoun",
    "A-": "adjective", "C-": "conjunction", "X-": "particle",
    "D-": "adverb", "P-": "preposition",
}

OSIS_NS = "http://www.bibletechnologies.net/2003/OSIS/namespace"


def get_ot_pos(morph):
    """Extrae POS del código morfológico WLC.
    morph='HR/Ncfsa' → último segmento 'Ncfsa' → primer char 'N' → noun
    morph='HVqp3ms' → sin /, después de H → 'V' → verb
    """
    if not morph:
        return "other"
    # Quitar prefijo H (Hebrew) o A (Aramaic)
    m = morph
    if m[0] in ("H", "A") and len(m) > 1:
        m = m[1:]
    # Tomar último segmento después de /
    segments = m.split("/")
    main_seg = segments[-1]
    if main_seg:
        return WLC_POS.get(main_seg[0], "other")
    return "other"


def get_ot_lemma(lemma_raw):
    """Extrae el lema principal del campo lemma WLC.
    Formato: 'b/7225' → '7225', '1254 a' → '1254a', 'd/8064' → '8064'
    """
    if not lemma_raw:
        return ""
    # Tomar último segmento después de /
    parts = lemma_raw.split("/")
    main = parts[-1].strip().replace(" ", "")
    return main


def parse_wlc_book(filepath, book_num, book_name):
    """Parsea un libro del AT en formato OSIS XML."""
    words = []
    tree = ET.parse(filepath)
    root = tree.getroot()
    ns = f"{{{OSIS_NS}}}"

    for chapter_el in root.iter(f"{ns}chapter"):
        ch_ref = chapter_el.get("osisID", "")
        try:
            ch_num = int(ch_ref.split(".")[-1])
        except (ValueError, IndexError):
            continue

        for verse_el in chapter_el.iter(f"{ns}verse"):
            vs_ref = verse_el.get("osisID", "")
            try:
                vs_num = int(vs_ref.split(".")[-1])
            except (ValueError, IndexError):
                continue

            word_pos = 0
            for w in verse_el.iter(f"{ns}w"):
                text = (w.text or "").strip()
                if not text:
                    continue
                word_pos += 1
                morph = w.get("morph", "")
                lemma_raw = w.get("lemma", "")
                pos = get_ot_pos(morph)
                lemma = get_ot_lemma(lemma_raw)

                words.append({
                    "corpus":   "OT",
                    "book":     book_name,
                    "book_num": book_num,
                    "chapter":  ch_num,
                    "verse":    vs_num,
                    "word_pos": word_pos,
                    "text":     text,
                    "lemma":    lemma,
                    "morph":    morph,
                    "pos":      pos,
                    "lang":     "heb",
                })
    return words


def parse_sblgnt_book(filepath, book_num, book_name):
    """Parsea un libro del NT en formato MorphGNT TXT.
    Columnas: BBCCVV  POS  PARSE  text  normalized  cased  lemma
    """
    words = []
    verse_word_count = {}

    with open(filepath, encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            ref = parts[0]
            pos_raw = parts[1]
            parse = parts[2]
            text = parts[3]       # forma superficial (con puntuación)
            norm = parts[4]       # forma normalizada (sin puntuación)
            lemma = parts[6]      # lema de diccionario

            try:
                ch = int(ref[2:4])
                vs = int(ref[4:6])
            except ValueError:
                continue

            key = (ch, vs)
            verse_word_count[key] = verse_word_count.get(key, 0) + 1
            wpos = verse_word_count[key]

            pos = SBLGNT_POS.get(pos_raw, "other")

            words.append({
                "corpus":   "NT",
                "book":     book_name,
                "book_num": book_num,
                "chapter":  ch,
                "verse":    vs,
                "word_pos": wpos,
                "text":     norm,     # usamos forma normalizada (sin puntuación)
                "lemma":    lemma,
                "morph":    parse,
                "pos":      pos,
                "lang":     "grc",
            })
    return words


def main():
    all_words = []
    stats = {"OT_books": 0, "NT_books": 0, "OT_words": 0, "NT_words": 0}

    print("Procesando AT Hebreo (WLC)...")
    for fname, (book_num, book_name) in sorted(WLC_BOOKS.items(), key=lambda x: x[1][0]):
        fpath = OT_DIR / fname
        if not fpath.exists():
            print(f"  [AVISO] No encontrado: {fname}")
            continue
        words = parse_wlc_book(fpath, book_num, book_name)
        all_words.extend(words)
        stats["OT_books"] += 1
        stats["OT_words"] += len(words)
        print(f"  {book_name:20s} -> {len(words):6,} palabras")

    print("\nProcesando NT Griego (SBLGNT)...")
    for fname, (book_num, book_name) in sorted(SBLGNT_BOOKS.items(), key=lambda x: x[1][0]):
        fpath = NT_DIR / fname
        if not fpath.exists():
            print(f"  [AVISO] No encontrado: {fname}")
            continue
        words = parse_sblgnt_book(fpath, book_num, book_name)
        all_words.extend(words)
        stats["NT_books"] += 1
        stats["NT_words"] += len(words)
        print(f"  {book_name:20s} -> {len(words):6,} palabras")

    # Guardar corpus unificado
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(all_words, f, ensure_ascii=False)

    total = stats["OT_words"] + stats["NT_words"]
    size_mb = OUT.stat().st_size / 1024 / 1024
    print(f"""
{'='*50}
  CORPUS UNIFICADO COMPLETO
{'='*50}
  Libros AT : {stats['OT_books']:3d}  |  Palabras AT : {stats['OT_words']:>8,}
  Libros NT : {stats['NT_books']:3d}  |  Palabras NT : {stats['NT_words']:>8,}
  TOTAL     : {stats['OT_books']+stats['NT_books']:3d}  |  TOTAL       : {total:>8,}
{'='*50}
  Guardado en: {OUT}
  Tamano: {size_mb:.1f} MB
""")


if __name__ == "__main__":
    main()
