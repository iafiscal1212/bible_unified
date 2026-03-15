#!/usr/bin/env python3
"""
orchestrator.py — Lanza los 6 módulos de análisis en paralelo.
Monitoriza progreso y genera summary_report.json.
"""
import subprocess, sys, time, json, os
from pathlib import Path
from datetime import datetime

BASE = Path(__file__).parent
RESULTS = BASE / "results"
LOGS = BASE / "logs"
RESULTS.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)

MODULES = [
    "analyze_frequencies.py",
    "analyze_morphology.py",
    "analyze_numerical.py",
    "analyze_structure.py",
    "analyze_cooccurrence.py",
    "analyze_positional.py",
]

RESULT_FILES = {
    "frequencies": RESULTS / "frequencies" / "frequency_analysis.json",
    "morphology": RESULTS / "morphology" / "morphology_analysis.json",
    "numerical": RESULTS / "numerical" / "numerical_analysis.json",
    "structure": RESULTS / "structure" / "structure_analysis.json",
    "cooccurrence": RESULTS / "cooccurrence" / "cooccurrence_analysis.json",
    "positional": RESULTS / "positional" / "positional_analysis.json",
}


def run_modules():
    """Lanza todos los módulos y monitoriza."""
    start = time.time()
    procs = {}

    print(f"{'='*60}")
    print(f"  BIBLE RESEARCH — ORCHESTRATOR")
    print(f"  Inicio: {datetime.now().isoformat()}")
    print(f"  Módulos: {len(MODULES)}")
    print(f"{'='*60}\n")

    # Lanzar cada módulo
    for mod in MODULES:
        mod_path = BASE / mod
        name = mod.replace("analyze_", "").replace(".py", "")
        log_file = open(LOGS / f"{name}.log", "a")
        print(f"  [LAUNCH] {name}")
        proc = subprocess.Popen(
            [sys.executable, str(mod_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(BASE),
        )
        procs[name] = {"proc": proc, "start": time.time(), "log": log_file}

    # Monitorizar
    print(f"\n  Esperando finalización...\n")
    completed = set()
    while len(completed) < len(procs):
        for name, info in procs.items():
            if name in completed:
                continue
            ret = info["proc"].poll()
            if ret is not None:
                elapsed = time.time() - info["start"]
                stdout = info["proc"].stdout.read().decode("utf-8", errors="replace")
                status = "OK" if ret == 0 else f"ERROR(rc={ret})"
                print(f"  [{status}] {name:15s} — {elapsed:.1f}s")
                if stdout.strip():
                    print(f"           {stdout.strip()}")
                info["log"].write(stdout)
                info["log"].close()
                completed.add(name)
        if len(completed) < len(procs):
            time.sleep(1)

    total_time = time.time() - start
    print(f"\n{'='*60}")
    print(f"  Todos los módulos completados en {total_time:.1f}s")
    print(f"{'='*60}\n")

    return total_time


def generate_summary():
    """Consolida hallazgos principales de cada módulo."""
    summary = {
        "generated_at": datetime.now().isoformat(),
        "modules": {},
    }

    for name, fpath in RESULT_FILES.items():
        if not fpath.exists():
            summary["modules"][name] = {"status": "MISSING"}
            continue
        try:
            with open(fpath, encoding="utf-8") as f:
                data = json.load(f)
            # Extraer hallazgos clave de cada módulo
            highlights = extract_highlights(name, data)
            summary["modules"][name] = {
                "status": "OK",
                "file": str(fpath),
                "highlights": highlights,
            }
        except Exception as e:
            summary["modules"][name] = {"status": f"ERROR: {e}"}

    out = RESULTS / "summary_report.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  Summary: {out}")
    return summary


def extract_highlights(name, data):
    """Extrae métricas clave de cada módulo para el resumen."""
    h = {}
    if name == "frequencies":
        h["total_tokens"] = data.get("total_tokens")
        h["total_word_types"] = data.get("total_word_types")
        h["total_lemma_types"] = data.get("total_lemma_types")
        h["hapax_words"] = data.get("global_hapax_words")
        h["hapax_lemmas"] = data.get("global_hapax_lemmas")
        h["zipf"] = data.get("zipf")
        h["ot_tokens"] = data.get("ot_tokens")
        h["nt_tokens"] = data.get("nt_tokens")
        h["top10_words"] = data.get("top50_words_global", [])[:10]
        h["top10_lemmas"] = data.get("top50_lemmas_global", [])[:10]

    elif name == "morphology":
        h["global_pos"] = data.get("global_pos_distribution")
        h["global_vn_ratio"] = data.get("global_verb_noun_ratio")
        h["ot_vn_ratio"] = data.get("ot_verb_noun_ratio")
        h["nt_vn_ratio"] = data.get("nt_verb_noun_ratio")
        h["top10_bigrams"] = data.get("top30_bigrams_global", [])[:10]

    elif name == "numerical":
        h["ot_total"] = data.get("ot_total_value")
        h["nt_total"] = data.get("nt_total_value")
        h["grand_total"] = data.get("grand_total")
        h["ot_nt_ratio"] = data.get("ot_nt_ratio")
        h["verse_stats"] = data.get("verse_value_global_stats")

    elif name == "structure":
        h["verse_length"] = data.get("global_verse_length")
        h["proportions"] = data.get("proportions")
        h["book_sizes_top5"] = sorted(
            data.get("book_sizes", []), key=lambda x: -x[1]
        )[:5]

    elif name == "cooccurrence":
        h["total_verses"] = data.get("global_n_verses")
        h["unique_pairs"] = data.get("global_unique_pairs")
        h["top10_pairs"] = data.get("top50_cooccurring_pairs", [])[:10]
        h["top10_pmi"] = data.get("global_top_pmi", [])[:10]

    elif name == "positional":
        h["first_lemma_top10"] = data.get("global_first_lemma_top30", [])[:10]
        h["last_lemma_top10"] = data.get("global_last_lemma_top30", [])[:10]
        h["first_pos"] = data.get("global_first_pos")
        h["last_pos"] = data.get("global_last_pos")

    return h


def main():
    total_time = run_modules()
    print("Generando summary report...")
    summary = generate_summary()
    print(f"\n{'='*60}")
    print(f"  BIBLE RESEARCH — COMPLETO")
    print(f"  Tiempo total: {total_time:.1f}s")
    print(f"  Módulos OK: {sum(1 for m in summary['modules'].values() if m.get('status') == 'OK')}/{len(MODULES)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
