#!/usr/bin/env python3
"""
Fase 13 — Orchestrator: Cuatro Investigaciones en Paralelo
Lanza 4 scripts simultáneamente con subprocess.Popen.
"""

import json
import logging
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
RESULTS_DIR = BASE / "results"
LOG_DIR = BASE / "logs"

LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "fase13_orchestrator.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


SCRIPTS = [
    {
        "name": "n_optimality",
        "file": "n_optimality.py",
        "question": "¿Por qué N=3 es el óptimo para el modelo jerárquico?",
        "results_dir": "n_optimality",
    },
    {
        "name": "anomalous_books",
        "file": "anomalous_books.py",
        "question": "¿Los libros AT anómalos tienen explicación composicional?",
        "results_dir": "anomalous",
    },
    {
        "name": "ot_quotes_in_nt",
        "file": "ot_quotes_in_nt.py",
        "question": "¿Las citas del AT en el NT heredan estructura AT-like localmente?",
        "results_dir": "ot_quotes",
    },
    {
        "name": "apocryphal_gospels",
        "file": "apocryphal_gospels.py",
        "question": "¿Los evangelios excluidos del canon difieren en H de los canónicos?",
        "results_dir": "apocryphal",
    },
]


def main():
    t0 = time.time()
    log.info("=" * 70)
    log.info("FASE 13 — Cuatro Investigaciones en Paralelo")
    log.info("4 scripts en paralelo")
    log.info("=" * 70)

    # Launch all 4 scripts simultaneously
    processes = {}
    log_files = {}

    for script in SCRIPTS:
        log_path = LOG_DIR / f"fase13_{script['name']}.log"
        log_file = open(log_path, "w")
        log_files[script["name"]] = log_file

        log.info(f"  Lanzando {script['file']}...")
        proc = subprocess.Popen(
            [sys.executable, str(BASE / script["file"])],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=str(BASE),
        )
        processes[script["name"]] = proc
        log.info(f"    PID={proc.pid}")

    # Wait with logging every 30 seconds
    all_done = False
    while not all_done:
        time.sleep(30)
        elapsed = time.time() - t0
        all_done = True
        status = []
        for name, proc in processes.items():
            ret = proc.poll()
            if ret is None:
                all_done = False
                status.append(f"{name}: running")
            else:
                status.append(f"{name}: done (rc={ret})")
        log.info(f"  [{elapsed:.0f}s] {' | '.join(status)}")

    # Close log files
    for lf in log_files.values():
        lf.close()

    elapsed = time.time() - t0
    log.info(f"\n  Todos los scripts completados en {elapsed:.1f}s")

    # Collect results
    results = {
        "phase": 13,
        "title": "Cuatro Investigaciones en Paralelo",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_elapsed_seconds": round(elapsed, 1),
        "scripts": [],
    }

    for script in SCRIPTS:
        script_result = {
            "name": script["name"],
            "question": script["question"],
            "return_code": processes[script["name"]].returncode,
            "status": "completed" if processes[script["name"]].returncode == 0 else "failed",
        }

        results_dir = RESULTS_DIR / script["results_dir"]
        if results_dir.exists():
            json_files = list(results_dir.glob("*.json"))
            script_result["n_output_files"] = len(json_files)
            script_result["output_files"] = [f.name for f in json_files]

            # Extract key results per script
            for jf in json_files:
                try:
                    with open(jf, "r") as f:
                        data = json.load(f)

                    fname = jf.name

                    # Script 1: n_optimality
                    if "retrodiction_by_n" in fname:
                        if isinstance(data, dict):
                            best = max(data.values(),
                                       key=lambda x: x.get("all_match_pct", 0)
                                       if isinstance(x, dict) else 0)
                            if isinstance(best, dict):
                                script_result["optimal_N_AT"] = best.get("N")
                                script_result["optimal_retrodiction_pct"] = best.get("all_match_pct")

                    elif "n_optimal_by_corpus" in fname:
                        if isinstance(data, dict):
                            universality = data.get("universality", {})
                            script_result["universal_N"] = universality.get("all_same_N")
                            script_result["optimal_Ns"] = universality.get("optimal_Ns")

                    elif "mutual_information" in fname:
                        if isinstance(data, dict):
                            mi_thresholds = {}
                            for corpus, vals in data.items():
                                if isinstance(vals, dict):
                                    mi_thresholds[corpus] = vals.get("N_threshold_10pct")
                            if mi_thresholds:
                                script_result["MI_thresholds"] = mi_thresholds

                    # Script 2: anomalous_books
                    elif "verdict" in fname and "anomalous" in str(jf.parent):
                        if isinstance(data, dict):
                            summary = data.get("summary", {})
                            script_result["n_genuinely_anomalous"] = summary.get("n_genuinely_anomalous")
                            script_result["n_size_effect"] = summary.get("n_size_effect")
                            for book in ["Nahum", "Obadiah", "1 Chronicles"]:
                                if book in data:
                                    script_result[f"{book}_explanation"] = data[book].get("explanation")

                    # Script 3: ot_quotes
                    elif "local_h_comparison" in fname:
                        if isinstance(data, dict):
                            script_result["quotes_elevate_H"] = data.get("quotes_elevate_H")
                            script_result["H_with_quotes"] = data.get("H_local_with_quotes_mean")
                            script_result["H_without_quotes"] = data.get("H_local_without_quotes_mean")
                            script_result["mann_whitney_p"] = data.get("mann_whitney_p")

                    elif "quote_identification" in fname:
                        if isinstance(data, dict):
                            script_result["n_quote_verses"] = data.get("final_quote_count")
                            script_result["method"] = data.get("method_used")

                    # Script 4: apocryphal
                    elif "corpus_availability" in fname:
                        if isinstance(data, dict):
                            n_found = sum(1 for v in data.values()
                                          if isinstance(v, dict) and v.get("found"))
                            script_result["apocryphal_found"] = n_found
                            script_result["apocryphal_details"] = {
                                k: v.get("found", False) for k, v in data.items()
                                if isinstance(v, dict)
                            }

                    elif "h_comparison" in fname:
                        if isinstance(data, dict):
                            apo = data.get("apocryphal", {})
                            for name, metrics in apo.items():
                                if isinstance(metrics, dict):
                                    script_result[f"apocryphal_{name}_H"] = metrics.get("H")
                                    script_result[f"apocryphal_{name}_cluster"] = metrics.get("cluster")

                    elif "proxy_analysis" in fname:
                        if isinstance(data, dict):
                            test = data.get("early_vs_late_test", {})
                            if test:
                                script_result["Mark_H"] = test.get("Mark_H")
                                script_result["Pastorals_H"] = test.get("Pastorals_H")
                                script_result["delta_H_early_late"] = test.get("delta_H")

                except Exception as e:
                    log.warning(f"  Error leyendo {jf}: {e}")

        results["scripts"].append(script_result)

    # Save summary
    summary_path = RESULTS_DIR / "fase13_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    log.info(f"\n{'=' * 70}")
    log.info(f"FASE 13 completada en {elapsed:.1f}s")
    log.info(f"Resumen en {summary_path}")

    for sr in results["scripts"]:
        log.info(f"\n  {sr['name']}: {sr['status']} (rc={sr['return_code']})")
        if "optimal_N_AT" in sr:
            log.info(f"    N óptimo AT: {sr['optimal_N_AT']} "
                     f"(retrodición={sr['optimal_retrodiction_pct']}%)")
        if "universal_N" in sr:
            log.info(f"    Universal: {sr['universal_N']}, Ns={sr.get('optimal_Ns')}")
        if "n_genuinely_anomalous" in sr:
            log.info(f"    Genuinamente anómalos: {sr['n_genuinely_anomalous']}")
        if "quotes_elevate_H" in sr:
            log.info(f"    Citas elevan H: {sr['quotes_elevate_H']} "
                     f"(p={sr.get('mann_whitney_p')})")
        if "apocryphal_found" in sr:
            log.info(f"    Apócrifos encontrados: {sr['apocryphal_found']}")
        if "Mark_H" in sr:
            log.info(f"    Marcos H={sr['Mark_H']}, Pastorales H={sr.get('Pastorals_H')}")


if __name__ == "__main__":
    main()
