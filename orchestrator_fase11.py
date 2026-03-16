#!/usr/bin/env python3
"""
Fase 11 — Orchestrator: El Origen Composicional de H
Lanza 3 scripts en paralelo con subprocess.Popen.
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
        logging.FileHandler(LOG_DIR / "fase11_orchestrator.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


SCRIPTS = [
    {
        "name": "composition_features",
        "file": "composition_features.py",
        "question": "¿Qué features compositivas predicen H?",
        "results_dir": "composition",
    },
    {
        "name": "nt_canonical_order",
        "file": "nt_canonical_order.py",
        "question": "¿El orden canónico del NT es único en producir H alto?",
        "results_dir": "nt_order",
    },
    {
        "name": "convergence_mechanism",
        "file": "convergence_mechanism.py",
        "question": "¿Qué tienen en común AT, Corán y Rig Veda a nivel compositivo?",
        "results_dir": "convergence",
    },
]


def main():
    t0 = time.time()
    log.info("=" * 70)
    log.info("FASE 11 — El Origen Composicional de H")
    log.info("3 scripts en paralelo")
    log.info("=" * 70)

    # Lanzar los 3 scripts simultáneamente
    processes = {}
    log_files = {}

    for script in SCRIPTS:
        log_path = LOG_DIR / f"fase11_{script['name']}.log"
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

    # Esperar con logging cada 30 segundos
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

    # Cerrar log files
    for lf in log_files.values():
        lf.close()

    elapsed = time.time() - t0
    log.info(f"\n  Todos los scripts completados en {elapsed:.1f}s")

    # Recopilar resultados
    results = {
        "phase": 11,
        "title": "El Origen Composicional de H",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_elapsed_seconds": elapsed,
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

            # Load key results
            for jf in json_files:
                try:
                    with open(jf, "r") as f:
                        data = json.load(f)
                    # Extract summary info based on filename
                    if "ranking" in jf.name or "correlation" in jf.name:
                        if isinstance(data, dict) and "top_predictors" in data:
                            script_result["top_predictors"] = data["top_predictors"][:3]
                        elif isinstance(data, list):
                            script_result["top_correlations"] = data[:3]
                    elif "regression" in jf.name:
                        if isinstance(data, list) and data:
                            best = data[-1]
                            script_result["best_model"] = {
                                "features": best.get("features"),
                                "r2": best.get("r2"),
                                "loo_rmse": best.get("loo_rmse"),
                            }
                    elif "synthetic" in jf.name:
                        script_result["synthetic_conclusion"] = data.get("conclusion")
                    elif "h_by_order" in jf.name:
                        hist = data.get("historical_orders", {})
                        script_result["historical_orders"] = {
                            k: {"H": v.get("H"), "name": v.get("name")}
                            for k, v in hist.items()
                        }
                        perm = data.get("permutation_test", {})
                        script_result["permutation_test"] = {
                            "H_canonical": perm.get("H_canonical"),
                            "H_random_mean": perm.get("H_random_mean"),
                            "p_value": perm.get("p_value"),
                        }
                        greedy = data.get("greedy_search", {})
                        script_result["greedy_max_H"] = greedy.get("greedy_max_H")
                        script_result["greedy_max_order_first5"] = \
                            greedy.get("greedy_max_order", [])[:5]
                    elif "length_dist" in jf.name:
                        if isinstance(data, list):
                            script_result["corpus_stats"] = [
                                {k: v for k, v in s.items()
                                 if k in ("label", "n_units", "cv", "H", "mean")}
                                for s in data if "error" not in s
                            ]
                    elif "common_property" in jf.name:
                        conclusions = {}
                        for k, v in data.items():
                            if isinstance(v, dict) and "conclusion" in v:
                                conclusions[k] = v["conclusion"]
                        if conclusions:
                            script_result["sequence_vs_unit"] = conclusions
                    elif "recitation" in jf.name:
                        if "recitation_hypothesis" in data:
                            rh = data["recitation_hypothesis"]
                            script_result["recitation_status"] = rh.get("status")
                            script_result["recitation_note"] = rh.get("note")
                except Exception as e:
                    log.warning(f"  Error leyendo {jf}: {e}")

        results["scripts"].append(script_result)

    # Generate summary
    summary_path = RESULTS_DIR / "fase11_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    log.info(f"\n{'=' * 70}")
    log.info(f"FASE 11 completada en {elapsed:.1f}s")
    log.info(f"Resumen en {summary_path}")

    for script_res in results["scripts"]:
        log.info(f"\n  {script_res['name']}: {script_res['status']}")
        if "top_predictors" in script_res:
            for tp in script_res["top_predictors"]:
                log.info(f"    Top predictor: {tp['feature']}, r={tp['pearson_r']}")
        if "historical_orders" in script_res:
            for k, v in script_res["historical_orders"].items():
                log.info(f"    {v['name']}: H={v['H']}")
        if "sequence_vs_unit" in script_res:
            for k, v in script_res["sequence_vs_unit"].items():
                log.info(f"    {k}: {v}")


if __name__ == "__main__":
    main()
