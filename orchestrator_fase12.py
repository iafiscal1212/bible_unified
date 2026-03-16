#!/usr/bin/env python3
"""
Fase 12 — Orchestrator: ¿Qué produce AC(1) alto? El Mecanismo Composicional
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
        logging.FileHandler(LOG_DIR / "fase12_orchestrator.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


SCRIPTS = [
    {
        "name": "recitation_hypothesis",
        "file": "recitation_hypothesis.py",
        "question": "¿Las restricciones métricas de recitación oral producen AC(1) alto?",
        "results_dir": "recitation",
    },
    {
        "name": "intermediate_scales",
        "file": "intermediate_scales.py",
        "question": "¿Qué produce correlaciones a escalas intermedias que AR(1) no captura?",
        "results_dir": "intermediate",
    },
    {
        "name": "compositional_rule",
        "file": "compositional_rule.py",
        "question": "¿Se puede reconstruir la regla composicional que produce H>0.9?",
        "results_dir": "compositional_rule",
    },
]


def main():
    t0 = time.time()
    log.info("=" * 70)
    log.info("FASE 12 — ¿Qué produce AC(1) alto? El Mecanismo Composicional")
    log.info("3 scripts en paralelo")
    log.info("=" * 70)

    # Launch all 3 scripts simultaneously
    processes = {}
    log_files = {}

    for script in SCRIPTS:
        log_path = LOG_DIR / f"fase12_{script['name']}.log"
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
        "phase": 12,
        "title": "¿Qué produce AC(1) alto? El Mecanismo Composicional",
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

            # Extract key results
            for jf in json_files:
                try:
                    with open(jf, "r") as f:
                        data = json.load(f)

                    if "verdict" in jf.name:
                        if isinstance(data, dict):
                            script_result["verdict"] = data.get("conclusion")
                            script_result["evidence_summary"] = [
                                f"{e['test']}: {e['direction']}"
                                for e in data.get("evidence", [])
                            ]

                    elif "model_comparison" in jf.name:
                        if isinstance(data, dict):
                            script_result["models"] = {
                                k: {"H": v.get("H_mean"), "AC1": v.get("AC1_mean"),
                                    "retrodiction": v.get("retrodiction_pct"),
                                    "n_params": v.get("n_params")}
                                for k, v in data.items()
                            }

                    elif "power_law" in jf.name:
                        if isinstance(data, dict):
                            script_result["power_law_betas"] = {
                                k: v.get("beta") for k, v in data.items()
                            }

                    elif "hmm_fit" in jf.name:
                        if isinstance(data, dict):
                            script_result["hmm_summary"] = {
                                k: {"mu": v.get("mu"),
                                    "mean_regime_len": v.get("mean_regime_length"),
                                    "H_hmm_synth": v.get("H_from_hmm_synthetic", {}).get("mean")}
                                for k, v in data.items()
                                if isinstance(v, dict) and "mu" in v
                            }

                    elif "retrodiction" in jf.name and "best" not in jf.name:
                        if isinstance(data, dict):
                            script_result["retrodiction"] = {
                                "model": data.get("model"),
                                "all_match_pct": data.get("all_match_pct"),
                                "H_match_pct": data.get("H_match_pct"),
                            }

                    elif "ot_poetry" in jf.name:
                        if isinstance(data, dict):
                            for genre in ["poetry", "prose", "prophetic"]:
                                if genre in data:
                                    g = data[genre]
                                    script_result[f"at_{genre}_ac1"] = g.get("ac1_words")

                    elif "block_scale" in jf.name:
                        if isinstance(data, dict):
                            for label in ["AT", "Corán"]:
                                if label in data:
                                    script_result[f"w_star_{label}"] = data[label].get("w_star")

                except Exception as e:
                    log.warning(f"  Error leyendo {jf}: {e}")

        results["scripts"].append(script_result)

    # Save summary
    summary_path = RESULTS_DIR / "fase12_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    log.info(f"\n{'=' * 70}")
    log.info(f"FASE 12 completada en {elapsed:.1f}s")
    log.info(f"Resumen en {summary_path}")

    for sr in results["scripts"]:
        log.info(f"\n  {sr['name']}: {sr['status']}")
        if "verdict" in sr:
            log.info(f"    Veredicto: {sr['verdict']}")
        if "models" in sr:
            for m, v in sr["models"].items():
                log.info(f"    {m}: H={v.get('H')}, retro={v.get('retrodiction')}%")
        if "at_poetry_ac1" in sr:
            log.info(f"    AT poesía AC(1)={sr['at_poetry_ac1']}, "
                     f"prosa AC(1)={sr.get('at_prose_ac1')}")


if __name__ == "__main__":
    main()
