#!/usr/bin/env python3
"""
Fase 15 — Orchestrator: Tres Investigaciones en Paralelo
Lanza 3 scripts simultáneamente con subprocess.Popen.
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
        logging.FileHandler(LOG_DIR / "fase15_orchestrator.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


SCRIPTS = [
    {
        "name": "pos_entropy_mechanism",
        "file": "pos_entropy_mechanism.py",
        "question": "¿Qué propiedad composicional produce alta/baja entropía de POS?",
        "results_dir": "pos_entropy",
    },
    {
        "name": "daniel_analysis",
        "file": "daniel_analysis.py",
        "question": "¿Por qué Daniel es el único libro AT fronterizo (P_AT=0.567)?",
        "results_dir": "daniel",
    },
    {
        "name": "d_parameter",
        "file": "d_parameter_interpretation.py",
        "question": "¿Qué significa el parámetro d del modelo ARFIMA compositivamente?",
        "results_dir": "d_parameter",
    },
]


def main():
    t0 = time.time()
    log.info("=" * 70)
    log.info("FASE 15 — Tres Investigaciones en Paralelo")
    log.info("3 scripts en paralelo")
    log.info("=" * 70)

    processes = {}
    log_files = {}

    for script in SCRIPTS:
        log_path = LOG_DIR / f"fase15_{script['name']}.log"
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

    for lf in log_files.values():
        lf.close()

    elapsed = time.time() - t0
    log.info(f"\n  Todos los scripts completados en {elapsed:.1f}s")

    results = {
        "phase": 15,
        "title": "Tres Investigaciones en Paralelo — Fase 15",
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

            for jf in json_files:
                try:
                    with open(jf, "r") as f:
                        data = json.load(f)

                    fname = jf.name

                    # Script 1: pos_entropy
                    if "feature_correlations" in fname:
                        if isinstance(data, dict):
                            sc = data.get("strongest_correlate", {})
                            script_result["strongest_correlate"] = sc.get("feature")
                            script_result["strongest_r"] = sc.get("r")
                            at_nt = data.get("AT_vs_NT_pos_entropy", {})
                            script_result["AT_pe_mean"] = at_nt.get("AT_mean")
                            script_result["NT_pe_mean"] = at_nt.get("NT_mean")
                            script_result["AT_lower_pe"] = at_nt.get("AT_lower")

                    elif "mechanistic_interpretation" in fname:
                        if isinstance(data, dict):
                            script_result["cause_or_correlate"] = data.get("cause_or_correlate")
                            script_result["direction"] = data.get("direction")

                    elif "genre_breakdown" in fname:
                        if isinstance(data, dict):
                            anova = data.get("anova", {})
                            script_result["genre_anova_p"] = anova.get("p_value")
                            script_result["genre_explains_pe"] = anova.get("genre_explains_pe")

                    # Script 2: daniel
                    elif "hebrew_vs_aramaic" in fname:
                        if isinstance(data, dict):
                            heb = data.get("hebrew", {})
                            ara = data.get("aramaic", {})
                            script_result["daniel_heb_pe"] = heb.get("pos_entropy")
                            script_result["daniel_ara_pe"] = ara.get("pos_entropy")
                            script_result["daniel_heb_H"] = heb.get("H")
                            script_result["daniel_ara_H"] = ara.get("H")

                    elif "verdict" in fname and "daniel" in str(jf.parent):
                        if isinstance(data, dict):
                            script_result["primary_factor"] = data.get("primary_factor")
                            script_result["factors"] = data.get("factors_identified")
                            script_result["daniel_summary"] = data.get("summary")

                    elif "dating_evidence" in fname:
                        if isinstance(data, dict):
                            script_result["closer_by_H"] = data.get("closer_by_H")
                            script_result["closer_by_pe"] = data.get("closer_by_pe")

                    # Script 3: d_parameter
                    elif "compositional_interpretation" in fname:
                        if isinstance(data, dict):
                            script_result["d_interpretation"] = (
                                data.get("interpretation", "")[:200]
                            )
                            ca = data.get("cluster_analysis", {})
                            script_result["d_separates_clusters"] = ca.get("d_separates_clusters")

                    elif "geometric_interpretation" in fname:
                        if isinstance(data, dict):
                            for corpus in ["AT", "NT", "Corán", "Rig_Veda"]:
                                if corpus in data:
                                    script_result[f"{corpus}_residual"] = data[corpus].get("residual")

                    elif "topic_modeling" in fname:
                        if isinstance(data, dict):
                            bc = data.get("best_correlation", {})
                            script_result["d_vs_topics_r"] = bc.get("d_vs_n_eff_r") if bc else None

                except Exception as e:
                    log.warning(f"  Error leyendo {jf}: {e}")

        results["scripts"].append(script_result)

    summary_path = RESULTS_DIR / "fase15_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    log.info(f"\n{'=' * 70}")
    log.info(f"FASE 15 completada en {elapsed:.1f}s")
    log.info(f"Resumen en {summary_path}")

    for sr in results["scripts"]:
        log.info(f"\n  {sr['name']}: {sr['status']} (rc={sr['return_code']})")
        if "strongest_correlate" in sr:
            log.info(f"    Correlato más fuerte: {sr['strongest_correlate']} "
                     f"(r={sr.get('strongest_r')})")
            log.info(f"    AT pe={sr.get('AT_pe_mean')}, NT pe={sr.get('NT_pe_mean')}")
        if "primary_factor" in sr:
            log.info(f"    Factor primario: {sr['primary_factor']}")
            log.info(f"    Daniel Heb pe={sr.get('daniel_heb_pe')}, "
                     f"Ara pe={sr.get('daniel_ara_pe')}")
        if "d_interpretation" in sr:
            log.info(f"    d: {sr['d_interpretation'][:100]}...")


if __name__ == "__main__":
    main()
