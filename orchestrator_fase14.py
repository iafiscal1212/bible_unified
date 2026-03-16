#!/usr/bin/env python3
"""
Fase 14 — Orchestrator: Cuatro Investigaciones en Paralelo
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
        logging.FileHandler(LOG_DIR / "fase14_orchestrator.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


SCRIPTS = [
    {
        "name": "homeric_vs_vedic",
        "file": "homeric_vs_vedic_recitation.py",
        "question": "¿Qué diferencia la recitación homérica de la védica?",
        "results_dir": "recitation_mechanism",
    },
    {
        "name": "unified_model",
        "file": "unified_model.py",
        "question": "¿Un modelo AR(1)-ARFIMA unifica AT/Corán/Rig Veda?",
        "results_dir": "unified_model",
    },
    {
        "name": "refined_classifier",
        "file": "refined_authenticity_tool.py",
        "question": "¿Cuál es la accuracy del clasificador multi-feature AT vs NT?",
        "results_dir": "refined_classifier",
    },
    {
        "name": "excluded_canon",
        "file": "excluded_canon.py",
        "question": "¿Los textos excluidos del canon difieren en H del NT canónico?",
        "results_dir": "excluded_canon",
    },
]


def main():
    t0 = time.time()
    log.info("=" * 70)
    log.info("FASE 14 — Cuatro Investigaciones en Paralelo")
    log.info("4 scripts en paralelo")
    log.info("=" * 70)

    # Launch all 4 scripts simultaneously
    processes = {}
    log_files = {}

    for script in SCRIPTS:
        log_path = LOG_DIR / f"fase14_{script['name']}.log"
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
        "phase": 14,
        "title": "Cuatro Investigaciones en Paralelo — Fase 14",
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

                    # Script 1: homeric_vs_vedic
                    if "verdict" in fname and "recitation" in str(jf.parent):
                        if isinstance(data, dict):
                            script_result["supported_hypotheses"] = data.get("supported_hypotheses")
                            script_result["best_explanation"] = data.get("best_explanation")

                    elif "cv_comparison" in fname:
                        if isinstance(data, dict):
                            for corpus in ["Homer", "Rig_Veda", "AT_all", "Corán"]:
                                if corpus in data and isinstance(data[corpus], dict):
                                    script_result[f"{corpus}_AC1"] = data[corpus].get("AC1_words")
                                    script_result[f"{corpus}_CV"] = data[corpus].get("CV_words")

                    # Script 2: unified_model
                    elif "fitted_params" in fname:
                        if isinstance(data, dict):
                            for corpus in ["AT", "NT", "Corán", "Rig_Veda"]:
                                if corpus in data and isinstance(data[corpus], dict):
                                    params = data[corpus].get("params", {})
                                    script_result[f"{corpus}_phi"] = params.get("phi")
                                    script_result[f"{corpus}_d"] = params.get("d")

                    elif "retrodiction" in fname and "unified" in str(jf.parent):
                        if isinstance(data, dict):
                            for corpus, vals in data.items():
                                if isinstance(vals, dict):
                                    script_result[f"{corpus}_retrodiction_pct"] = vals.get("match_pct")

                    elif "verdict" in fname and "unified" in str(jf.parent):
                        if isinstance(data, dict):
                            script_result["d_universal"] = data.get("d_universal")
                            script_result["d_range"] = data.get("d_range")

                    # Script 3: refined_classifier
                    elif "classifier_results" in fname:
                        if isinstance(data, dict):
                            best_acc = 0
                            best_name = None
                            for clf_name, clf_res in data.items():
                                if isinstance(clf_res, dict):
                                    acc = clf_res.get("accuracy", 0)
                                    if acc > best_acc:
                                        best_acc = acc
                                        best_name = clf_name
                            script_result["best_classifier"] = best_name
                            script_result["best_accuracy"] = best_acc
                            if best_name and best_name in data:
                                script_result["n_misclassified"] = data[best_name].get("n_misclassified")
                                script_result["misclassified"] = data[best_name].get("misclassified")
                                script_result["top_feature"] = None
                                fi = data[best_name].get("feature_importances")
                                if fi:
                                    script_result["top_feature"] = max(fi.items(), key=lambda x: x[1])[0]

                    elif "verdict" in fname and "classifier" in str(jf.parent):
                        if isinstance(data, dict):
                            script_result["n_ambiguous"] = data.get("n_ambiguous")
                            script_result["ambiguous_books"] = [a["book"] for a in data.get("ambiguous_books", [])]

                    # Script 4: excluded_canon
                    elif "corpus_availability" in fname and "excluded" in str(jf.parent):
                        if isinstance(data, dict):
                            n_found = sum(1 for v in data.values()
                                          if isinstance(v, dict) and v.get("found"))
                            script_result["texts_found"] = n_found

                    elif "excluded_vs_canonical" in fname:
                        if isinstance(data, dict):
                            excluded = data.get("excluded", {})
                            comparison = data.get("comparison", {})
                            for name, stats in excluded.items():
                                if isinstance(stats, dict):
                                    script_result[f"{name}_H"] = stats.get("H")
                                    script_result[f"{name}_AC1"] = stats.get("AC1")
                            for name, comp in comparison.items():
                                if isinstance(comp, dict):
                                    script_result[f"{name}_classification"] = comp.get("classification")

                    elif "verdict" in fname and "excluded" in str(jf.parent):
                        if isinstance(data, dict):
                            script_result["excluded_conclusion"] = data.get("conclusion")
                            script_result["n_NT_like"] = data.get("n_NT_like")

                except Exception as e:
                    log.warning(f"  Error leyendo {jf}: {e}")

        results["scripts"].append(script_result)

    # Save summary
    summary_path = RESULTS_DIR / "fase14_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    log.info(f"\n{'=' * 70}")
    log.info(f"FASE 14 completada en {elapsed:.1f}s")
    log.info(f"Resumen en {summary_path}")

    for sr in results["scripts"]:
        log.info(f"\n  {sr['name']}: {sr['status']} (rc={sr['return_code']})")
        if "best_explanation" in sr:
            log.info(f"    Mejor explicación: {sr['best_explanation']}")
            log.info(f"    Hipótesis soportadas: {sr.get('supported_hypotheses')}")
        if "d_universal" in sr:
            log.info(f"    d universal: {sr['d_universal']} (range={sr.get('d_range')})")
        if "best_classifier" in sr:
            log.info(f"    Mejor clf: {sr['best_classifier']} "
                     f"(acc={sr.get('best_accuracy')}, "
                     f"misclassified={sr.get('n_misclassified')})")
        if "excluded_conclusion" in sr:
            log.info(f"    Conclusión: {sr['excluded_conclusion']}")


if __name__ == "__main__":
    main()
