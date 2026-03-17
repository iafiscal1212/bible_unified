#!/usr/bin/env python3
"""
Fase 16 — Orchestrator: Cuatro Investigaciones en Paralelo
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
        logging.FileHandler(LOG_DIR / "fase16_orchestrator.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


SCRIPTS = [
    {
        "name": "phi_mechanism",
        "file": "phi_mechanism.py",
        "question": "¿Qué produce φ alto (coherencia local / AC1)?",
        "results_dir": "phi_mechanism",
    },
    {
        "name": "book_of_mormon",
        "file": "book_of_mormon.py",
        "question": "¿Cómo se compara el Book of Mormon (s.XIX) con textos antiguos?",
        "results_dir": "bom",
    },
    {
        "name": "unavailable_corpora",
        "file": "unavailable_corpora.py",
        "question": "¿Qué corpora faltan y cuál sería su impacto?",
        "results_dir": "unavailable_corpora",
    },
    {
        "name": "phi_vs_d_space",
        "file": "phi_vs_d_space.py",
        "question": "¿Puede el espacio (φ, d) clasificar AT vs NT a nivel de libro?",
        "results_dir": "phi_d_space",
    },
]


def main():
    t0 = time.time()
    log.info("=" * 70)
    log.info("FASE 16 — Cuatro Investigaciones en Paralelo")
    log.info("4 scripts en paralelo")
    log.info("=" * 70)

    processes = {}
    log_files = {}

    for script in SCRIPTS:
        log_path = LOG_DIR / f"fase16_{script['name']}.log"
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

    # Polling loop
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

    # Build summary
    results = {
        "phase": 16,
        "title": "Cuatro Investigaciones en Paralelo — Fase 16",
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

                    # Script 1: phi_mechanism
                    if "phi_mechanism_verdict" in fname:
                        if isinstance(data, dict):
                            script_result["n_mechanisms_tested"] = data.get("n_mechanisms_tested")
                            script_result["n_supported"] = data.get("n_supported")
                            script_result["primary_mechanisms"] = data.get("primary_mechanisms")
                            script_result["summary"] = data.get("summary")

                    elif "transition_distributions" in fname:
                        if isinstance(data, dict):
                            by_genre = data.get("by_genre", {})
                            for genre in ["liturgical", "narrative"]:
                                if genre in by_genre:
                                    script_result[f"{genre}_mean_delta"] = by_genre[genre].get("mean_delta")
                            test = by_genre.get("liturgical_vs_narrative_test", {})
                            script_result["liturgical_lower_p"] = test.get("p_value")

                    # Script 2: book_of_mormon
                    elif "bom_metrics" in fname:
                        if isinstance(data, dict):
                            script_result["bom_n_verses"] = data.get("n_verses")
                            script_result["bom_H"] = data.get("H")
                            script_result["bom_AC1"] = data.get("AC1")
                            script_result["bom_DFA"] = data.get("DFA")
                            script_result["bom_CV"] = data.get("CV")

                    elif "english_comparison" in fname:
                        if isinstance(data, dict):
                            for corpus in ["OT_eng", "NT_eng", "BOM"]:
                                if corpus in data:
                                    script_result[f"{corpus}_H"] = data[corpus].get("H")
                                    script_result[f"{corpus}_AC1"] = data[corpus].get("AC1")

                    elif "classifier_result" in fname:
                        if isinstance(data, dict):
                            eng = data.get("english_comparison", {})
                            script_result["H_closer_to"] = eng.get("H_closer_to")
                            script_result["AC1_closer_to"] = eng.get("AC1_closer_to")

                    # Script 3: unavailable_corpora
                    elif "systematic_gap_analysis" in fname:
                        if isinstance(data, dict):
                            script_result["n_corpora_documented"] = data.get("n_corpora")
                            summary = data.get("summary", {})
                            script_result["key_gap"] = summary.get("key_gap", "")[:200]
                            script_result["priority_1"] = summary.get("priority_1")

                    # Script 4: phi_vs_d_space
                    elif "decision_boundary" in fname:
                        if isinstance(data, dict):
                            script_result["svm_equation"] = data.get("equation")
                            script_result["train_accuracy"] = data.get("training_accuracy")
                            script_result["loo_accuracy"] = data.get("loo_accuracy")

                    elif "books_near_boundary" in fname:
                        if isinstance(data, dict):
                            script_result["n_misclassified"] = data.get("n_misclassified")
                            daniel = data.get("daniel", {})
                            script_result["daniel_rank"] = daniel.get("rank_by_proximity")
                            script_result["daniel_in_top_5"] = daniel.get("in_top_5")
                            top5 = data.get("top_10_near_boundary", [])[:5]
                            script_result["top_5_boundary"] = [
                                {"book": b["book"], "dist": b["distance_to_boundary"]}
                                for b in top5
                            ]

                except Exception as e:
                    log.warning(f"  Error leyendo {jf}: {e}")

        results["scripts"].append(script_result)

    summary_path = RESULTS_DIR / "fase16_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    log.info(f"\n{'=' * 70}")
    log.info(f"FASE 16 completada en {elapsed:.1f}s")
    log.info(f"Resumen en {summary_path}")

    for sr in results["scripts"]:
        log.info(f"\n  {sr['name']}: {sr['status']} (rc={sr['return_code']})")
        if "summary" in sr:
            log.info(f"    {sr.get('summary', '')[:120]}")
        if "bom_n_verses" in sr:
            log.info(f"    BOM: {sr['bom_n_verses']} verses, H={sr.get('bom_H')}, AC1={sr.get('bom_AC1')}")
        if "n_corpora_documented" in sr:
            log.info(f"    {sr['n_corpora_documented']} corpora documented")
        if "loo_accuracy" in sr:
            log.info(f"    SVM LOO accuracy: {sr['loo_accuracy']}")


if __name__ == "__main__":
    main()
