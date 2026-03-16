#!/usr/bin/env python3
"""
Fase 9 — Orchestrator
Lanza 4 scripts simultáneamente y genera resumen consolidado.
"""

import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent
LOG_DIR = BASE / "logs"
RESULTS_DIR = BASE / "results"

LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "orchestrator_fase9.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

SCRIPTS = [
    {
        "name": "nt_special_case",
        "file": "nt_special_case.py",
        "question": "¿Por qué el NT tiene H alto pero no estructura MPS?",
    },
    {
        "name": "lxx_vs_mt",
        "file": "lxx_vs_mt.py",
        "question": "¿Dos líneas de transmisión independientes producen el mismo H?",
    },
    {
        "name": "transmission_typology",
        "file": "transmission_typology.py",
        "question": "¿La tipología de transmisión tiene subtipos con propiedades H distintas?",
    },
    {
        "name": "generative_processes",
        "file": "generative_processes.py",
        "question": "¿AT, Corán y Rig Veda son el mismo proceso generativo?",
    },
]


def main():
    t0 = time.time()
    log.info("=" * 70)
    log.info("FASE 9 — ORCHESTRATOR")
    log.info(f"Lanzando {len(SCRIPTS)} scripts simultáneamente")
    log.info("=" * 70)

    python = sys.executable
    processes = {}
    log_files = {}

    # Lanzar todos
    for script in SCRIPTS:
        script_path = BASE / script["file"]
        log_path = LOG_DIR / f"fase9_{script['name']}.log"

        log.info(f"  Lanzando {script['file']}...")

        log_fh = open(log_path, "w")
        proc = subprocess.Popen(
            [python, str(script_path)],
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            cwd=str(BASE),
        )
        processes[script["name"]] = {
            "proc": proc,
            "start_time": time.time(),
            "log_fh": log_fh,
            "log_path": log_path,
        }
        log.info(f"    PID={proc.pid}")

    log.info(f"\nTodos lanzados. Monitoreando cada 30s...")

    # Monitorear
    completed = set()
    results_status = {}

    while len(completed) < len(SCRIPTS):
        time.sleep(30)
        elapsed = time.time() - t0

        for script in SCRIPTS:
            name = script["name"]
            if name in completed:
                continue

            proc_info = processes[name]
            retcode = proc_info["proc"].poll()

            if retcode is not None:
                proc_info["log_fh"].close()
                script_elapsed = time.time() - proc_info["start_time"]
                completed.add(name)

                if retcode == 0:
                    log.info(f"  ✓ {name} completado en {script_elapsed:.1f}s")
                    results_status[name] = {
                        "status": "completed",
                        "elapsed_seconds": script_elapsed,
                        "return_code": retcode,
                    }
                else:
                    log.error(f"  ✗ {name} FALLÓ (retcode={retcode}) en {script_elapsed:.1f}s")
                    # Leer últimas líneas del log
                    try:
                        with open(proc_info["log_path"]) as f:
                            lines = f.readlines()
                        last_lines = "".join(lines[-10:])
                    except:
                        last_lines = "(no log)"
                    results_status[name] = {
                        "status": "failed",
                        "elapsed_seconds": script_elapsed,
                        "return_code": retcode,
                        "last_log_lines": last_lines,
                    }
            else:
                log.info(f"  ⏳ {name} en progreso ({elapsed:.0f}s total)")

    total_elapsed = time.time() - t0
    log.info(f"\n{'=' * 70}")
    log.info(f"Todos los scripts terminaron en {total_elapsed:.1f}s")

    # Generar resumen
    summary = generate_summary(results_status, total_elapsed)

    summary_path = RESULTS_DIR / "fase9_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    log.info(f"Resumen guardado en {summary_path}")

    # Contar éxitos/fallos
    n_ok = sum(1 for v in results_status.values() if v["status"] == "completed")
    n_fail = sum(1 for v in results_status.values() if v["status"] == "failed")
    log.info(f"Resultado: {n_ok} exitosos, {n_fail} fallidos")

    if n_fail > 0:
        for name, info in results_status.items():
            if info["status"] == "failed":
                log.error(f"  FALLIDO: {name}")
                if "last_log_lines" in info:
                    log.error(f"    Últimas líneas:\n{info['last_log_lines']}")


def generate_summary(results_status, total_elapsed):
    """Genera el resumen consolidado de Fase 9."""
    summary = {
        "phase": 9,
        "timestamp": datetime.now().isoformat(),
        "total_elapsed_seconds": total_elapsed,
        "scripts_run": len(SCRIPTS),
        "scripts_succeeded": sum(1 for v in results_status.values() if v["status"] == "completed"),
        "scripts_failed": [k for k, v in results_status.items() if v["status"] == "failed"],
        "investigations": [],
    }

    # 1. NT Special Case
    nt_data = _load_json(RESULTS_DIR / "nt_special" / "chronological_groups.json")
    nt_comp = _load_json(RESULTS_DIR / "nt_special" / "nt_composition_analysis.json")
    nt_trad = _load_json(RESULTS_DIR / "nt_special" / "tradition_comparison.json")

    nt_finding = {
        "name": "nt_special_case",
        "question": "¿Por qué el NT tiene H alto pero no estructura MPS?",
        "status": results_status.get("nt_special_case", {}).get("status", "unknown"),
    }

    if nt_comp:
        nt_finding["key_findings"] = {
            "cv_verse_len": nt_comp.get("coefficient_of_variation_verse_len"),
            "n_genre_transitions": nt_comp.get("n_genre_transitions_canonical"),
            "n_genres": nt_comp.get("n_distinct_genres"),
            "order_test": nt_comp.get("order_test"),
            "hypothesis": nt_comp.get("hypothesis"),
        }
    if nt_data:
        nt_finding["chronological_groups"] = {
            k: {
                "label": v.get("label"),
                "H": v.get("H"),
                "mps_p": v.get("mps_p"),
                "n_verses": v.get("n_verses"),
            }
            for k, v in nt_data.items()
        }
    if nt_trad:
        nt_finding["tagnt_status"] = nt_trad.get("status", "unknown")

    summary["investigations"].append(nt_finding)

    # 2. LXX vs MT
    lxx_data = _load_json(RESULTS_DIR / "lxx_mt" / "lxx_mt_comparison.json")
    lxx_avail = _load_json(RESULTS_DIR / "lxx_mt" / "lxx_availability.json")

    lxx_finding = {
        "name": "lxx_vs_mt",
        "question": "¿Dos líneas de transmisión independientes producen el mismo H?",
        "status": results_status.get("lxx_vs_mt", {}).get("status", "unknown"),
    }

    if lxx_data:
        if isinstance(lxx_data, dict) and "status" in lxx_data:
            lxx_finding["key_findings"] = {
                "lxx_status": lxx_data.get("status"),
                "note": lxx_data.get("note"),
                "mt_pentateuch": lxx_data.get("mt_pentateuch_complete"),
            }
        elif isinstance(lxx_data, list):
            lxx_finding["key_findings"] = {
                "n_books_compared": len(lxx_data),
                "comparisons": lxx_data,
            }

    if lxx_avail:
        lxx_finding["availability"] = [
            {"source": a.get("source"), "cloned": a.get("cloned"),
             "n_files": a.get("total_text_files", 0)}
            for a in (lxx_avail if isinstance(lxx_avail, list) else [])
        ]

    summary["investigations"].append(lxx_finding)

    # 3. Tipología
    typo_cluster = _load_json(RESULTS_DIR / "typology" / "clustering_results.json")
    typo_predict = _load_json(RESULTS_DIR / "typology" / "predictive_dimensions.json")
    typo_refined = _load_json(RESULTS_DIR / "typology" / "refined_typology.json")

    typo_finding = {
        "name": "transmission_typology",
        "question": "¿La tipología tiene subtipos con propiedades H distintas?",
        "status": results_status.get("transmission_typology", {}).get("status", "unknown"),
    }

    if typo_cluster:
        typo_finding["clustering"] = {
            "k2": typo_cluster.get("clusters_k2"),
            "k3": typo_cluster.get("clusters_k3"),
        }
    if typo_predict:
        typo_finding["best_predictor"] = typo_predict.get("best_predictor")
        typo_finding["ranking"] = typo_predict.get("ranking")
    if typo_refined:
        typo_finding["refined_typology"] = typo_refined.get("category_summary")

    summary["investigations"].append(typo_finding)

    # 4. Procesos generativos
    gen_arfima = _load_json(RESULTS_DIR / "generative" / "arfima_parameters.json")
    gen_verdict = _load_json(RESULTS_DIR / "generative" / "verdict.json")
    gen_process = _load_json(RESULTS_DIR / "generative" / "same_process_test.json")

    gen_finding = {
        "name": "generative_processes",
        "question": "¿AT, Corán y Rig Veda son el mismo proceso generativo?",
        "status": results_status.get("generative_processes", {}).get("status", "unknown"),
    }

    if gen_arfima:
        gen_finding["arfima_models"] = {
            k: {"model": v.get("model"), "d": v.get("d_used"), "H": v.get("H_rs")}
            for k, v in gen_arfima.items()
        }
    if gen_verdict:
        gen_finding["verdict"] = {
            "category": gen_verdict.get("category"),
            "description": gen_verdict.get("description"),
        }
    if gen_process:
        gen_finding["same_process_test"] = {
            k: {"frac_both": v.get("frac_both_match"), "verdict": v.get("verdict")}
            for k, v in gen_process.items() if isinstance(v, dict) and "verdict" in v
        }

    summary["investigations"].append(gen_finding)

    # Preguntas abiertas
    summary["open_questions"] = [
        {
            "question": "¿H(LXX) ≈ H(MT) para libros individuales del Pentateuco?",
            "what_would_answer_it": "Un corpus digital de la LXX con segmentación por versículo compatible con WLC/BHSA (ej. CATSS con verse alignment).",
        },
        {
            "question": "¿El Pentateuco Samaritano confirma H5a?",
            "what_would_answer_it": "Corpus digital etiquetado morfológicamente del SP con segmentación por versículo.",
        },
        {
            "question": "¿Los manuscritos pre-uthmanicos del Corán tienen el mismo H?",
            "what_would_answer_it": "Digitalización de los palimpsestos de Ṣanʿāʾ como corpus analizable.",
        },
    ]

    return summary


def _load_json(path):
    """Carga un JSON si existe, devuelve None si no."""
    try:
        if path.exists():
            with open(path) as f:
                return json.load(f)
    except Exception as e:
        log.warning(f"  Error cargando {path}: {e}")
    return None


if __name__ == "__main__":
    main()
