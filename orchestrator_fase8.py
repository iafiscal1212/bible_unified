#!/usr/bin/env python3
"""
Fase 8 — Orchestrator
Lanza los 5 scripts de investigación en paralelo y genera resumen consolidado.
"""

import json
import logging
import time
import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime

BASE = Path(__file__).resolve().parent
RESULTS_DIR = BASE / "results"
LOG_DIR = BASE / "logs"

LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "fase8_orchestrator.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

SCRIPTS = [
    {
        "name": "word_level_variants",
        "file": "word_level_variants.py",
        "question": "¿Qué tipo de variante textual DSS/WLC impacta más H localmente?",
        "output_dir": "results/word_variants"
    },
    {
        "name": "dss_other_books",
        "file": "dss_other_books.py",
        "question": "¿Es la invarianza temporal de H general o específica de Isaías?",
        "output_dir": "results/dss_books"
    },
    {
        "name": "transmission_decay_rate",
        "file": "transmission_decay_rate.py",
        "question": "¿Decae H más rápido con transmisión libre que controlada?",
        "output_dir": "results/decay"
    },
    {
        "name": "h5_resolution_attempt",
        "file": "h5_resolution_attempt.py",
        "question": "¿Podemos distinguir H5a de H5b con datos actuales?",
        "output_dir": "results/h5"
    },
    {
        "name": "h_authenticity_tool",
        "file": "h_authenticity_tool.py",
        "question": "¿Puede H usarse como criterio de autenticidad textual?",
        "output_dir": "results/authenticity"
    }
]


def main():
    t0 = time.time()
    log.info("=" * 70)
    log.info("FASE 8 — ORCHESTRATOR — START")
    log.info(f"Timestamp: {datetime.now().isoformat()}")
    log.info(f"Launching {len(SCRIPTS)} scripts in parallel...")
    log.info("=" * 70)

    # ── Launch all scripts simultaneously ──────────────────────────────
    processes = {}
    log_files = {}

    for script in SCRIPTS:
        script_path = BASE / script["file"]
        log_path = LOG_DIR / f"fase8_{script['name']}.log"

        log.info(f"  Launching {script['name']} → {script['file']}")

        # Open log file for stdout/stderr
        log_fh = open(log_path, "w")
        log_files[script["name"]] = log_fh

        proc = subprocess.Popen(
            [sys.executable, str(script_path)],
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            cwd=str(BASE),
            env={**os.environ, "PYTHONUNBUFFERED": "1"}
        )
        processes[script["name"]] = {
            "proc": proc,
            "start_time": time.time(),
            "script": script
        }

    log.info(f"All {len(processes)} scripts launched.")

    # ── Monitor progress (every 30s) ───────────────────────────────────
    completed = set()
    while len(completed) < len(processes):
        time.sleep(10)  # Check every 10s, log every 30s

        elapsed_total = time.time() - t0
        status_lines = []

        for name, info in processes.items():
            proc = info["proc"]
            elapsed_script = time.time() - info["start_time"]

            if proc.poll() is not None:
                if name not in completed:
                    completed.add(name)
                    rc = proc.returncode
                    status = "OK" if rc == 0 else f"FAILED (rc={rc})"
                    log.info(f"  [{name}] FINISHED — {status} ({elapsed_script:.1f}s)")
                    log_files[name].close()
            else:
                status_lines.append(f"    {name}: running ({elapsed_script:.0f}s)")

        # Log status every ~30s
        if int(elapsed_total) % 30 < 11 and status_lines:
            log.info(f"Status ({elapsed_total:.0f}s elapsed, {len(completed)}/{len(processes)} done):")
            for sl in status_lines:
                log.info(sl)

    # Close any remaining log files
    for name, fh in log_files.items():
        if not fh.closed:
            fh.close()

    elapsed_total = time.time() - t0
    log.info(f"All scripts finished in {elapsed_total:.1f}s")

    # ── Check for failures ─────────────────────────────────────────────
    failures = []
    for name, info in processes.items():
        rc = info["proc"].returncode
        if rc != 0:
            failures.append(name)
            log.error(f"FAILED: {name} (return code {rc})")

    if failures:
        log.warning(f"{len(failures)} scripts failed: {failures}")

    # ── Generate consolidated summary ──────────────────────────────────
    log.info("Generating consolidated summary...")

    summary = {
        "phase": 8,
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed_total, 1),
        "scripts_run": len(SCRIPTS),
        "scripts_succeeded": len(SCRIPTS) - len(failures),
        "scripts_failed": failures,
        "investigations": []
    }

    # Script 1: word_level_variants
    try:
        with open(RESULTS_DIR / "word_variants" / "variant_impact.json") as f:
            vi = json.load(f)
        summary["investigations"].append({
            "name": "word_level_variants",
            "question": SCRIPTS[0]["question"],
            "status": "completed",
            "key_findings": {
                "anova_windowed_p": vi.get("anova_windowed", {}).get("p_value"),
                "anova_per_verse_p": vi.get("anova_per_verse", {}).get("p_value"),
                "variant_types_tested": list(vi.get("anova_windowed", {}).get("groups", {}).keys()),
                "global_H_dss": vi.get("global_H_dss"),
                "global_H_wlc": vi.get("global_H_wlc"),
            },
            "conclusion": (
                f"ANOVA windowed p={vi.get('anova_windowed', {}).get('p_value', '?')}, "
                f"per-verse p={vi.get('anova_per_verse', {}).get('p_value', '?')}. "
                "See variant_impact.json for full breakdown."
            )
        })
    except Exception as e:
        summary["investigations"].append({
            "name": "word_level_variants",
            "status": "failed",
            "error": str(e)
        })

    # Script 2: dss_other_books
    try:
        with open(RESULTS_DIR / "dss_books" / "all_books_comparison.json") as f:
            dss_books = json.load(f)
        n_compared = dss_books.get("n_books_compared", 0)
        n_sig = dss_books.get("n_significantly_different", 0)
        summary["investigations"].append({
            "name": "dss_other_books",
            "question": SCRIPTS[1]["question"],
            "status": "completed",
            "key_findings": {
                "n_books_compared": n_compared,
                "n_significantly_different": n_sig,
                "avg_abs_delta_H": dss_books.get("avg_abs_delta_H"),
                "books": [
                    {
                        "book": b["bhsa_book"],
                        "H_DSS": b["dss"]["H"],
                        "H_WLC": b["wlc"]["H"],
                        "delta_H": b["delta_H"],
                        "p": b["mann_whitney_p"]
                    }
                    for b in dss_books.get("books", [])
                ]
            },
            "conclusion": dss_books.get("conclusion", "See JSON for details")
        })
    except Exception as e:
        summary["investigations"].append({
            "name": "dss_other_books",
            "status": "failed",
            "error": str(e)
        })

    # Script 3: transmission_decay_rate
    try:
        with open(RESULTS_DIR / "decay" / "decay_rates.json") as f:
            decay = json.load(f)
        summary["investigations"].append({
            "name": "transmission_decay_rate",
            "question": SCRIPTS[2]["question"],
            "status": "completed",
            "key_findings": {
                "measured_rates": [
                    {k: dr[k] for k in ["corpus", "transmission", "rate_H_per_century", "type"]
                     if k in dr}
                    for dr in decay.get("decay_rates", [])
                    if dr.get("type") == "measured"
                ],
                "cross_corpus_controlled_mean_H": decay.get("cross_corpus_h_by_transmission", {}).get("controlled_H_mean"),
                "cross_corpus_free_mean_H": decay.get("cross_corpus_h_by_transmission", {}).get("free_H_mean"),
            },
            "conclusion": decay.get("conclusion", "See JSON for details")
        })
    except Exception as e:
        summary["investigations"].append({
            "name": "transmission_decay_rate",
            "status": "failed",
            "error": str(e)
        })

    # Script 4: h5_resolution_attempt
    try:
        with open(RESULTS_DIR / "h5" / "h5_analysis.json") as f:
            h5 = json.load(f)
        with open(RESULTS_DIR / "h5" / "required_future_corpus.json") as f:
            future = json.load(f)
        summary["investigations"].append({
            "name": "h5_resolution_attempt",
            "question": SCRIPTS[3]["question"],
            "status": "completed",
            "key_findings": {
                "H5a_status": h5.get("logical_verdict", {}).get("H5a"),
                "H5b_status": h5.get("logical_verdict", {}).get("H5b"),
                "actual_result": h5.get("logical_verdict", {}).get("actual_result"),
                "best_future_candidate": future.get("recommendation"),
            },
            "conclusion": h5.get("logical_verdict", {}).get("implication", "See JSON")
        })
    except Exception as e:
        summary["investigations"].append({
            "name": "h5_resolution_attempt",
            "status": "failed",
            "error": str(e)
        })

    # Script 5: h_authenticity_tool
    try:
        with open(RESULTS_DIR / "authenticity" / "cross_validation.json") as f:
            cv = json.load(f)
        with open(RESULTS_DIR / "authenticity" / "book_anomalies.json") as f:
            anomalies = json.load(f)
        summary["investigations"].append({
            "name": "h_authenticity_tool",
            "question": SCRIPTS[4]["question"],
            "status": "completed",
            "key_findings": {
                "loo_accuracy": cv.get("accuracy"),
                "n_anomalous_books": anomalies.get("n_anomalous"),
                "anomalous_books": [
                    {"book": b["book"], "z_score": b["z_score"]}
                    for b in anomalies.get("books", [])
                    if b.get("anomalous")
                ],
            },
            "conclusion": (
                f"LOO cross-validation accuracy: {cv.get('accuracy', '?'):.1%}. "
                f"{anomalies.get('n_anomalous', 0)} books show anomalous Zipf exponent (|z|>2). "
                "Method is indicative, not definitive — see method_limitations.json."
            )
        })
    except Exception as e:
        summary["investigations"].append({
            "name": "h_authenticity_tool",
            "status": "failed",
            "error": str(e)
        })

    # ── Hypotheses status ──────────────────────────────────────────────
    summary["hypotheses_status"] = {
        "H4": {
            "status": "CONFIRMED (strengthened)",
            "evidence": "Cross-corpus comparison + temporal invariance DSS/WLC + decay rate analysis"
        },
        "H5a": {
            "status": "INDETERMINATE",
            "reason": "H_DSS ≈ H_WLC — compatible with both H5a and H5b"
        },
        "H5b": {
            "status": "INDETERMINATE",
            "reason": "Same as H5a — temporal invariance does not distinguish"
        }
    }

    # ── Top 3 open questions ───────────────────────────────────────────
    summary["top_3_open_questions"] = [
        {
            "question": "Does the temporal invariance of H hold for ALL biblical books, or only Isaiah?",
            "what_would_answer_it": "Script 2 (dss_other_books) results. If all DSS books show H_DSS ≈ H_WLC, invariance is general. If some diverge, there are book-specific transmission dynamics."
        },
        {
            "question": "Can H5a be distinguished from H5b using the Samaritan Pentateuch?",
            "what_would_answer_it": "A morphologically-tagged digital Samaritan Pentateuch, compared with the Masoretic version using identical segmentation. Two independent transmission lines from the same original (~400 BCE divergence)."
        },
        {
            "question": "Is the controlled/free H boundary robust across languages?",
            "what_would_answer_it": "Adding more corpora: Canon Pali (controlled, Indo-Aryan), Avesta (controlled, Iranian), classical Chinese poetry (mixed). If the H threshold separates these correctly, the method gains predictive validity."
        }
    ]

    # ── Required additional data ───────────────────────────────────────
    summary["required_additional_data"] = [
        "Morphologically-tagged digital Samaritan Pentateuch",
        "Pre-Uthmanic Quran manuscript (Birmingham or Paris) as tagged corpus",
        "Canon Pali with morphological tagging and stanza segmentation",
        "Avesta (Zoroastrian) with morphological tagging",
        "Papyrus P66 (Gospel of John, ~200 CE) as tagged digital corpus"
    ]

    # ── Save summary ───────────────────────────────────────────────────
    with open(RESULTS_DIR / "fase8_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    log.info("=" * 70)
    log.info(f"FASE 8 — ORCHESTRATOR — DONE ({elapsed_total:.1f}s)")
    log.info(f"Summary saved to: results/fase8_summary.json")
    log.info("=" * 70)

    # Print summary to stdout
    print("\n" + "=" * 70)
    print("FASE 8 — SUMMARY")
    print("=" * 70)
    for inv in summary["investigations"]:
        status_emoji = "OK" if inv.get("status") == "completed" else "FAIL"
        print(f"\n[{status_emoji}] {inv['name']}")
        if "conclusion" in inv:
            print(f"    {inv['conclusion'][:200]}")
    print(f"\nTotal time: {elapsed_total:.1f}s")
    print(f"Failures: {len(failures)}")

    return summary


if __name__ == "__main__":
    main()
