#!/usr/bin/env python3
"""
Fase 17 — Orchestrator: Libro de los Muertos + Canon Pali
Lanza 2 scripts simultáneamente con subprocess.Popen.
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
        logging.FileHandler(LOG_DIR / "fase17_orchestrator.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


SCRIPTS = [
    {
        "name": "book_of_dead",
        "file": "book_of_dead.py",
        "question": "¿El Libro de los Muertos (litúrgico sin revelación) es AT-like o NT-like?",
        "results_dir": "book_of_dead",
    },
    {
        "name": "pali_canon",
        "file": "pali_canon.py",
        "question": "¿El Canon Pali (oral controlado, autoridad humana) es AT-like o NT-like?",
        "results_dir": "pali_canon",
    },
]


def build_typology_table(results):
    """Build updated typology table with all corpora."""
    table = {
        "AT": {
            "transmission": "controlled scribal (1000+ years)",
            "authority": "divine revelation",
            "function": "liturgical + legal + narrative",
            "language_family": "Afroasiatic (Semitic)",
            "H_range": "high (0.88)",
            "AC1_range": "high (0.34)",
        },
        "NT": {
            "transmission": "rapid copying (decades-centuries)",
            "authority": "apostolic witness",
            "function": "epistolary + narrative + apocalyptic",
            "language_family": "Indo-European (Greek)",
            "H_range": "moderate (0.78)",
            "AC1_range": "low (0.09)",
        },
        "Corán": {
            "transmission": "controlled oral then scribal",
            "authority": "divine revelation (direct)",
            "function": "liturgical + legal",
            "language_family": "Afroasiatic (Semitic)",
            "H_range": "high (0.90)",
            "AC1_range": "very high (0.47)",
        },
        "Rig_Veda": {
            "transmission": "oral controlled (millenia)",
            "authority": "divine/inspired (ṛṣi)",
            "function": "liturgical/hymnic",
            "language_family": "Indo-European (Indo-Aryan)",
            "H_range": "low (0.56)",
            "AC1_range": "none (-0.02)",
        },
    }

    # Add new corpora from results
    for script in results.get("scripts", []):
        if script["name"] == "book_of_dead" and script.get("status") == "completed":
            table["Book_of_Dead"] = {
                "transmission": "controlled scribal (~1500 years)",
                "authority": "priestly/ritual (no divine revelation)",
                "function": "funerary liturgical",
                "language_family": "Afroasiatic (Egyptian, non-Semitic)",
                "H": script.get("bod_H"),
                "AC1": script.get("bod_AC1"),
                "classified_as": script.get("bod_predicted"),
                "P_AT": script.get("bod_P_AT"),
                "nearest_corpus": script.get("bod_nearest"),
                "typology_cell": "liturgical controlled WITHOUT revelation",
            }

        if script["name"] == "pali_canon" and script.get("status") == "completed":
            table["Pali_Canon"] = {
                "transmission": "oral controlled then scribal (~400 years oral)",
                "authority": "human (Buddha, not divine)",
                "function": "didactic/doctrinal",
                "language_family": "Indo-European (Indo-Aryan)",
                "H": script.get("pali_H"),
                "AC1": script.get("pali_AC1"),
                "classified_as": script.get("pali_predicted"),
                "P_AT": script.get("pali_P_AT"),
                "nearest_corpus": script.get("pali_nearest"),
                "typology_cell": "oral controlled with HUMAN authority",
            }

    return table


def main():
    t0 = time.time()
    log.info("=" * 70)
    log.info("FASE 17 — Libro de los Muertos + Canon Pali")
    log.info("2 scripts en paralelo")
    log.info("=" * 70)

    processes = {}
    log_files = {}

    for script in SCRIPTS:
        log_path = LOG_DIR / f"fase17_{script['name']}.log"
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

    for lf in log_files.values():
        lf.close()

    elapsed = time.time() - t0
    log.info(f"\n  Todos los scripts completados en {elapsed:.1f}s")

    # Build summary
    results = {
        "phase": 17,
        "title": "Libro de los Muertos + Canon Pali — Fase 17",
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

                    # Book of Dead
                    if script["name"] == "book_of_dead":
                        if "corpus_metrics" in fname:
                            if isinstance(data, dict) and "error" not in data:
                                script_result["bod_n_texts"] = data.get("n_texts")
                                script_result["bod_n_sentences"] = data.get("n_sentences")
                                script_result["bod_n_tokens"] = data.get("n_tokens")
                                script_result["bod_H"] = data.get("H")
                                script_result["bod_AC1"] = data.get("AC1")
                                script_result["bod_DFA"] = data.get("DFA")
                                script_result["bod_CV"] = data.get("CV")
                                script_result["bod_mean"] = data.get("mean_verse_len")

                        elif "classifier_result" in fname:
                            if isinstance(data, dict):
                                script_result["bod_predicted"] = data.get("predicted_class")
                                script_result["bod_P_AT"] = data.get("P_AT")
                                script_result["bod_P_NT"] = data.get("P_NT")

                        elif "phi_d_placement" in fname:
                            if isinstance(data, dict):
                                script_result["bod_nearest"] = data.get("nearest_corpus")
                                script_result["bod_d_approx"] = data.get("d_approx")

                        elif "pos_distribution" in fname:
                            if isinstance(data, dict):
                                script_result["bod_pos_entropy"] = data.get("pos_entropy")

                        elif "comparison_with_psalms" in fname:
                            if isinstance(data, dict):
                                sim = data.get("similarity_within_1sd", {})
                                script_result["bod_similar_to_psalms"] = all(
                                    s.get("similar", False) for s in sim.values()
                                ) if sim else None

                    # Pali Canon
                    elif script["name"] == "pali_canon":
                        if "combined_metrics" in fname:
                            if isinstance(data, dict) and "error" not in data:
                                script_result["pali_n_segments"] = data.get("n_segments")
                                script_result["pali_n_words"] = data.get("n_words")
                                script_result["pali_H"] = data.get("H")
                                script_result["pali_AC1"] = data.get("AC1")
                                script_result["pali_DFA"] = data.get("DFA")
                                script_result["pali_CV"] = data.get("CV")
                                script_result["pali_mean"] = data.get("mean_verse_len")

                        elif "classifier_result" in fname:
                            if isinstance(data, dict):
                                script_result["pali_predicted"] = data.get("predicted_class")
                                script_result["pali_P_AT"] = data.get("P_AT")
                                script_result["pali_P_NT"] = data.get("P_NT")
                                pc = data.get("per_collection", {})
                                for cn in ["DN", "MN"]:
                                    if cn in pc:
                                        script_result[f"pali_{cn}_predicted"] = pc[cn].get("predicted")
                                        script_result[f"pali_{cn}_P_AT"] = pc[cn].get("P_AT")

                        elif "phi_d_placement" in fname and "pali" in str(jf.parent):
                            if isinstance(data, dict):
                                script_result["pali_nearest"] = data.get("nearest_corpus")
                                script_result["pali_d_approx"] = data.get("d_approx")

                        elif "comparison_rigveda" in fname:
                            if isinstance(data, dict):
                                script_result["pali_vs_rv_similar"] = data.get("n_similar_metrics", 0)
                                script_result["pali_vs_rv_interpretation"] = (
                                    data.get("interpretation", "")[:200]
                                )

                        elif "comparison_nt" in fname:
                            if isinstance(data, dict):
                                script_result["pali_H_vs_pauline_z"] = data.get("H_z_score")
                                script_result["pali_H_within_pauline"] = data.get("H_within_2sd")

                except Exception as e:
                    log.warning(f"  Error leyendo {jf}: {e}")

        results["scripts"].append(script_result)

    # ── Typology table ────────────────────────────────────────────────
    typology = build_typology_table(results)
    results["typology_table"] = typology

    # ── Synthesis ─────────────────────────────────────────────────────
    synthesis = {
        "questions_answered": [],
        "implications": [],
    }

    for sr in results["scripts"]:
        if sr["name"] == "book_of_dead" and sr.get("bod_predicted"):
            pred = sr["bod_predicted"]
            p_at = sr.get("bod_P_AT", 0)
            synthesis["questions_answered"].append({
                "question": "¿El Libro de los Muertos (litúrgico sin revelación) es AT-like?",
                "answer": (
                    f"Classified as {pred}-like (P_AT={p_at}). "
                    f"{'Liturgical function + controlled transmission ARE sufficient without revelation.' if pred == 'AT' else 'Revelation adds something beyond liturgical function.'}"
                ),
            })

        if sr["name"] == "pali_canon" and sr.get("pali_predicted"):
            pred = sr["pali_predicted"]
            p_at = sr.get("pali_P_AT", 0)
            synthesis["questions_answered"].append({
                "question": "¿El Canon Pali (oral controlado, autoridad humana) es AT-like?",
                "answer": (
                    f"Classified as {pred}-like (P_AT={p_at}). "
                    f"{'Oral controlled transmission with religious authority IS sufficient.' if pred == 'AT' else 'Divine revelation distinguishes AT from human-authority texts.'}"
                ),
            })

    # Implications for typology
    bod_pred = None
    pali_pred = None
    for sr in results["scripts"]:
        if sr["name"] == "book_of_dead":
            bod_pred = sr.get("bod_predicted")
        if sr["name"] == "pali_canon":
            pali_pred = sr.get("pali_predicted")

    if bod_pred and pali_pred:
        if bod_pred == "AT" and pali_pred == "AT":
            synthesis["implications"].append(
                "Both non-revelatory controlled-transmission texts are AT-like. "
                "This suggests controlled transmission (not revelation) is the "
                "primary driver of AT-like statistical signatures."
            )
        elif bod_pred == "NT" and pali_pred == "NT":
            synthesis["implications"].append(
                "Both non-revelatory texts are NT-like. "
                "This suggests divine revelation claim IS associated with "
                "distinctive statistical patterns that non-revelatory texts lack."
            )
        else:
            synthesis["implications"].append(
                f"Mixed result: Book of Dead is {bod_pred}-like, "
                f"Pali Canon is {pali_pred}-like. "
                "This suggests the AT/NT distinction depends on specific "
                "combination of factors, not a single variable."
            )

    results["synthesis"] = synthesis

    summary_path = RESULTS_DIR / "fase17_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    log.info(f"\n{'=' * 70}")
    log.info(f"FASE 17 completada en {elapsed:.1f}s")
    log.info(f"Resumen en {summary_path}")

    for sr in results["scripts"]:
        log.info(f"\n  {sr['name']}: {sr['status']} (rc={sr['return_code']})")
        if "bod_predicted" in sr:
            log.info(f"    BoD: {sr['bod_predicted']}-like (P_AT={sr.get('bod_P_AT')})")
            log.info(f"    BoD H={sr.get('bod_H')}, AC1={sr.get('bod_AC1')}")
            log.info(f"    Nearest: {sr.get('bod_nearest')}")
        if "pali_predicted" in sr:
            log.info(f"    Pali: {sr['pali_predicted']}-like (P_AT={sr.get('pali_P_AT')})")
            log.info(f"    Pali H={sr.get('pali_H')}, AC1={sr.get('pali_AC1')}")
            log.info(f"    Nearest: {sr.get('pali_nearest')}")

    for q in synthesis.get("questions_answered", []):
        log.info(f"\n  Q: {q['question']}")
        log.info(f"  A: {q['answer']}")

    for imp in synthesis.get("implications", []):
        log.info(f"\n  Implication: {imp}")


if __name__ == "__main__":
    main()
