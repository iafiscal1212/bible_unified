#!/usr/bin/env python3
"""
Fase 18 — Orchestrator: Mishnah (Sefaria) + Test H4'
Lanza 2 scripts: primero mishnah_sefaria.py, luego transmission_origin_test.py
(secuencial porque el test necesita los resultados de la Mishnah).
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
        logging.FileHandler(LOG_DIR / "fase18_orchestrator.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


SCRIPTS = [
    {
        "name": "mishnah_sefaria",
        "file": "mishnah_sefaria.py",
        "question": "¿La Mishnah (transmisión controlada tardía) es AT-like o NT-like?",
        "results_dir": "mishnah",
        "depends_on": None,
    },
    {
        "name": "transmission_origin_test",
        "file": "transmission_origin_test.py",
        "question": "¿H4' (control desde el origen → AT-like) se sostiene con 9 corpora?",
        "results_dir": "transmission_origin",
        "depends_on": "mishnah_sefaria",
    },
]


def main():
    t0 = time.time()
    log.info("=" * 70)
    log.info("FASE 18 — Mishnah + Test H4' (transmisión controlada desde el origen)")
    log.info("2 scripts secuenciales (el test depende de los resultados de Mishnah)")
    log.info("=" * 70)

    script_results = {}

    for script in SCRIPTS:
        log.info(f"\n  Lanzando {script['file']}...")
        log_path = LOG_DIR / f"fase18_{script['name']}.log"

        with open(log_path, "w") as log_file:
            proc = subprocess.Popen(
                [sys.executable, str(BASE / script["file"])],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                cwd=str(BASE),
            )
            log.info(f"    PID={proc.pid}")

            # Polling loop
            while proc.poll() is None:
                time.sleep(30)
                elapsed = time.time() - t0
                log.info(f"    [{elapsed:.0f}s] {script['name']}: running...")

            elapsed = time.time() - t0
            rc = proc.returncode
            log.info(f"    [{elapsed:.0f}s] {script['name']}: done (rc={rc})")
            script_results[script["name"]] = rc

    elapsed = time.time() - t0
    log.info(f"\n  Todos los scripts completados en {elapsed:.1f}s")

    # ── Build summary ────────────────────────────────────────────────
    results = {
        "phase": 18,
        "title": "Mishnah + Test H4' — Fase 18",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_elapsed_seconds": round(elapsed, 1),
        "scripts": [],
    }

    for script in SCRIPTS:
        sr = {
            "name": script["name"],
            "question": script["question"],
            "return_code": script_results[script["name"]],
            "status": "completed" if script_results[script["name"]] == 0 else "failed",
        }

        rdir = RESULTS_DIR / script["results_dir"]
        if rdir.exists():
            json_files = list(rdir.glob("*.json"))
            sr["n_output_files"] = len(json_files)
            sr["output_files"] = [f.name for f in json_files]

            for jf in json_files:
                try:
                    with open(jf) as f:
                        data = json.load(f)

                    fname = jf.name

                    # ── Mishnah metrics ──
                    if script["name"] == "mishnah_sefaria":
                        if "mishnah_metrics" in fname:
                            if isinstance(data, dict) and "error" not in data:
                                sr["mish_n_tractates"] = data.get("n_tractates")
                                sr["mish_n_mishnayot"] = data.get("n_mishnayot")
                                sr["mish_n_words"] = data.get("n_words")
                                sr["mish_H"] = data.get("H")
                                sr["mish_AC1"] = data.get("AC1")
                                sr["mish_DFA"] = data.get("DFA")
                                sr["mish_CV"] = data.get("CV")
                                sr["mish_mean"] = data.get("mean_verse_len")
                            else:
                                sr["mish_error"] = data.get("error", "unknown")

                        elif "classifier_result" in fname:
                            if isinstance(data, dict):
                                sr["mish_predicted"] = data.get("predicted_class")
                                sr["mish_P_AT"] = data.get("P_AT")
                                sr["mish_P_NT"] = data.get("P_NT")

                        elif "phi_d_placement" in fname:
                            if isinstance(data, dict):
                                sr["mish_nearest"] = data.get("nearest_corpus")
                                sr["mish_d_approx"] = data.get("d_approx")

                        elif "comparison_at_legal" in fname:
                            if isinstance(data, dict):
                                sr["mish_vs_legal_n_similar"] = data.get("n_similar_metrics")
                                sr["mish_vs_legal_interp"] = (
                                    data.get("interpretation", "")[:200]
                                )

                    # ── Transmission test ──
                    elif script["name"] == "transmission_origin_test":
                        if "verdict" in fname:
                            if isinstance(data, dict):
                                sr["h4prime_supported"] = data.get("h4_prime_supported")
                                sr["h4prime_verdict"] = data.get("verdict", "")[:300]
                                sr["h4prime_n_corpora"] = data.get("n_corpora")

                        elif "correlations" in fname:
                            if isinstance(data, dict):
                                sr["rpb_H"] = data.get("control_from_origin_vs_H", {}).get("r")
                                sr["rpb_H_p"] = data.get("control_from_origin_vs_H", {}).get("p")
                                sr["rpb_AC1"] = data.get("control_from_origin_vs_AC1", {}).get("r")
                                sr["rpb_AC1_p"] = data.get("control_from_origin_vs_AC1", {}).get("p")

                        elif "fisher_test" in fname:
                            if isinstance(data, dict):
                                sr["fisher_p"] = data.get("p_value")
                                sr["fisher_odds"] = data.get("odds_ratio")

                        elif "regression" in fname:
                            if isinstance(data, dict):
                                sr["reg_delay_H_slope"] = data.get("delay_vs_H", {}).get("slope")
                                sr["reg_delay_H_p"] = data.get("delay_vs_H", {}).get("p_value")

                except Exception as e:
                    log.warning(f"  Error leyendo {jf}: {e}")

        results["scripts"].append(sr)

    # ── Synthesis ────────────────────────────────────────────────────
    synthesis = {
        "questions_answered": [],
        "implications": [],
    }

    for sr in results["scripts"]:
        if sr["name"] == "mishnah_sefaria" and sr.get("mish_predicted"):
            pred = sr["mish_predicted"]
            p_at = sr.get("mish_P_AT", 0)
            h_val = sr.get("mish_H", "?")
            ac1_val = sr.get("mish_AC1", "?")
            synthesis["questions_answered"].append({
                "question": "¿La Mishnah (transmisión controlada tardía) es AT-like o NT-like?",
                "answer": (
                    f"Classified as {pred}-like (P_AT={p_at}). "
                    f"H={h_val}, AC1={ac1_val}. "
                    f"{'Mishnah is NT-like despite controlled transmission — supports H4 prime (delay matters).' if pred == 'NT' else 'Mishnah is AT-like — challenges H4 prime hypothesis.'}"
                ),
            })

        if sr["name"] == "transmission_origin_test" and sr.get("h4prime_supported") is not None:
            synthesis["questions_answered"].append({
                "question": "¿H4' (control desde el origen → AT-like) se sostiene?",
                "answer": sr.get("h4prime_verdict", "See verdict.json for details"),
            })

            # Fisher test significance
            fisher_p = sr.get("fisher_p")
            if fisher_p is not None:
                synthesis["implications"].append(
                    f"Fisher exact test: p={fisher_p:.4f}. "
                    f"{'Significant at α=0.05 — control_from_origin is associated with AT-like classification.' if fisher_p < 0.05 else 'Not significant at α=0.05 — insufficient evidence for association.'}"
                )

            rpb_h = sr.get("rpb_H")
            if rpb_h is not None:
                synthesis["implications"].append(
                    f"Point-biserial r(control_from_origin, H) = {rpb_h:.3f} "
                    f"(p={sr.get('rpb_H_p', '?'):.4f})"
                )

    # Cross-script synthesis
    mish_sr = next((s for s in results["scripts"] if s["name"] == "mishnah_sefaria"), {})
    test_sr = next((s for s in results["scripts"] if s["name"] == "transmission_origin_test"), {})

    mish_pred = mish_sr.get("mish_predicted")
    h4_supported = test_sr.get("h4prime_supported")

    if mish_pred == "NT" and h4_supported:
        synthesis["implications"].append(
            "KEY FINDING: Mishnah is NT-like despite controlled transmission. "
            "This confirms H4': what matters is not just controlled transmission, "
            "but controlled transmission FROM THE ORIGIN. "
            "The Mishnah had centuries of free oral debate before codification — "
            "the statistical signature was already set."
        )
    elif mish_pred == "AT":
        synthesis["implications"].append(
            "UNEXPECTED: Mishnah classified as AT-like. "
            "This would challenge H4' — controlled transmission with delay "
            "still produces AT-like signatures. "
            "Consider: is the delay insufficient, or does the genre (legal) dominate?"
        )

    results["synthesis"] = synthesis

    # ── Typology table (updated with Mishnah) ────────────────────────
    results["typology_table"] = build_typology_table(results)

    summary_path = RESULTS_DIR / "fase18_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    log.info(f"\n{'=' * 70}")
    log.info(f"FASE 18 completada en {elapsed:.1f}s")
    log.info(f"Resumen en {summary_path}")

    for sr in results["scripts"]:
        log.info(f"\n  {sr['name']}: {sr['status']} (rc={sr['return_code']})")
        if "mish_predicted" in sr:
            log.info(f"    Mishnah: {sr['mish_predicted']}-like (P_AT={sr.get('mish_P_AT')})")
            log.info(f"    H={sr.get('mish_H')}, AC1={sr.get('mish_AC1')}")
        if "h4prime_supported" in sr:
            log.info(f"    H4' supported: {sr['h4prime_supported']}")
            log.info(f"    Fisher p={sr.get('fisher_p')}")

    for q in synthesis.get("questions_answered", []):
        log.info(f"\n  Q: {q['question']}")
        log.info(f"  A: {q['answer']}")

    for imp in synthesis.get("implications", []):
        log.info(f"\n  → {imp}")


def build_typology_table(results):
    """Build updated typology table with all 9+ corpora."""
    table = {
        "AT": {
            "transmission": "controlled scribal (1000+ years)",
            "authority": "divine revelation",
            "function": "liturgical + legal + narrative",
            "language_family": "Afroasiatic (Semitic)",
            "H_range": "high (0.88)",
            "AC1_range": "high (0.34)",
            "control_from_origin": True,
        },
        "NT": {
            "transmission": "rapid copying (decades-centuries)",
            "authority": "apostolic witness",
            "function": "epistolary + narrative + apocalyptic",
            "language_family": "Indo-European (Greek)",
            "H_range": "moderate (0.78)",
            "AC1_range": "low (0.09)",
            "control_from_origin": False,
        },
        "Corán": {
            "transmission": "controlled oral then scribal",
            "authority": "divine revelation (direct)",
            "function": "liturgical + legal",
            "language_family": "Afroasiatic (Semitic)",
            "H_range": "high (0.90)",
            "AC1_range": "very high (0.47)",
            "control_from_origin": True,
        },
        "Rig_Veda": {
            "transmission": "oral controlled (millenia)",
            "authority": "divine/inspired (ṛṣi)",
            "function": "liturgical/hymnic",
            "language_family": "Indo-European (Indo-Aryan)",
            "H_range": "low (0.56)",
            "AC1_range": "none (-0.02)",
            "control_from_origin": True,
        },
        "Book_of_Dead": {
            "transmission": "controlled scribal (~1500 years)",
            "authority": "priestly/ritual (no divine revelation)",
            "function": "funerary liturgical",
            "language_family": "Afroasiatic (Egyptian)",
            "control_from_origin": True,
        },
        "Pali_Canon": {
            "transmission": "oral controlled then scribal (~400 years oral)",
            "authority": "human (Buddha, not divine)",
            "function": "didactic/doctrinal",
            "language_family": "Indo-European (Indo-Aryan)",
            "control_from_origin": True,
        },
    }

    # Add Mishnah from results
    for sr in results.get("scripts", []):
        if sr["name"] == "mishnah_sefaria" and sr.get("status") == "completed":
            table["Mishnah"] = {
                "transmission": "oral controlled → codified (~200 CE)",
                "authority": "rabbinic (human, post-prophetic)",
                "function": "legal/halakhic",
                "language_family": "Afroasiatic (Semitic, Mishnaic Hebrew)",
                "control_from_origin": False,
                "control_delay_years": 400,
                "H": sr.get("mish_H"),
                "AC1": sr.get("mish_AC1"),
                "classified_as": sr.get("mish_predicted"),
                "P_AT": sr.get("mish_P_AT"),
                "nearest_corpus": sr.get("mish_nearest"),
                "typology_cell": "controlled transmission WITH DELAY (free debate → codification)",
            }

    return table


if __name__ == "__main__":
    main()
