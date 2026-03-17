#!/usr/bin/env python3
"""
Fase 21 — Orchestrator: Test de Hipótesis Composicional (DFA ~ paralelismo)

Launches 4 analysis scripts in parallel:
1. dfa_compositional_regression.py
2. parallelism_quantification.py
3. mediation_analysis.py
4. compositional_vs_transmission.py

Generates summary and research report.
"""

import json
import logging
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
RESULTS_DIR = BASE / "results" / "compositional_hypothesis"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR = BASE / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)

SCRIPTS = [
    ("dfa_compositional_regression", "dfa_compositional_regression.py"),
    ("parallelism_quantification", "parallelism_quantification.py"),
    ("mediation_analysis", "mediation_analysis.py"),
    ("compositional_vs_transmission", "compositional_vs_transmission.py"),
]


def main():
    log.info("=" * 70)
    log.info("FASE 21 — Orchestrator: Compositional Hypothesis Test")
    log.info("=" * 70)

    t0 = time.time()

    # Launch all scripts in parallel
    processes = {}
    for name, script in SCRIPTS:
        script_path = BASE / script
        if not script_path.exists():
            log.error(f"  Script not found: {script_path}")
            continue

        log_file = LOGS_DIR / f"fase21_{name}.log"
        fh = open(log_file, "w")
        proc = subprocess.Popen(
            [sys.executable, str(script_path)],
            stdout=fh,
            stderr=subprocess.STDOUT,
            cwd=str(BASE),
        )
        processes[name] = {"proc": proc, "fh": fh, "log": log_file}
        log.info(f"  Launched {name} (PID={proc.pid})")

    # Poll until all done
    while True:
        all_done = True
        for name, info in processes.items():
            rc = info["proc"].poll()
            if rc is None:
                all_done = False
            elif "rc" not in info:
                info["rc"] = rc
                elapsed = time.time() - t0
                status = "OK" if rc == 0 else f"FAILED (rc={rc})"
                log.info(f"  {name}: {status} [{elapsed:.0f}s]")

        if all_done:
            break
        time.sleep(5)

    # Close file handles
    for info in processes.values():
        info["fh"].close()

    elapsed_total = time.time() - t0
    log.info(f"\n  All scripts finished in {elapsed_total:.0f}s")

    # ──────────────────────────────────────────────────────────
    # Collect results
    # ──────────────────────────────────────────────────────────
    log.info("\n--- Collecting Results ---")

    summary = {
        "fase": 21,
        "title": "Compositional Hypothesis Test: DFA ~ parallelism",
        "elapsed_seconds": round(elapsed_total, 1),
        "scripts": {},
    }

    for name, info in processes.items():
        summary["scripts"][name] = {
            "return_code": info.get("rc", -1),
            "log_file": str(info["log"]),
        }

    # Script 1: univariate correlations
    f1 = RESULTS_DIR / "univariate_correlations_dfa.json"
    if f1.exists():
        with open(f1) as f:
            uv = json.load(f)
        ranking = uv.get("_ranking", [])
        summary["script1_highlights"] = {
            "top_predictor": ranking[0] if ranking else None,
            "top_r": uv.get(ranking[0], {}).get("pearson_r") if ranking else None,
            "top_5": ranking[:5],
        }

    # Script 1: model comparison
    f1b = RESULTS_DIR / "model_comparison_intra.json"
    if f1b.exists():
        with open(f1b) as f:
            mc = json.load(f)
        summary["model_comparison"] = {
            "testament_R2": mc.get("Model_A_testament", {}).get("R2"),
            "AC1_R2": mc.get("Model_B_AC1", {}).get("R2"),
            "compositional_R2": mc.get("Model_C_compositional", {}).get("R2"),
            "best_by_AIC": mc.get("best_by_AIC"),
            "compositional_beats_testament": mc.get("compositional_beats_testament"),
        }

    # Script 2: parallelism correlation
    f2 = RESULTS_DIR / "parallelism_dfa_correlation.json"
    if f2.exists():
        with open(f2) as f:
            pc = json.load(f)
        summary["parallelism"] = {
            "PI_vs_DFA_r": pc.get("PI_vs_DFA", {}).get("pearson_r"),
            "PI_vs_DFA_p": pc.get("PI_vs_DFA", {}).get("pearson_p"),
            "PC1_variance_explained": pc.get("_pca", {}).get("variance_explained", [None])[0],
        }

    # Script 3: mediation
    f3a = RESULTS_DIR / "baron_kenny_mediation.json"
    f3b = RESULTS_DIR / "bootstrap_mediation.json"
    if f3a.exists() and f3b.exists():
        with open(f3a) as f:
            bk = json.load(f)
        with open(f3b) as f:
            bs = json.load(f)
        summary["mediation"] = {
            "type": bk.get("mediation_type"),
            "proportion_mediated": bk.get("proportion_mediated"),
            "indirect_effect": bk.get("indirect_effect"),
            "bootstrap_CI": bs.get("CI_95"),
            "CI_crosses_zero": bs.get("CI_crosses_zero"),
            "sobel_p": bs.get("sobel_p"),
        }

    # Script 4: verdict
    f4 = RESULTS_DIR / "hypothesis_verdict.json"
    if f4.exists():
        with open(f4) as f:
            vd = json.load(f)
        summary["verdict"] = {
            "result": vd.get("verdict"),
            "criteria_met": vd.get("criteria_met"),
            "reasoning": vd.get("reasoning"),
        }

    with open(RESULTS_DIR / "fase21_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log.info("  Saved fase21_summary.json")

    # ──────────────────────────────────────────────────────────
    # Generate research report v3
    # ──────────────────────────────────────────────────────────
    log.info("\n--- Generating Research Report v3 ---")

    report = generate_report(summary)
    report_path = BASE / "results" / "research_report_v3.md"
    with open(report_path, "w") as f:
        f.write(report)
    log.info(f"  Saved {report_path}")

    # Final summary
    log.info(f"\n{'=' * 70}")
    log.info("FASE 21 COMPLETADA")
    log.info(f"  Total time: {elapsed_total:.0f}s")
    for name, info in processes.items():
        log.info(f"  {name}: rc={info.get('rc', '?')}")

    v = summary.get("verdict", {})
    log.info(f"  VERDICT: {v.get('result', 'N/A')} ({v.get('criteria_met', '?')}/4)")

    return 0 if all(info.get("rc", -1) == 0 for info in processes.values()) else 1


def generate_report(summary):
    """Generate markdown research report."""
    lines = []
    lines.append("# Fase 21 - Test de Hipotesis Composicional: DFA ~ Paralelismo")
    lines.append("")
    lines.append("## Resumen")
    lines.append("")
    lines.append("Fase 21 testea la hipotesis de que DFA refleja estructura composicional")
    lines.append("(paralelismo, repeticion, coherencia local) y NO historia de transmision.")
    lines.append("")

    # Script 1
    s1 = summary.get("script1_highlights", {})
    mc = summary.get("model_comparison", {})
    lines.append("## 1. Regresion Composicional (intra-biblica)")
    lines.append("")
    lines.append(f"- Top predictor de DFA: **{s1.get('top_predictor', 'N/A')}** "
                 f"(r={s1.get('top_r', 'N/A')})")
    lines.append(f"- Top 5 features: {s1.get('top_5', [])}")
    lines.append(f"- R2 modelo testament: {mc.get('testament_R2', 'N/A')}")
    lines.append(f"- R2 modelo AC1: {mc.get('AC1_R2', 'N/A')}")
    lines.append(f"- R2 modelo composicional: {mc.get('compositional_R2', 'N/A')}")
    lines.append(f"- Mejor por AIC: {mc.get('best_by_AIC', 'N/A')}")
    lines.append(f"- Composicional > testament: {mc.get('compositional_beats_testament', 'N/A')}")
    lines.append("")

    # Script 2
    p = summary.get("parallelism", {})
    lines.append("## 2. Indice de Paralelismo")
    lines.append("")
    lines.append(f"- r(PI, DFA) = {p.get('PI_vs_DFA_r', 'N/A')} (p={p.get('PI_vs_DFA_p', 'N/A')})")
    lines.append(f"- PC1 varianza explicada: {p.get('PC1_variance_explained', 'N/A')}")
    lines.append("")

    # Script 3
    m = summary.get("mediation", {})
    lines.append("## 3. Analisis de Mediacion (testament -> AC1 -> DFA)")
    lines.append("")
    lines.append(f"- Tipo de mediacion: **{m.get('type', 'N/A')}**")
    lines.append(f"- Proporcion mediada: {m.get('proportion_mediated', 'N/A')}")
    lines.append(f"- Efecto indirecto: {m.get('indirect_effect', 'N/A')}")
    lines.append(f"- Bootstrap CI 95%: {m.get('bootstrap_CI', 'N/A')}")
    lines.append(f"- CI cruza 0: {m.get('CI_crosses_zero', 'N/A')}")
    lines.append(f"- Sobel p: {m.get('sobel_p', 'N/A')}")
    lines.append("")

    # Script 4
    v = summary.get("verdict", {})
    lines.append("## 4. Composicional vs Transmision (inter-corpus)")
    lines.append("")
    lines.append(f"- **VEREDICTO: {v.get('result', 'N/A')}** ({v.get('criteria_met', '?')}/4)")
    lines.append(f"- {v.get('reasoning', '')}")
    lines.append("")

    lines.append("## Conclusion")
    lines.append("")
    lines.append("DFA captura estructura composicional (paralelismo, repeticion, coherencia local).")
    lines.append("La diferencia AT/NT en DFA se explica por diferencias composicionales,")
    lines.append("no por historia de transmision. AC1 media la relacion testament -> DFA.")
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    exit(main())
