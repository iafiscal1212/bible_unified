#!/usr/bin/env python3
"""
Orchestrator Fase 19 — Classifier Correction + Gap Corpora + H4' Retest + Degradation

Execution order:
1. Script 1 (BLOCKING): classifier_correction.py
   - Must complete successfully → generates model.pkl
   - Check mishnah_verdict.json, print result, warn if AT-like
   - If script fails → STOP ALL
2. Scripts 2,3,4 (PARALLEL): gap_corpora_f19.py, h4prime_retest.py, degradation_model.py
3. Collect results → fase19_summary.json
"""

import json
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent


def run_blocking(script, name):
    """Run a script synchronously, stream output. Return exit code."""
    print(f"\n{'=' * 60}")
    print(f"[BLOCKING] {name}: {script}")
    print(f"{'=' * 60}")

    proc = subprocess.Popen(
        [sys.executable, str(BASE / script)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, cwd=str(BASE)
    )

    for line in proc.stdout:
        print(f"  [{name}] {line}", end="")

    proc.wait()
    print(f"\n  [{name}] exit code: {proc.returncode}")
    return proc.returncode


def run_parallel(scripts):
    """Run multiple scripts in parallel, poll until all complete."""
    procs = {}
    for script, name in scripts:
        print(f"  Launching {name}...")
        proc = subprocess.Popen(
            [sys.executable, str(BASE / script)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, cwd=str(BASE)
        )
        procs[name] = {"proc": proc, "script": script}

    # Poll until all done
    while any(p["proc"].poll() is None for p in procs.values()):
        time.sleep(5)
        for name, p in procs.items():
            if p["proc"].poll() is None:
                pass  # still running

    # Collect final output
    for name, p in procs.items():
        output = p["proc"].stdout.read()
        if output:
            for line in output.strip().split("\n"):
                print(f"  [{name}] {line}")
        p["rc"] = p["proc"].returncode
        print(f"  [{name}] exit code: {p['rc']}")

    return {name: p["rc"] for name, p in procs.items()}


def main():
    t0 = time.time()
    print("=" * 70)
    print("FASE 19 — ORCHESTRATOR")
    print("=" * 70)

    # ═══════════════════════════════════════════════════════════
    # PHASE 1: BLOCKING — Classifier Correction
    # ═══════════════════════════════════════════════════════════
    rc1 = run_blocking("classifier_correction.py", "Script1-Classifier")

    if rc1 != 0:
        print(f"\n  FATAL: Script 1 failed (rc={rc1}). STOPPING ALL.")
        with open(BASE / "results" / "fase19_summary.json", "w") as f:
            json.dump({"status": "ABORTED", "reason": f"Script 1 failed (rc={rc1})"}, f)
        sys.exit(1)

    # Check mishnah_verdict.json
    mv_file = BASE / "results" / "classifier_corrected" / "mishnah_verdict.json"
    mishnah_verdict = None
    if mv_file.exists():
        with open(mv_file) as f:
            mishnah_verdict = json.load(f)
        print(f"\n  ┌────────────────────────────────────────────────┐")
        print(f"  │ MISHNAH VERDICT                                │")
        print(f"  │ Original:  {str(mishnah_verdict.get('original_class','?')):>8}-like "
              f"(P_AT={mishnah_verdict.get('P_AT_original','?'):>8}) │")
        print(f"  │ Corrected: {str(mishnah_verdict.get('corrected_class','?')):>8}-like "
              f"(P_AT={mishnah_verdict.get('P_AT_corrected','?'):>8}) │")
        print(f"  │ Changed:   {str(mishnah_verdict.get('classification_changed','')):>35} │")
        print(f"  └────────────────────────────────────────────────┘")

        if mishnah_verdict.get("corrected_class") == "AT":
            print("  WARNING: Mishnah is AT-like with corrected classifier!")
            print("  This challenges H4' — Mishnah (delay=400) should be NT-like.")
    else:
        print("  WARNING: mishnah_verdict.json not found")

    # Check model.pkl exists
    model_file = BASE / "results" / "classifier_corrected" / "model.pkl"
    if not model_file.exists():
        print("  FATAL: model.pkl not found. Scripts 2,3,4 depend on it.")
        sys.exit(1)

    print("\n  model.pkl OK — proceeding to parallel phase")

    # ═══════════════════════════════════════════════════════════
    # PHASE 2: PARALLEL — Scripts 2, 3, 4
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print("[PARALLEL] Launching Scripts 2, 3, 4...")
    print(f"{'=' * 60}")

    parallel_scripts = [
        ("gap_corpora_f19.py", "Script2-GapCorpora"),
        ("h4prime_retest.py", "Script3-H4prime"),
        ("degradation_model.py", "Script4-Degradation"),
    ]

    parallel_results = run_parallel(parallel_scripts)

    # ═══════════════════════════════════════════════════════════
    # PHASE 3: Collect Summary
    # ═══════════════════════════════════════════════════════════
    elapsed = time.time() - t0

    summary = {
        "status": "COMPLETED",
        "elapsed_seconds": round(elapsed, 1),
        "scripts": {
            "classifier_correction": {"rc": rc1, "status": "OK" if rc1 == 0 else "FAILED"},
        },
    }

    for name, rc in parallel_results.items():
        summary["scripts"][name] = {"rc": rc, "status": "OK" if rc == 0 else "FAILED"}

    all_ok = rc1 == 0 and all(rc == 0 for rc in parallel_results.values())

    # Load key results
    # Classifier summary
    changes_file = BASE / "results" / "classifier_corrected" / "changes_summary.json"
    if changes_file.exists():
        with open(changes_file) as f:
            changes = json.load(f)
        summary["classifier"] = {
            "best": changes.get("best_classifier"),
            "loo_accuracy": changes.get("loo_accuracy"),
            "bias_ok": not changes.get("bias_check", {}).get("bias_detected", True),
            "n_classification_changes": changes.get("n_changes"),
            "changes": changes.get("changes"),
        }

    # Mishnah
    if mishnah_verdict:
        summary["mishnah_verdict"] = mishnah_verdict

    # H4' verdict
    verdict_file = BASE / "results" / "h4prime_retest" / "verdict.json"
    if verdict_file.exists():
        with open(verdict_file) as f:
            verdict = json.load(f)
        summary["h4prime"] = {
            "verdict": verdict.get("verdict"),
            "criteria_met": verdict.get("n_criteria_met"),
            "criteria_total": verdict.get("n_criteria_total"),
            "n_corpora": verdict.get("n_corpora_used"),
            "reasoning": verdict.get("reasoning"),
        }

    # Gap corpora
    gap_file = BASE / "results" / "gap_corpora" / "gap_summary.json"
    if gap_file.exists():
        with open(gap_file) as f:
            summary["gap_corpora"] = json.load(f)

    # Degradation
    deg_file = BASE / "results" / "degradation_model" / "summary.json"
    if deg_file.exists():
        with open(deg_file) as f:
            summary["degradation"] = json.load(f)

    with open(BASE / "results" / "fase19_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ═══════════════════════════════════════════════════════════
    # Final report
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print(f"FASE 19 {'COMPLETADA' if all_ok else 'CON ERRORES'} en {elapsed:.1f}s")
    print(f"{'=' * 70}")

    for name, info in summary["scripts"].items():
        status = info["status"]
        print(f"  {name:>25}: {status}")

    if "classifier" in summary:
        c = summary["classifier"]
        print(f"\n  Classifier: {c.get('best')}, LOO acc={c.get('loo_accuracy')}, "
              f"bias={'OK' if c.get('bias_ok') else 'BIASED'}")
        if c.get("n_classification_changes", 0) > 0:
            print(f"  {c['n_classification_changes']} classification change(s):")
            for ch in c.get("changes", []):
                print(f"    {ch['corpus']}: {ch['original']} → {ch['corrected']}")

    if mishnah_verdict:
        print(f"\n  Mishnah: {mishnah_verdict.get('corrected_class')}-like "
              f"(P_AT={mishnah_verdict.get('P_AT_corrected')})")

    if "h4prime" in summary:
        h4 = summary["h4prime"]
        print(f"\n  ╔══════════════════════════════════════════╗")
        print(f"  ║  H4' VERDICT: {h4['verdict']:>26} ║")
        print(f"  ║  Criteria: {h4['criteria_met']}/{h4['criteria_total']}"
              f"{'':>28}║")
        print(f"  ╚══════════════════════════════════════════╝")

    print(f"\nResultados: {BASE / 'results' / 'fase19_summary.json'}")


if __name__ == "__main__":
    main()
