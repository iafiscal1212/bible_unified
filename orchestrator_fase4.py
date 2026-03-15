#!/usr/bin/env python3
"""
orchestrator_fase4.py — Lanza las 5 investigaciones cuántico-inspiradas en paralelo.
Todo numpy/scipy — cero frameworks cuánticos.
"""
import subprocess, sys, time, json, os
from pathlib import Path
from datetime import datetime

BASE = Path(__file__).parent
RESULTS = BASE / "results"
LOGS = BASE / "logs"
LOGS.mkdir(parents=True, exist_ok=True)

MODULES = [
    "mps_representation.py",
    "von_neumann_entropy.py",
    "quantum_mutual_information.py",
    "quantum_walk.py",
    "mps_compression_ratio.py",
]

RESULT_FILES = {
    "mps_representation": [
        RESULTS / "mps" / "correlation_matrix.json",
        RESULTS / "mps" / "svd_spectrum.json",
        RESULTS / "mps" / "bond_dimension.json",
        RESULTS / "mps" / "reconstruction_error.json",
    ],
    "von_neumann_entropy": [
        RESULTS / "von_neumann" / "density_matrices_summary.json",
        RESULTS / "von_neumann" / "entropy_comparison.json",
        RESULTS / "von_neumann" / "delta_s_by_book.json",
        RESULTS / "von_neumann" / "correlations.json",
    ],
    "quantum_mutual_information": [
        RESULTS / "qmi" / "quantum_mi_matrix.json",
        RESULTS / "qmi" / "classical_mi_matrix.json",
        RESULTS / "qmi" / "delta_i_matrix.json",
        RESULTS / "qmi" / "network_modularity.json",
        RESULTS / "qmi" / "community_structure.json",
    ],
    "quantum_walk": [
        RESULTS / "qwalk" / "top_quantum_lemas_OT.json",
        RESULTS / "qwalk" / "top_quantum_lemas_NT.json",
        RESULTS / "qwalk" / "delta_p_ot.json",
        RESULTS / "qwalk" / "delta_p_nt.json",
    ],
    "mps_compression_ratio": [
        RESULTS / "mps_compression" / "compression_ratios.json",
        RESULTS / "mps_compression" / "permutation_test_chi.json",
        RESULTS / "mps_compression" / "bits_comparison.json",
    ],
}


def run_modules():
    start = time.time()
    procs = {}

    print(f"{'='*60}")
    print(f"  BIBLE RESEARCH — FASE 4: ANÁLISIS CUÁNTICO-INSPIRADO")
    print(f"  (numpy/scipy puro — cero frameworks cuánticos)")
    print(f"  Inicio: {datetime.now().isoformat()}")
    print(f"  Investigaciones: {len(MODULES)}")
    print(f"{'='*60}\n")

    for mod in MODULES:
        mod_path = BASE / mod
        name = mod.replace(".py", "")
        log_file = LOGS / f"fase4_{name}.log"
        log_fh = open(log_file, "w")
        print(f"  [LAUNCH] {name} → {log_file}")
        proc = subprocess.Popen(
            [sys.executable, str(mod_path)],
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            cwd=str(BASE),
        )
        procs[name] = {"proc": proc, "start": time.time(), "log_file": log_file, "log_fh": log_fh}

    print(f"\n  Esperando finalización...\n")
    completed = set()
    last_progress = time.time()

    while len(completed) < len(procs):
        for name, info in procs.items():
            if name in completed:
                continue
            ret = info["proc"].poll()
            if ret is not None:
                elapsed = time.time() - info["start"]
                info["log_fh"].close()
                status = "OK" if ret == 0 else f"ERROR(rc={ret})"
                print(f"  [{status}] {name:30s} — {elapsed:.1f}s")
                try:
                    with open(info["log_file"]) as f:
                        lines = f.readlines()
                        if lines:
                            print(f"           {lines[-1].strip()}")
                except Exception:
                    pass
                completed.add(name)

        now = time.time()
        if now - last_progress >= 30 and len(completed) < len(procs):
            elapsed_total = now - start
            running = [n for n in procs if n not in completed]
            print(f"  [PROGRESS {elapsed_total:.0f}s] Running: {', '.join(running)}")
            for name in running:
                try:
                    with open(procs[name]["log_file"]) as f:
                        lines = f.readlines()
                        if lines:
                            print(f"    {name}: {lines[-1].strip()}")
                except Exception:
                    pass
            last_progress = now

        if len(completed) < len(procs):
            time.sleep(2)

    total_time = time.time() - start
    print(f"\n{'='*60}")
    print(f"  Todas las investigaciones completadas en {total_time:.1f}s")
    print(f"{'='*60}\n")
    return total_time


def extract_highlights(name, files):
    data = {}
    for f in files:
        if f.exists():
            with open(f) as fh:
                data[f.stem] = json.load(fh)

    h = {"status": "OK", "files": [str(f) for f in files if f.exists()]}

    if name == "mps_representation":
        bd = data.get("bond_dimension", {})
        re = data.get("reconstruction_error", {})
        h["key_finding"] = {
            "chi_OT_99": bd.get("OT", {}).get("chi_990", {}).get("bond_dimension"),
            "chi_NT_99": bd.get("NT", {}).get("chi_990", {}).get("bond_dimension"),
            "chi_global_99": bd.get("global", {}).get("chi_990", {}).get("bond_dimension"),
            "recon_error_OT": re.get("OT", {}).get("relative_frobenius_error"),
            "recon_error_NT": re.get("NT", {}).get("relative_frobenius_error"),
        }
        chi_ot = h["key_finding"]["chi_OT_99"]
        chi_nt = h["key_finding"]["chi_NT_99"]
        h["adds_info"] = "yes" if chi_ot and chi_nt and chi_ot != chi_nt else "indeterminate"

    elif name == "von_neumann_entropy":
        ec = data.get("entropy_comparison", {})
        ot_nt = ec.get("ot_vs_nt", {})
        h["key_finding"] = {
            "ot_mean_delta_s": ot_nt.get("ot_mean_delta_s"),
            "nt_mean_delta_s": ot_nt.get("nt_mean_delta_s"),
            "ot_vs_nt_p": ot_nt.get("mannwhitney_p"),
            "significant": ot_nt.get("significant"),
            "by_genre": ec.get("by_genre"),
        }
        h["adds_info"] = "yes" if ot_nt.get("significant") else "no"

    elif name == "quantum_mutual_information":
        mod = data.get("network_modularity", {})
        di = data.get("delta_i_matrix", {})
        h["key_finding"] = {
            "quantum_modularity_genre": mod.get("quantum_network", {}).get("modularity_genre"),
            "classical_modularity_genre": mod.get("classical_network", {}).get("modularity_genre"),
            "quantum_modularity_corpus": mod.get("quantum_network", {}).get("modularity_corpus"),
            "classical_modularity_corpus": mod.get("classical_network", {}).get("modularity_corpus"),
            "mean_delta_i": di.get("mean_delta"),
        }
        qm = mod.get("quantum_network", {}).get("modularity_genre", 0)
        cm = mod.get("classical_network", {}).get("modularity_genre", 0)
        h["adds_info"] = "yes" if qm > cm * 1.1 else "no"

    elif name == "quantum_walk":
        ot_top = data.get("top_quantum_lemas_OT", {})
        nt_top = data.get("top_quantum_lemas_NT", {})
        dp_ot = data.get("delta_p_ot", {})
        dp_nt = data.get("delta_p_nt", {})
        h["key_finding"] = {
            "top10_OT": [x.get("lemma") for x in ot_top.get("top10_quantum_preferred", [])],
            "top10_NT": [x.get("lemma") for x in nt_top.get("top10_quantum_preferred", [])],
            "mean_abs_delta_OT": dp_ot.get("stats", {}).get("mean_abs_delta"),
            "mean_abs_delta_NT": dp_nt.get("stats", {}).get("mean_abs_delta"),
        }
        h["adds_info"] = "yes"

    elif name == "mps_compression_ratio":
        cr = data.get("compression_ratios", {})
        pt = data.get("permutation_test_chi", {})
        h["key_finding"] = {
            "OT": {
                "chi": cr.get("OT", {}).get("bond_dimension_chi"),
                "ratio_classical_vs_mps": cr.get("OT", {}).get("ratio_classical_vs_mps"),
                "mps_better": cr.get("OT", {}).get("mps_more_compressible"),
            },
            "NT": {
                "chi": cr.get("NT", {}).get("bond_dimension_chi"),
                "ratio_classical_vs_mps": cr.get("NT", {}).get("ratio_classical_vs_mps"),
                "mps_better": cr.get("NT", {}).get("mps_more_compressible"),
            },
            "permutation_OT_p": pt.get("OT", {}).get("p_value"),
            "permutation_NT_p": pt.get("NT", {}).get("p_value"),
            "permutation_OT_sig": pt.get("OT", {}).get("significant"),
            "permutation_NT_sig": pt.get("NT", {}).get("significant"),
        }
        ot_sig = pt.get("OT", {}).get("significant", False)
        nt_sig = pt.get("NT", {}).get("significant", False)
        h["adds_info"] = "yes" if ot_sig or nt_sig else "no"

    return h


def generate_summary():
    summary = {
        "generated_at": datetime.now().isoformat(),
        "phase": "Fase 4 — Análisis Cuántico-Inspirado (numpy/scipy puro)",
        "investigations": {},
    }

    for name, files in RESULT_FILES.items():
        try:
            highlights = extract_highlights(name, files)
            summary["investigations"][name] = highlights
        except Exception as e:
            summary["investigations"][name] = {"status": f"ERROR: {e}"}

    # Count how many add info
    adds_yes = sum(1 for v in summary["investigations"].values()
                   if v.get("adds_info") == "yes")
    adds_no = sum(1 for v in summary["investigations"].values()
                  if v.get("adds_info") == "no")
    summary["overall_verdict"] = {
        "investigations_adding_info": adds_yes,
        "investigations_not_adding": adds_no,
        "conclusion": (
            f"Of 5 quantum-inspired analyses, {adds_yes} add information "
            f"beyond classical methods and {adds_no} do not."
        ),
    }

    out = RESULTS / "fase4_summary.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  Fase 4 Summary: {out}")
    return summary


def main():
    total_time = run_modules()
    print("Generando fase 4 summary report...")
    summary = generate_summary()

    print(f"\n{'='*60}")
    print(f"  FASE 4 — RESULTADOS CLAVE")
    print(f"{'='*60}")
    for inv_name, inv_data in summary["investigations"].items():
        adds = inv_data.get("adds_info", "?")
        icon = "✓" if adds == "yes" else "✗" if adds == "no" else "?"
        print(f"\n  [{icon}] {inv_name}")
        kf = inv_data.get("key_finding", {})
        for k, v in kf.items():
            if isinstance(v, dict):
                print(f"    {k}:")
                for k2, v2 in v.items():
                    print(f"      {k2}: {v2}")
            elif isinstance(v, list):
                print(f"    {k}: {v[:5]}{'...' if len(v) > 5 else ''}")
            else:
                print(f"    {k}: {v}")

    ov = summary.get("overall_verdict", {})
    print(f"\n{'='*60}")
    print(f"  VEREDICTO: {ov.get('conclusion', '?')}")
    print(f"  FASE 4 COMPLETA — {total_time:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
