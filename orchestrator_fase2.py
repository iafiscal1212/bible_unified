#!/usr/bin/env python3
"""
orchestrator_fase2.py — Lanza las 5 investigaciones profundas en paralelo.
"""
import subprocess, sys, time, json, os
from pathlib import Path
from datetime import datetime

BASE = Path(__file__).parent
RESULTS = BASE / "results"
LOGS = BASE / "logs"

MODULES = [
    "deep_zipf.py",
    "deep_numerical.py",
    "deep_bimodal.py",
    "deep_vn_ratio.py",
    "deep_proportions.py",
]

RESULT_FILES = {
    "deep_zipf": [
        RESULTS / "deep_zipf" / "zipf_by_book.json",
        RESULTS / "deep_zipf" / "zipf_surface_vs_lemma.json",
        RESULTS / "deep_zipf" / "ks_test_results.json",
    ],
    "deep_numerical": [
        RESULTS / "deep_numerical" / "permutation_test.json",
        RESULTS / "deep_numerical" / "bootstrap_ci.json",
        RESULTS / "deep_numerical" / "theoretical_vs_observed.json",
        RESULTS / "deep_numerical" / "ratio_by_book.json",
    ],
    "deep_bimodal": [
        RESULTS / "deep_bimodal" / "gmm_fit.json",
        RESULTS / "deep_bimodal" / "genre_classification.json",
        RESULTS / "deep_bimodal" / "length_by_genre.json",
        RESULTS / "deep_bimodal" / "mannwhitney_test.json",
    ],
    "deep_vn_ratio": [
        RESULTS / "deep_vn_ratio" / "vn_by_book.json",
        RESULTS / "deep_vn_ratio" / "changepoints.json",
        RESULTS / "deep_vn_ratio" / "correlations.json",
        RESULTS / "deep_vn_ratio" / "autocorrelation.json",
    ],
    "deep_proportions": [
        RESULTS / "deep_proportions" / "level_ratios.json",
        RESULTS / "deep_proportions" / "irrational_distances.json",
        RESULTS / "deep_proportions" / "book_size_distribution.json",
        RESULTS / "deep_proportions" / "selfsimilarity_test.json",
    ],
}


def run_modules():
    start = time.time()
    procs = {}

    print(f"{'='*60}")
    print(f"  BIBLE RESEARCH — FASE 2: ANÁLISIS PROFUNDO")
    print(f"  Inicio: {datetime.now().isoformat()}")
    print(f"  Investigaciones: {len(MODULES)}")
    print(f"{'='*60}\n")

    for mod in MODULES:
        mod_path = BASE / mod
        name = mod.replace(".py", "")
        print(f"  [LAUNCH] {name}")
        proc = subprocess.Popen(
            [sys.executable, str(mod_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(BASE),
        )
        procs[name] = {"proc": proc, "start": time.time()}

    print(f"\n  Esperando finalización...\n")
    completed = set()
    while len(completed) < len(procs):
        for name, info in procs.items():
            if name in completed:
                continue
            ret = info["proc"].poll()
            if ret is not None:
                elapsed = time.time() - info["start"]
                stdout = info["proc"].stdout.read().decode("utf-8", errors="replace")
                status = "OK" if ret == 0 else f"ERROR(rc={ret})"
                print(f"  [{status}] {name:20s} — {elapsed:.1f}s")
                if stdout.strip():
                    print(f"           {stdout.strip()}")
                completed.add(name)
        if len(completed) < len(procs):
            time.sleep(1)

    total_time = time.time() - start
    print(f"\n{'='*60}")
    print(f"  Todas las investigaciones completadas en {total_time:.1f}s")
    print(f"{'='*60}\n")
    return total_time


def extract_deep_highlights(name, files):
    """Extrae hallazgos clave de cada investigación."""
    data = {}
    for f in files:
        if f.exists():
            with open(f) as fh:
                data[f.stem] = json.load(fh)

    h = {"status": "OK", "files": [str(f) for f in files if f.exists()]}

    if name == "deep_zipf":
        svl = data.get("zipf_surface_vs_lemma", {})
        bc = svl.get("by_corpus", {})
        da = svl.get("delta_analysis", {})
        h["key_finding"] = {
            "ot_surface_s": bc.get("OT", {}).get("surface", {}).get("s"),
            "ot_lemma_s": bc.get("OT", {}).get("lemma", {}).get("s"),
            "nt_surface_s": bc.get("NT", {}).get("surface", {}).get("s"),
            "nt_lemma_s": bc.get("NT", {}).get("lemma", {}).get("s"),
            "ot_delta_s_mean": da.get("ot_delta_s_mean"),
            "ot_delta_s_cv": da.get("ot_delta_s_cv"),
            "nt_delta_s_mean": da.get("nt_delta_s_mean"),
            "mannwhitney": da.get("mannwhitney_ot_vs_nt"),
        }

    elif name == "deep_numerical":
        perm = data.get("permutation_test", {})
        boot = data.get("bootstrap_ci", {})
        theo = data.get("theoretical_vs_observed", {})
        h["key_finding"] = {
            "observed_ratio": perm.get("observed_ratio"),
            "permutation_p_value": perm.get("p_value"),
            "bootstrap_95_ci": boot.get("ci_95"),
            "ci_contains_1": boot.get("contains_1"),
            "theoretical_ratio": theo.get("theoretical_ratio_uniform"),
        }

    elif name == "deep_bimodal":
        gmm = data.get("gmm_fit", {})
        lbg = data.get("length_by_genre", {})
        mw = data.get("mannwhitney_test", {})
        h["key_finding"] = {
            "gmm_global": gmm.get("global"),
            "genre_stats": {g: {"mean": v.get("mean"), "n": v.get("n_verses")}
                            for g, v in lbg.items()},
            "mannwhitney_tests": {k: {"p": v.get("p_value"), "sig": v.get("significant_005")}
                                   for k, v in mw.items()},
        }

    elif name == "deep_vn_ratio":
        cp = data.get("changepoints", {})
        corr = data.get("correlations", {})
        h["key_finding"] = {
            "changepoints": cp.get("changepoints", [])[:5],
            "correlations": corr,
        }

    elif name == "deep_proportions":
        lr = data.get("level_ratios", {})
        ir = data.get("irrational_distances", {})
        ss = data.get("selfsimilarity_test", {})
        h["key_finding"] = {
            "total_ratios": lr.get("total_ratio"),
            "nearest_irrationals": {k: {"value": v.get("observed_value"),
                                         "nearest": v.get("nearest_constant"),
                                         "distance": v.get("nearest_distance")}
                                     for k, v in ir.items()},
            "self_similarity": {
                "ks_stat": ss.get("ks_stat"),
                "p_value": ss.get("p_value"),
                "self_similar": ss.get("self_similar"),
            },
        }

    return h


def generate_summary():
    summary = {
        "generated_at": datetime.now().isoformat(),
        "phase": "Fase 2 — Análisis Profundo",
        "investigations": {},
    }

    for name, files in RESULT_FILES.items():
        try:
            highlights = extract_deep_highlights(name, files)
            summary["investigations"][name] = highlights
        except Exception as e:
            summary["investigations"][name] = {"status": f"ERROR: {e}"}

    out = RESULTS / "deep_summary.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  Deep Summary: {out}")
    return summary


def main():
    total_time = run_modules()
    print("Generando deep summary report...")
    summary = generate_summary()

    # Print summary to terminal
    print(f"\n{'='*60}")
    print(f"  FASE 2 — RESULTADOS CLAVE")
    print(f"{'='*60}")
    for inv_name, inv_data in summary["investigations"].items():
        print(f"\n  --- {inv_name} ---")
        kf = inv_data.get("key_finding", {})
        for k, v in kf.items():
            if isinstance(v, dict):
                print(f"    {k}:")
                for k2, v2 in v.items():
                    print(f"      {k2}: {v2}")
            else:
                print(f"    {k}: {v}")

    print(f"\n{'='*60}")
    print(f"  FASE 2 COMPLETA — {total_time:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
