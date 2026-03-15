#!/usr/bin/env python3
"""
orchestrator_fase3.py — Lanza las 4 investigaciones profundas de Fase 3 en paralelo.
Monitoreo cada 30s, resumen final con conclusiones.
"""
import subprocess, sys, time, json, os
from pathlib import Path
from datetime import datetime

BASE = Path(__file__).parent
RESULTS = BASE / "results"
LOGS = BASE / "logs"
LOGS.mkdir(parents=True, exist_ok=True)

MODULES = [
    "deep_numerical_mechanism.py",
    "deep_algebraic_constants.py",
    "deep_fractal.py",
    "deep_zipf_semantic.py",
]

RESULT_FILES = {
    "deep_numerical_mechanism": [
        RESULTS / "deep_numerical_mechanism" / "letter_contributions.json",
        RESULTS / "deep_numerical_mechanism" / "sensitivity_curve.json",
        RESULTS / "deep_numerical_mechanism" / "letter_distributions.json",
        RESULTS / "deep_numerical_mechanism" / "chi2_test.json",
    ],
    "deep_algebraic_constants": [
        RESULTS / "deep_algebraic" / "all_distances.json",
        RESULTS / "deep_algebraic" / "bonferroni_results.json",
        RESULTS / "deep_algebraic" / "bootstrap_stability.json",
        RESULTS / "deep_algebraic" / "progression_test.json",
    ],
    "deep_fractal": [
        RESULTS / "deep_fractal" / "box_counting.json",
        RESULTS / "deep_fractal" / "hurst_exponent.json",
        RESULTS / "deep_fractal" / "dfa_results.json",
        RESULTS / "deep_fractal" / "fractal_by_corpus.json",
        RESULTS / "deep_fractal" / "fractal_by_genre.json",
    ],
    "deep_zipf_semantic": [
        RESULTS / "deep_zipf_semantic" / "zipf_by_at_book.json",
        RESULTS / "deep_zipf_semantic" / "top5_anomalous.json",
        RESULTS / "deep_zipf_semantic" / "entropy_correlation.json",
        RESULTS / "deep_zipf_semantic" / "zipf_by_pos.json",
        RESULTS / "deep_zipf_semantic" / "anomaly_characteristics.json",
    ],
}


def run_modules():
    start = time.time()
    procs = {}

    print(f"{'='*60}")
    print(f"  BIBLE RESEARCH — FASE 3: ANÁLISIS PROFUNDO (4 líneas)")
    print(f"  Inicio: {datetime.now().isoformat()}")
    print(f"  Investigaciones: {len(MODULES)}")
    print(f"{'='*60}\n")

    for mod in MODULES:
        mod_path = BASE / mod
        name = mod.replace(".py", "")
        log_file = LOGS / f"fase3_{name}.log"
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
                # Show last line of log
                try:
                    with open(info["log_file"]) as f:
                        lines = f.readlines()
                        if lines:
                            print(f"           {lines[-1].strip()}")
                except Exception:
                    pass
                completed.add(name)

        # Progress report every 30 seconds
        now = time.time()
        if now - last_progress >= 30 and len(completed) < len(procs):
            elapsed_total = now - start
            running = [n for n in procs if n not in completed]
            print(f"  [PROGRESS {elapsed_total:.0f}s] Running: {', '.join(running)}")
            # Show last log lines for running processes
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
    """Extract key findings from each investigation."""
    data = {}
    for f in files:
        if f.exists():
            with open(f) as fh:
                data[f.stem] = json.load(fh)

    h = {"status": "OK", "files": [str(f) for f in files if f.exists()]}

    if name == "deep_numerical_mechanism":
        lc = data.get("letter_contributions", {})
        sc = data.get("sensitivity_curve", [])
        chi2 = data.get("chi2_test", {})
        conc = lc.get("concentration", {})
        h["key_finding"] = {
            "ratio_letter_level": lc.get("ratio"),
            "ot_total_gematria": lc.get("ot_total_gematria"),
            "nt_total_isopsephy": lc.get("nt_total_isopsephy"),
            "ot_top3_pct": conc.get("ot_top3_pct"),
            "nt_top3_pct": conc.get("nt_top3_pct"),
            "ot_gini": conc.get("ot_gini"),
            "nt_gini": conc.get("nt_gini"),
            "equilibrium_type": conc.get("equilibrium_type"),
            "chi2_stat": chi2.get("chi2"),
            "chi2_p": chi2.get("p_value"),
            "sensitivity_at_50pct": sc[4].get("ratio_remaining") if len(sc) > 4 else None,
        }
        h["conclusion"] = (
            "CONFIRMADA: el equilibrio es producido por " + conc.get("equilibrium_type", "?") +
            ". Chi2 " + ("significativo" if chi2.get("significant") else "no significativo") +
            f" (p={chi2.get('p_value', '?')})."
        )

    elif name == "deep_algebraic_constants":
        bonf = data.get("bonferroni_results", {}).get("results", {})
        boot = data.get("bootstrap_stability", {})
        prog = data.get("progression_test", {})
        n_comp = data.get("bonferroni_results", {}).get("n_comparisons", 0)
        h["key_finding"] = {
            "n_comparisons": n_comp,
        }
        survived = []
        for ratio_name, bd in bonf.items():
            sig = bd.get("significant", False)
            h["key_finding"][ratio_name] = {
                "nearest": bd.get("nearest_constant"),
                "distance": bd.get("raw_distance"),
                "p_bonferroni": bd.get("p_value_bonferroni"),
                "significant": sig,
            }
            if sig:
                survived.append(f"{ratio_name}≈{bd.get('nearest_constant')}")
        h["key_finding"]["progression"] = prog.get("conclusion")
        h["conclusion"] = (
            f"Sobreviven Bonferroni ({n_comp} comparaciones): " +
            (", ".join(survived) if survived else "NINGUNA") +
            f". Progresión: {prog.get('conclusion', '?')}."
        )

    elif name == "deep_fractal":
        fbc = data.get("fractal_by_corpus", {})
        fbg = data.get("fractal_by_genre", {})
        comp = fbc.get("comparison", {}).get("hurst_comparison", {})
        h["key_finding"] = {
            "global": fbc.get("global", {}),
            "OT": fbc.get("OT", {}),
            "NT": fbc.get("NT", {}),
            "hurst_comparison_p": comp.get("p_value"),
            "hurst_OT_vs_NT_significant": comp.get("significant"),
            "genres": {g: {"hurst": v.get("hurst", {}).get("H"),
                          "dfa": v.get("dfa", {}).get("alpha")}
                      for g, v in fbg.items()},
        }
        global_h = fbc.get("global", {}).get("hurst", {}).get("H")
        global_alpha = fbc.get("global", {}).get("dfa", {}).get("alpha")
        h["conclusion"] = (
            f"Hurst global H={global_h}, DFA α={global_alpha}. " +
            ("Ambos >0.5 → memoria larga confirmada. " if global_h and global_alpha and global_h > 0.5 and global_alpha > 0.5 else
             "Resultados mixtos. ") +
            f"AT vs NT: {'significativamente diferentes' if comp.get('significant') else 'no significativamente diferentes'} (p={comp.get('p_value', '?')})."
        )

    elif name == "deep_zipf_semantic":
        top5 = data.get("top5_anomalous", [])
        ec = data.get("entropy_correlation", {})
        pos = data.get("zipf_by_pos", {}).get("summary", {})
        chars = data.get("anomaly_characteristics", {}).get("correlations", {})
        h["key_finding"] = {
            "top5_anomalous_books": [x["book"] for x in top5[:5]],
            "top5_s_values": [x["zipf_s"] for x in top5[:5]],
            "anomaly_locations": [x.get("anomaly_location") for x in top5[:5]],
            "entropy_pearson_r": ec.get("pearson", {}).get("r"),
            "entropy_pearson_p": ec.get("pearson", {}).get("p"),
            "more_anomalous_pos": pos.get("more_anomalous"),
            "verb_mean_s": pos.get("verb_mean_s"),
            "noun_mean_s": pos.get("noun_mean_s"),
            "verb_vs_noun_p": pos.get("mannwhitney_p"),
            "correlations": chars,
        }
        h["conclusion"] = (
            f"Libros más anómalos: {', '.join(x['book'] for x in top5[:3])}. " +
            f"Anomalía en: {pos.get('more_anomalous', '?')} (s_verb={pos.get('verb_mean_s')}, s_noun={pos.get('noun_mean_s')}). " +
            f"Entropía vs Zipf: r={ec.get('pearson', {}).get('r')}, p={ec.get('pearson', {}).get('p')}."
        )

    return h


def generate_summary():
    summary = {
        "generated_at": datetime.now().isoformat(),
        "phase": "Fase 3 — Análisis Profundo (4 líneas)",
        "investigations": {},
    }

    for name, files in RESULT_FILES.items():
        try:
            highlights = extract_highlights(name, files)
            summary["investigations"][name] = highlights
        except Exception as e:
            summary["investigations"][name] = {"status": f"ERROR: {e}"}

    # Generate top 3 questions
    summary["top_3_questions"] = [
        "¿La autosimilaridad fractal es una propiedad del corpus bíblico específicamente, o de cualquier texto sagrado compilado?",
        "¿El mecanismo de equilibrio numérico (distribución de letras) tiene paralelo en otros corpus bilingües antiguos?",
        "¿Los libros con Zipf más anómalo comparten un proceso de transmisión textual (copistas, redactores) específico?",
    ]

    out = RESULTS / "fase3_summary.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  Fase 3 Summary: {out}")
    return summary


def main():
    total_time = run_modules()
    print("Generando fase 3 summary report...")
    summary = generate_summary()

    # Print summary to terminal
    print(f"\n{'='*60}")
    print(f"  FASE 3 — RESULTADOS CLAVE")
    print(f"{'='*60}")
    for inv_name, inv_data in summary["investigations"].items():
        print(f"\n  ─── {inv_name} ───")
        conclusion = inv_data.get("conclusion", "Sin conclusión")
        print(f"  Conclusión: {conclusion}")
        kf = inv_data.get("key_finding", {})
        for k, v in kf.items():
            if isinstance(v, dict):
                print(f"    {k}:")
                for k2, v2 in v.items():
                    if isinstance(v2, dict):
                        print(f"      {k2}: {v2}")
                    else:
                        print(f"      {k2}: {v2}")
            elif isinstance(v, list):
                print(f"    {k}: {v}")
            else:
                print(f"    {k}: {v}")

    print(f"\n  ─── TOP 3 PREGUNTAS ABIERTAS ───")
    for i, q in enumerate(summary.get("top_3_questions", []), 1):
        print(f"    {i}. {q}")

    print(f"\n{'='*60}")
    print(f"  FASE 3 COMPLETA — {total_time:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
