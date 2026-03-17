#!/usr/bin/env python3
"""
Fase 19 — Script 4: degradation_model.py

Model H degradation under controlled vs free transmission:
- Load measured decay rates from DSS Isaiah comparison and Phase 10
- Simulate controlled and free degradation trajectories
- Test irreversibility: can late standardization recover AT-like H?
- Predict H values at specific delays (65, 70, 170 years)

Yasna INVALIDADO — NO incluir.
"""

import json
import logging
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent
RESULTS_DIR = BASE / "results" / "degradation_model"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Load rates
# ═══════════════════════════════════════════════════════════════

def load_rates():
    """Load decay rates from Phase 7 (DSS) and Phase 10 (decay)."""
    rates = {}

    # DSS comparison (the only MEASURED rate)
    dss_file = BASE / "results" / "dss" / "dss_isaiah_comparison.json"
    if dss_file.exists():
        with open(dss_file) as f:
            dss = json.load(f)
        corrected = dss.get("CORRECTED_COMPARISON", {})
        dss_v = corrected.get("dss_verses", {})
        wlc_v = corrected.get("wlc_verses", {})
        delta_H = corrected.get("delta_H", 0)
        rates["controlled_measured"] = {
            "source": "DSS Isaiah vs WLC (1,100 years)",
            "H_ancient": dss_v.get("hurst_H"),
            "H_recent": wlc_v.get("hurst_H"),
            "delta_H": delta_H,
            "delta_centuries": 11.0,
            "rate_per_century": round(abs(delta_H) / 11.0, 6),
        }
        log.info(f"  DSS controlled rate: {rates['controlled_measured']['rate_per_century']} H/century")

    # Decay rates from Phase 10
    decay_file = BASE / "results" / "decay" / "decay_rates.json"
    if decay_file.exists():
        with open(decay_file) as f:
            decay = json.load(f)
        for entry in decay.get("decay_rates", []):
            corpus = entry["corpus"]
            rates[corpus] = {
                "transmission": entry.get("transmission"),
                "rate_upper": entry.get("rate_H_per_century_upper_bound",
                                        abs(entry.get("rate_H_per_century", 0))),
                "rate_lower": entry.get("rate_H_per_century_lower_bound", 0),
                "type": entry.get("type"),
                "H_current": entry.get("H_current"),
            }

    return rates


# ═══════════════════════════════════════════════════════════════
# Simulation
# ═══════════════════════════════════════════════════════════════

def simulate_degradation(H_initial, rate_per_century, delay_years):
    """Simulate H degradation over delay_years at given rate."""
    delay_centuries = delay_years / 100.0
    return round(H_initial - rate_per_century * delay_centuries, 4)


def run_simulations(rates):
    """Run degradation simulations for a range of delays."""
    delays = [0, 10, 20, 30, 50, 65, 70, 80, 100, 120, 150, 170, 200, 250, 300, 400]

    # Controlled rate from DSS (measured)
    controlled_rate = rates.get("controlled_measured", {}).get("rate_per_century", 0.000755)

    # Free rate estimates: collect upper bounds
    free_rates = []
    for name, data in rates.items():
        if isinstance(data, dict) and data.get("transmission") == "free":
            r = data.get("rate_upper", 0)
            if 0 < r < 1:  # filter out absurd values
                free_rates.append(r)

    free_rate_mean = float(np.mean(free_rates)) if free_rates else 0.027
    free_rate_max = float(np.max(free_rates)) if free_rates else 0.052

    # AT H distribution: use corpus-level values
    fp_file = BASE / "results" / "unified_model" / "fitted_params.json"
    H_at = 0.85  # default
    if fp_file.exists():
        with open(fp_file) as f:
            fp = json.load(f)
        at_target = fp.get("AT", {}).get("target", {})
        H_at = at_target.get("H", 0.8766)

    # AT/NT boundary: approximate from classifier
    H_boundary = 0.75

    simulations = {
        "parameters": {
            "controlled_rate_per_century": round(controlled_rate, 6),
            "free_rate_mean_per_century": round(free_rate_mean, 6),
            "free_rate_max_per_century": round(free_rate_max, 6),
            "H_initial_AT": round(H_at, 4),
            "H_AT_NT_boundary_approx": H_boundary,
            "free_rates_used": free_rates,
        },
        "trajectories": [],
    }

    for delay in delays:
        H_ctrl = simulate_degradation(H_at, controlled_rate, delay)
        H_free_m = simulate_degradation(H_at, free_rate_mean, delay)
        H_free_x = simulate_degradation(H_at, free_rate_max, delay)

        simulations["trajectories"].append({
            "delay_years": delay,
            "H_controlled": H_ctrl,
            "H_free_mean": H_free_m,
            "H_free_worst": H_free_x,
            "controlled_AT_like": H_ctrl > H_boundary,
            "free_mean_AT_like": H_free_m > H_boundary,
            "free_worst_AT_like": H_free_x > H_boundary,
        })

    # Critical delays: when H crosses boundary
    def critical_delay(rate):
        if rate <= 0:
            return None
        return round((H_at - H_boundary) / rate * 100, 0)

    simulations["critical_delays"] = {
        "controlled": critical_delay(controlled_rate),
        "free_mean": critical_delay(free_rate_mean),
        "free_worst": critical_delay(free_rate_max),
        "unit": "years",
        "interpretation": "Delay in years at which H crosses AT/NT boundary",
    }

    return simulations


# ═══════════════════════════════════════════════════════════════
# Irreversibility test
# ═══════════════════════════════════════════════════════════════

def test_irreversibility():
    """Can late standardization recover AT-like H?"""
    reclass_file = BASE / "results" / "classifier_corrected" / "reclassification_all_corpora.json"
    if not reclass_file.exists():
        return {"status": "no_data"}

    with open(reclass_file) as f:
        reclassified = json.load(f)

    # Corpora that had free transmission then late standardization
    late_controlled = {
        "NT": {
            "delay": 300,
            "description": "Free copying 50–350 CE, Byzantine standardization ~350 CE",
        },
        "Mishnah": {
            "delay": 400,
            "description": "Free oral debate ~200 BCE–200 CE, Judah haNasi codification ~200 CE",
        },
    }

    # Also check Homero if available (but likely excluded due to missing features)
    if "Homero" in reclassified:
        late_controlled["Homero"] = {
            "delay": 400,
            "description": "Free oral ~750 BCE, Alexandrian standardization ~350 BCE",
        }

    results = {}
    for name, info in late_controlled.items():
        if name in reclassified and reclassified[name].get("status") == "classified":
            data = reclassified[name]
            results[name] = {
                "delay": info["delay"],
                "description": info["description"],
                "predicted": data["predicted_class"],
                "P_AT": data["P_AT"],
                "H": data.get("features", {}).get("H"),
                "recovered_AT_like": data["predicted_class"] == "AT",
            }

    n_tested = len(results)
    n_recovered = sum(1 for r in results.values() if r["recovered_AT_like"])

    if n_tested == 0:
        conclusion = "NO_DATA"
        interpretation = "No late-standardized corpora available for testing."
    elif n_recovered == 0:
        conclusion = "IRREVERSIBLE"
        interpretation = (
            f"0/{n_tested} late-standardized corpora recovered AT-like signatures. "
            "Free transmission degrades H irreversibly — later codification "
            "cannot restore the original long-range correlation structure."
        )
    else:
        conclusion = f"PARTIALLY_REVERSIBLE"
        interpretation = (
            f"{n_recovered}/{n_tested} late-standardized corpora show AT-like H. "
            "Some recovery may be possible, but the sample is too small to confirm."
        )

    return {
        "question": "Can late standardization (after free transmission) recover AT-like H?",
        "corpora": results,
        "n_tested": n_tested,
        "n_recovered_AT": n_recovered,
        "conclusion": conclusion,
        "interpretation": interpretation,
    }


# ═══════════════════════════════════════════════════════════════
# Predictions for gap delays
# ═══════════════════════════════════════════════════════════════

def predict_gap_delays(simulations):
    """Predict H for gap corpus delays (65, 70, 170 years)."""
    params = simulations["parameters"]
    H0 = params["H_initial_AT"]
    boundary = params["H_AT_NT_boundary_approx"]

    predictions = {}
    for delay in [65, 70, 170]:
        H_ctrl = simulate_degradation(H0, params["controlled_rate_per_century"], delay)
        H_free_m = simulate_degradation(H0, params["free_rate_mean_per_century"], delay)
        H_free_x = simulate_degradation(H0, params["free_rate_max_per_century"], delay)

        predictions[f"delay_{delay}"] = {
            "delay_years": delay,
            "H_if_controlled": H_ctrl,
            "H_if_free_mean": H_free_m,
            "H_if_free_worst": H_free_x,
            "boundary": boundary,
            "controlled_prediction": "AT-like" if H_ctrl > boundary else "NT-like",
            "free_mean_prediction": "AT-like" if H_free_m > boundary else "NT-like",
            "free_worst_prediction": "AT-like" if H_free_x > boundary else "NT-like",
        }

    return predictions


# ═══════════════════════════════════════════════════════════════
# Compare predictions with observed
# ═══════════════════════════════════════════════════════════════

def compare_predicted_vs_observed(predictions):
    """Compare model predictions with actual classified corpora."""
    comparisons = {}

    # 1 Clemente (delay=65)
    clem_file = BASE / "results" / "gap_corpora" / "1_clemente_metrics.json"
    if clem_file.exists():
        with open(clem_file) as f:
            data = json.load(f)
        pred = predictions.get("delay_65", {})
        observed_pred = data.get("predicted_class")
        comparisons["1_Clemente"] = {
            "delay": 65,
            "observed_H": data.get("H"),
            "observed_class": observed_pred,
            "model_controlled_H": pred.get("H_if_controlled"),
            "model_free_mean_H": pred.get("H_if_free_mean"),
            "consistent_with_controlled": pred.get("controlled_prediction") == observed_pred,
            "consistent_with_free": pred.get("free_mean_prediction") == observed_pred,
        }

    # Didache (delay=70)
    did_file = BASE / "results" / "classifier_corrected" / "reclassification_all_corpora.json"
    if did_file.exists():
        with open(did_file) as f:
            reclassified = json.load(f)
        if "Didache" in reclassified and reclassified["Didache"].get("status") == "classified":
            did = reclassified["Didache"]
            pred = predictions.get("delay_70", {})
            comparisons["Didache"] = {
                "delay": 70,
                "observed_H": did.get("features", {}).get("H"),
                "observed_class": did["predicted_class"],
                "model_controlled_H": pred.get("H_if_controlled"),
                "model_free_mean_H": pred.get("H_if_free_mean"),
                "consistent_with_controlled": pred.get("controlled_prediction") == did["predicted_class"],
                "consistent_with_free": pred.get("free_mean_prediction") == did["predicted_class"],
            }

    # Tosefta (delay=170)
    tosefta_file = BASE / "results" / "gap_corpora" / "tosefta_corrected.json"
    if tosefta_file.exists():
        with open(tosefta_file) as f:
            data = json.load(f)
        tosefta_pred = data.get("predicted_class_corrected") or data.get("predicted_class")
        pred = predictions.get("delay_170", {})
        if tosefta_pred:
            comparisons["Tosefta"] = {
                "delay": 170,
                "observed_H": data.get("H"),
                "observed_class": tosefta_pred,
                "model_controlled_H": pred.get("H_if_controlled"),
                "model_free_mean_H": pred.get("H_if_free_mean"),
                "consistent_with_controlled": pred.get("controlled_prediction") == tosefta_pred,
                "consistent_with_free": pred.get("free_mean_prediction") == tosefta_pred,
            }

    return comparisons


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    log.info("=" * 70)
    log.info("FASE 19 — Script 4: Modelo de degradación")
    log.info("=" * 70)

    # 1. Load rates
    log.info("\n[1] Cargando tasas de degradación...")
    rates = load_rates()
    log.info(f"  {len(rates)} entradas")

    with open(RESULTS_DIR / "loaded_rates.json", "w") as f:
        json.dump(rates, f, indent=2, ensure_ascii=False, default=str)

    # 2. Simulations
    log.info("\n[2] Simulando trayectorias de degradación...")
    simulations = run_simulations(rates)
    cd = simulations["critical_delays"]
    log.info(f"  Delay crítico (controlled): {cd['controlled']} años")
    log.info(f"  Delay crítico (free mean): {cd['free_mean']} años")
    log.info(f"  Delay crítico (free worst): {cd['free_worst']} años")

    with open(RESULTS_DIR / "simulations.json", "w") as f:
        json.dump(simulations, f, indent=2, ensure_ascii=False)

    # 3. Irreversibility test
    log.info("\n[3] Test de irreversibilidad...")
    irreversibility = test_irreversibility()
    log.info(f"  Conclusión: {irreversibility.get('conclusion')}")

    with open(RESULTS_DIR / "irreversibility.json", "w") as f:
        json.dump(irreversibility, f, indent=2, ensure_ascii=False)

    # 4. Gap corpus predictions
    log.info("\n[4] Predicciones para delays 65, 70, 170 años...")
    predictions = predict_gap_delays(simulations)
    for k, v in predictions.items():
        log.info(f"  {k}: ctrl→{v['controlled_prediction']}, "
                 f"free→{v['free_mean_prediction']}")

    with open(RESULTS_DIR / "gap_predictions.json", "w") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    # 5. Compare with observed
    log.info("\n[5] Comparando predicciones vs observado...")
    comparisons = compare_predicted_vs_observed(predictions)
    for name, comp in comparisons.items():
        log.info(f"  {name}: observed={comp['observed_class']}, "
                 f"ctrl_model={'OK' if comp['consistent_with_controlled'] else 'MISS'}, "
                 f"free_model={'OK' if comp['consistent_with_free'] else 'MISS'}")

    with open(RESULTS_DIR / "predicted_vs_observed.json", "w") as f:
        json.dump(comparisons, f, indent=2, ensure_ascii=False)

    # 6. Summary
    summary = {
        "controlled_rate_measured": rates.get("controlled_measured", {}).get("rate_per_century"),
        "critical_delays": simulations["critical_delays"],
        "irreversibility": irreversibility.get("conclusion"),
        "gap_predictions": predictions,
        "predicted_vs_observed": comparisons,
    }

    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    log.info(f"\n{'=' * 70}")
    log.info("Script 4 completado.")


if __name__ == "__main__":
    main()
