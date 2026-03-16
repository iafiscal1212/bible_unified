#!/usr/bin/env python3
"""
Fase 14 — Script 2: Unified Model
Modelo unificado AR(1)-ARFIMA para AT/Corán/Rig Veda.

Ajusta (φ, d, σ_ε, σ_η) por corpus:
- φ = AR(1) coefficient (short-range)
- d = fractional differencing (long-range)
- σ_ε = innovation noise
- σ_η = topic-level noise

Boundary analysis: ¿qué separa AT de NT en el espacio de parámetros?
"""

import json
import logging
import time
import re
import numpy as np
from pathlib import Path
from scipy import stats as sp_stats
from scipy.optimize import minimize
from collections import defaultdict

BASE = Path(__file__).resolve().parent
RESULTS_DIR = BASE / "results" / "unified_model"
LOG_DIR = BASE / "logs"
BIBLE_JSON = BASE / "bible_unified.json"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "fase14_unified_model.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


# ── Reusable functions (from compositional_rule.py) ────────────────────

def hurst_exponent_rs(series):
    series = np.asarray(series, dtype=float)
    n = len(series)
    if n < 20:
        return float("nan")
    min_block, max_block = 10, n // 2
    sizes, rs_values = [], []
    block = min_block
    while block <= max_block:
        sizes.append(block)
        n_blocks = n // block
        rs_list = []
        for i in range(n_blocks):
            seg = series[i * block:(i + 1) * block]
            devs = np.cumsum(seg - seg.mean())
            R = devs.max() - devs.min()
            S = seg.std(ddof=1)
            if S > 0:
                rs_list.append(R / S)
        if rs_list:
            rs_values.append(np.mean(rs_list))
        block = int(block * 1.5)
        if block == sizes[-1]:
            block += 1
    if len(sizes) < 3:
        return float("nan")
    slope, _, r, _, _ = sp_stats.linregress(np.log(sizes), np.log(rs_values))
    return float(slope)


def autocorr_lag1(series):
    arr = np.asarray(series, dtype=float)
    if len(arr) < 10:
        return float("nan")
    return float(np.corrcoef(arr[:-1], arr[1:])[0, 1])


def dfa_exponent(series):
    """Detrended Fluctuation Analysis."""
    series = np.asarray(series, dtype=float)
    n = len(series)
    if n < 50:
        return float("nan")
    y = np.cumsum(series - series.mean())
    sizes = []
    flucts = []
    s = 10
    while s <= n // 4:
        sizes.append(s)
        n_segs = n // s
        f2 = []
        for i in range(n_segs):
            seg = y[i * s:(i + 1) * s]
            x = np.arange(s)
            coef = np.polyfit(x, seg, 1)
            trend = np.polyval(coef, x)
            f2.append(np.mean((seg - trend) ** 2))
        flucts.append(np.sqrt(np.mean(f2)))
        s = int(s * 1.5)
        if s == sizes[-1]:
            s += 1
    if len(sizes) < 3:
        return float("nan")
    slope, _, _, _, _ = sp_stats.linregress(np.log(sizes), np.log(flucts))
    return float(slope)


# ── ARFIMA simulation ──────────────────────────────────────────────────

def arfima_weights(d, n_weights=200):
    """Compute truncated MA(∞) weights for fractional differencing."""
    w = np.zeros(n_weights)
    w[0] = 1.0
    for k in range(1, n_weights):
        w[k] = w[k - 1] * (d + k - 1) / k
    return w


def generate_arfima_ar1(n, phi, d, sigma_eps, sigma_eta, topic_dur=30, rng=None):
    """Generate series from AR(1) + ARFIMA(0,d,0) + hierarchical topics.

    Model:
    1. Topic-level: μ_topic ~ N(global_mean, σ_η), duration ~ Geom(1/topic_dur)
    2. AR(1) within-topic: x_t = φ·x_{t-1} + ε_t, ε ~ ARFIMA(0,d,0) innovations
    """
    if rng is None:
        rng = np.random.default_rng()

    global_mean = 12.0  # typical verse length

    # Generate topic boundaries
    topics = []
    pos = 0
    while pos < n:
        dur = max(5, rng.geometric(1.0 / topic_dur))
        mu = max(3, global_mean + rng.normal(0, sigma_eta))
        topics.append((pos, min(pos + dur, n), mu))
        pos += dur

    # Generate ARFIMA innovations
    white = rng.normal(0, sigma_eps, n + 200)
    weights = arfima_weights(d, 200)
    innovations = np.convolve(white, weights, mode='full')[:n]

    # AR(1) with topics
    series = np.zeros(n)
    series[0] = global_mean
    for start, end, mu in topics:
        for t in range(start, end):
            if t == 0:
                series[t] = mu + innovations[t]
            else:
                prev = series[t - 1] - mu
                series[t] = mu + phi * prev + innovations[t]
    series = np.maximum(1, np.round(series)).astype(float)
    return series


# ── Fitting ────────────────────────────────────────────────────────────

def compute_targets(series):
    """Compute target statistics from empirical series."""
    return {
        "H": hurst_exponent_rs(series),
        "AC1": autocorr_lag1(series),
        "DFA": dfa_exponent(series),
        "mean": float(np.mean(series)),
        "std": float(np.std(series)),
        "cv": float(np.std(series) / np.mean(series)) if np.mean(series) > 0 else 0,
    }


def loss_function(params, target, n_series=50, n_len=1000, rng_seed=42):
    """Loss = sum of squared deviations of simulated stats from target."""
    phi, d, sigma_eps, sigma_eta = params

    # Parameter constraints
    if not (-0.99 < phi < 0.99):
        return 1e6
    if not (-0.49 < d < 0.49):
        return 1e6
    if sigma_eps <= 0 or sigma_eta <= 0:
        return 1e6

    rng = np.random.default_rng(rng_seed)
    hs, ac1s, dfas = [], [], []
    for _ in range(n_series):
        try:
            s = generate_arfima_ar1(n_len, phi, d, sigma_eps, sigma_eta, rng=rng)
            hs.append(hurst_exponent_rs(s))
            ac1s.append(autocorr_lag1(s))
            dfas.append(dfa_exponent(s))
        except Exception:
            continue

    if len(hs) < 10:
        return 1e6

    # Filter NaNs
    hs = [h for h in hs if not np.isnan(h)]
    ac1s = [a for a in ac1s if not np.isnan(a)]
    dfas = [d_ for d_ in dfas if not np.isnan(d_)]

    if not hs or not ac1s:
        return 1e6

    loss = 0
    loss += (np.mean(hs) - target["H"]) ** 2 * 10  # weight H more
    loss += (np.mean(ac1s) - target["AC1"]) ** 2 * 5
    if target["DFA"] and not np.isnan(target["DFA"]) and dfas:
        loss += (np.mean(dfas) - target["DFA"]) ** 2 * 5
    return loss


def fit_model(target, label, n_restarts=8):
    """Fit ARFIMA-AR1 model to target statistics."""
    log.info(f"  Fitting {label}... (target H={target['H']:.3f}, AC1={target['AC1']:.3f})")

    best_loss = 1e10
    best_params = None
    rng = np.random.default_rng(123)

    for restart in range(n_restarts):
        # Random initial guess
        phi0 = rng.uniform(0.0, 0.5)
        d0 = rng.uniform(0.0, 0.4)
        se0 = rng.uniform(0.5, 3.0)
        sh0 = rng.uniform(0.5, 3.0)
        x0 = [phi0, d0, se0, sh0]

        try:
            result = minimize(
                loss_function,
                x0,
                args=(target, 30, 800, 42 + restart),
                method="Nelder-Mead",
                options={"maxiter": 300, "xatol": 0.01, "fatol": 0.001},
            )
            if result.fun < best_loss:
                best_loss = result.fun
                best_params = result.x
                log.info(f"    restart {restart}: loss={result.fun:.4f}, "
                         f"params=[{', '.join(f'{p:.3f}' for p in result.x)}]")
        except Exception as e:
            log.warning(f"    restart {restart} failed: {e}")

    if best_params is None:
        return None

    phi, d, sigma_eps, sigma_eta = best_params

    # Validate: generate series with fitted params and check
    rng2 = np.random.default_rng(999)
    val_hs, val_ac1s, val_dfas = [], [], []
    for _ in range(100):
        s = generate_arfima_ar1(800, phi, d, sigma_eps, sigma_eta, rng=rng2)
        val_hs.append(hurst_exponent_rs(s))
        val_ac1s.append(autocorr_lag1(s))
        val_dfas.append(dfa_exponent(s))

    val_hs = [h for h in val_hs if not np.isnan(h)]
    val_ac1s = [a for a in val_ac1s if not np.isnan(a)]
    val_dfas = [d_ for d_ in val_dfas if not np.isnan(d_)]

    return {
        "label": label,
        "params": {
            "phi": round(phi, 4),
            "d": round(d, 4),
            "sigma_eps": round(sigma_eps, 4),
            "sigma_eta": round(sigma_eta, 4),
        },
        "loss": round(best_loss, 6),
        "target": {k: round(v, 4) if isinstance(v, float) and not np.isnan(v) else v
                   for k, v in target.items()},
        "fitted": {
            "H_mean": round(np.mean(val_hs), 4) if val_hs else None,
            "H_std": round(np.std(val_hs), 4) if val_hs else None,
            "AC1_mean": round(np.mean(val_ac1s), 4) if val_ac1s else None,
            "AC1_std": round(np.std(val_ac1s), 4) if val_ac1s else None,
            "DFA_mean": round(np.mean(val_dfas), 4) if val_dfas else None,
        },
    }


# ── Corpus loading ─────────────────────────────────────────────────────

def load_bible_verses():
    """Load AT and NT verse-length series."""
    with open(BIBLE_JSON, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    OT_BOOKS = {"Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy",
                "Joshua", "Judges", "Ruth", "1 Samuel", "2 Samuel",
                "1 Kings", "2 Kings", "1 Chronicles", "2 Chronicles",
                "Ezra", "Nehemiah", "Esther", "Job", "Psalms",
                "Proverbs", "Ecclesiastes", "Song of Solomon",
                "Isaiah", "Jeremiah", "Lamentations", "Ezekiel",
                "Daniel", "Hosea", "Joel", "Amos", "Obadiah",
                "Jonah", "Micah", "Nahum", "Habakkuk", "Zephaniah",
                "Haggai", "Zechariah", "Malachi"}
    NT_BOOKS = {"Matthew", "Mark", "Luke", "John", "Acts",
                "Romans", "1 Corinthians", "2 Corinthians", "Galatians",
                "Ephesians", "Philippians", "Colossians",
                "1 Thessalonians", "2 Thessalonians",
                "1 Timothy", "2 Timothy", "Titus", "Philemon",
                "Hebrews", "James", "1 Peter", "2 Peter",
                "1 John", "2 John", "3 John", "Jude", "Revelation"}

    at_verses = defaultdict(int)
    nt_verses = defaultdict(int)
    book_verses = defaultdict(lambda: defaultdict(int))

    for w in corpus:
        book = w.get("book", "")
        key = (book, w.get("chapter", 0), w.get("verse", 0))
        if book in OT_BOOKS:
            at_verses[key] += 1
        elif book in NT_BOOKS:
            nt_verses[key] += 1
        book_verses[book][key] += 1

    at_lens = np.array([at_verses[k] for k in sorted(at_verses.keys())], dtype=float)
    nt_lens = np.array([nt_verses[k] for k in sorted(nt_verses.keys())], dtype=float)

    # Per-book series
    book_lens = {}
    for book, verses in book_verses.items():
        lens = [verses[k] for k in sorted(verses.keys())]
        if len(lens) >= 50:
            book_lens[book] = np.array(lens, dtype=float)

    return at_lens, nt_lens, book_lens


def load_quran():
    """Load Quran verse lengths."""
    quran_file = BASE / "results" / "comparison_corpora" / "quran_morphology.txt"
    if not quran_file.exists():
        return None
    pat = re.compile(r'\((\d+):(\d+):(\d+)(?::\d+)?\)')
    verses = defaultdict(set)
    with open(quran_file, "r", encoding="utf-8") as f:
        for line in f:
            m = pat.search(line)
            if m:
                sura, aya, word = int(m.group(1)), int(m.group(2)), int(m.group(3))
                verses[(sura, aya)].add(word)
    lens = [len(verses[k]) for k in sorted(verses.keys())]
    return np.array(lens, dtype=float) if lens else None


def load_rigveda_synthetic():
    """Calibrated synthetic Rig Veda (from Phase 6 parameters)."""
    rng = np.random.default_rng(123)
    lens = []
    total = 0
    while total < 10552:
        meter = rng.choice(["tristubh", "gayatri", "jagati", "anustubh"],
                           p=[0.40, 0.25, 0.10, 0.25])
        base = {"tristubh": 9, "gayatri": 7, "jagati": 10, "anustubh": 8}[meter]
        std = {"tristubh": 2, "gayatri": 1.5, "jagati": 2, "anustubh": 1.5}[meter]
        wc = max(1, int(rng.normal(base, std)))
        lens.append(wc)
        total += 1
    return np.array(lens, dtype=float)


# ── Boundary analysis ──────────────────────────────────────────────────

def boundary_analysis(fitted_params):
    """Analyze parameter boundaries between corpora."""
    log.info("\n=== Boundary Analysis ===")
    results = {}

    names = list(fitted_params.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            n1, n2 = names[i], names[j]
            p1, p2 = fitted_params[n1]["params"], fitted_params[n2]["params"]
            diffs = {}
            for param in ["phi", "d", "sigma_eps", "sigma_eta"]:
                v1, v2 = p1.get(param, 0), p2.get(param, 0)
                diffs[param] = round(v2 - v1, 4)
            results[f"{n1}_vs_{n2}"] = {
                "params_1": p1,
                "params_2": p2,
                "differences": diffs,
                "max_diff_param": max(diffs.keys(), key=lambda k: abs(diffs[k])),
                "max_diff_value": max(abs(v) for v in diffs.values()),
            }
            log.info(f"  {n1} vs {n2}: max diff in "
                     f"{results[f'{n1}_vs_{n2}']['max_diff_param']} "
                     f"(Δ={results[f'{n1}_vs_{n2}']['max_diff_value']:.4f})")
    return results


# ── Parameter space sweep ──────────────────────────────────────────────

def parameter_space_sweep():
    """Sweep (φ, d) space and compute H, AC1 for each."""
    log.info("\n=== Parameter Space Sweep ===")
    rng = np.random.default_rng(42)
    phi_range = np.linspace(-0.3, 0.8, 12)
    d_range = np.linspace(-0.2, 0.45, 12)

    grid = []
    for phi in phi_range:
        for d in d_range:
            hs, ac1s = [], []
            for _ in range(20):
                try:
                    s = generate_arfima_ar1(500, phi, d, 1.5, 2.0, rng=rng)
                    hs.append(hurst_exponent_rs(s))
                    ac1s.append(autocorr_lag1(s))
                except Exception:
                    pass
            hs = [h for h in hs if not np.isnan(h)]
            ac1s = [a for a in ac1s if not np.isnan(a)]
            grid.append({
                "phi": round(phi, 3),
                "d": round(d, 3),
                "H_mean": round(np.mean(hs), 4) if hs else None,
                "AC1_mean": round(np.mean(ac1s), 4) if ac1s else None,
            })
    log.info(f"  Grid: {len(grid)} points computed")
    return grid


# ── Main ───────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    log.info("=" * 70)
    log.info("FASE 14 — Script 2: Unified Model")
    log.info("=" * 70)

    # Load corpora
    log.info("\nCargando corpora...")
    at_lens, nt_lens, book_lens = load_bible_verses()
    quran_lens = load_quran()
    rv_lens = load_rigveda_synthetic()

    log.info(f"  AT: {len(at_lens)} versos")
    log.info(f"  NT: {len(nt_lens)} versos")
    log.info(f"  Corán: {len(quran_lens) if quran_lens is not None else 0} aleyas")
    log.info(f"  RV: {len(rv_lens)} padas")

    # Compute target statistics
    targets = {}
    for label, series in [("AT", at_lens), ("NT", nt_lens),
                           ("Corán", quran_lens), ("Rig_Veda", rv_lens)]:
        if series is not None and len(series) > 50:
            targets[label] = compute_targets(series)
            log.info(f"  {label}: H={targets[label]['H']:.3f}, "
                     f"AC1={targets[label]['AC1']:.3f}, "
                     f"DFA={targets[label]['DFA']:.3f}")

    # Save targets
    targets_serial = {k: {k2: round(v2, 4) if isinstance(v2, float) else v2
                          for k2, v2 in v.items()} for k, v in targets.items()}
    with open(RESULTS_DIR / "corpus_targets.json", "w") as f:
        json.dump(targets_serial, f, indent=2, ensure_ascii=False)

    # Fit model per corpus
    log.info("\n=== Fitting models ===")
    fitted = {}
    for label, target in targets.items():
        if target["H"] and not np.isnan(target["H"]):
            result = fit_model(target, label)
            if result:
                fitted[label] = result
                log.info(f"  {label}: loss={result['loss']:.4f}, "
                         f"φ={result['params']['phi']:.3f}, "
                         f"d={result['params']['d']:.3f}")

    with open(RESULTS_DIR / "fitted_params.json", "w") as f:
        json.dump(fitted, f, indent=2, ensure_ascii=False)

    # Boundary analysis
    if len(fitted) >= 2:
        boundaries = boundary_analysis(fitted)
        with open(RESULTS_DIR / "boundary_analysis.json", "w") as f:
            json.dump(boundaries, f, indent=2, ensure_ascii=False)

    # Parameter space sweep
    grid = parameter_space_sweep()
    with open(RESULTS_DIR / "parameter_space.json", "w") as f:
        json.dump(grid, f, indent=2, ensure_ascii=False)

    # Retrodiction test: how well do fitted params reproduce target stats?
    log.info("\n=== Retrodiction Test ===")
    retrodiction = {}
    rng = np.random.default_rng(777)
    for label, fit in fitted.items():
        p = fit["params"]
        target = targets[label]
        n_ok = 0
        n_trials = 100
        for _ in range(n_trials):
            s = generate_arfima_ar1(800, p["phi"], p["d"], p["sigma_eps"], p["sigma_eta"],
                                    rng=rng)
            h = hurst_exponent_rs(s)
            ac1 = autocorr_lag1(s)
            if not np.isnan(h) and not np.isnan(ac1):
                h_ok = abs(h - target["H"]) < 0.15
                ac1_ok = abs(ac1 - target["AC1"]) < 0.15
                if h_ok and ac1_ok:
                    n_ok += 1
        retrodiction[label] = {
            "n_trials": n_trials,
            "n_match": n_ok,
            "match_pct": round(100 * n_ok / n_trials, 1),
        }
        log.info(f"  {label}: {n_ok}/{n_trials} = {100 * n_ok / n_trials:.1f}%")

    with open(RESULTS_DIR / "retrodiction.json", "w") as f:
        json.dump(retrodiction, f, indent=2, ensure_ascii=False)

    # Summary verdict
    log.info("\n=== VEREDICTO ===")
    verdict = {
        "n_corpora_fitted": len(fitted),
        "corpora": list(fitted.keys()),
    }

    if "AT" in fitted and "NT" in fitted:
        at_p, nt_p = fitted["AT"]["params"], fitted["NT"]["params"]
        verdict["AT_NT_key_difference"] = {
            "param": max(["phi", "d", "sigma_eps", "sigma_eta"],
                         key=lambda k: abs(at_p[k] - nt_p[k])),
            "AT_value": at_p[max(["phi", "d", "sigma_eps", "sigma_eta"],
                                 key=lambda k: abs(at_p[k] - nt_p[k]))],
            "NT_value": nt_p[max(["phi", "d", "sigma_eps", "sigma_eta"],
                                 key=lambda k: abs(at_p[k] - nt_p[k]))],
        }

    # Check universality: all corpora have similar d?
    d_values = [fitted[c]["params"]["d"] for c in fitted if "d" in fitted[c]["params"]]
    if len(d_values) >= 2:
        verdict["d_range"] = [round(min(d_values), 4), round(max(d_values), 4)]
        verdict["d_universal"] = bool(max(d_values) - min(d_values) < 0.1)

    best_retro = max(retrodiction.items(), key=lambda x: x[1]["match_pct"]) if retrodiction else None
    if best_retro:
        verdict["best_retrodiction"] = {
            "corpus": best_retro[0],
            "match_pct": best_retro[1]["match_pct"],
        }

    log.info(f"  Corpora fitted: {verdict['n_corpora_fitted']}")
    if "d_universal" in verdict:
        log.info(f"  d universal: {verdict['d_universal']} (range={verdict['d_range']})")

    with open(RESULTS_DIR / "verdict.json", "w") as f:
        json.dump(verdict, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    log.info(f"\n{'=' * 70}")
    log.info(f"Script 2 completado en {elapsed:.1f}s")


if __name__ == "__main__":
    main()
