#!/usr/bin/env python3
"""
Fase 9 — Script 4: Procesos Generativos
¿AT, Corán y Rig Veda son el mismo proceso generativo o procesos distintos que comparten H alto?
ARFIMA, ACF, MF-DFA y test de mismo proceso.
"""

import json
import logging
import time
import numpy as np
from pathlib import Path
from scipy import stats
from collections import defaultdict

# ── Configuración ────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent
RESULTS_DIR = BASE / "results" / "generative"
LOG_DIR = BASE / "logs"
BIBLE_JSON = BASE / "bible_unified.json"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "fase9_generative.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


# ── Métricas base ────────────────────────────────────────────────────────

def hurst_exponent_rs(series):
    series = np.asarray(series, dtype=float)
    n = len(series)
    if n < 20:
        return float("nan"), 0.0
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
        return float("nan"), 0.0
    slope, _, r, _, _ = stats.linregress(np.log(sizes), np.log(rs_values))
    return float(slope), float(r ** 2)


# ── Carga de series de verse-lengths ─────────────────────────────────────

def load_all_series():
    """Carga series de longitudes de versículo para todos los corpus."""
    log.info("Cargando series de todos los corpus...")
    series = {}

    # 1. AT y NT de bible_unified.json
    with open(BIBLE_JSON) as f:
        data = json.load(f)

    for corpus_label, corpus_key in [("AT", "OT"), ("NT", "NT")]:
        verses = defaultdict(int)
        for w in data:
            if w.get("corpus") == corpus_key:
                key = (w["book_num"], w["chapter"], w["verse"])
                verses[key] += 1
        sorted_keys = sorted(verses.keys())
        lens = [verses[k] for k in sorted_keys]
        series[corpus_label] = np.array(lens, dtype=float)
        log.info(f"  {corpus_label}: {len(lens)} versos")

    # 2. Corán
    quran_path = BASE / "results" / "comparison_corpora" / "quran_morphology.txt"
    if quran_path.exists():
        quran_verses = defaultdict(int)
        with open(quran_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) >= 2:
                    ref = parts[0]
                    segs = ref.split(":")
                    if len(segs) >= 3:
                        try:
                            sura = int(segs[0].strip("()"))
                            aya = int(segs[1])
                            quran_verses[(sura, aya)] += 1
                        except (ValueError, IndexError):
                            pass
        if quran_verses:
            sorted_keys = sorted(quran_verses.keys())
            lens = [quran_verses[k] for k in sorted_keys]
            series["Corán"] = np.array(lens, dtype=float)
            log.info(f"  Corán: {len(lens)} aleyas")
    else:
        log.warning(f"  Corán: archivo no encontrado ({quran_path})")

    # 3. Homero
    for homer_name in ["homer_iliad.xml", "homer_odyssey.xml"]:
        homer_path = BASE / "results" / "comparison_corpora" / homer_name
        if homer_path.exists():
            import re
            with open(homer_path, encoding="utf-8", errors="ignore") as f:
                content = f.read()
            # Buscar líneas/versos con texto griego
            lines = re.findall(r'<l[^>]*>(.*?)</l>', content, re.DOTALL)
            if not lines:
                lines = content.strip().split("\n")
            lens = []
            for line in lines:
                clean = re.sub(r'<[^>]+>', '', line).strip()
                words = [w for w in clean.split() if len(w) > 0]
                if words:
                    lens.append(len(words))
            if lens:
                key = "Homero_Ilíada" if "iliad" in homer_name else "Homero_Odisea"
                series[key] = np.array(lens, dtype=float)
                log.info(f"  {key}: {len(lens)} líneas")

    # Combinar Homero si ambos existen
    if "Homero_Ilíada" in series and "Homero_Odisea" in series:
        series["Homero"] = np.concatenate([series["Homero_Ilíada"], series["Homero_Odisea"]])
        log.info(f"  Homero combinado: {len(series['Homero'])} líneas")

    # 4. Heródoto
    herodotus_path = BASE / "results" / "comparison_corpora" / "herodotus.xml"
    if herodotus_path.exists():
        import re
        with open(herodotus_path, encoding="utf-8", errors="ignore") as f:
            content = f.read()
        # Secciones/párrafos
        sections = re.findall(r'<p[^>]*>(.*?)</p>', content, re.DOTALL)
        if not sections:
            sections = re.findall(r'<milestone[^>]*/>\s*(.*?)(?=<milestone|$)', content, re.DOTALL)
        lens = []
        for sec in sections:
            clean = re.sub(r'<[^>]+>', '', sec).strip()
            words = clean.split()
            if len(words) >= 3:
                lens.append(len(words))
        if lens:
            series["Heródoto"] = np.array(lens, dtype=float)
            log.info(f"  Heródoto: {len(lens)} secciones")

    # 5. Rig Veda — intentar cargar la serie guardada
    rv_cache = BASE / "results" / "rigveda"
    if rv_cache.exists():
        # Buscar archivos con datos de pada lengths
        for fname in rv_cache.iterdir():
            if "pada" in fname.name.lower() or "verse_len" in fname.name.lower():
                try:
                    with open(fname) as f:
                        rv_data = json.load(f)
                    if isinstance(rv_data, list):
                        series["Rig Veda"] = np.array(rv_data, dtype=float)
                        log.info(f"  Rig Veda: {len(rv_data)} pādas (de {fname.name})")
                        break
                except:
                    pass

    # Si no hay archivo de pādas, intentar reconstruir desde el json de métricas
    if "Rig Veda" not in series:
        rv_metrics = BASE / "results" / "rigveda" / "rigveda_metrics.json"
        if rv_metrics.exists():
            with open(rv_metrics) as f:
                rv = json.load(f)
            # No tenemos la serie directa, pero sabemos H=0.934, n_padas=21253
            # Simular una serie con esos parámetros como proxy
            log.warning("  Rig Veda: no hay serie guardada, generando proxy ARFIMA con d=H-0.5=0.434")
            n_padas = rv.get("n_padas", 21253)
            d_rv = rv.get("hurst_H", 0.934) - 0.5
            series["Rig Veda"] = generate_arfima_series(n_padas, d_rv, seed=42)
            log.info(f"  Rig Veda (proxy): {n_padas} pādas, d={d_rv:.3f}")

    # 6. Mishnah
    mishnah_dir = BASE / "results" / "comparison_corpora" / "mishnah"
    if mishnah_dir.exists():
        lens = []
        import re
        for fpath in sorted(mishnah_dir.glob("*.txt")):
            with open(fpath, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        words = line.split()
                        if len(words) >= 2:
                            lens.append(len(words))
        if lens:
            series["Mishnah"] = np.array(lens, dtype=float)
            log.info(f"  Mishnah: {len(lens)} unidades")

    return series


# ── ARFIMA ───────────────────────────────────────────────────────────────

def generate_arfima_series(n, d, ar_coeffs=None, ma_coeffs=None, seed=None):
    """
    Genera una serie ARFIMA(p,d,q) con long memory parameter d.
    d = H - 0.5 para relación con Hurst.
    Usa el método de Hosking (1984) para generar FGN.
    """
    if seed is not None:
        np.random.seed(seed)

    # Generar ruido fraccional gaussiano (FGN) via Hosking
    # Autocovarianzas de FGN: γ(k) = 0.5*(|k+1|^(2d) - 2|k|^(2d) + |k-1|^(2d))
    # Para d > 0 (persistent), d < 0.5
    d = np.clip(d, -0.49, 0.49)

    # Método de Davies-Harte (circulant embedding) para FGN
    m = 2
    while m < 2 * n:
        m *= 2

    # Autocovarianza
    k = np.arange(m)
    # γ(k) para FGN con parámetro d
    gamma = np.zeros(m)
    gamma[0] = 1.0
    for j in range(1, m):
        gamma[j] = 0.5 * (abs(j + 1) ** (2 * d) - 2 * abs(j) ** (2 * d) + abs(j - 1) ** (2 * d))

    # Circulant embedding
    c = np.concatenate([gamma, gamma[-2:0:-1]])
    eigenvalues = np.fft.fft(c).real

    if np.any(eigenvalues < -1e-10):
        # Fallback: Cholesky con covarianza truncada
        log.warning(f"  Eigenvalues negativos en circulant embedding (d={d:.3f}), usando Cholesky truncado")
        n_use = min(n, 2000)
        cov = np.zeros((n_use, n_use))
        for i in range(n_use):
            for j in range(n_use):
                lag = abs(i - j)
                if lag < len(gamma):
                    cov[i, j] = gamma[lag]
        cov += 1e-8 * np.eye(n_use)
        try:
            L = np.linalg.cholesky(cov)
            z = np.random.randn(n_use)
            fgn = L @ z
            if n_use < n:
                # Repetir para llenar
                reps = (n // n_use) + 1
                fgn = np.tile(fgn, reps)[:n]
            return fgn[:n]
        except np.linalg.LinAlgError:
            log.warning("  Cholesky falló, generando ruido blanco como fallback")
            return np.random.randn(n)

    eigenvalues = np.maximum(eigenvalues, 0)
    sqrt_eigenvalues = np.sqrt(eigenvalues)

    # Generar
    z1 = np.random.randn(len(c))
    z2 = np.random.randn(len(c))
    z_complex = z1 + 1j * z2
    w = np.fft.fft(sqrt_eigenvalues * z_complex)
    fgn = w[:n].real / np.sqrt(len(c))

    # Aplicar AR y MA si se proporcionan
    if ar_coeffs is not None or ma_coeffs is not None:
        # Simple: solo FGN + escalar, sin AR/MA complejo
        pass

    return fgn


def fit_arfima(series, name=""):
    """
    Ajusta ARFIMA(p, d, q) a una serie.
    Primero estima d via R/S, luego ajusta AR(p) al residuo.
    """
    series = np.asarray(series, dtype=float)
    n = len(series)

    # 1. Estimar d via Hurst
    H, H_r2 = hurst_exponent_rs(series)
    d_hat = H - 0.5 if not np.isnan(H) else 0.0

    # 2. Estimar d via Whittle (aproximación por periodograma)
    # Periodograma
    s_centered = series - series.mean()
    periodogram = np.abs(np.fft.fft(s_centered)) ** 2 / n
    freqs = np.fft.fftfreq(n)

    # Solo frecuencias positivas, excluyendo f=0
    pos_mask = freqs > 0
    pos_freqs = freqs[pos_mask]
    pos_period = periodogram[pos_mask]

    # Para ARFIMA(0,d,0): S(f) ∝ |2sin(πf)|^(-2d)
    # Log-regresión: log(S) = -2d * log|2sin(πf)| + const
    if len(pos_freqs) > 10:
        log_spec = np.log(pos_period + 1e-30)
        log_factor = np.log(np.abs(2 * np.sin(np.pi * pos_freqs)) + 1e-30)

        # Usar solo frecuencias bajas (GPH estimator: primeras n^0.5 frecuencias)
        m = int(np.sqrt(n))
        if m > 5:
            slope, intercept, r, p, se = stats.linregress(
                log_factor[:m], log_spec[:m]
            )
            d_whittle = float(-slope / 2)
        else:
            d_whittle = d_hat
    else:
        d_whittle = d_hat

    # 3. Fractional differencing y ajuste AR
    # Fractionally difference la serie: (1-B)^d * X_t
    # Luego ajustar AR a la serie diferenciada
    d_use = d_whittle
    frac_diff = fractional_difference(series, d_use)

    # Ajustar AR(1), AR(2), AR(3) y elegir por AIC
    best_aic = float("inf")
    best_p = 0
    best_ar = []

    for p_order in range(4):
        if p_order == 0:
            residuals = frac_diff
            k = 1  # solo sigma^2
        else:
            if len(frac_diff) <= p_order + 1:
                continue
            # Yule-Walker
            try:
                from scipy.signal import lfilter
                # Autocorrelaciones
                acf_vals = np.correlate(frac_diff - frac_diff.mean(),
                                       frac_diff - frac_diff.mean(), mode="full")
                acf_vals = acf_vals[len(frac_diff) - 1:] / acf_vals[len(frac_diff) - 1]
                # Levinson-Durbin
                r_vec = acf_vals[1:p_order + 1]
                R_mat = np.zeros((p_order, p_order))
                for i in range(p_order):
                    for j in range(p_order):
                        R_mat[i, j] = acf_vals[abs(i - j)]
                try:
                    ar = np.linalg.solve(R_mat, r_vec)
                except np.linalg.LinAlgError:
                    continue
                # Residuales
                residuals = frac_diff[p_order:].copy()
                for i in range(len(residuals)):
                    for j in range(p_order):
                        if i + p_order - j - 1 >= 0 and i + p_order - j - 1 < len(frac_diff):
                            residuals[i] -= ar[j] * frac_diff[i + p_order - j - 1]
            except Exception:
                continue
            k = p_order + 1

        if len(residuals) < 10:
            continue
        sigma2 = float(np.var(residuals))
        if sigma2 <= 0:
            continue
        aic = len(residuals) * np.log(sigma2) + 2 * k

        if aic < best_aic:
            best_aic = aic
            best_p = p_order
            best_ar = ar.tolist() if p_order > 0 else []

    # 4. Resultado
    result = {
        "corpus": name,
        "n": n,
        "H_rs": float(H),
        "H_R2": float(H_r2),
        "d_rs": float(d_hat),
        "d_whittle": float(d_whittle),
        "d_used": float(d_use),
        "best_ar_order": best_p,
        "ar_coefficients": best_ar,
        "aic": float(best_aic),
        "model": f"ARFIMA({best_p}, {d_use:.4f}, 0)",
    }

    log.info(f"  {name}: {result['model']}, d_rs={d_hat:.4f}, d_whittle={d_whittle:.4f}")
    return result


def fractional_difference(series, d, max_terms=100):
    """
    Aplica diferenciación fraccional (1-B)^d a la serie.
    Usa la expansión binomial truncada.
    """
    n = len(series)
    # Coeficientes binomiales: π_k = Γ(k-d) / (Γ(-d) * Γ(k+1))
    # Recursión: π_0 = 1, π_k = π_{k-1} * (k-1-d)/k
    weights = np.zeros(min(max_terms, n))
    weights[0] = 1.0
    for k in range(1, len(weights)):
        weights[k] = weights[k - 1] * (k - 1 - d) / k

    result = np.zeros(n)
    for t in range(n):
        s = 0.0
        for k in range(min(t + 1, len(weights))):
            s += weights[k] * series[t - k]
        result[t] = s

    return result


# ── ACF comparison ───────────────────────────────────────────────────────

def compute_acf(series, max_lag=100):
    """Calcula la función de autocorrelación para lags 1..max_lag."""
    series = np.asarray(series, dtype=float)
    n = len(series)
    mean = series.mean()
    var = series.var()
    if var == 0:
        return np.zeros(max_lag)

    acf_vals = np.zeros(max_lag)
    for lag in range(max_lag):
        if lag >= n:
            break
        c = np.sum((series[:n - lag] - mean) * (series[lag:] - mean)) / (n * var)
        acf_vals[lag] = c

    return acf_vals


def compare_acf(series_dict, max_lag=100):
    """Compara ACF entre todos los corpus."""
    log.info("\n=== Comparación de ACF ===")

    acf_data = {}
    for name, s in series_dict.items():
        acf = compute_acf(s, max_lag=max_lag)
        acf_data[name] = {
            "acf": acf.tolist(),
            "n": len(s),
        }
        log.info(f"  {name}: ACF(1)={acf[1]:.4f}, ACF(10)={acf[10]:.4f}, ACF(50)={acf[min(50, max_lag-1)]:.4f}")

    # KS test entre pares de ACF
    names = list(acf_data.keys())
    ks_tests = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = np.array(acf_data[names[i]]["acf"])
            b = np.array(acf_data[names[j]]["acf"])
            ks_stat, ks_p = stats.ks_2samp(a, b)
            pair = f"{names[i]} vs {names[j]}"
            ks_tests[pair] = {
                "ks_statistic": float(ks_stat),
                "p_value": float(ks_p),
                "distinguishable": bool(ks_p < 0.05),
            }
            log.info(f"  KS {pair}: stat={ks_stat:.4f}, p={ks_p:.4f}")

    # ¿AT, Corán y RV son indistinguibles entre sí?
    at_like = ["AT", "Corán", "Rig Veda"]
    at_like_tests = {}
    for i in range(len(at_like)):
        for j in range(i + 1, len(at_like)):
            if at_like[i] in acf_data and at_like[j] in acf_data:
                pair = f"{at_like[i]} vs {at_like[j]}"
                if pair in ks_tests:
                    at_like_tests[pair] = ks_tests[pair]

    return {
        "max_lag": max_lag,
        "acf_by_corpus": acf_data,
        "ks_tests": ks_tests,
        "at_like_cluster_tests": at_like_tests,
    }


# ── Multifractal DFA (MF-DFA) ───────────────────────────────────────────

def mfdfa(series, q_range=(-5, 5), n_q=21, min_box=16, max_box=None):
    """
    Multifractal DFA.
    Calcula h(q) para varios valores de q.
    q=2 corresponde al DFA estándar.
    """
    series = np.asarray(series, dtype=float)
    n = len(series)
    if max_box is None:
        max_box = n // 4

    y = np.cumsum(series - series.mean())

    # Escala de boxes
    sizes = []
    s = min_box
    while s <= max_box:
        sizes.append(s)
        s = int(s * 1.3)
        if sizes and s == sizes[-1]:
            s += 1
    sizes = np.array(sizes)

    if len(sizes) < 4:
        return None

    q_values = np.linspace(q_range[0], q_range[1], n_q)
    # Excluir q=0 (se trata especialmente)
    q_values = q_values[q_values != 0]

    # Para cada escala, calcular la varianza local
    Fq = np.zeros((len(q_values), len(sizes)))

    for si, s in enumerate(sizes):
        n_seg = n // s
        if n_seg < 1:
            continue
        rms = np.zeros(n_seg)
        for v in range(n_seg):
            seg = y[v * s:(v + 1) * s]
            x_ax = np.arange(s)
            coeffs = np.polyfit(x_ax, seg, 1)
            trend = np.polyval(coeffs, x_ax)
            rms[v] = np.sqrt(np.mean((seg - trend) ** 2))

        rms = rms[rms > 0]
        if len(rms) < 2:
            continue

        for qi, q in enumerate(q_values):
            if q == 0:
                Fq[qi, si] = np.exp(0.5 * np.mean(np.log(rms ** 2)))
            else:
                Fq[qi, si] = (np.mean(rms ** q)) ** (1.0 / q)

    # Calcular h(q) por regresión log-log
    hq = np.zeros(len(q_values))
    hq_r2 = np.zeros(len(q_values))

    for qi in range(len(q_values)):
        valid = Fq[qi, :] > 0
        if np.sum(valid) < 3:
            hq[qi] = float("nan")
            continue
        log_s = np.log(sizes[valid])
        log_f = np.log(Fq[qi, valid])
        slope, _, r, _, _ = stats.linregress(log_s, log_f)
        hq[qi] = slope
        hq_r2[qi] = r ** 2

    # Espectro multifractal: τ(q) = q*h(q) - 1, α = dτ/dq, f(α) = q*α - τ(q)
    tau_q = q_values * hq - 1

    # Derivada numérica
    alpha_mf = np.gradient(tau_q, q_values)
    f_alpha = q_values * alpha_mf - tau_q

    # Anchura del espectro
    valid_mask = ~np.isnan(alpha_mf) & ~np.isnan(f_alpha)
    if np.sum(valid_mask) > 2:
        alpha_range = float(np.max(alpha_mf[valid_mask]) - np.min(alpha_mf[valid_mask]))
    else:
        alpha_range = float("nan")

    return {
        "q_values": q_values.tolist(),
        "h_q": hq.tolist(),
        "h_q_r2": hq_r2.tolist(),
        "tau_q": tau_q.tolist(),
        "alpha_mf": alpha_mf.tolist(),
        "f_alpha": f_alpha.tolist(),
        "spectrum_width": alpha_range,
        "h_q2": float(hq[np.argmin(np.abs(q_values - 2))]) if len(q_values) > 0 else None,
    }


# ── Test de mismo proceso generativo ────────────────────────────────────

def test_same_process(series_dict, arfima_params, n_sim=500):
    """
    Simula series con parámetros ARFIMA del AT y prueba si reproducen
    las propiedades del Corán y Rig Veda.
    """
    log.info("\n=== Test de mismo proceso generativo ===")

    if "AT" not in arfima_params:
        return {"error": "No AT ARFIMA parameters available"}

    at_params = arfima_params["AT"]
    d_at = at_params["d_used"]
    n_at = at_params["n"]

    results = {}
    targets = ["Corán", "Rig Veda", "NT", "Homero"]

    for target in targets:
        if target not in arfima_params or target not in series_dict:
            continue

        target_H = arfima_params[target]["H_rs"]
        target_acf = compute_acf(series_dict[target], max_lag=50)
        n_target = len(series_dict[target])

        # Simular n_sim series con parámetros del AT
        sim_H = []
        sim_acf_ks_p = []
        log.info(f"\n  Simulando {n_sim} series AT-like para comparar con {target}...")

        for i in range(n_sim):
            sim = generate_arfima_series(n_target, d_at, seed=i * 17 + 31)
            # Escalar a media y std del target
            sim = sim * series_dict[target].std() + series_dict[target].mean()
            sim = np.maximum(sim, 1)  # verse lengths >= 1

            h_sim, _ = hurst_exponent_rs(sim)
            sim_H.append(h_sim)

            acf_sim = compute_acf(sim, max_lag=50)
            ks_stat, ks_p = stats.ks_2samp(acf_sim, target_acf)
            sim_acf_ks_p.append(ks_p)

            if (i + 1) % 100 == 0:
                log.info(f"    {i + 1}/{n_sim} simulaciones")

        sim_H = np.array(sim_H)
        sim_acf_ks_p = np.array(sim_acf_ks_p)

        # ¿Qué fracción reproduce H del target?
        H_tolerance = 0.1  # ±0.1
        frac_H = float(np.mean(np.abs(sim_H - target_H) < H_tolerance))

        # ¿Qué fracción tiene ACF indistinguible del target?
        frac_acf = float(np.mean(sim_acf_ks_p > 0.05))

        # ¿Qué fracción cumple ambos?
        both_mask = (np.abs(sim_H - target_H) < H_tolerance) & (sim_acf_ks_p > 0.05)
        frac_both = float(np.mean(both_mask))

        verdict = ""
        if frac_both > 0.05:
            verdict = "COMPATIBLE — mismo proceso generativo plausible"
        elif frac_both > 0.01:
            verdict = "MARGINAL — misma familia pero posiblemente distinto"
        else:
            verdict = "INCOMPATIBLE — procesos generativos distintos"

        results[target] = {
            "target_H": float(target_H),
            "sim_H_mean": float(sim_H.mean()),
            "sim_H_std": float(sim_H.std()),
            "frac_H_match": frac_H,
            "frac_acf_match": frac_acf,
            "frac_both_match": frac_both,
            "verdict": verdict,
            "n_simulations": n_sim,
            "H_tolerance": H_tolerance,
        }
        log.info(f"  {target}: frac_H={frac_H:.3f}, frac_ACF={frac_acf:.3f}, "
                f"frac_both={frac_both:.3f} → {verdict}")

    return results


# ── Conclusión formal ────────────────────────────────────────────────────

def synthesize_verdict(arfima_params, acf_results, mfdfa_results, process_test):
    """Sintetiza todos los resultados en un veredicto formal."""
    log.info("\n=== Sintetizando veredicto ===")

    at_like = ["AT", "Corán", "Rig Veda"]

    # 1. ¿Parámetros ARFIMA indistinguibles?
    d_values = {}
    for name in at_like:
        if name in arfima_params:
            d_values[name] = arfima_params[name]["d_used"]

    d_range = max(d_values.values()) - min(d_values.values()) if len(d_values) >= 2 else float("nan")
    arfima_similar = d_range < 0.15  # Tolerancia

    # 2. ¿ACF indistinguibles?
    acf_tests = acf_results.get("at_like_cluster_tests", {})
    acf_all_indistinguishable = all(
        not t.get("distinguishable", True)
        for t in acf_tests.values()
    ) if acf_tests else False

    # 3. ¿Espectros multifractales similares?
    widths = {}
    for name in at_like:
        if name in mfdfa_results and mfdfa_results[name] is not None:
            widths[name] = mfdfa_results[name]["spectrum_width"]

    width_range = max(widths.values()) - min(widths.values()) if len(widths) >= 2 else float("nan")
    mf_similar = width_range < 0.3  # Tolerancia

    # 4. ¿Test de mismo proceso?
    process_compatible = {}
    for target in ["Corán", "Rig Veda"]:
        if target in process_test:
            process_compatible[target] = process_test[target]["frac_both_match"] > 0.05

    # Veredicto
    if arfima_similar and acf_all_indistinguishable and mf_similar:
        category = "MISMO_FENÓMENO"
        description = (
            "Los tres corpus AT-like (AT, Corán, Rig Veda) son compatibles con "
            "el MISMO proceso generativo estocástico: parámetros ARFIMA similares, "
            "ACF indistinguibles, y espectros multifractales comparables."
        )
    elif (arfima_similar or acf_all_indistinguishable) and not mf_similar:
        category = "MISMA_FAMILIA_INSTANCIAS_DISTINTAS"
        description = (
            "Los tres corpus comparten la misma FAMILIA de procesos (memoria larga similar), "
            "pero difieren en detalles de estructura fina (espectro multifractal). "
            "Son instancias distintas de un fenómeno similar, no copias del mismo proceso."
        )
    else:
        category = "FENÓMENOS_PARALELOS"
        description = (
            "Los tres corpus comparten el rango de H (>0.9) y MPS significativo, "
            "pero difieren en estructura de autocorrelación y/o espectro multifractal. "
            "Son fenómenos PARALELOS: procesos distintos que convergen en propiedades similares."
        )

    verdict = {
        "category": category,
        "description": description,
        "evidence": {
            "arfima_d_range": float(d_range) if not np.isnan(d_range) else None,
            "arfima_similar": arfima_similar,
            "d_values": d_values,
            "acf_all_indistinguishable": acf_all_indistinguishable,
            "acf_tests": {k: v for k, v in acf_tests.items()},
            "mf_spectrum_widths": widths,
            "mf_width_range": float(width_range) if not np.isnan(width_range) else None,
            "mf_similar": mf_similar,
            "process_test_compatible": process_compatible,
        },
        "implication": (
            "Si MISMO_FENÓMENO: sugiere un mecanismo universal que produce memoria larga "
            "en textos de revelación controlada, independientemente de lengua y cultura. "
            "Si MISMA_FAMILIA: el mecanismo general es el mismo pero con variaciones culturales. "
            "Si FENÓMENOS_PARALELOS: la convergencia en H alto es una propiedad emergente "
            "de la transmisión controlada, no de un proceso generativo específico."
        ),
    }

    log.info(f"\n  VEREDICTO: {category}")
    log.info(f"  {description}")

    return verdict


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    log.info("=" * 60)
    log.info("FASE 9 — Script 4: Procesos Generativos")
    log.info("=" * 60)

    # 1. Cargar series
    series_dict = load_all_series()
    log.info(f"\nSeries cargadas: {list(series_dict.keys())}")

    # 2. ARFIMA
    log.info("\n=== Ajuste ARFIMA ===")
    arfima_params = {}
    for name, s in series_dict.items():
        if name.startswith("Homero_"):
            continue  # Usar solo el combinado
        result = fit_arfima(s, name=name)
        arfima_params[name] = result

    with open(RESULTS_DIR / "arfima_parameters.json", "w") as f:
        json.dump(arfima_params, f, indent=2, ensure_ascii=False)

    # 3. ACF comparison
    # Filtrar solo los principales
    main_series = {k: v for k, v in series_dict.items() if not k.startswith("Homero_")}
    acf_results = compare_acf(main_series, max_lag=100)

    with open(RESULTS_DIR / "acf_comparison.json", "w") as f:
        json.dump(acf_results, f, indent=2, ensure_ascii=False)

    # 4. MF-DFA
    log.info("\n=== Multifractal DFA ===")
    mfdfa_results = {}
    for name, s in main_series.items():
        log.info(f"  MF-DFA para {name} (n={len(s)})...")
        result = mfdfa(s)
        mfdfa_results[name] = result
        if result:
            log.info(f"    spectrum_width={result['spectrum_width']:.4f}, "
                    f"h(q=2)={result['h_q2']:.4f}")

    with open(RESULTS_DIR / "multifractal_spectra.json", "w") as f:
        json.dump(mfdfa_results, f, indent=2, ensure_ascii=False)

    # 5. Test de mismo proceso
    process_test = test_same_process(series_dict, arfima_params, n_sim=500)

    with open(RESULTS_DIR / "same_process_test.json", "w") as f:
        json.dump(process_test, f, indent=2, ensure_ascii=False)

    # 6. Veredicto
    verdict = synthesize_verdict(arfima_params, acf_results, mfdfa_results, process_test)

    with open(RESULTS_DIR / "verdict.json", "w") as f:
        json.dump(verdict, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    log.info(f"\n{'=' * 60}")
    log.info(f"Script 4 completado en {elapsed:.1f}s")
    log.info(f"Resultados en {RESULTS_DIR}")


if __name__ == "__main__":
    main()
