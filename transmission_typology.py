#!/usr/bin/env python3
"""
Fase 9 — Script 3: Tipología de Transmisión Refinada
¿La clasificación "controlada vs libre" tiene subtipos con propiedades H distintas?
Clustering jerárquico + correlación con dimensiones tipológicas.
"""

import json
import logging
import time
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist

# ── Configuración ────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent
RESULTS_DIR = BASE / "results" / "typology"
LOG_DIR = BASE / "logs"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "fase9_typology.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ── Datos de corpus (cargados de resultados anteriores) ──────────────────

def load_all_corpus_metrics():
    """Carga métricas de todos los corpus de fases anteriores."""
    fase5 = BASE / "results" / "fase5_comparison.json"
    rigveda = BASE / "results" / "rigveda" / "rigveda_metrics.json"

    corpora = {}

    # Fase 5: AT, NT, Corán, Homero, Heródoto, Mishnah
    if fase5.exists():
        with open(fase5) as f:
            data = json.load(f)
        for c in data:
            name = c["corpus"]
            corpora[name] = {
                "H": c.get("hurst_H"),
                "alpha": c.get("dfa_alpha"),
                "chi": c.get("bond_dim_chi"),
                "mps_p": c.get("mps_permtest_p"),
                "mps_significant": c.get("mps_significant"),
                "delta_S": c.get("delta_S_mean"),
                "n_words": c.get("n_words"),
                "n_units": c.get("n_units"),
                "lang": c.get("lang"),
                "type": c.get("type"),
            }

    # Rig Veda
    if rigveda.exists():
        with open(rigveda) as f:
            rv = json.load(f)
        corpora["Rig Veda (Sánscrito)"] = {
            "H": rv.get("hurst_H"),
            "alpha": rv.get("dfa_alpha"),
            "chi": rv.get("bond_dim_chi"),
            "mps_p": rv.get("mps_permtest_p"),
            "mps_significant": rv.get("mps_significant"),
            "delta_S": rv.get("delta_S_mean"),
            "n_words": rv.get("n_words"),
            "n_units": rv.get("n_padas"),
            "lang": "san",
            "type": rv.get("type", "religioso"),
        }

    return corpora


# ── Tipología multidimensional ───────────────────────────────────────────

# Dimensiones tipológicas (derivadas de la literatura académica)
TYPOLOGY = {
    "AT (Hebreo)": {
        "authority_type": "dictado_directo",      # śruti-like
        "authority_code": 3,                       # 3=dictado, 2=testimonio, 1=interpretación, 0=narrativa
        "control_delay_years": 0,                  # Rango: 0 (tradición) a ~500 (crítica: Esdras)
        "control_delay_code": 0,                   # 0=inmediato, 1=generación, 2=siglos
        "control_type": "scribal_contada",
        "control_intensity": 3,                    # 3=máximo (masora), 2=alto, 1=medio, 0=ninguno
        "isolation": False,                        # Múltiples líneas: DSS, LXX, SP
        "isolation_code": 0,
        "composition_period_bce": (-1200, -400),   # Rango amplio
        "transmission_start_bce": -500,            # Esdras como punto conservador
        "n_independent_lines": 3,                  # MT, LXX, SP
    },
    "NT (Griego)": {
        "authority_type": "testimonio_mediado",
        "authority_code": 2,
        "control_delay_years": 300,                # Desde composición (~50-100 d.C.) hasta Nicea (325)
        "control_delay_code": 2,
        "control_type": "institucional_monástica",
        "control_intensity": 2,
        "isolation": False,
        "isolation_code": 0,
        "composition_period_bce": (-50, -120),     # 50-120 d.C. = -50 a -120 en BCE invertido
        "transmission_start_bce": 325,             # Nicea (usamos CE directamente aquí)
        "n_independent_lines": 5,                  # NA, Byz, Western, Caesarean, Alexandrian
    },
    "Corán (Árabe)": {
        "authority_type": "dictado_directo",
        "authority_code": 3,
        "control_delay_years": 20,                 # ~632 (muerte profeta) a ~650 (Uthmán)
        "control_delay_code": 0,
        "control_type": "memorización_huffaz",
        "control_intensity": 3,
        "isolation": True,                         # Uthmán quemó variantes
        "isolation_code": 1,
        "composition_period_bce": (-610, -632),    # 610-632 d.C.
        "transmission_start_bce": -650,
        "n_independent_lines": 1,                  # Una sola canonización
    },
    "Homero (Griego)": {
        "authority_type": "narrativa_oral",
        "authority_code": 0,
        "control_delay_years": 400,                # Composición ~800 a.C., escritura ~400 a.C.
        "control_delay_code": 2,
        "control_type": "ninguno_hasta_alejandrinos",
        "control_intensity": 0,
        "isolation": False,
        "isolation_code": 0,
        "composition_period_bce": (800, 700),
        "transmission_start_bce": 400,
        "n_independent_lines": 0,                  # Tradición oral no controlada
    },
    "Heródoto (Griego)": {
        "authority_type": "narrativa_historica",
        "authority_code": 0,
        "control_delay_years": 200,
        "control_delay_code": 2,
        "control_type": "ninguno",
        "control_intensity": 0,
        "isolation": False,
        "isolation_code": 0,
        "composition_period_bce": (440, 425),
        "transmission_start_bce": 200,
        "n_independent_lines": 0,
    },
    "Mishnah (Hebreo)": {
        "authority_type": "interpretación_legal",
        "authority_code": 1,
        "control_delay_years": 0,                  # R. Judah haNasi compiló y fijó
        "control_delay_code": 0,
        "control_type": "scribal_académica",
        "control_intensity": 2,
        "isolation": False,
        "isolation_code": 0,
        "composition_period_bce": (-200, -200),    # ~200 d.C.
        "transmission_start_bce": -200,
        "n_independent_lines": 2,                  # Tradiciones palestina y babilónica
    },
    "Rig Veda (Sánscrito)": {
        "authority_type": "dictado_directo",        # śruti
        "authority_code": 3,
        "control_delay_years": 0,                   # Transmisión oral desde composición
        "control_delay_code": 0,
        "control_type": "oral_verificada_pathas",
        "control_intensity": 3,                     # Máximo: pada, krama, jaṭā, ghana pathas
        "isolation": False,
        "isolation_code": 0,
        "composition_period_bce": (1500, 1200),
        "transmission_start_bce": 1500,
        "n_independent_lines": 5,                   # 5 shakhas supervivientes de Ṛgveda
    },
}


def build_corpus_matrix(corpora):
    """Construye la matriz corpus × (métricas + dimensiones tipológicas)."""
    log.info("\n=== Construyendo matriz corpus × dimensiones ===")

    matrix = {}
    for name, metrics in corpora.items():
        if name not in TYPOLOGY:
            log.warning(f"  {name}: no tiene tipología definida, skip")
            continue

        typo = TYPOLOGY[name]
        entry = {
            # Métricas numéricas (de los datos)
            "H": metrics["H"],
            "alpha": metrics["alpha"],
            "chi": metrics["chi"],
            "mps_p": metrics["mps_p"],
            "mps_significant": metrics["mps_significant"],
            "delta_S": metrics["delta_S"],
            "n_words": metrics["n_words"],
            "n_units": metrics["n_units"],
            "lang": metrics["lang"],
            # Dimensiones tipológicas (de la literatura académica)
            **typo,
        }
        matrix[name] = entry
        log.info(f"  {name}: H={metrics['H']:.3f}, auth={typo['authority_code']}, "
                f"delay={typo['control_delay_years']}, intensity={typo['control_intensity']}")

    return matrix


def hierarchical_clustering(corpora):
    """Clustering jerárquico sobre métricas H, α, MPS_p."""
    log.info("\n=== Clustering jerárquico ===")

    names = []
    features = []

    for name, metrics in corpora.items():
        if metrics["H"] is None or metrics["alpha"] is None or metrics["mps_p"] is None:
            continue
        names.append(name)
        # Normalizar: H y alpha están en [0,2], mps_p en [0,1]
        features.append([
            metrics["H"],
            metrics["alpha"],
            1.0 - metrics["mps_p"],  # Invertir: 1=significativo, 0=no
        ])

    X = np.array(features)
    n = len(names)

    if n < 3:
        return {"error": "Insuficientes corpus para clustering", "n": n}

    # Normalizar a z-scores
    X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)

    # Distancia euclidiana + linkage Ward
    dists = pdist(X_norm, metric="euclidean")
    Z = linkage(dists, method="ward")

    # Cortar en 2 y 3 clusters
    labels_2 = fcluster(Z, t=2, criterion="maxclust")
    labels_3 = fcluster(Z, t=3, criterion="maxclust")

    results = {
        "method": "ward_euclidean",
        "features_used": ["H", "alpha", "1-mps_p"],
        "n_corpora": n,
        "corpora": {},
        "linkage_matrix": Z.tolist(),
    }

    for i, name in enumerate(names):
        results["corpora"][name] = {
            "features": features[i],
            "features_normalized": X_norm[i].tolist(),
            "cluster_k2": int(labels_2[i]),
            "cluster_k3": int(labels_3[i]),
        }

    # Describir clusters
    for k in [2, 3]:
        labels = labels_2 if k == 2 else labels_3
        cluster_key = f"cluster_k{k}"
        cluster_desc = {}
        for cl in range(1, k + 1):
            members = [names[i] for i in range(n) if labels[i] == cl]
            member_H = [features[i][0] for i in range(n) if labels[i] == cl]
            cluster_desc[f"cluster_{cl}"] = {
                "members": members,
                "mean_H": float(np.mean(member_H)),
                "n": len(members),
            }
        results[f"clusters_k{k}"] = cluster_desc

    log.info(f"  k=2: {results['clusters_k2']}")
    log.info(f"  k=3: {results['clusters_k3']}")

    return results


def analyze_predictive_dimensions(matrix, clustering):
    """
    Correlaciona cada dimensión tipológica con la asignación de cluster.
    ¿Qué dimensión predice mejor el cluster matemático?
    """
    log.info("\n=== Análisis de dimensiones predictivas ===")

    if "error" in clustering:
        return {"error": clustering["error"]}

    # Dimensiones numéricas a probar
    dimensions = [
        "authority_code",
        "control_delay_years",
        "control_delay_code",
        "control_intensity",
        "isolation_code",
        "n_independent_lines",
    ]

    # Usar cluster_k2 como variable dependiente
    corpora_in_cluster = clustering.get("corpora", {})
    names = list(corpora_in_cluster.keys())

    results = {}
    for dim in dimensions:
        dim_values = []
        cluster_labels = []
        for name in names:
            if name in matrix and dim in matrix[name]:
                dim_values.append(float(matrix[name][dim]))
                cluster_labels.append(corpora_in_cluster[name]["cluster_k2"])

        if len(set(cluster_labels)) < 2 or len(dim_values) < 4:
            results[dim] = {"note": "insufficient_variation"}
            continue

        # Point-biserial correlation (cluster label vs dimension value)
        # Remap cluster labels to 0/1
        unique_labels = sorted(set(cluster_labels))
        binary = [0 if l == unique_labels[0] else 1 for l in cluster_labels]

        r, p = stats.pointbiserialr(binary, dim_values)

        # También Mann-Whitney entre los dos clusters
        group_0 = [dim_values[i] for i in range(len(binary)) if binary[i] == 0]
        group_1 = [dim_values[i] for i in range(len(binary)) if binary[i] == 1]

        if len(group_0) >= 2 and len(group_1) >= 2:
            mw_stat, mw_p = stats.mannwhitneyu(group_0, group_1, alternative="two-sided")
        else:
            mw_stat, mw_p = float("nan"), float("nan")

        results[dim] = {
            "point_biserial_r": float(r),
            "point_biserial_p": float(p),
            "mann_whitney_p": float(mw_p),
            "group_0_mean": float(np.mean(group_0)) if group_0 else None,
            "group_1_mean": float(np.mean(group_1)) if group_1 else None,
            "group_0_members": [names[i] for i in range(len(binary)) if binary[i] == 0],
            "group_1_members": [names[i] for i in range(len(binary)) if binary[i] == 1],
        }
        log.info(f"  {dim}: r={r:.3f}, p={p:.4f}, MW_p={mw_p:.4f}")

    # Ranking por |r|
    ranked = sorted(
        [(dim, abs(results[dim].get("point_biserial_r", 0)))
         for dim in dimensions if "point_biserial_r" in results.get(dim, {})],
        key=lambda x: x[1], reverse=True
    )
    results["ranking"] = [{"dimension": d, "abs_r": float(r)} for d, r in ranked]

    if ranked:
        best = ranked[0]
        results["best_predictor"] = {
            "dimension": best[0],
            "abs_r": float(best[1]),
            "interpretation": (
                f"La dimensión '{best[0]}' es la más predictiva del clustering matemático "
                f"(|r|={best[1]:.3f}). "
            ),
        }

    return results


def build_refined_typology(matrix, clustering, predictive):
    """Construye la tipología refinada basada en los resultados."""
    log.info("\n=== Construyendo tipología refinada ===")

    best_dim = predictive.get("best_predictor", {}).get("dimension", "authority_code")

    # Mapear corpus a su tipo refinado
    refined = {}
    for name, data in matrix.items():
        H = data.get("H")
        mps_sig = data.get("mps_significant")
        auth = data.get("authority_type")
        delay = data.get("control_delay_years")
        intensity = data.get("control_intensity")

        # Clasificación refinada
        if H is not None and H > 0.9 and mps_sig:
            category = "revelación_controlada_inmediata"
            description = "Dictado directo + transmisión controlada desde el origen"
        elif H is not None and H > 0.9 and not mps_sig:
            category = "heterogéneo_compilado"
            description = "H alto por heterogeneidad inter-libro, no por estructura interna"
        elif H is not None and H > 0.6 and mps_sig:
            category = "interpretación_controlada"
            description = "Transmisión controlada pero no dictado directo"
        elif H is not None and H < 0.7:
            category = "transmisión_libre"
            description = "Sin control efectivo durante transmisión temprana"
        else:
            category = "indeterminado"
            description = "No clasificable con las métricas actuales"

        refined[name] = {
            "category": category,
            "description": description,
            "H": H,
            "mps_significant": mps_sig,
            "authority_type": auth,
            "control_delay_years": delay,
            "control_intensity": intensity,
        }
        log.info(f"  {name}: → {category}")

    # Resumen de categorías
    categories = {}
    for name, data in refined.items():
        cat = data["category"]
        if cat not in categories:
            categories[cat] = {"members": [], "description": data["description"]}
        categories[cat]["members"].append(name)

    return {
        "best_predictor_dimension": best_dim,
        "corpus_classification": refined,
        "category_summary": categories,
        "note": (
            "Esta tipología es EXPLORATORIA y se basa en 7 corpus. "
            "No se afirma causalidad. La clasificación emerge de la combinación "
            "de métricas matemáticas (H, MPS) con dimensiones tipológicas académicas."
        ),
    }


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    log.info("=" * 60)
    log.info("FASE 9 — Script 3: Tipología de Transmisión")
    log.info("=" * 60)

    # 1. Cargar métricas
    corpora = load_all_corpus_metrics()
    log.info(f"Corpus cargados: {len(corpora)}")
    for name, m in corpora.items():
        log.info(f"  {name}: H={m.get('H')}, α={m.get('alpha')}, MPS_p={m.get('mps_p')}")

    # 2. Construir matriz
    matrix = build_corpus_matrix(corpora)

    # 3. Clustering jerárquico
    clustering = hierarchical_clustering(corpora)

    # 4. Dimensiones predictivas
    predictive = analyze_predictive_dimensions(matrix, clustering)

    # 5. Tipología refinada
    refined = build_refined_typology(matrix, clustering, predictive)

    # ── Guardar resultados ───────────────────────────────────────────────

    with open(RESULTS_DIR / "corpus_matrix.json", "w") as f:
        json.dump(matrix, f, indent=2, ensure_ascii=False, default=str)

    with open(RESULTS_DIR / "clustering_results.json", "w") as f:
        json.dump(clustering, f, indent=2, ensure_ascii=False, default=str)

    with open(RESULTS_DIR / "predictive_dimensions.json", "w") as f:
        json.dump(predictive, f, indent=2, ensure_ascii=False, default=str)

    with open(RESULTS_DIR / "refined_typology.json", "w") as f:
        json.dump(refined, f, indent=2, ensure_ascii=False, default=str)

    elapsed = time.time() - t0
    log.info(f"\n{'=' * 60}")
    log.info(f"Script 3 completado en {elapsed:.1f}s")
    log.info(f"Resultados en {RESULTS_DIR}")


if __name__ == "__main__":
    main()
