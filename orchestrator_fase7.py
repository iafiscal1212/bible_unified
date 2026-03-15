#!/usr/bin/env python3
"""
Orchestrator Fase 7 — Lanza dos tareas en paralelo:
  1. analyze_dss.py — Dead Sea Scrolls analysis
  2. generate_report.py — Documento de investigación consolidado

Monitoriza ambos cada 30 segundos.
Cuando analyze_dss.py termina, regenera el reporte con los datos DSS.
"""

import subprocess
import sys
import time
import logging
from pathlib import Path

BASE = Path(__file__).resolve().parent
LOG_DIR = BASE / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "fase7_orchestrator.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


def tail_log(logfile, n=3):
    """Muestra las últimas n líneas de un log."""
    try:
        with open(logfile, "r") as f:
            lines = f.readlines()
            return lines[-n:] if len(lines) >= n else lines
    except FileNotFoundError:
        return ["  (log no existe aún)"]


def main():
    log.info("=" * 70)
    log.info("FASE 7 — Orchestrator: DSS + Reporte en paralelo")
    log.info("=" * 70)

    python = sys.executable

    # Lanzar ambos procesos en paralelo
    log.info("Lanzando analyze_dss.py...")
    p_dss = subprocess.Popen(
        [python, str(BASE / "analyze_dss.py")],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    log.info("Lanzando generate_report.py (primera pasada, sin DSS)...")
    p_report = subprocess.Popen(
        [python, str(BASE / "generate_report.py")],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # Monitorizar cada 30 segundos
    dss_done = False
    report_done = False
    start = time.time()

    while not (dss_done and report_done):
        time.sleep(30)
        elapsed = time.time() - start

        # Verificar estado
        if not dss_done:
            ret = p_dss.poll()
            if ret is not None:
                dss_done = True
                log.info(f"  [✓] analyze_dss.py terminó (código={ret}, {elapsed:.0f}s)")
                # Mostrar tail del log
                for line in tail_log(LOG_DIR / "fase7_dss.log", 5):
                    log.info(f"    DSS> {line.rstrip()}")
            else:
                log.info(f"  [⏳] analyze_dss.py corriendo ({elapsed:.0f}s)...")
                for line in tail_log(LOG_DIR / "fase7_dss.log", 2):
                    log.info(f"    DSS> {line.rstrip()}")

        if not report_done:
            ret = p_report.poll()
            if ret is not None:
                report_done = True
                log.info(f"  [✓] generate_report.py terminó (código={ret}, {elapsed:.0f}s)")
            else:
                log.info(f"  [⏳] generate_report.py corriendo ({elapsed:.0f}s)...")

    # Cuando DSS termine, regenerar el reporte con los datos de DSS incluidos
    log.info("\nAmbos procesos terminados. Regenerando reporte con datos DSS...")
    result = subprocess.run(
        [python, str(BASE / "generate_report.py")],
        capture_output=True, text=True, timeout=120
    )
    if result.returncode == 0:
        log.info("  [✓] Reporte regenerado exitosamente con datos DSS.")
    else:
        log.warning(f"  [!] Error regenerando reporte: {result.stderr[:500]}")

    # Resumen final
    elapsed_total = time.time() - start
    log.info(f"\n{'=' * 70}")
    log.info(f"FASE 7 COMPLETADA en {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)")
    log.info(f"  Outputs:")
    log.info(f"    results/dss/dss_isaiah_comparison.json")
    log.info(f"    results/research_report.md")
    log.info(f"{'=' * 70}")


if __name__ == "__main__":
    main()
