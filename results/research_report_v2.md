# Fase 20 — Corrección de análisis previos: DFA como métrica principal

## Resumen ejecutivo

- **Threshold DFA óptimo**: 0.5 (F1=0.8387)
- **Zona de ambigüedad**: [0.445, 0.555]
- **AT DFA medio**: 0.7168 ± 0.2236
- **NT DFA medio**: 0.4893 ± 0.1189
- **Mann-Whitney**: p=7.4e-05, Cohen d=1.27

## Mishnah (DFA verdict)

- DFA = 0.6545
- Clasificación: **AT**
- Mishnah DFA=0.6545 classified as AT by DFA threshold. z-score vs AT=-0.279, vs NT=1.389. Closer to AT distribution.

## Estructura inter-libro

- AT permutation test (H): p=0.972
- AT permutation test (DFA): p=?
- NT permutation test (H): p=0.924

## Análisis controlado por género

- AT_narrative vs NT_narrative DFA: p=0.004435, Cohen d=1.49
- Significativo: True

## H4' DFA Retest

- **Veredicto: INDETERMINADA**
- Criterios cumplidos: 0/4
- H4' DFA retest: 0/4 criteria met → INDETERMINADA. MW p=0.530303, Spearman rho=-0.164 (p=0.61048), LOO: 0/12 robust.

## Corrección de análisis previos

La Fase 19 reveló que H, AC1 y CV no separan AT/NT a nivel libro (p>0.05). Solo DFA es robusto (p=0.000074, Cohen d=1.30). El clasificador original estaba contaminado por mean_verse_len y pos_entropy. Esta fase rehace el análisis completo usando DFA como métrica principal y controlando por género literario.

---
*Generado automáticamente por orchestrator_fase20.py*