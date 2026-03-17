# Fase 21 - Test de Hipotesis Composicional: DFA ~ Paralelismo

## Resumen

Fase 21 testea la hipotesis de que DFA refleja estructura composicional
(paralelismo, repeticion, coherencia local) y NO historia de transmision.

## 1. Regresion Composicional (intra-biblica)

- Top predictor de DFA: **AC1** (r=0.8033)
- Top 5 features: ['AC1', 'CV', 'pos_entropy', 'mean_delta', 'proper_ratio']
- R2 modelo testament: 0.2642
- R2 modelo AC1: 0.6452
- R2 modelo composicional: 0.6981
- Mejor por AIC: C_compositional
- Composicional > testament: True

## 2. Indice de Paralelismo

- r(PI, DFA) = 0.818 (p=0.0)
- PC1 varianza explicada: 0.8004

## 3. Analisis de Mediacion (testament -> AC1 -> DFA)

- Tipo de mediacion: **partial**
- Proporcion mediada: 0.4554
- Efecto indirecto: 0.1036
- Bootstrap CI 95%: [0.0225, 0.1968]
- CI cruza 0: False
- Sobel p: 0.022809

## 4. Composicional vs Transmision (inter-corpus)

- **VEREDICTO: COMPOSICIONAL (sugerida)** (3/4)
- 3/4 criteria met. R2 COMP=0.694 vs TRANS=0.027. LOO MAE COMP=0.088 vs TRANS=0.153. rho(AC1,DFA)=0.552 vs rho(delay,DFA)=0.013.

## Conclusion

DFA captura estructura composicional (paralelismo, repeticion, coherencia local).
La diferencia AT/NT en DFA se explica por diferencias composicionales,
no por historia de transmision. AC1 media la relacion testament -> DFA.
