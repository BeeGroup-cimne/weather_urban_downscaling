# AUDITORÍA DE TRAZABILIDAD — Repo weather_urban_downscaling
## Fecha: 2026-05-23 | Versión: manuscrito UCLIM-S-26-01796

---

## 1. RESUMEN EJECUTIVO

**Hallazgo principal: el paper mezcla resultados de DOS familias de checkpoints incompatibles.**

- **Tiles_*** (10 checkpoints, tile-based training, secuencia fija T=6)
- **Ablation_*** (36 checkpoints, full-frame training, T∈{6,12,24} × seeds S42/S43/S44)

La configuración principal (`eval_config.yaml`) evalúa Exp1/Exp2/CS1/CS2 con checkpoints `Tiles_*`, pero Exp3 con `Ablation_*`. El paper las trata como la misma arquitectura "Mamba", pero son modelos entrenados con pipelines distintos. Algunas figuras se generaron con una tercera configuración (`refresh_seq24s43`) que mezcla semillas.

**Riesgo de integridad científica: ALTO.** Cualquier revisor que pida reproducibilidad detectará estas discrepancias.

---

## 2. INVENTARIO DE CHECKPOINTS (50 archivos .h5)

### 2.1 Familia Tiles (tile-based) — 10 checkpoints

| Archivo | Tamaño |
|---------|--------|
| Tiles_LSTM_S42_best.h5 | 55 MB |
| Tiles_MAMBA_S42_best.h5 | 8 MB |
| Tiles_TRANSFORMER_S42_best.h5 | 13 MB |
| Tiles_UNET_S42_best.h5 | 23 MB |
| Tiles_UNET_S43_best.h5 | 23 MB |

**Características:** Entrenados con pipeline tile-based (`train_tiles.py`). Cada modelo tiene UNA sola secuencia temporal (T=6). Solo Mamba tiene 1 seed (S42); UNet tiene 2 seeds.

### 2.2 Familia Ablation (full-frame, "Legacy") — 36 checkpoints

| Arquitectura | Seeds | Seq lengths | Total |
|-------------|-------|-------------|-------|
| UNet | S42, S43, S44 | T=6 (fijo) | 6 (best+last × 3) |
| LSTM | S42, S43, S44 | T=6 (fijo) | 6 |
| Transformer | S42, S43, S44 | T=6 (fijo) | 6 |
| Mamba | S42, S43, S44 | T=6, T=12, T=24 | 18 |

**Características:** Entrenados con pipeline full-frame (`run_ablation.py`). Mamba es la única arquitectura con ablación de secuencia. Cargables con `ModelZoo.build_hybrid_unet_mamba` + `load_weights` (NO `tf.keras.models.load_model`).

---

## 3. CONFIGURACIONES DE EVALUACIÓN (3 archivos)

### 3.1 eval_config.yaml (PRINCIPAL — usada para publicación)

| Experimento | Checkpoints | Seed | Models |
|------------|-------------|------|--------|
| Exp1 (spatial) | **Tiles_*** S42 (+UNet S43) | 42 | unet, lstm, transformer, mamba |
| Exp2 (groundtruth) | **Tiles_*** S42 | 42 | unet, lstm, transformer, mamba |
| Exp3 (bottleneck) | **Ablation_MAMBA_S42** + **S42_SEQ12** | 42 | mamba_seq6, mamba_seq12 |
| CS1 (heatwave) | **Tiles_*** S42 | 42 | unet, lstm, transformer, mamba |
| CS2 (robustness) | **Tiles_*** S42 | 42 | unet, lstm, transformer, mamba |

**⚠️ PROBLEMA #1: Mamba seq24 AUSENTE en Exp3.** Solo compara seq6 vs seq12. ¿De dónde salen los resultados de T=24 en el paper?

**⚠️ PROBLEMA #2: Tiles_MAMBA_S42 no tiene variantes de seq_len.** Todas las evaluaciones de Exp1/Exp2/CS1/CS2 usan el MISMO checkpoint Mamba (T=6 fijo). Pero el paper reporta Mamba T=6, T=12, T=24 como si fueran modelos distintos evaluados en los mismos experimentos.

### 3.2 eval_config_refresh_seq24s43_server.yaml ("refresh" — Mar 15, 2026)

| Experimento | Checkpoints | Seed |
|------------|-------------|------|
| Exp1 | Ablation_MAMBA_Legacy_S42_SEQ24 + S43_SEQ24 | 42, 43 |
| Exp2 | Ablation_MAMBA_Legacy_S43_SEQ24 | 43 |
| Exp3 | S42 (seq6), S42 (seq12), **S43 (seq24)** | mixto |
| CS1/CS2 | S43_SEQ24 | 43 |

**⚠️ PROBLEMA #3: Mezcla de semillas en Exp3.** seq6=S42, seq12=S42, seq24=S43. La comparación seq12 vs seq24 confunde efecto de secuencia con efecto de semilla.

**⚠️ PROBLEMA #4: Esta es la config que generó la Figura 4 stale** (anotaciones "refresh"/"Mamba refresh"). Los valores NO coinciden con la Tabla 3 (que viene de eval_config.yaml).

### 3.3 eval_config_exp3_fullfidelity_n3_server.yaml (n=3)

- 9 checkpoints Mamba (3 seeds × 3 seq lengths), todos Ablation_*
- Exp1/Exp2/CS1/CS2 **DESHABILITADOS**
- Solo genera Exp3 con n=3 por seq_len

---

## 4. DISCREPANCIAS DETECTADAS

### 🔴 CRÍTICAS

| # | Discrepancia | Evidencia | Impacto |
|---|-------------|-----------|---------|
| **D1** | Figura 4 ≠ Tabla 3 (Exp1 metrics) | Fig 4: ConvLSTM 0.142, Mamba 0.144, UNet 0.163. Tabla 3: Mamba 0.138, ConvLSTM 0.162, UNet 0.213. La figura tiene anotaciones "refresh" → fue generada con `refresh_seq24s43` (Ablation_*), no con `eval_config.yaml` (Tiles_*). | **FATAL**. El paper reporta dos conjuntos de resultados distintos para el mismo experimento. |
| **D2** | Figura 5 caption "Mamba (T=6)" vs imagen "seq24 (S43)" | La imagen viene de la config `refresh_seq24s43` que usa Ablation_MAMBA_Legacy_S43_SEQ24. El caption se escribió para Tiles_MAMBA_S42 (T=6). | **FATAL**. Figura y caption son de modelos diferentes. |
| **D3** | Paper dice "Mamba same parameter count as ConvLSTM", pero Tabla 2 muestra Mamba 0.677M vs ConvLSTM 4.612M | Tiles_MAMBA_S42 = 8 MB, Tiles_LSTM_S42 = 55 MB. Ratio real ~7×. | **GRAVE**. La premisa de "controlled benchmark with fixed backbone" es falsa si los parámetros difieren 7×. |
| **D4** | Mamba seq24 no está en eval_config.yaml Exp3, pero el paper reporta resultados de T=24 | eval_config.yaml solo compara seq6 vs seq12. Los resultados de T=24 vienen de `refresh_seq24s43` o `n3_server`, que usan checkpoints distintos. | **GRAVE**. No hay trazabilidad de qué config produjo los números de T=24. |
| **D5** | Tiles_ vs Ablation_ tratados como intercambiables | El paper describe UNA arquitectura ("hybrid U-Net with Mamba bottleneck"), pero evalúa Tiles_MAMBA_S42 en Exp1/Exp2/CS1/CS2 y Ablation_MAMBA_Legacy_* en Exp3. Son checkpoints de pipelines de entrenamiento distintos. | **GRAVE**. Los resultados no son comparables entre experimentos. |

### 🟠 SIGNIFICATIVAS

| # | Discrepancia | Evidencia |
|---|-------------|-----------|
| **D6** | evaluate_for_paper.py referencia modelos inexistentes | `UNet_best.h5`, `ConvLSTM_best.h5`, `Transformer_best.h5` no existen en experiments/models/. No hay symlinks. |
| **D7** | Doble split ambiguo | Train=Ene-Oct, Val=Nov, Test=Dic. Pero Exp1 evalúa en verano (dentro de training). CS1 dice "held-out summer" pero no está documentado cómo. |
| **D8** | eval_config_refresh mezcla semillas en Exp3 | seq6/seq12=S42, seq24=S43. Confunde efecto de arquitectura con efecto de inicialización. |
| **D9** | CV reportado (23.3%→2.3%) es sobre n=3 | Solo 3 seeds en Ablation_MAMBA_Legacy. Estadísticamente frágil para un headline. |

---

## 5. DIAGRAMA DE TRAZABILIDAD ROTO

```
Paper reporta:
  "Mamba T=12 achieves RMSE 0.677°C, SSIM 0.848"

¿Qué checkpoint generó esto?
  → No documentado explícitamente en ningún config
  
Candidatos:
  a) Tiles_MAMBA_S42 (T=6 fijo, usado en Exp1/Exp2/CS1)
  b) Ablation_MAMBA_Legacy_S42_SEQ12 (T=12, usado en Exp3)
  c) Ablation_MAMBA_Legacy_S43_SEQ24 (T=24, usado en refresh)
  
El headline number (0.677) probablemente viene de (b) en Exp3,
pero Exp3 usa Ablation_* mientras Exp1/Exp2 usan Tiles_*.
El paper no distingue entre estas familias.
```

---

## 6. PLAN DE REMEDIACIÓN

### Paso 1: Elegir UNA familia de checkpoints canónica

**Recomendación: Ablation_* (full-frame)**

- Tienen ablación de secuencia completa (T=6,12,24) con 3 seeds
- Son los únicos que pueden sostener los claims del paper
- Cargables vía `ModelZoo.build_hybrid_unet_mamba` + `load_weights`

### Paso 2: Regenerar TODAS las evaluaciones con UN solo config

Crear `config/eval_config_CANONICAL.yaml`:
- Exp1: Ablation_UNET S42, Ablation_LSTM S42, Ablation_TRANSFORMER S42, Ablation_MAMBA S42 (T=6), S42_SEQ12 (T=12), S42_SEQ24 (T=24)
- Exp2: mismos checkpoints
- Exp3: todos los Ablation_MAMBA (3 seeds × 3 seq)
- CS1, CS2: Ablation_* checkpoints

### Paso 3: Verificar consistencia figura↔tabla↔checkpoint

Cada figura debe tener un comentario YAML embebido:
```yaml
# figure: F4_experiment1_metrics
# config: eval_config_CANONICAL.yaml
# date: 2026-05-23
# checkpoints:
#   unet: Ablation_UNET_Legacy_S42_best.h5
#   lstm: Ablation_LSTM_Legacy_S42_best.h5
#   ...
```

### Paso 4: Arreglar la narrativa de parámetros

- Mamba: 0.677M → reportar como ventaja ("7× fewer parameters")
- Eliminar claim de "same parameter count"
- O implementar versión de Mamba con ~4.6M parámetros para fair comparison

### Paso 5: Generar manifest de reproducibilidad

CSV: `experiments/manifest.csv` con columnas:
`figure, table, experiment, model, checkpoint_path, config_file, eval_date, verified`

---

## 7. ARCHIVOS QUE REQUIEREN ACCIÓN INMEDIATA

| Archivo | Acción |
|---------|--------|
| `Paper/imagenes/F4_experiment1_metrics.pdf` | Regenerar desde config canónico |
| `Paper/imagenes/F5_experiment1_qualitative.pdf` | Corregir caption/modelo |
| `config/eval_config.yaml` | Archivar como `_DEPRECATED` |
| `config/eval_config_refresh_seq24s43_server.yaml` | Archivar |
| `scripts/evaluation/evaluate_for_paper.py` | Actualizar paths de checkpoints |
| `Paper/main_v1_urban_climate.tex` | Corregir claim de parámetros (§3.2) |

---

*Auditoría generada por phdresearch. Requiere validación manual de los valores numéricos.*
