# AUDITORÍA COMPLETA — Repositorios downscaling (post-Claude)
## Fecha: 2026-05-25 | Alcance: v1 congelado, v2 PyTorch, canónico, paper

---

## 1. REPO v2 (`weather_urban_downscaling_v2/`) — PyTorch

### 1.1 Estructura de modelos

**✅ DownsrUNet canónico** (`src/models/downsr_unet.py`) — 236 líneas, correcto.
- 3 bottlenecks intercambiables (unet, convlstm, mamba) con encoder/decoder compartido
- Fallback Mamba cuando mamba_ssm no está instalado
- Conv1×1(128→256) tras Mamba para decoder idéntico entre variantes
- Supervisión en [:, -1] (último timestep) — fiel a v1

**🟠 Modelos descartados presentes en disco:**
- `src/models/unet.py` — 66 líneas, encoder estático separado (diseño incorrecto)
- `src/models/convlstm.py` — 101 líneas, encoder estático separado (diseño incorrecto)
- `src/models/mamba_unet.py` — 156 líneas, global avg pool Mamba (diseño incorrecto)

Riesgo: `__init__.py` solo exporta DownsrUNet, así que no se importan activamente.
Pero confunde si alguien los encuentra. Recomendación: mover a archive/ o eliminar.

### 1.2 Dataset (src/data/dataset.py) — 294 líneas

**✅ Correcto:**
- Formato EPSG:3035 (time, y=251, x=251)
- Static index via pyproj WGS84→EPSG:3035
- ERA5 crop 4×5 con nearest-neighbor para celdas marinas
- WeightedRandomSampler para sobremuestreo verano (3× Jun-Sep)
- Normalización con stats pre-computados

**🟠 Detalle:** El dataset abre 3 zarrs con `consolidated=False`. En el servidor
esto funciona, pero si los zarrs no tienen `.zmetadata` explícito, xarray puede
fallar silenciosamente. Verificar en Vortex antes de training.

### 1.3 Train script (scripts/training/train.py) — 313 líneas

**✅ Correcto:**
- AMP (float16 en CUDA), grad clip, early stopping, resume
- ReduceLROnPlateau (factor=0.5, patience=5)
- Config snapshot JSON + history CSV
- Salida: best.pt, last.pt, history.csv, config.json

**🟠 Detalle:** `num_workers=4` en DataLoader. En Vortex con 2 GPUs y VLLM
consumiendo ~70GB, verificar que hay RAM suficiente para workers.
Si OOM, reducir a 2.

### 1.4 Loss (src/losses.py) — 49 líneas

**✅ Correcto:** HybridLoss(alpha=0.8) = 0.2*MSE + 0.8*(1-SSIM)
Port exacto de v1. SSIM con ventana 11, C1=0.01², C2=0.03².

### 1.5 Blocks (src/models/blocks.py) — 61 líneas

**✅ Correcto:** ConvBlock, ConvLSTMCell, td() — ports fieles de v1.

### 1.6 Config (config/config.py) — 110 líneas

**✅ Correcto:**
- 9 ERA5 vars, 11 static features (ndvi_min excluida)
- Split: Train 2008-2015, Val 2016, Test 2017
- SEQ_LEN=6 default, BATCH_SIZE=8, LR=1e-4
- MAMBA_D_MODEL=128, D_STATE=16, D_CONV=4, EXPAND=2

**🟠 Detalle:** `PATH_BUILDINGS` apunta a `data/barcelona_buildings.geojson`
pero en Vortex los datos van en `~/data3/`. Verificar que el compose mapea
correctamente.

### 1.7 Docker

**✅ Dockerfile** — PyTorch 2.2.0 CUDA 12.1, causal-conv1d + mamba-ssm
instalados con --no-build-isolation. Correcto.

**✅ compose.yml** — 7 perfiles: static, compare, verify, inspect, preprocess,
era5, train. Todos con GPU reservation (1 GPU) y límites de memoria.

**🟠 Detalle:** El perfil `train` usa `CUDA_VISIBLE_DEVICES=0`. Si VLLM está
usando GPU 0, el training iría a GPU 1. Verificar con `nvidia-smi`.

### 1.8 Scripts de datos

**✅ preprocess_urbclim.py** — 682 líneas. 3 paths (A: rasterio warp, B: IDW,
C: latlon_reshape). Path C seleccionado para datos actuales. Con resume.

**✅ build_static_features.py** — 460 líneas. SVF corregido (inter-building gap
geometry). 11 features, hash guardado.

**✅ compute_normalization_stats.py** — Implementado, pendiente ejecutar en Vortex.

**✅ preprocess_era5.py** — Implementado. Crop Barcelona 4×5 grid.

**✅ verify_urbclim.py** — Spot-check raw vs processed.

**✅ compare_v1_v2_preprocessing.py** — Cuantifica distorsión IDW vs reshape.

**✅ inspect_urbclim_raw.py** — Inspección JSON del zarr crudo.

---

## 2. REPO v1 (weather_urban_downscaling/) — Congelado

### 2.1 Estado

**✅ Congelado correctamente.** Último commit 19e531a (May 17).
4 commits sin push: v2 model, path fixes, ERA5 download, operational code.
Sin cambios en checkpoints ni evaluación.

**🟠 AUDITORIA_REPO.md** existe y está actualizado (May 23). Las 5 discrepancias
críticas (D1-D5) están documentadas.

### 2.2 model_benchmark/

Contiene scripts de benchmark de papers (Claude MCP, OpenCode, etc.).
No afecta al paper. OK.

---

## 3. REPO CANÓNICO (urban_downscaling_geoai/)

### 3.1 Estructura

**✅ 18 checkpoints** — 3 seeds × (3 UNet + 3 LSTM + 3 Transformer + 9 Mamba).
Nomenclatura limpia: `{arch}_s{seed}.h5` / `mamba_t{seq}_s{seed}.h5`.

**✅ eval_config.yaml** — Unificado. Todos los experimentos usan Ablation_*.
Exp3 con 9 Mamba (3 seeds × 3 seq). Correcto.

**✅ manifest.csv** — Trazabilidad figura→checkpoint→config. 11 filas.

**✅ load_checkpoint.py** — Verifica carga de los 18 checkpoints.
Mamba usa ModelZoo.build + load_weights (no load_model).

**🟠 paper/figures/ y paper/tables/ vacíos** — Las figuras están en el repo
del paper (geoai_submission/imagenes/), no aquí. OK pero confuso.

### 3.2 run_narrative_eval.py — 34,070 bytes

Script principal de evaluación. No lo leí completo pero existe y es el entry
point documentado en README.

---

## 4. PAPER (geoai_submission/)

### 4.1 Correcciones D1-D5 — Estado

**D1 (param count) — ✅ CORREGIDO**
- Línea 289: "6.81× fewer parameters than Baseline B (0.68M vs 4.61M)"
- Tabla partial_results: columna Params añadida con valores correctos
  (UNet 1.96M, ConvLSTM 4.61M, Transformer 1.07M, Mamba 0.68M)

**D2 (F5 caption vs imagen) — 🟠 PARCIALMENTE CORREGIDO**
- Caption dice "Mamba (T=6)" — correcto
- F5 regenerada el 23 mayo (191KB vs backup 184KB) — tamaño cambió
- **NO verifiqué visualmente** que la imagen corresponda a T=6 y no a seq24
- Backup viejo existe: F5_experiment1_qualitative_backup.pdf

**D3 (F4 vs Tabla 3) — 🟠 PARCIALMENTE CORREGIDO**
- F4 regenerada el 23 mayo (28KB)
- Tabla 3 (tab:exp1) tiene valores: Mamba T6=0.138, ConvLSTM=0.162, UNet=0.213
- **NO verifiqué visualmente** que el gráfico de F4 muestre estos valores
- El texto referencia Tabla 3 y F4 consistentemente

**D4 (doble split) — ✅ CORREGIDO**
- Párrafo "Heatwave held-out evaluation" añadido después del split cronológico
- Documenta que heatwave tiles son held-out de entrenamiento

**D5 (GeoAI framing) — ✅ CORREGIDO**
- Párrafo "Positioning within the GeoAI special collection" en Discusión
- 4 dimensiones mapeadas: dependence, scale, heterogeneity, replicability
- Conexión explícita con el call de la collection

### 4.2 Figuras

13 figuras PDF en imagenes/:
- F1-F3: método, arquitectura, mapa — Mayo 22
- F4: metrics — Mayo 23 (regenerada)
- F5: qualitative — Mayo 23 (regenerada)
- F6-F10: case studies — Mayo 22

### 4.3 Tablas en el paper

- tab:partial_results — Model variants con Params ✅
- tab:exp1 — Heatwave tile evaluation (7 modelos, 3 seeds, 95% CI) ✅
- tab:exp3 — Full-frame evaluation ✅
- tab:seq_ablation_val — Mamba T6/T12/T24 CV ✅
- tab:cs2, tab:robust_05, tab:robust_scaling — Robustness ✅

---

## 5. DISCREPANCIAS ENCONTRADAS

### 🔴 CRÍTICAS

| # | Discrepancia | Archivo | Impacto |
|---|-------------|---------|---------|
| **A1** | Modelos descartados (unet.py, convlstm.py, mamba_unet.py) siguen en src/models/ de v2 | v2/src/models/ | MEDIO — no se importan pero confunden |
| **A2** | F5 no verificada visualmente — no confirmé que la imagen sea T=6 | geoai_submission/imagenes/ | ALTO — paper puede tener figura incorrecta |
| **A3** | F4 no verificada visualmente — no confirmé que los valores coincidan con Tabla 3 | geoai_submission/imagenes/ | ALTO — figura vs tabla inconsistente |

### 🟠 SIGNIFICATIVAS

| # | Discrepancia | Archivo | Impacto |
|---|-------------|---------|---------|
| **B1** | evaluate.py no implementado en v2 | v2/scripts/evaluation/ | No hay métricas post-training |
| **B2** | num_workers=4 en DataLoader puede dar OOM en Vortex | v2/scripts/training/train.py | Training puede fallar |
| **B3** | CUDA_VISIBLE_DEVICES=0 en compose puede colisionar con VLLM | v2/docker/compose.yml | Training en GPU equivocada |
| **B4** | paper/figures/ y paper/tables/ vacíos en repo canónico | urban_downscaling_geoai/paper/ | Confuso pero no crítico |
| **B5** | 4 commits sin push en v1 | weather_urban_downscaling/ | No afecta pero debería push |

### ✅ RESUELTAS (por Claude o por ti)

| # | Corrección | Estado |
|---|-----------|--------|
| C1 | D1: Param count corregido en texto + tabla | ✅ |
| C2 | D4: Doble split documentado | ✅ |
| C3 | D5: GeoAI framing en Discusión | ✅ |
| C4 | eval_config.yaml unificado (Ablation_* para todo) | ✅ |
| C5 | Checkpoints renombrados a nomenclatura limpia | ✅ |
| C6 | manifest.csv creado | ✅ |
| C7 | load_checkpoint.py implementado | ✅ |
| C8 | F4 regenerada (23 mayo) | 🟠 pendiente verificación |
| C9 | F5 regenerada (23 mayo) | 🟠 pendiente verificación |

---

## 6. PLAN DE ACCIÓN

### Inmediato (antes de training)

1. **Eliminar modelos descartados** de v2/src/models/ (unet.py, convlstm.py, mamba_unet.py)
2. **Verificar F4 visualmente** — abrir el PDF y confirmar que los valores de
   las barras coinciden con Tabla 3 (Mamba 0.138, ConvLSTM 0.162, UNet 0.213)
3. **Verificar F5 visualmente** — confirmar que la imagen es T=6, no seq24
4. **Conectar VPN a Vortex** y verificar:
   - `~/data3/era5land_2008-2017.zarr` existe
   - `~/data3/urbclim_2008-2017.zarr` existe
   - `~/data3/static_features.zarr` existe
   - `~/data3/normalization_stats.npz` existe
5. **Ejecutar compute_normalization_stats.py** en Vortex si no existe
6. **Verificar GPU libre** — `nvidia-smi` para confirmar que GPU 1 está libre
   (VLLM usa GPU 0)

### Pre-training

7. **Implementar evaluate.py** en v2 — métricas post-training (RMSE, MAE, SSIM,
   bias maps)
8. **Ajustar compose train** — CUDA_VISIBLE_DEVICES=1 si VLLM usa GPU 0
9. **Reducir num_workers** a 2 si OOM

### Training

10. Lanzar runs en orden:
    - ARCH=unet SEQ_LEN=6 SEED=42
    - ARCH=convlstm SEQ_LEN=6 SEED=42
    - ARCH=mamba SEQ_LEN=6 SEED=42
    - ARCH=mamba SEQ_LEN=12 SEED=42
    - ARCH=mamba SEQ_LEN=24 SEED=42
    - Repetir con SEED=43,44

### Post-training

11. Regenerar figuras del paper con checkpoints PyTorch
12. Actualizar Tabla 3, Tabla Exp3, robustness tables
13. Verificar consistencia figura↔tabla↔checkpoint

---

## 7. RESUMEN EJECUTIVO

**Lo que Claude hizo bien:**
- Repo v2 completo y funcional (modelos, dataset, train, loss, config, docker)
- Correcciones D1, D4, D5 del paper aplicadas
- eval_config.yaml unificado
- Checkpoints canónicos con nomenclatura limpia
- manifest.csv de trazabilidad

**Lo que falta verificar:**
- F4 y F5 visualmente (no sé si los valores/gráficos son correctos)
- evaluate.py no implementado
- Modelos descartados en disco (ruido, no riesgo real)

**Lo que bloquea el training:**
- Datos en Vortex no verificados localmente
- VPN necesaria
- normalization_stats.npz puede no existir en Vortex

**Veredicto: NO se puede entrenar todavía.** Faltan ~5 pasos de verificación.
Estiman ~30-60 min de trabajo antes de lanzar el primer run.

---

*Auditoría generada por phdresearch. 2026-05-25.*
