# Tabla Comparativa de Features: Zheng et al. (2025) vs Pipeline de Kerin Cardona

---

## Resumen de Arquitecturas

| Aspecto | Zheng et al. (2025) | Kerin Cardona (Downscaling Urbano) |
|---------|---------------------|------------------------------------|
| **Objetivo** | LST downscaling (100m → 30m) con deep learning + view factors | Downscaling de T2m ERA5-Land (~9km) a UrbClim (~1km) con Mamba/UNet |
| **Modelo** | DeepLabV3 + regresión con características geoespaciales | UNet / Mamba (SSM) + ConvLSTM + Transformer (múltiples arquitecturas) |
| **Input dinámico** | No tiene (solo features estáticas + LST observado) | Sí: 9 canales atmosféricos de ERA5-Land (u10, v10, d2m, t2m, lai_hv, lai_lv, tp, ssrd, fal) |
| **Escala LR** | 100m (Landsat) | ~9km (ERA5-Land, grid 4×3) |
| **Escala HR** | 30m (Landsat B7) | ~1km (UrbClim, grid 251×251) |
| **Target** | LST (Land Surface Temperature) | T2m (2m air temperature) |

---

## Tabla Comparativa de Features

| Feature | Usado por Zheng | Usado por Kerin | Fuente de Datos | Solapamiento | Notas de Integración Posible |
|---------|:---------------:|:---------------:|-----------------|:------------:|------------------------------|
| **LULC / Land Use** | ✅ (Landsat 8/9, DeepLabV3) | ✅ (vía `residential_index`, `industrial_index`, `services_index`, `leisure_index`) | Zheng: Landsat 8/9 + DeepLabV3. Kerin: building footprints (OSM/Barcelona GeoJSON) combinados en índices compuestos. | **Medio** | Los índices de Kerin (residential, industrial, services, leisure) son proxies de uso de suelo, pero no equivalen a un LULC categórico. Se podría integrar LULC de DeepLabV3 directamente como canal estático adicional. |
| **Elevation** | ✅ (ASTER GDEM) | ✅ (`elevation`) | Zheng: ASTER GDEM (30m). Kerin: DEM genérico en el dataset estático. | **Alto** | Misma variable, escalas diferentes. La elevación de Kerin es a ~1km (UrbClim), la de Zheng es 30m downscaled a 30m. Integración directa si se unifica la fuente a ASTER GDEM global. |
| **Slope** | ✅ (derivado de DEM) | ❌ No usado explícitamente | Zheng: derivado de ASTER GDEM. Kerin: no lo computa. | **None** | Pendiente podría añadirse como feature estático derivado de elevación. En ciudad llana (Barcelona) tiene baja relevancia; en topografía compleja sería útil. |
| **SVF (Sky View Factor)** | ✅ (GIS, DeepLabV3) | ✅ (`svf`, `svf_canyon`) | Zheng: extraído de clasificación LULC + geometría 3D. Kerin: modelo de cañón urbano (`svf_canyon = cos(arctan(2*h/w))`) con blending open-terrain. | **Alto** | Mismo concepto, diferente método. Kerin usa un modelo analítico continuo (canyon + open blend), Zheng probablemente usa método más directo (ray-tracing o proporciones desde LULC). Se podrían ensamblar ambos enfoques. |
| **TVF (Tree View Factor)** | ✅ (DeepLabV3 + GIS) | ❌ No usado explícitamente | Zheng: extraído de segmentación semántica de árboles (DeepLabV3). Kerin: solo `ndvi_mean` y `ndvi_min` como proxies de vegetación. | **Bajo** | TVF es más específico que NDVI. Kerin podría incorporarlo derivándolo de fracción de copa arbórea (tree canopy fraction) desde LULC o desde datos LiDAR/OSM si están disponibles. |
| **BVF (Building View Factor)** | ✅ (DeepLabV3 + GIS) | ❌ No usado explícitamente | Zheng: extraído de segmentación de edificios. Kerin: lo aproxima mediante `building_density`, `avg_height`, `h_over_w`. | **Medio** | BVF es un view factor específico de edificios; Kerin usa `building_density` + `h_over_w` como aproximación. Integración: añadir BVF directo si se dispone de clasificación LULC o modelos 3D. |
| **avg_height** | ❌ No usado | ✅ (`avg_height`, `height_index`, `max_levels`) | Kerin: de building footprints GeoJSON (altura = levels × 3m). | **None** | Feature específica de Kerin, derivada de datos catastrales. Zheng no la usa (trabaja con LULC 2D + view factors). |
| **building_density** | ❌ No usado | ✅ (`building_density`) | Kerin: área construida / área del buffer (radio 150m) desde OSM/Barcelona GeoJSON. | **None** | Feature morfológica clave en Kerin, ausente en Zheng. |
| **impervious_fraction** | ❌ No usado | ✅ (`impervious_fraction`) | Kerin: `0.65*density + 0.35*(1-ndvi_mean)`. | **None** | Proxy de superficie impermeable útil para modelos urbanos. |
| **roughness** | ❌ No usado | ✅ (`roughness`) | Kerin: `0.15 * avg_height * (1 - exp(-sqrt(4*density)))`. | **None** | Rugosidad aerodinámica, relevante para downscaling atmosférico pero no para LST. |
| **h_over_w** | ❌ No usado | ✅ (`h_over_w`) | Kerin: avg_height / street_width_m. | **None** | Ratio altura/ancho de calle, relacionado con SVF. |
| **street_width_m** | ❌ No usado | ✅ (`street_width_m`) | Kerin: derivado de building_density y buffer. | **None** | Anchura de calle estimada geométricamente. |
| **NDVI (mean/min)** | ❌ No usado como feature | ✅ (`ndvi_mean`, `ndvi_min`) | Kerin: del dataset estático de estaciones (fuente remota, probablemente MODIS/Landsat). | **None** | Vegetación como feature morfológica. Relevante para refrigeración urbana. |
| **u10, v10 (wind)** | ❌ No usado | ✅ (LR dinámico) | ERA5-Land, ~9km. | **None** | Features atmosféricas dinámicas temporales. No aplican en contexto de Zheng (LST estático). |
| **t2m (temperature)** | ❌ No usado (LST es target) | ✅ (LR dinámico) | ERA5-Land, ~9km. | **None** | Input atmosférico coarse para downscaling. |
| **d2m (dewpoint)** | ❌ No usado | ✅ (LR dinámico) | ERA5-Land, ~9km. | **None** | Humedad como predictor atmosférico. |
| **tp (precipitation)** | ❌ No usado | ✅ (LR dinámico) | ERA5-Land, ~9km. | **None** | Precipitación total (casi siempre 0 en Barcelona en verano para las muestras). |
| **ssrd (solar radiation)** | ❌ No usado | ✅ (LR dinámico) | ERA5-Land, ~9km (convertido de J/m² a Wh/m²). | **None** | Radiación solar, clave para calentamiento diurno. |
| **lai_hv, lai_lv (LAI)** | ❌ No usado | ✅ (LR dinámico, canales neutrales) | ERA5-Land. | **None** | Leaf Area Index. En el conjunto de entrenamiento estos canales son casi constantes (neutral fill en forecast). |
| **fal (albedo)** | ❌ No usado | ✅ (LR dinámico, canal neutral) | ERA5-Land. | **None** | Albedo de pronóstico, también casi constante. |

---

## Solapamiento General

| Categoría | Cantidad de Features |
|-----------|:--------------------:|
| Features con solapamiento **Alto** | 2 (elevation, SVF) |
| Features con solapamiento **Medio** | 2 (LULC/land-use indices, BVF/building density) |
| Features con solapamiento **Bajo** | 1 (TVF/NDVI) |
| Features con solapamiento **None** | 16 (slope, avg_height, building_density, impervious_fraction, roughness, h_over_w, street_width_m, ndvi_mean/min, + 9 canales atmosféricos dinámicos) |

---

## Análisis de Integración Posible

### Features de Zheng que podrían añadirse al pipeline de Kerin:

1. **Slope (pendiente)**: Fácil de añadir. Se deriva del mismo DEM de elevación. Coste computacional mínimo. Beneficio: mejora en downscaling en zonas montañosas (ya que Barcelona tiene el Collserola al norte).

2. **TVF (Tree View Factor)**: Beneficio potencial alto para capturar sombra arbórea y refrigeración. Se puede aproximar desde `ndvi_mean` y `height_index`, pero TVF directo desde segmentación LULC sería más preciso.

3. **BVF (Building View Factor)**: Ya está implícito en `building_density` + `h_over_w` + `svf`. Añadirlo como canal separado podría ayudar, pero el beneficio marginal es bajo dado el solapamiento.

4. **LULC categórico (de DeepLabV3)**: Añadir un mapa de LULC categórico (residencial, industrial, vegetación, agua, etc.) como canal one-hot o embedding sería la integración más impactante. Los índices compuestos de Kerin (residential_index, etc.) ya capturan esta información pero de forma continua. Un LULC discreto aportaría claridad semántica.

### Features de Kerin que podrían incorporarse al enfoque de Zheng:

1. **avg_height + building_density**: Mejoran la estimación de rugosidad urbana y profundidad de cañón, relevantes para LST.

2. **h_over_w + street_width_m**: Relacionados con SVF pero más interpretables geométricamente.

3. **NDVI mean/min**: Proxies de vegetación que complementarían TVF.

### Features dinámicas (LR) de Kerin no aplicables a Zheng:

Los 9 canales atmosféricos (u10, v10, d2m, t2m, lai_hv, lai_lv, tp, ssrd, fal) son específicos del downscaling meteorológico. Zheng trabaja con LST desde Landsat (instante satelital), no con downscaling temporal. Sin embargo, si Zheng extendiera su método a múltiples tiempos, estas variables atmosféricas serían predictores útiles.

---

## Nota sobre la estructura del modelo de Kerin

El pipeline de Kerin usa **dos tipos de features simultáneamente**:

1. **Features estáticas (morfológicas)**: 13 canales en la resolución HR (251×251). Se concatenan con las features dinámicas en el encoder del modelo.

2. **Features dinámicas (atmosféricas)**: 9 canales en resolución LR (4×3 para Barcelona) de ERA5-Land, con secuencia temporal de 6 timesteps.

Ambos se fusionan en la primera capa convolucional del UNet/Mamba:
- `in_channels = in_channels_dyn (9) + in_channels_static (13)`
- Las features estáticas primero se redimensionan (downsample) a la resolución LR para la concatenación.

Esto significa que **cualquier feature estática adicional** (como slope, TVF, BVF, LULC categórico) se puede añadir simplemente incrementando `STATIC_CHANNELS` y agregando el canal correspondiente al dataset estático sin cambiar la arquitectura del modelo.

---

*Fecha del análisis: 18 de mayo de 2026*
*Repositorio analizado: /Users/kerincardona/weather_urban_downscaling*
*Archivos clave revisados: config/config.py, src/data_loader.py, src/utils/feature_engineering_static.py, src/utils/static_data_concatenation.py, src/torch_engine/model_mamba.py, scripts/forecast/fetch_openmeteo_ecmwf.py*
