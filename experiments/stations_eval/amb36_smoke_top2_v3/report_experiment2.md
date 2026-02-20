# Experimento 2: validación externa en estaciones

- Modelos evaluados: mamba unet
- Métrica de comparación externa: estaciones reales vs modelo y UrbClim

## Ranking por segmento (menor MAE es mejor)

| segment | model | N | MAE_model | RMSE_model | Corr_model | ΔMAE vs UrbClim | ΔRMSE vs UrbClim | ΔCorr vs UrbClim |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| all | mamba | 50 | 3.139160 | 3.503408 | 0.468839 | -2.144196 | -2.266813 | -0.242114 |
| all | unet | 50 | 10.202467 | 10.390230 | 0.091958 | -9.207503 | -9.153635 | -0.618994 |
| day | mamba | 20 | 2.893731 | 3.119365 | 0.521023 | -2.050004 | -2.026869 | -0.161847 |
| day | unet | 20 | 9.423539 | 9.551383 | 0.168858 | -8.579812 | -8.458887 | -0.514012 |
| night | mamba | 30 | 3.302780 | 3.737578 | 0.325897 | -2.206992 | -2.413602 | -0.320702 |
| night | unet | 30 | 10.721751 | 10.913700 | 0.056558 | -9.625962 | -9.589724 | -0.590041 |
