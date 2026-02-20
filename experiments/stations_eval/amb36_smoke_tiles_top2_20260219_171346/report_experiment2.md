# Experimento 2: validación externa en estaciones

- Modelos evaluados: mamba unet
- Métrica de comparación externa: estaciones reales vs modelo y UrbClim

## Ranking por segmento (menor MAE es mejor)

| segment | model | N | MAE_model | RMSE_model | Corr_model | ΔMAE vs UrbClim | ΔRMSE vs UrbClim | ΔCorr vs UrbClim |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| all | mamba | 50 | 4.848984 | 6.314786 | 0.021003 | -3.854020 | -5.078192 | -0.689949 |
| all | unet | 50 | 9.326942 | 9.942327 | -0.073248 | -8.331978 | -8.705732 | -0.784200 |
| day | mamba | 20 | 4.461661 | 5.615917 | 0.247348 | -3.617934 | -4.523421 | -0.435522 |
| day | unet | 20 | 8.789531 | 9.260207 | 0.150170 | -7.945804 | -8.167711 | -0.532700 |
| night | mamba | 30 | 5.107200 | 6.740563 | -0.138927 | -4.011411 | -5.416587 | -0.785527 |
| night | unet | 30 | 9.685214 | 10.372181 | -0.243630 | -8.589426 | -9.048205 | -0.890229 |
