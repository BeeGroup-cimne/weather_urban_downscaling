# Experimento 2: validación externa en estaciones

- Modelos evaluados: mamba unet
- Métrica de comparación externa: estaciones reales vs modelo y UrbClim

## Ranking por segmento (menor MAE es mejor)

| segment | model | N | MAE_model | RMSE_model | Corr_model | ΔMAE vs UrbClim | ΔRMSE vs UrbClim | ΔCorr vs UrbClim |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| all | mamba | 7150 | 2.744926 | 3.369974 | 0.787363 | -1.109680 | -1.275061 | -0.043465 |
| all | unet | 7150 | 6.652191 | 7.534607 | -0.046538 | -5.016946 | -5.439694 | -0.877367 |
| day | mamba | 3600 | 2.322650 | 2.853879 | 0.774929 | -0.825216 | -0.916369 | -0.044750 |
| day | unet | 3600 | 5.068198 | 5.918310 | -0.065451 | -3.570764 | -3.980799 | -0.885130 |
| night | mamba | 3550 | 3.173149 | 3.822835 | 0.673967 | -1.398151 | -1.579551 | -0.057791 |
| night | unet | 3550 | 8.258494 | 8.878090 | -0.042169 | -6.483496 | -6.634807 | -0.773927 |
