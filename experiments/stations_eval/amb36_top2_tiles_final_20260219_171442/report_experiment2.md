# Experimento 2: validación externa en estaciones

- Modelos evaluados: mamba unet
- Métrica de comparación externa: estaciones reales vs modelo y UrbClim

## Ranking por segmento (menor MAE es mejor)

| segment | model | N | MAE_model | RMSE_model | Corr_model | ΔMAE vs UrbClim | ΔRMSE vs UrbClim | ΔCorr vs UrbClim |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| all | mamba | 7150 | 3.829585 | 5.120866 | 0.386311 | -2.194340 | -3.025953 | -0.444518 |
| all | unet | 7150 | 7.253335 | 8.292755 | 0.253621 | -5.618090 | -6.197841 | -0.577207 |
| day | mamba | 3600 | 3.127986 | 4.256812 | 0.407891 | -1.630552 | -2.319301 | -0.411788 |
| day | unet | 3600 | 6.300788 | 7.240624 | 0.263361 | -4.803355 | -5.303114 | -0.556318 |
| night | mamba | 3550 | 4.541066 | 5.868578 | 0.232039 | -2.766068 | -3.625295 | -0.499719 |
| night | unet | 3550 | 8.219296 | 9.238134 | 0.064806 | -6.444298 | -6.994851 | -0.666951 |
