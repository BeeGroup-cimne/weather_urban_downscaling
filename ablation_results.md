# Results Report: Tile Ablation (Legacy Models)

This report updates the previous summary with the **newly retrained Transformer** and keeps all models in the same comparison table.

## Updated Metrics (from latest tile logs)

Source logs:
- `experiments/logs/Tiles_UNET_log.csv`
- `experiments/logs/Tiles_LSTM_log.csv`
- `experiments/logs/Tiles_TRANSFORMER_log.csv`
- `experiments/logs/Tiles_MAMBA_log.csv`

| Model | Params | Epochs Logged | Best Val Loss | Best Val MAE |
| :--- | ---: | ---: | ---: | ---: |
| **MAMBA** | 676,641 | 35 | **0.044591** (ep 28) | **0.102873** (ep 28) |
| **UNET** | 1,957,985 | 35 | 0.103750 (ep 34) | 0.182687 (ep 29) |
| **LSTM** | 4,611,681 | 34 | 0.117599 (ep 32) | 0.201527 (ep 33) |
| **TRANSFORMER (retrained)** | 1,072,225 | 35 | 0.127285 (ep 30) | 0.209561 (ep 30) |

## Notes

- The Transformer row above is the updated run (`Tiles_TRANSFORMER_log.csv`).
- Ranking by validation loss in this run: **Mamba > UNet > LSTM > Transformer**.
- Ranking by validation MAE in this run: **Mamba > UNet > LSTM > Transformer**.

## Figures Explaining the Update

### 1) Updated training dynamics (all models)
![Updated curves](experiments/figures/tiles_ablation_updated_curves.png)

### 2) Updated best-metric comparison
![Updated best metrics](experiments/figures/tiles_ablation_updated_best_metrics.png)

### 3) Updated Transformer qualitative map (retrained model)
![Retrained transformer map](experiments/figures/tiles_post_train_publish_Tiles_TRANSFORMER_RETRAINED_PUB.png)

### 4) Unified 2x3 comparison panel (best for direct visual comparison)
![Unified 2x3 panel](experiments/figures/tiles_model_comparison_panel_2x3.png)

### 5) Hourly field evolution for the selected day (Transformer vs Ground Truth)
![Hourly evolution GIF](experiments/figures/fig07_hourly_field_evolution_transformer_2017-08-15_retrained.gif)
![Hourly evolution summary](experiments/figures/fig07_hourly_field_evolution_transformer_2017-08-15_retrained_summary.png)

### 6) Hourly field evolution for the top-2 models (Mamba and UNet)
![Mamba hourly evolution](experiments/figures/fig07_hourly_field_evolution_mamba_2017-08-15_best2.gif)
![UNet hourly evolution](experiments/figures/fig07_hourly_field_evolution_unet_2017-08-15_best2.gif)

### 7) Side-by-side hourly animation (Mamba vs UNet vs Ground Truth)
![Top-2 side-by-side hourly panel](experiments/figures/fig08_hourly_top2_side_by_side_mamba_unet_2017-08-15_best2.gif)
![Top-2 hourly MAE summary](experiments/figures/fig08_hourly_top2_side_by_side_mamba_unet_2017-08-15_best2_summary.png)

Quick read:
- Daily mean MAE: **Mamba 0.1083** vs **UNet 0.1375**
- Mamba is better in **22/24** hourly frames for this day
