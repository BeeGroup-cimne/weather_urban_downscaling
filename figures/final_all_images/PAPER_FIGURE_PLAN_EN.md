# Paper Figure Order and Captions (English, Publication Draft)

Date: 2026-03-09  
Scope: final narrative using consolidated assets from `figures/final_all_images/all_sources/`  
Rules: frozen training, post-training evaluation only, CS2A and CS2B explicitly separated.

## Main Manuscript (Recommended Order)

| ID | File (consolidated) | Draft title | Draft caption (English) |
|---|---|---|---|
| Fig. 1 | `all_sources/imagenes__imagenes__F1_method_overview.png` | End-to-end downscaling workflow | Overview of the post-training pipeline used in this study, from low-resolution ERA5-Land drivers and static GIS predictors to high-resolution urban temperature reconstruction and evaluation outputs. The workflow is fully frozen at model weights and focuses on deterministic inference, evaluation, and reporting. |
| Fig. 2 | `all_sources/imagenes__imagenes__F3_study_area_map.pdf` | Study area and station context | Geographic context of the Barcelona-domain urban grid used for downscaling and evaluation. The map situates the analysis region and supports interpretation of both domain-wide metrics and point-based station validation. |
| Fig. 3 | `all_sources/paper_final__experiments__eval_outputs__paper_figures_final__fig_exp1_spatial_performance.pdf` | Experiment 1: domain-wide spatial performance | Domain-level metrics (MAE, RMSE, SSIM) for the main models and interpolation baselines. Hybrid-Mamba ranks first in this setting (RMSE 0.1356, MAE 0.1086, SSIM 0.8097), followed by ConvLSTM and UNet variants. |
| Fig. 4 | `all_sources/repro_v2__figures__repro_v2__fig_v2_exp1_mamba_vs_bilinear_delta.pdf` | Experiment 1: qualitative correction field | Spatial comparison between Hybrid-Mamba and bilinear interpolation, including the correction field (Mamba minus bilinear). This figure highlights where learned downscaling introduces structured urban corrections beyond interpolation. |
| Fig. 5 | `all_sources/paper_final__experiments__eval_outputs__paper_figures_final__fig_exp2_station_rmse_segments.pdf` | Experiment 2: station RMSE by segment | Point-based validation against station observations for all/day/night segments. Baseline interpolation remains strongest in point metrics (e.g., all-segment RMSE: bilinear 1.6642, nearest 1.6934), while learned models rank lower. |
| Fig. 6 | `all_sources/paper_final__experiments__eval_outputs__paper_figures_final__fig_exp2_scatter_mamba_vs_bilinear.pdf` | Experiment 2: station scatter diagnostics | Observation-vs-prediction scatter comparison at station level (Hybrid-Mamba vs bilinear baseline). The plot shows the pointwise generalization gap that coexists with stronger field-level spatial performance of learned models. |
| Fig. 7 | `all_sources/paper_final__experiments__eval_outputs__paper_figures_final__fig_exp3_bottleneck_ablation.pdf` | Experiment 3: bottleneck temporal ablation | Full-frame bottleneck ablation comparing Mamba sequence lengths. `T=12` outperforms `T=6` (RMSE 1.6569 vs 2.0572; MAE 1.3951 vs 1.6057; SSIM 0.7960 vs 0.6674), supporting longer temporal context. |
| Fig. 8 | `all_sources/repro_v2__figures__repro_v2__fig_v2_cs1_heatwave_storyboard.pdf` | Case Study 1: extreme heatwave reconstruction | Qualitative and quantitative storyboard for a representative heatwave episode (day and night behavior). This case aligns with Exp1: Hybrid-Mamba is best in heatwave-focused evaluation (RMSE 0.1815; MAE 0.1479; SSIM 0.8124). |
| Fig. 9 | `all_sources/imagenes__imagenes__F7_casestudy2_cooling_maps.pdf` | Case Study 2A: nocturnal cooling and persistence | Legacy case-study view of areas with weak nocturnal heat dissipation and persistent hotspots. This analysis is complementary to robustness and should be discussed as CS2A (physical persistence narrative), not as Monte Carlo robustness. |
| Fig. 10 | `all_sources/paper_final__experiments__eval_outputs__paper_figures_final__fig_cs2_montecarlo_robustness.pdf` | Case Study 2B: Monte Carlo input robustness | Input-perturbation robustness under increasing noise amplitudes. Stability ranking by mean deviation vs clean predictions is led by ConvLSTM, then Mamba (LSTM 0.1274 C; Mamba 0.1465 C; Transformer 0.1573 C; UNet 0.1571 C). |

## Supplementary Figures (Recommended)

| ID | File (consolidated) | Suggested role |
|---|---|---|
| Fig. S1 | `all_sources/imagenes__imagenes__F6_casestudy2_persistence.png` | Additional CS2A persistence visualization (hotspot persistence emphasis). |
| Fig. S2 | `all_sources/repro_v2__figures__repro_v2__fig_v2_exp2_station_mae_heatmap.pdf` | Station-by-station Exp2 error heterogeneity. |
| Fig. S3 | `all_sources/repro_v2__figures__repro_v2__fig_v2_exp1_day_night_model_grid.pdf` | Multi-model day/night qualitative grid for Exp1. |
| Fig. S4 | `all_sources/paper_final__experiments__eval_outputs__paper_figures_final__fig_cs1_heatwave_performance.pdf` | Aggregate quantitative ranking for CS1. |
| Fig. S5 | `all_sources/legacy_pdf__figures__pdf__fig11_mamba_spatial_comparison.pdf` | Legacy spatial comparison panel (historical context). |
| Fig. S6 | `all_sources/legacy_pdf__figures__pdf__fig10_mamba_memory_ablation.pdf` | Memory/efficiency-oriented Mamba ablation visualization. |
| Fig. S7 | `all_sources/legacy_pdf__figures__pdf__fig05_timeseries_stations_real_improved.pdf` | Detailed station time-series behavior. |
| Fig. S8 | `all_sources/presentation__experiments__presentation_figures__fig08_hourly_top2_side_by_side_mamba_unet_2017-08-15_PUB.gif` | Dynamic qualitative comparison during heatwave evolution. |
| Fig. S9 | `all_sources/presentation__experiments__presentation_figures__fig07_hourly_field_evolution_mamba_2017-08-15_PUB.gif` | Dynamic Mamba field evolution under extreme conditions. |
| Fig. S10 | `all_sources/legacy_pdf__figures__pdf__fig09_train_val_curves.pdf` | Legacy optimization behavior and convergence context. |

## Caption Notes for Writing Consistency

1. Use **CS2A** only for cooling/persistence figures (`F6`, `F7` style assets).
2. Use **CS2B** only for Monte Carlo robustness figures (`fig_cs2_montecarlo_robustness`, `fig_v2_cs2_robustness_curve`).
3. Keep Exp2 interpretation explicit: interpolation baselines dominate point metrics despite weaker domain-scale spatial fidelity.
4. For Exp3 captions, acknowledge that current aggregate uses `n=1` per model in the publish run.

## Data Sources for Reported Numbers

- `experiments/eval_outputs/exp1_spatial_publish/metrics_aggregate.csv`
- `experiments/eval_outputs/exp2_groundtruth_publish/stations_eval_rank_by_segment.csv`
- `experiments/eval_outputs/exp3_bottleneck_publish/fullframe_eval_aggregate.csv`
- `experiments/eval_outputs/cs1_heatwave_publish/metrics_aggregate.csv`
- `experiments/eval_outputs/cs2_robustness_publish/robustness_summary.csv`
