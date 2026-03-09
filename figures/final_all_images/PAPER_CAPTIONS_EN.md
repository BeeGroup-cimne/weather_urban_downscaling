# Final Figure Captions (English, Copy-Paste Ready)

Date: 2026-03-09  
Project: Urban microclimate downscaling (frozen-training, post-training evaluation)

## Main Manuscript

**Figure 1. End-to-end downscaling workflow.**  
Schematic overview of the post-training pipeline used in this study. Dynamic ERA5-Land predictors and static GIS covariates are combined to reconstruct high-resolution urban temperature fields, followed by deterministic evaluation for spatial skill, station validation, ablation, and case studies.

**Figure 2. Study area and observational context.**  
Geographic domain used for high-resolution downscaling and evaluation, including the station-based observational context used for point-wise validation analyses.

**Figure 3. Experiment 1: domain-wide spatial performance.**  
Comparison of MAE, RMSE, and SSIM at the spatial-field level. Hybrid-Mamba ranks first overall (RMSE 0.1356, MAE 0.1086, SSIM 0.8097), indicating the strongest domain-scale reconstruction quality among tested models.

**Figure 4. Experiment 1: qualitative contrast and correction field (Hybrid-Mamba vs bilinear baseline).**  
Visual comparison of reconstructed temperature fields and spatial correction patterns. The correction map (Hybrid-Mamba minus bilinear) highlights structured urban-scale adjustments that are not captured by interpolation alone.

**Figure 5. Experiment 2: station validation by temporal segment.**  
Point-based RMSE against in-situ stations for all/day/night subsets. Interpolation baselines rank best on point-wise metrics (all-segment RMSE: bilinear 1.6642, nearest 1.6934), whereas learned models perform worse at station points.

**Figure 6. Experiment 2: station scatter diagnostics (Hybrid-Mamba vs bilinear).**  
Observed-versus-predicted station scatter illustrating calibration and dispersion behavior. The comparison emphasizes the field-vs-point tension: stronger spatial skill of learned models does not directly translate into superior station-point error.

**Figure 7. Experiment 3: temporal bottleneck ablation (Mamba T=6 vs T=12).**  
Full-frame ablation of temporal context length. The longer context (T=12) outperforms T=6 (RMSE 1.6569 vs 2.0572; MAE 1.3951 vs 1.6057; SSIM 0.7960 vs 0.6674), supporting the value of extended temporal memory.

**Figure 8. Case Study 1: extreme heatwave reconstruction.**  
Heatwave-focused qualitative and quantitative analysis across representative day/night conditions. Results are consistent with Experiment 1, with Hybrid-Mamba achieving the best case-study aggregate performance (RMSE 0.1815, MAE 0.1479, SSIM 0.8124).

**Figure 9. Case Study 2A: nocturnal cooling and persistence hotspots.**  
Legacy physically oriented analysis of low nocturnal heat dissipation and persistent hotspot regions. This case study addresses thermal persistence behavior and is reported separately from Monte Carlo robustness.

**Figure 10. Case Study 2B: Monte Carlo input robustness.**  
Sensitivity of predictions under controlled perturbations of dynamic inputs. Stability ranking by mean absolute deviation from clean predictions is led by ConvLSTM, followed by Hybrid-Mamba (LSTM 0.1274 C, Mamba 0.1465 C, Transformer 0.1573 C, UNet 0.1571 C).

## Supplementary Material

**Figure S1. Case Study 2A persistence detail.**  
Additional visualization of persistence structure and non-dissipative nighttime behavior, complementing Figure 9.

**Figure S2. Experiment 2 station MAE heatmap.**  
Station-by-station error heterogeneity across models, useful for diagnosing local performance variability and site-dependent bias.

**Figure S3. Experiment 1 day/night multi-model grid.**  
Side-by-side qualitative comparison across models for matched daytime and nighttime events.

**Figure S4. Case Study 1 aggregate performance ranking.**  
Quantitative model ranking under heatwave-focused evaluation.

**Figure S5. Legacy spatial comparison panel.**  
Historical qualitative panel retained for reproducibility and context with earlier analysis snapshots.

**Figure S6. Legacy Mamba memory-efficiency ablation.**  
Supplementary view of memory/performance trade-offs associated with temporal modeling choices.

**Figure S7. Station time-series detail.**  
Extended station-wise temporal traces for deeper inspection of diurnal behavior and residual structure.

**Figure S8. Dynamic side-by-side heatwave comparison (GIF).**  
Animated comparison of top-performing models during heatwave evolution.

**Figure S9. Dynamic Hybrid-Mamba heatwave evolution (GIF).**  
Animated field evolution showing temporal continuity and error structure under extreme conditions.

**Figure S10. Legacy training/validation curves.**  
Optimization-history context retained as supplementary material.

## Scope Consistency Note

- Use **CS2A** exclusively for cooling/persistence narrative (Figures 9, S1).  
- Use **CS2B** exclusively for Monte Carlo robustness narrative (Figure 10).  
- Do not merge CS2A and CS2B conclusions into a single unlabeled case-study statement.
