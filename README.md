# Weather Urban Downscaling

**Physics-Aware Spatiotemporal Downscaling of Urban Microclimates using Vision Mamba (U-Net + SSM)**

Reproducing results for the paper. Benchmarks Vision Mamba (Selective State Space Models) against ConvLSTM, Transformer, and static U-Net baselines for 100-m temperature field reconstruction over the Barcelona Metropolitan Area from ERA5-Land forcings.

**Key results** (Experiment 3, Full-Frame, Mamba T=12h): MAE **0.513 °C** · RMSE **0.677 °C** · SSIM **0.848** — a **50% spatial error reduction** vs. static U-Net baseline.

---

## Current Status (Consolidated)

This repository now includes **3 publishable experiments** runnable through single entry scripts:

1. **Experiment 1 (tiles + heatwave, multi-seed, multi-model, with baselines)**  
   Main script: `scripts/ablation/run_ablation_tiles_heatwave_caffeinate.sh`
2. **Experiment 2 (external validation against real weather stations)**  
   Main script: `scripts/evaluation/run_stations_eval_ablation.sh`
3. **Experiment 3 (full-frame top-model replication + ranking stability vs Exp. 1)**  
   Main script: `scripts/ablation/run_experiment3_fullframe_replica.sh`

Server-ready wrappers and bundling tools are also included:
- `scripts/ablation/run_ablation_tiles_heatwave_server.sh`
- `scripts/tools/make_server_bundle.sh`
- `scripts/tools/make_server_bundle.py`

---

## Current Model Zoo

Available models in `src/models_legacy.py`:

- `unet` (DL baseline)
- `lstm` (`convlstm` in inference/evaluation)
- `transformer`
- `mamba`
- `baseline_nearest` (LR→HR upsampling without training)
- `baseline_bilinear` (LR→HR upsampling without training)

The two baseline models are minimum controls used to quantify the added value of learned models.

---

## Available Experiments

| ID | Script | Goal | Models |
|---|---|---|---|
| E1 | `scripts/ablation/run_ablation_tiles_heatwave_caffeinate.sh` | Main publishable experiment (tiles, heat events, seeds, bootstrap CIs) | unet, lstm, transformer, mamba + baselines |
| E2 | `scripts/evaluation/run_stations_eval_ablation.sh` | External validation with real stations (day/night, heatwave/non-heatwave) | unet, lstm, transformer, mamba |
| E3 | `scripts/ablation/run_experiment3_fullframe_replica.sh` | Full-frame replication of top models and ranking stability vs E1 | default: transformer, mamba |
| A1 | `scripts/ablation/run_ablation.py` | Classic full-frame ablation (no tiles) | unet, lstm, transformer, mamba |
| A2 | `scripts/ablation/run_ablation_tiles.py` | Configurable tile-based ablation (fast/iterative) | unet, lstm, transformer, mamba |
| I1 | `scripts/inference/run_inference_tiles_fullframe.py` | Full-frame reconstruction from tile-trained models + comparison maps | all + baselines |
| V1 | `scripts/evaluation/evaluate_test_set.py` | MAE/RMSE/SSIM evaluation on train/val/test splits | unet, convlstm, transformer, mamba |
| S1 | `scripts/overfit_sanity.py` / `scripts/overfit_fixed_tile.py` | Training sanity checks | multiple |
| H1 | `scripts/derive_aemet_heatwaves.py` | Build heatwave event timestamps from stations | n/a |

---

## Quick Installation

### Requirements
- Python 3.10+
- TensorFlow 2.13+ (Mac Silicon / Linux CUDA)
- Geospatial dependencies from `requirements_*.txt`

### Local setup (example)

```bash
git clone https://github.com/your-org/weather_urban_downscaling.git
cd weather_urban_downscaling

python -m venv .venv
source .venv/bin/activate
pip install -r requirements_mac.txt
```

### Linux/CUDA

```bash
pip install -r requirements_tf.txt
```

---

## Required Data

Expected structure:

```text
data/
├── processed/
│   ├── estaciones_interpoladas_final.nc
│   ├── weather_static_FINAL_stations.zarr/
│   ├── stations_t2m.grib                  # optional for weighted_station
│   └── era5land/lr_2017.grib
└── raw/
    └── weather_stations.zarr/             # used by external validation / heatwaves
```

Auto-generated derived caches:
- `data/processed/weather_cache.zarr`
- `data/processed/static_processed.npy`
- `data/processed/stats_config.npz`

---

## Recommended Paper Execution Flow

### 1) Experiment 1: tiles + heatwave publish run

```bash
scripts/ablation/run_ablation_tiles_heatwave_caffeinate.sh
```

Linux server (without `caffeinate`):

```bash
scripts/ablation/run_ablation_tiles_heatwave_server.sh
```

Main outputs (`OUTDIR`, default `experiments/heatwaves/publish_run_*`):
- `metrics_raw.csv`
- `metrics_aggregate.csv`
- `metrics_aggregate_ci.csv`
- `metrics_by_model_seed.csv`
- `metrics_delta_vs_baseline_ci.csv`
- `training_summary.csv`
- `report_publish.md`
- `report_experiment1.md`

### 2) Experiment 2: real station validation

Optionally derive AEMET heatwave events first:

```bash
python scripts/derive_aemet_heatwaves.py \
  --zarr data/raw/weather_stations.zarr \
  --target-year 2017 \
  --out-dir experiments/heatwaves/aemet
```

Then evaluate:

```bash
STATIONS_GRIB=data/processed/stations_t2m.grib \
HEATWAVE_TIMES_FILE=experiments/heatwaves/aemet/event_times_2017.txt \
scripts/evaluation/run_stations_eval_ablation.sh
```

Main outputs (`OUTDIR`, default `experiments/stations_eval/ablation_*`):
- `stations_eval_models_summary.csv`
- `stations_eval_per_station_all_models.csv`
- `stations_eval_rank_by_segment.csv`
- `report_experiment2.md`

### 3) Experiment 3: full-frame top-model replication

```bash
EXP1_AGG_CSV=experiments/heatwaves/<run>/metrics_aggregate_ci.csv \
scripts/ablation/run_experiment3_fullframe_replica.sh
```

### 4) Deterministic post-training orchestrator (recommended for final paper runs)

Single entrypoint with explicit config:

```bash
./.venv/bin/python scripts/evaluation/run_narrative_eval.py \
  --config config/eval_config.yaml \
  --stages exp1,exp2,exp3,cs1,cs2
```

Run only Case Study 2B (Monte Carlo robustness):

```bash
./.venv/bin/python scripts/evaluation/run_narrative_eval.py \
  --config config/eval_config.yaml \
  --stages cs2
```

CS2B outputs:
- `experiments/eval_outputs/cs2_robustness_publish/robustness_summary.csv`
- `experiments/eval_outputs/cs2_robustness_publish/cs2_rank_stability.csv`
- `experiments/eval_outputs/cs2_robustness_publish/<model>/robustness_results.csv`
- `experiments/eval_outputs/cs2_robustness_publish/<model>/report_robustness.md`

Dual protocol for Experiment 2 (standard vs station-aligned footprint):

```bash
./.venv/bin/python scripts/evaluation/run_exp2_dual_protocol.py \
  --config config/eval_config.yaml \
  --reuse-existing
```

Build the master report across all experiments/case studies:

```bash
./.venv/bin/python scripts/evaluation/build_master_report.py \
  --config config/eval_config.yaml
```

Main outputs (`OUTDIR`, default `experiments/fullframe/experiment3_*`):
- `fullframe_eval_raw.csv`
- `fullframe_training_summary.csv`
- `fullframe_eval_aggregate.csv`
- `ranking_stability_vs_exp1.csv`
- `report_experiment3.md`

### Case Study 2 Naming (Important)

To avoid narrative ambiguity in the manuscript, use this convention:

- `CS2A` = **Night Cooling & Persistence** (legacy spatial analysis of low nocturnal heat dissipation / persistent hotspots).  
  Reference run:
  - `experiments/fullframe/casestudy2_20260223_022102/report_casestudy2.md`
  - `experiments/fullframe/casestudy2_20260223_022102/cs2_cooling_summary.csv`
- `CS2B` = **Input Robustness (Monte Carlo)** (current deterministic post-training robustness under input perturbations).  
  Reference run:
  - `experiments/eval_outputs/cs2_robustness_publish/robustness_summary.csv`
  - `experiments/eval_outputs/cs2_robustness_publish/cs2_rank_stability.csv`

Current `eval_config.yaml` / narrative pipeline (`--stages ... cs2`) corresponds to **CS2B**.

---

## Run Only One Model (Common Cases)

Only `transformer` in E1:

```bash
MODELS="transformer" scripts/ablation/run_ablation_tiles_heatwave_caffeinate.sh
```

Only `mamba` in E1:

```bash
MODELS="mamba" scripts/ablation/run_ablation_tiles_heatwave_caffeinate.sh
```

With the server wrapper:

```bash
MODELS="mamba" scripts/ablation/run_ablation_tiles_heatwave_server.sh
```

---

## Standardized Visualization (Fair Comparisons)

`scripts/inference/run_inference_tiles_fullframe.py` already supports consistent model-to-model comparison:

- LR is upscaled to HR for display.
- All 3 panels (LR / Pred / HR) are shown at the same visual size.
- Shared color scaling with `--scale-from`:
  - `all` (default)
  - `hr`
  - `hr_lr` (**recommended for model comparisons**)
- LR channel can be set via `--lr-channel` (auto-detects `t2m` when available).

Recommended example for comparable figures:

```bash
python scripts/inference/run_inference_tiles_fullframe.py \
  --model-type transformer \
  --model-path experiments/models/Tiles_TRANSFORMER_S42_best.h5 \
  --time 2017-08-15T15:00:00 \
  --patch-size 96 \
  --stride 48 \
  --lr-resample nearest \
  --scale-from hr_lr \
  --experiment-name "Tiles_TRANSFORMER_S42" \
  --out experiments/figures/tiles_publish.png
```

---

## Reproducibility (Paper-Ready)

The consolidated flow enforces:

- explicit temporal splits (`train/val/test`) with no leakage
- multi-seed training (defaults `42 43 44` in E1 and E3)
- bootstrap CI summaries (`consolidate_experiment1.py`, `consolidate_experiment3.py`)
- comparison against non-trained baselines
- external validation with real stations and segmentation (day/night, heatwave/non-heatwave)

### CI & Repro Manifest

Minimal CI workflow: `.github/workflows/ci-eval.yml`

What it checks:
- Python compile check of evaluation orchestration scripts
- deterministic smoke test (`run_narrative_eval.py --stages ""`)
- reproducibility manifest generation

Generate manifest locally:

```bash
./.venv/bin/python scripts/evaluation/build_repro_manifest.py \
  --config config/eval_config.yaml \
  --out experiments/eval_outputs/repro_manifest.json
```

Strict mode (fail if required stage outputs are missing):

```bash
./.venv/bin/python scripts/evaluation/build_repro_manifest.py \
  --config config/eval_config.yaml \
  --out experiments/eval_outputs/repro_manifest.json \
  --strict
```

Publication gate (required artifacts + basic sanity checks):

```bash
./.venv/bin/python scripts/evaluation/validate_publication_artifacts.py \
  --config config/eval_config.yaml \
  --out experiments/eval_outputs/publication_gate_report.json
```

Build deterministic publication bundle:

```bash
./.venv/bin/python scripts/evaluation/build_publication_bundle.py \
  --config config/eval_config.yaml \
  --out dist/publication_eval_bundle.tar.gz
```

---

## Server Support

### Build a server bundle

```bash
scripts/tools/make_server_bundle.sh dist/weather_urban_downscaling_server_bundle.tar.gz
```

This includes scripts/config and excludes heavy outputs (models/figures/datasets).

### Useful environment variables

- `PYTHON_BIN` (explicit Python path on server)
- `USE_GPU_CONFIG=1` (enable server config)
- `OUTDIR=...` (reproducible output path)
- `MODELS="..."`, `SEEDS="..."` (experiment sweep control)

---

## Quick Troubleshooting

- If input data changed and cache is stale, remove `data/processed/weather_cache.zarr`.
- If packages are missing (for example `numpy`), validate the active environment first.
- On Mac, use the `caffeinate` wrapper for long runs.

---

## License

MIT. See `LICENSE`.
