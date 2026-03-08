# Server README — Consolidated Experiments (Paper-Ready)

Operational guide to run the 3 consolidated paper experiments on server infrastructure.

---

## 1) Scope

Run reproducibly:

1. **E1**: `tiles + heatwave` ablation, multi-model/multi-seed + baselines + bootstrap CI  
   `scripts/ablation/run_ablation_tiles_heatwave_server.sh`
2. **E2**: External validation against real stations (segmented)  
   `scripts/evaluation/run_stations_eval_ablation.sh`
3. **E3**: `full-frame` top-model replication + ranking stability vs E1  
   `scripts/ablation/run_experiment3_fullframe_replica.sh`

---

## 2) Server prerequisites

- Linux + NVIDIA GPU (recommended H100/H200/A100 class)
- Python 3.10+ with dependencies from `requirements_tf.txt`
- Dataset in `data/` (see structure in `README.md`)
- Enough free storage for multiple `experiments/` runs

Common environment variables:

- `PYTHON_BIN` (optional if auto-detection is not enough)
- `USE_GPU_CONFIG=1` (recommended on server)
- `OUTDIR=...` (recommended for traceability)

---

## 3) Recommended server execution order

### Step A — Experiment 1 (primary)

```bash
USE_GPU_CONFIG=1 \
OUTDIR=experiments/heatwaves/publish_run_$(date +%Y%m%d_%H%M%S) \
scripts/ablation/run_ablation_tiles_heatwave_server.sh
```

Defaults:
- Models: `unet lstm transformer mamba`
- Seeds: `42 43 44`
- Baselines: nearest + bilinear

Key artifacts in `OUTDIR`:
- `metrics_raw.csv`
- `metrics_aggregate.csv`
- `metrics_aggregate_ci.csv`
- `metrics_by_model_seed.csv`
- `metrics_delta_vs_baseline_ci.csv`
- `training_summary.csv`
- `report_publish.md`
- `report_experiment1.md`

### Step B — Experiment 2 (real station validation)

Optional: derive heatwave timestamps from AEMET stations:

```bash
python scripts/derive_aemet_heatwaves.py \
  --zarr data/raw/weather_stations.zarr \
  --target-year 2017 \
  --out-dir experiments/heatwaves/aemet
```

Evaluation:

```bash
USE_GPU_CONFIG=1 \
STATIONS_GRIB=data/processed/stations_t2m.grib \
HEATWAVE_TIMES_FILE=experiments/heatwaves/aemet/event_times_2017.txt \
OUTDIR=experiments/stations_eval/ablation_$(date +%Y%m%d_%H%M%S) \
scripts/evaluation/run_stations_eval_ablation.sh
```

Key artifacts:
- `stations_eval_models_summary.csv`
- `stations_eval_per_station_all_models.csv`
- `stations_eval_rank_by_segment.csv`
- `report_experiment2.md`

### Step C — Experiment 3 (full-frame top-model replication)

```bash
USE_GPU_CONFIG=1 \
MODELS="transformer mamba" \
SEEDS="42 43 44" \
EXP1_AGG_CSV=experiments/heatwaves/<run_e1>/metrics_aggregate_ci.csv \
OUTDIR=experiments/fullframe/experiment3_$(date +%Y%m%d_%H%M%S) \
scripts/ablation/run_experiment3_fullframe_replica.sh
```

Key artifacts:
- `fullframe_eval_raw.csv`
- `fullframe_training_summary.csv`
- `fullframe_eval_aggregate.csv`
- `ranking_stability_vs_exp1.csv`
- `report_experiment3.md`

---

## 4) Run only one model (server)

Only `transformer`:

```bash
MODELS="transformer" scripts/ablation/run_ablation_tiles_heatwave_server.sh
```

Only `mamba`:

```bash
MODELS="mamba" scripts/ablation/run_ablation_tiles_heatwave_server.sh
```

> Important: if `MODELS` is not set, the wrapper runs all models (`unet lstm transformer mamba`).

---

## 5) Comparable visualization across models

For fair map comparison, use:
- `--scale-from hr_lr`
- `--lr-resample nearest`
- same `time`, `patch-size`, and `stride`

Example:

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

## 6) Monitoring and recovery

- Training logs per model/seed: `experiments/logs/`
- Checkpoints: `experiments/models/`
- Consolidated summaries: experiment `OUTDIR` folders

If a run is interrupted:
- relaunch the same script with the same `OUTDIR`
- keep existing artifacts for traceability

---

## 7) Bundle for another server

```bash
scripts/tools/make_server_bundle.sh dist/weather_urban_downscaling_server_bundle.tar.gz
```

Includes code/scripts and excludes heavy outputs.

---

## 8) Practical execution recommendation (datacenter GPU)

- Run **E1 → E2 → E3** in that order.
- For faster iteration, do a short E1 run first (fewer seeds/models), then execute the full final run.
- Keep local runs for quick sanity checks; final publishable runs should be executed on server.

---

## 9) Deterministic post-training flow (recommended)

Single config-driven orchestrator:

```bash
./.venv/bin/python scripts/evaluation/run_narrative_eval.py \
  --config config/eval_config.yaml \
  --stages exp1,exp2,exp3,cs1,cs2
```

Dual protocol for E2 (standard vs station-aligned):

```bash
./.venv/bin/python scripts/evaluation/run_exp2_dual_protocol.py \
  --config config/eval_config.yaml \
  --reuse-existing
```

Master report update after all runs:

```bash
./.venv/bin/python scripts/evaluation/build_master_report.py \
  --config config/eval_config.yaml
```

Reproducibility manifest (checksums of config/checkpoints/key outputs):

```bash
./.venv/bin/python scripts/evaluation/build_repro_manifest.py \
  --config config/eval_config.yaml \
  --out experiments/eval_outputs/repro_manifest.json \
  --strict
```
