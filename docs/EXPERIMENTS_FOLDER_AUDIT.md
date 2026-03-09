# Experiments Folder Audit

Date: 2026-03-09  
Scope: folder-by-folder audit under `experiments/` for paper relevance, reproducibility, and maintenance.

## Folder: `experiments/ab_compare`

### What it contains
- Two variants:
  - `A_main_like/`
  - `B_current/`
- Each variant stores:
  - `ab_compare_*.png` (visual snapshot)
  - `static_processed.npy`
  - `stats_config.npz`
  - `weather_cache.zarr/` (full cache structure)

### Technical findings
- Only one script references this folder:
  - `scripts/tools/ab_compare_pipeline.py`
- No references found in:
  - `README.md`
  - narrative/evaluation reports
  - paper figure manifests
- `A_main_like` vs `B_current` evidence:
  - `static_processed.npy` is identical (same MD5).
  - `stats_config.npz` differs (means/stds changed between variants).
  - `ab_compare_*.png` differs slightly (same shape, low non-zero pixel ratio).
  - `weather_cache.zarr/.zmetadata` is identical between A and B.

### Interpretation
- This folder is a diagnostics sandbox for comparing data-pipeline settings
  (`main-like` vs `current` behavior), not a publication artifact source.
- It does not feed the deterministic post-training narrative pipeline.
- It is useful for debugging lineage/history, but not needed for paper execution.

### Decision
- **Status:** `Archive / Optional keep`
- **Needed for paper narrative:** `No`
- **Needed for reproducible eval pipeline:** `No`
- **Keep script:** `Yes` (`scripts/tools/ab_compare_pipeline.py`, for future diagnostics)

### Recommendation
1. Keep `scripts/tools/ab_compare_pipeline.py` in repo.
2. Move `experiments/ab_compare/` outputs to an archive location (or keep excluded from core paper branch) to reduce operational noise.
3. Do not delete immediately until complete folder-by-folder audit is finished.

### Action flag
- `pending_user_decision`: archive now vs keep until full audit ends.

## Folder: `experiments/eval_outputs`

### What it contains
- Publish runs (narrative pipeline):
  - `exp1_spatial_publish/`
  - `exp2_groundtruth_publish/`
  - `exp2_dual_protocol_publish/`
  - `exp3_bottleneck_publish/`
  - `cs1_heatwave_publish/`
  - `cs2_robustness_publish/`
- Final reports/manifests:
  - `report_master_narrative.md`
  - `report_paper_ready.md`
  - `repro_manifest.json`
  - `publication_gate_report.json`
- Final curated figures:
  - `paper_figures_final/`
- Temporary runs:
  - `_smoke_*`, `*_smoke`
  - `cs2_smoke/`
  - `exp2_station_aligned_publish/` (empty)

### Technical findings
- Active `eval_config.yaml` points to publish directories above (not to smoke folders).
- Main size contributors:
  - `cs1_heatwave_publish`: ~491M
    - `figures/` alone: ~490M, 3240 files (`1080 .png`, `1080 .npy`, `1080 .csv`)
  - `exp1_spatial_publish`: ~19M (`figures/` + metrics)
  - `exp2_groundtruth_publish`: ~15M (model-level station CSV/PNG outputs)
  - `paper_figures_final`: ~9.8M
- Smoke folders are tiny and clearly non-final:
  - `_smoke_cs2_baseline_bilinear`: 8K
  - `_smoke_cs2_mamba`: 8K
  - `_smoke_narrative_cs2`: 16K
  - `cs2_smoke`: 16K
  - `exp3_smoke`: 20K
  - `exp1_smoke`: 424K
  - `cs1_smoke`: 412K
- `exp2_station_aligned_publish` is empty (0B).

### Interpretation
- `eval_outputs` is the core post-training evidence folder for the paper.
- Publish directories + reports/manifests + `paper_figures_final` are required for paper reproducibility.
- Smoke folders are execution debris and can be archived/removed safely.
- `cs1_heatwave_publish/figures` is oversized but currently useful for qualitative regeneration scripts
  (`generate_publication_figures.py` searches PNGs there).

### Decision by subfolder
- **Keep (required):**
  - `exp1_spatial_publish/`
  - `exp2_groundtruth_publish/`
  - `exp2_dual_protocol_publish/`
  - `exp3_bottleneck_publish/`
  - `cs1_heatwave_publish/` (see pruning note below)
  - `cs2_robustness_publish/`
  - `paper_figures_final/`
  - root reports/manifests (`report_*`, `repro_manifest.json`, `publication_gate_report.json`)
- **Archive/remove (safe):**
  - `_smoke_cs2_baseline_bilinear/`
  - `_smoke_cs2_mamba/`
  - `_smoke_narrative_cs2/`
  - `cs2_smoke/`
  - `cs1_smoke/`
  - `exp1_smoke/`
  - `exp3_smoke/`
  - `exp2_station_aligned_publish/` (empty)
- **Optional prune (after a dedicated pruning plan):**
  - `cs1_heatwave_publish/figures`
  - Keep a minimal representative subset if regeneration requirements are formalized.

### Recommendation
1. Keep publish evidence unchanged for now.
2. Remove/relocate all smoke subfolders in one cleanup step.
3. Plan a second-pass compaction for `cs1_heatwave_publish/figures` to reduce 490M without
   breaking figure regeneration expectations.

### Action flag
- `pending_user_decision`: execute smoke cleanup now vs defer until all `experiments/` folders are audited.

## Folder: `experiments/features_fullframe`

### What it contains
- 7 files `.npy` (`251x251`, `float32`), all tagged `fullframe_PUB_*`:
  - `fullframe_PUB_TRUTH.npy`
  - `fullframe_PUB_BASELINE_BILINEAR.npy`
  - `fullframe_PUB_Tiles_LSTM.npy`
  - `fullframe_PUB_Tiles_UNET.npy`
  - `fullframe_PUB_Ablation_LSTM.npy`
  - `fullframe_PUB_Ablation_UNET.npy`
  - `fullframe_PUB_Ablation_MAMBA_SEQ12.npy`

### Technical findings
- Small footprint (~1.8M total).
- Files are distinct (different checksums, not duplicates).
- Referenced by:
  - `scripts/figures/generate_fullframe_preds.py` (producer)
  - `scripts/figures/generate_presentation_figures.py` (consumer)
- Not referenced directly by the deterministic `eval_outputs` narrative pipeline.

### Interpretation
- This is a compact feature snapshot bank for figure generation/presentation, not a training artifact.
- It provides fast reproducible visual inputs without rerunning heavier inference.
- Given tiny size and explicit script usage, removal would hurt convenience more than it helps.

### Decision
- **Status:** `Keep`
- **Needed for paper narrative (direct):** `Optional`
- **Needed for figure regeneration convenience:** `Yes`
- **Storage burden:** `Low`

### Recommendation
1. Keep folder as-is.
2. Label it as "presentation helper assets" in future cleanup docs (no urgent action required).

### Action flag
- `no_action_required` (retain).

## Folder: `experiments/fullframe`

### What it contains
- Active folders after cleanup:
  - `casestudy2_20260223_022102/` (legacy CS2A canonical run)
  - `casestudy2_regen_20260305/` (regen assets for CS2 qualitative outputs)
  - `experiment3_multiseed_20260303_221555/` (best E3 multiseed aggregate)

### Technical findings
- Current footprint: ~`85M` (reduced from ~`121M` after superseded-run archive).
- Canonical references exist for CS2A:
  - `README.md` and `docs/CASE_STUDY2_DISAMBIGUATION.md` point to
    `casestudy2_20260223_022102`.
  - `config/eval_config.yaml` comments reference the same CS2A folder.
- `experiment3_multiseed_20260303_221555` is the only retained E3 folder with coherent
  multiseed aggregate table in `fullframe_eval_aggregate.csv`.
- Prior noise detected and cleaned:
  - malformed 0B file
    `fullframe_eval_aggregate.csv --split test --out-stem fig04_mae_comparison_final`
  - multiple empty `evals/` directories
  - `.DS_Store`

### Interpretation
- `fullframe` should be treated as a **legacy narrative companion**:
  - CS2A canonical evidence (night cooling/persistence),
  - one retained E3 multiseed reference.
- Deterministic post-training publication pipeline remains under `experiments/eval_outputs/`.

### Decision
- **Keep (required/important legacy):**
  - `casestudy2_20260223_022102/`
  - `casestudy2_regen_20260305/`
  - `experiment3_multiseed_20260303_221555/`
- **Archived locally as superseded:**
  - `casestudy2_20260223_014912/`
  - `experiment3_20260226_172922/`
  - `experiment3_20260226_174011/`
  - `experiment3_20260226_174501/`
  - `experiment3_evalonly_20260226_175940/`
  - `experiment3_retake_full_20260227_131305/`
  - `experiment3_retake_full_20260227_131945/`
  - `experiment3_retake_smoke_20260227_130957/`
  - `experiment3_ssim_check_20260227_131523/`
  - target: `experiments/archive/fullframe_superseded_20260309/`

### Recommendation
1. Keep CS2A canonical folder unchanged until manuscript acceptance.
2. Keep only `experiment3_multiseed_20260303_221555` as E3 legacy reference in active tree.
3. If git-tracked archival is desired, revise `.gitignore` policy (`experiments/` is currently ignored).

### Action flag
- `completed_for_fullframe`: safe cleanup + local archive applied.

## Folder: `experiments/paper_figures`

### What it contains
- Architecture and infographic artifact bank:
  - `architectures/`, `architectures_tb/`, `architectures_tb_nosize/`
  - `block_diagrams/`, `flow/`, `f1_v2/`, `f2_v2/`
- Types: `png`, `pdf`, `svg`, `dot`, markdown docs.

### Technical findings
- Current footprint: ~`47M`.
- No direct execution dependency in deterministic eval pipeline (`eval_config.yaml` and
  `run_narrative_eval.py` do not consume this folder).
- Not part of atlas manifest (`manifest_all_sources.csv` has `0` rows from this path).
- Reproducibility gap:
  - README files reference scripts
    (`scripts/generate_paper_architecture_figures.py`,
    `scripts/generate_paper_block_diagrams.py`,
    `scripts/generate_paper_flow_figures.py`)
    that are not present in current repository snapshot.
- Duplication evidence:
  - `architectures_tb` and `architectures_tb_nosize` include identical DOT sources and several
    byte-identical SVGs (content-hash duplicate groups).
- Safe cleanup applied:
  - removed `.DS_Store`.

### Interpretation
- Folder has high narrative value for methods/appendix figures, but regeneration is currently
  partially undocumented due missing generator scripts.

### Decision
- **Keep (publication asset bank):**
  - all current subfolders under `experiments/paper_figures/`
- **Delete-safe applied:**
  - `.DS_Store` removed

### Recommendation
1. Keep current files as frozen assets for paper assembly.
2. In a later hardening pass, either restore generator scripts or update README commands to valid entrypoints.
3. Optionally deduplicate `architectures_tb` vs `architectures_tb_nosize` after paper submission.

### Action flag
- `in_progress`: hygiene applied; pending script-regeneration documentation fix.

## Folder: `experiments/presentation`

### What it contains
- Single file:
  - `mindcity_eng_lite_v2.pptx` (~`8.1M`)

### Technical findings
- Not consumed by deterministic evaluation scripts.
- Mentioned indirectly in narrative docs (`docs/presentaciones/*`) as source/support material.
- No structural duplication or execution risk found.

### Interpretation
- This is a standalone communication asset, not a pipeline artifact.

### Decision
- **Keep (optional narrative asset):**
  - `mindcity_eng_lite_v2.pptx`
- **Delete-safe:**
  - none

### Recommendation
1. Retain as external supporting material.
2. Keep out of core eval reproducibility claims.

### Action flag
- `no_action_required` (retain).

## Folder: `experiments/reports`

### What it contained
- Legacy consolidated report bundle:
  - `consolidated_20260227_115630/` (`consolidated_report.md`, rank CSVs)

### Technical findings
- Legacy report references old output roots:
  - `experiments/stations_eval/experiment2_20260227_111926`
  - `experiments/heatwaves/casestudy1_20260227_103825`
- Current deterministic reporting pipeline uses:
  - `experiments/eval_outputs/report_master_narrative.md`
  - `experiments/eval_outputs/report_paper_ready.md`
  - via `scripts/evaluation/build_master_report.py` and publication validators.

### Interpretation
- Folder was a pre-orchestrator consolidation step and is superseded by `eval_outputs` reports.

### Decision
- **Archive candidate:** `consolidated_20260227_115630/` ✅ moved locally on 2026-03-09 to
  `experiments/archive/reports_superseded_20260309/`
- `experiments/reports/` active path recreated as empty directory.

### Recommendation
1. Keep all final manuscript citations pointed to `experiments/eval_outputs/report_*`.
2. Use archived consolidated report only for historical traceability.

### Action flag
- `completed_for_reports`: legacy bundle archived locally; active path left clean.

## Folder: `experiments/robustness`

### What it contained
- One legacy Monte Carlo run:
  - `robustness_20260304_214556/`

### Technical findings
- Current canonical CS2B robustness is under:
  - `experiments/eval_outputs/cs2_robustness_publish/`
  - referenced by `README.md`, `config/eval_config.yaml`, and narrative scripts.
- Legacy folder is not referenced by active orchestrator config.

### Interpretation
- `experiments/robustness` is a legacy staging location superseded by `eval_outputs` CS2B outputs.

### Decision
- **Archive candidate:** `robustness_20260304_214556/` ✅ moved locally on 2026-03-09 to
  `experiments/archive/robustness_superseded_20260309/`
- `experiments/robustness/` active path recreated as empty directory.

### Recommendation
1. Keep CS2B publication references exclusively under `experiments/eval_outputs/cs2_robustness_publish`.
2. Preserve archived legacy run only for provenance comparison.

### Action flag
- `completed_for_robustness`: legacy run archived locally; active path left clean.

## Folder: `experiments/stations_eval`

### What it contains
- Active station-eval assets after cleanup:
  - canonical data bundle: `data/`
  - historical full model comparison: `experiment2_20260227_111926/`
  - ablation runs: `ablation_*`, `ablation_multiseed_20260303_182241/`
  - real-observation figure prep runs: `fig05_real_*`
  - top2 historical runs: `amb36_top2_full_*`, `amb36_top2_tiles_final_*`

### Technical findings
- Current footprint: ~`64M` (down from ~`66M` after smoke/partial archive).
- Active deterministic linkage:
  - `config/eval_config.yaml` uses:
    - `data.stations_obs_csv = experiments/stations_eval/data/meteocat_amb36_air_temperature_hourly_2017.csv`
    - `narrative.exp2_dual_protocol.protocols.station_aligned.output_dir =
      experiments/stations_eval/fig05_real_prep_full2017_20260306`
- Cleanup/archival actions applied:
  - archived smoke runs:
    - `amb36_smoke_top2/`
    - `amb36_smoke_top2_v2/`
    - `amb36_smoke_top2_v3/`
    - `amb36_smoke_tiles_top2_20260219_171346/`
  - archived partial run:
    - `ablation_20260301_011139/` (incomplete, missing full model coverage)
  - target: `experiments/archive/stations_eval_superseded_20260309/`

### Reproducibility/risk findings
- Multiple `ablation_YYYY...` folders remain and are not all equivalent; this can confuse
  “official Exp2 legacy” selection if no canonical tag is provided.
- `experiment2_20260227_111926` and `fig05_real_prep_full2017_20260306` serve distinct purposes
  (historical broad baseline vs station-aligned protocol reuse).

### Interpretation
- `stations_eval` remains partially active due explicit config dependency on `data/` and
  `fig05_real_prep_full2017_20260306`.
- Most smoke/partial clutter has been removed from the active path.

### Decision
- **Keep (required):**
  - `data/`
  - `fig05_real_prep_full2017_20260306/`
- **Keep (legacy but useful):**
  - `experiment2_20260227_111926/`
  - `ablation_multiseed_20260303_182241/`
  - `ablation_20260227_165406/`, `ablation_20260227_171440/`, `ablation_20260227_172902/`,
    `ablation_20260301_013523/`
  - `fig05_real_prep_20260306/`, `fig05_real_prep_20260306_bias/`, `fig05_real_improved_20260306/`
  - `amb36_top2_full_20260219_165137/`, `amb36_top2_tiles_final_20260219_171442/`
- **Archived locally (superseded/partial):**
  - listed above in cleanup actions.

### Recommendation
1. Tag one folder as canonical legacy Exp2 (`experiment2_20260227_111926`) in docs to reduce ambiguity.
2. Keep `fig05_real_prep_full2017_20260306` fixed while dual-protocol comparison depends on it.
3. If desired later, archive additional old `ablation_*` runs once final manuscript tables are frozen.

### Action flag
- `in_progress`: smoke/partial cleanup applied; optional second-pass archival pending.

## Folder: `experiments/figures`

### What it contains
- Mixed artifact hub (365 files total) with:
  - legacy single-image outputs at root (`tiles_*`, `result_*`, `fig07/fig08*`, `*_architecture.png`)
  - model-batch comparisons (`batch_compare_all/`, `batch_compare_all_v2/`)
  - P95 evaluations (`p95_eval_*`, `p95_mamba_seq_compare_*`)
  - presentation assets (`ppt_preview/`, `ppt_update/`)

### Technical findings
- Size profile (largest subfolders):
  - `p95_eval_20260216_153933/` ~12M
  - `p95_eval_20260216_210822/` ~12M
  - `batch_compare_all_v2/` ~7.5M
  - `batch_compare_all/` ~6.1M
  - `p95_mamba_seq_compare_20260226_171451/` ~5.9M
- Empty/noise folders:
  - `p95_eval_20260216_210309/` (0B)
  - `.ipynb_checkpoints/`
- Pipeline references exist (not dead data):
  - `scripts/evaluation/run_p95_eval_caffeinate.sh`
  - `scripts/evaluation/run_p95_eval_mamba_seq_compare.sh`
  - `scripts/evaluation/sync_image_atlas.py`
  - `scripts/tools/make_server_bundle.sh` and `.py`
- Atlas integration is substantial:
  - `figures/final_all_images/manifest_all_sources.csv` has 257 assets total.
  - 142 entries are sourced from `experiments/figures`.
- Internal duplication/versioning:
  - `batch_compare_all/` and `batch_compare_all_v2/` overlap heavily.
  - `p95_eval_20260216_153933/` and `p95_eval_20260216_210822/` share 73 basename matches.
  - many `.npy` are byte-identical across variants (hash collisions), indicating repeated arrays under different filenames.

### Reproducibility/risk findings
- `run_casestudy1_eval.sh` still contains non-deterministic fallback:
  - if `mamba_seq_csv` is not provided, it picks latest by glob:
    `experiments/figures/p95_mamba_seq_compare_*/paper_summary_mamba_seq.csv`
- The two P95 eval runs are not equivalent:
  - `p95_eval_20260216_153933/summary.csv` shows model labeling/pathology (many rows effectively mamba-like).
  - `p95_eval_20260216_210822/` includes `paper_summary_by_model.csv` and coherent per-model aggregation.
- Batch compare v1 vs v2:
  - `batch_compare_all_v2/` is the corrected superset (includes `ConvLSTM_best` and transformer attempt).
  - `batch_compare_all/` has unsupported/skipped entries and earlier behavior.

### Interpretation
- This folder is a hybrid of:
  - active archive material needed by the image-atlas narrative,
  - historical intermediate runs from iterative figure engineering.
- It is useful for narrative traceability, but currently noisy for strict reproducibility.

### Decision by subfolder
- **Keep (active archive):**
  - root curated files under `experiments/figures/` consumed by atlas
  - `p95_eval_20260216_210822/`
  - `p95_mamba_seq_compare_20260226_171451/`
  - `batch_compare_all_v2/`
  - `ppt_preview/`, `ppt_update/` (presentation lineage)
- **Archive (historical superseded):**
  - `p95_eval_20260216_153933/` ✅ moved on 2026-03-09 to
    `experiments/archive/figures_superseded_20260309/p95_eval_20260216_153933/`
  - `batch_compare_all/` ✅ moved on 2026-03-09 to
    `experiments/archive/figures_superseded_20260309/batch_compare_all/`
- **Delete-safe (noise):**
  - `p95_eval_20260216_210309/` (empty) ✅ removed on 2026-03-09
  - `.ipynb_checkpoints/` ✅ removed on 2026-03-09

### Recommendation
1. Keep archived superseded sets in `experiments/archive/figures_superseded_20260309/` until paper acceptance.
2. For deterministic CS1 postprocessing, require explicit `mamba_seq_csv` path (no latest-glob fallback) in the next evaluation hardening pass.
3. If storage cleanup is needed later, archive/compress additional historical figure runs instead of deleting.

### Action flag
- `completed_for_figures`: safe cleanup + soft archive applied.

## Folder: `experiments/heatwaves`

### What it contains
- Event-definition assets:
  - `aemet/` (event times, station metadata, thresholds)
- Legacy Case Study 1 runs:
  - `casestudy1_20260223_012118/`
  - `casestudy1_20260223_025616/`
  - `casestudy1_20260227_103545/`
  - `casestudy1_20260227_103825/`
  - `casestudy1_multiseed_20260304_005009/` (`S42/S43/S44`)
- Legacy Experiment 1 publish runs:
  - `publish_run_20260220_220458/`
  - `publish_run_multiseeds_20260303_175704/`
  - `publish_run_multiseeds_20260303_181038/`
  - `publish_run_multiseeds_20260303_175636/` (empty)
  - `publish_run_multiseeds_20260303_175645/` (empty)

### Technical findings
- Total composition (all nested files): `570 npy`, `504 png`, `437 csv`.
- Size hotspots:
  - `casestudy1_multiseed_20260304_005009`: ~49M
  - `publish_run_multiseeds_20260303_175704`: ~48M
  - `publish_run_multiseeds_20260303_181038`: ~48M
  - `casestudy1_20260223_025616`: ~36M
- Empty folders (safe noise):
  - `publish_run_multiseeds_20260303_175636/figures`
  - `publish_run_multiseeds_20260303_175645/figures`
- Deterministic pipeline linkage:
  - `config/eval_config.yaml` only requires `experiments/heatwaves/aemet/event_times_2017.txt`.
  - Narrative outputs now target `experiments/eval_outputs/*` (not `experiments/heatwaves/*`).
- Legacy hardcoded dependencies still exist:
  - `scripts/figures/generate_presentation_figures.py` pins `publish_run_20260220_220458`.
  - `docker/compose.server-exp3-eval.yml` points `EXP1_AGG_CSV` to `publish_run_20260220_220458`.
  - `docs/SERVER_README.md` still documents old `experiments/heatwaves/publish_run_*` flow.
- Atlas integration:
  - `figures/final_all_images/manifest_all_sources.csv` contains `0` entries from `experiments/heatwaves`.

### Reproducibility/risk findings
- Folder contains multiple historical variants for similar analyses (CS1 and publish runs), increasing ambiguity on “official” evidence.
- `casestudy1_multiseed_20260304_005009/S43` and `S44` reports keep title string “(S42)” (metadata inconsistency in report text).
- Two multiseed publish runs are both non-empty and differ in aggregate CSVs:
  - `175704`: includes `mamba` aggregated over more samples.
  - `181038`: includes explicit `mamba_seq12` line and different aggregate CI tables.

### Interpretation
- `aemet/` is active scientific input and should remain in place.
- Most other subfolders are legacy outputs from pre-orchestration iterations; useful for lineage, but not required by the current deterministic paper pipeline under `experiments/eval_outputs`.
- One legacy run (`publish_run_20260220_220458`) remains indirectly “active” due hardcoded references in presentation/server helper scripts.

### Decision by subfolder
- **Keep (required):**
  - `aemet/`
- **Keep temporarily (until ref migration):**
  - `publish_run_20260220_220458/` (referenced by helper scripts/docs)
- **Archive candidates (legacy historical):**
  - `casestudy1_20260223_012118/`
  - `casestudy1_20260223_025616/`
  - `casestudy1_20260227_103545/`
  - `casestudy1_20260227_103825/`
  - `casestudy1_multiseed_20260304_005009/`
  - `publish_run_multiseeds_20260303_175704/`
  - `publish_run_multiseeds_20260303_181038/`
- **Delete-safe (noise):**
  - `publish_run_multiseeds_20260303_175636/` (empty) ✅ removed on 2026-03-09
  - `publish_run_multiseeds_20260303_175645/` (empty) ✅ removed on 2026-03-09

### Recommendation
1. Keep `aemet/` untouched (active input dependency).
2. Before archiving `publish_run_20260220_220458/`, migrate hardcoded references in:
   `scripts/figures/generate_presentation_figures.py`, `docker/compose.server-exp3-eval.yml`, and `docs/SERVER_README.md`.
3. Archive legacy CS1/publish multiseed folders after the reference migration, keeping them outside active execution paths.
4. Delete the two empty multiseed run folders immediately when cleanup is approved.

### Action flag
- `in_progress`: delete-safe applied; pending decision for archive of legacy `heatwaves` outputs after reference migration.

## Folder: `experiments/logs`

### What it contains
- Training-history CSV logs (small):
  - `Tiles_*_log.csv`
  - `Ablation_*_log.csv`
  - `UNet_gpu_optimized_log.csv`
- Raw terminal log dumps:
  - `ablation_mamba_seq6.log` (~7.4M)
  - `ablation_tiles_all_20260216_212132.log` (empty)
- Notebook noise:
  - `.ipynb_checkpoints/`

### Technical findings
- File composition: `11 csv`, `2 log`, `1 .gitkeep`.
- Size profile:
  - `ablation_mamba_seq6.log`: ~7.4M (dominant)
  - all CSV logs: ~4K–8K each
  - `ablation_tiles_all_20260216_212132.log`: `0B`
- Script linkage exists:
  - Packaging includes `experiments/logs/` (`scripts/tools/make_server_bundle.sh` and `.py`).
  - Training scripts write logs here (`scripts/training/*.py`).
  - Figure scripts read logs from here (`scripts/figures/fig09_train_val_curves.py`, `generate_presentation_figures.py`, `fig10_mamba_memory_ablation.py`).
- Naming mismatch risk in figure scripts:
  - scripts expect `*_S42_log.csv` files (e.g. `Tiles_UNET_S42_log.csv`), but these files are absent in repo.
  - available files are mostly non-seeded names (`Tiles_UNET_log.csv`, `Tiles_LSTM_log.csv`, etc.).
- Atlas integration:
  - `figures/final_all_images/manifest_all_sources.csv` has `0` entries from `experiments/logs`.

### Reproducibility/risk findings
- Figure-generation scripts that depend on missing `*_S42_log.csv` degrade gracefully (skip/warn), but produce incomplete or no training-curve panels.
- This does not affect deterministic post-training evaluation stages (`exp1/exp2/exp3/cs1/cs2`), but affects reproducibility of specific presentation figures.

### Interpretation
- `experiments/logs` is a lightweight provenance folder for training dynamics and auxiliary figures.
- It is not a core dependency for the frozen post-training narrative execution, but it is useful for methods/training-curve storytelling.
- Current log filename conventions are inconsistent with some consumer scripts.

### Decision
- **Keep (recommended):**
  - all non-empty CSV training logs
  - `ablation_mamba_seq6.log` (archive-grade provenance; optional to relocate later)
- **Delete-safe (noise):**
  - `.ipynb_checkpoints/` ✅ removed on 2026-03-09
  - `ablation_tiles_all_20260216_212132.log` (empty) ✅ removed on 2026-03-09

### Recommendation
1. Keep current CSV logs as provenance evidence for training-curve figures.
2. In a later hardening pass, normalize log filename expectations in figure scripts (or add deterministic aliases) to remove `*_S42` mismatch.
3. Optionally archive `ablation_mamba_seq6.log` out of active folder if storage hygiene is prioritized.
4. Delete empty/noise items when cleanup is approved.

### Action flag
- `in_progress`: delete-safe applied; pending decision on optional archive of large raw log (`ablation_mamba_seq6.log`).

## Folder: `experiments/models`

### What it contains
- Main checkpoint bank at root:
  - `40` files `*.h5` (`*_best.h5` and `*_last.h5`)
  - families: `Tiles_*_S42/S43`, `Ablation_*_Legacy_S42/S43/S44`, `Ablation_*_SEQ12`
- Backup snapshots:
  - `backup_seq12_20260228_181606/`
  - `backup_seq12_20260228_225733/`
  - `backup_seq12_20260303_152857/`
  - each backup contains `10 h5 + 5 csv`
- Misc files:
  - `lstm_summary.txt`, `trans_summary.txt`
  - `old_models.py` (empty, 0B)

### Technical findings
- Folder size: ~`1.5G`.
- Backup overhead:
  - three `backup_seq12_*` directories total ~`624M`.
  - each backup is byte-identical to top-level files of the same names (`mismatch_count=0`).
- Top-level checkpoint usage in deterministic pipeline:
  - `config/eval_config.yaml` references only explicit `*_best.h5` files (seeded names).
- Legacy path mismatch risk:
  - some scripts still default to non-seeded names not present in folder
    (e.g. `Tiles_UNET_best.h5`, `Tiles_MAMBA_best.h5` in `run_p95_eval_caffeinate.sh`).
- `*_last.h5` files:
  - present for all families, but no active references found in scripts/config/docs.

### Reproducibility/risk findings
- Core reproducible eval pipeline is correctly anchored to explicit seeded `*_best.h5`.
- Legacy scripts with non-seeded defaults can fail if run without overrides.
- Large duplicated backups add noise and storage pressure without adding unique checkpoint content.

### Interpretation
- `experiments/models` is a critical folder and must be preserved for frozen-training reproducibility.
- Unique value is concentrated in top-level seeded `*_best.h5` checkpoints.
- Backup folders are archival copies, not active dependencies.

### Decision
- **Keep (required):**
  - top-level seeded `*_best.h5` used by `eval_config.yaml`
  - top-level seeded `*_last.h5` (optional for lineage; not currently referenced)
- **Archive candidates (storage cleanup):**
  - `backup_seq12_20260228_181606/` ✅ moved on 2026-03-09 to
    `experiments/archive/models_superseded_20260309/backup_seq12_20260228_181606/`
  - `backup_seq12_20260228_225733/` ✅ moved on 2026-03-09 to
    `experiments/archive/models_superseded_20260309/backup_seq12_20260228_225733/`
  - `backup_seq12_20260303_152857/` ✅ moved on 2026-03-09 to
    `experiments/archive/models_superseded_20260309/backup_seq12_20260303_152857/`
- **Delete-safe (noise):**
  - `old_models.py` (empty 0B) ✅ removed on 2026-03-09

### Recommendation
1. Keep top-level checkpoints unchanged (especially all `*_best.h5` used by narrative config).
2. Keep archived backups in `experiments/archive/models_superseded_20260309/` until paper acceptance.
3. Harden legacy scripts by replacing non-seeded defaults with existing seeded checkpoint names.
4. Preserve `*_last.h5` for lineage unless an explicit model-pruning policy is approved.

### Action flag
- `in_progress`: archive + delete-safe applied; pending only legacy script hardening for non-seeded checkpoint defaults.

## Folder: `experiments/presentation_figures`

### What it contains
- Curated presentation-ready figures (single flat folder):
  - `14 png`
  - `3 gif`
- Includes static panels (`fig_a`...`fig_h`, `fig_mamba_*`, `fig12_heatwave_case_study.png`)
  and dynamic CS1 animations (`fig07*`, `fig08*`).

### Technical findings
- Folder is compact (~26M total).
- No internal duplicates by content hash inside this folder.
- It is actively integrated in final atlas tooling:
  - `scripts/evaluation/sync_image_atlas.py` includes `experiments/presentation_figures`.
  - `figures/final_all_images/manifest_all_sources.csv` contains `17` entries from this folder.
- Provenance scripts that generate into this folder:
  - `scripts/figures/generate_presentation_figures.py`
  - `scripts/figures/fig09_train_val_curves.py`
  - `scripts/figures/fig10_mamba_memory_ablation.py`
  - `scripts/figures/fig11_mamba_spatial_comparison.py`
  - `scripts/figures/fig12_heatwave_case_study.py`
- Cross-folder overlap:
  - `fig12_heatwave_case_study.png` also exists in `figures/png/`, but content differs (not byte-identical), so they are distinct versions.

### Interpretation
- This folder acts as a curated “presentation secondary layer” already wired into narrative atlas assets.
- It is not execution-critical for deterministic evaluation, but is publication/storytelling-critical.

### Decision
- **Keep (required for narrative assets):**
  - entire `experiments/presentation_figures/`
- **Delete-safe:**
  - none identified

### Recommendation
1. Keep as-is; do not prune this folder during active paper assembly.
2. If unification is desired later, define a canonical owner for duplicated figure names across `experiments/presentation_figures` vs `figures/png`.

### Action flag
- `no_action_required` (retain).
