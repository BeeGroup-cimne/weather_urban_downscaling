# Case Study 2 Disambiguation (CS2A vs CS2B)

This note formalizes the Case Study 2 naming to prevent publication ambiguity.

## Canonical Naming

- `CS2A` = **Night Cooling & Persistence** (legacy)
- `CS2B` = **Input Robustness (Monte Carlo)** (current narrative pipeline)

## CS2A: Night Cooling & Persistence (Legacy)

Research question:
- Where does nocturnal heat fail to dissipate, and how persistent are hotspots across nights?

Primary artifacts:
- `experiments/fullframe/casestudy2_20260223_022102/report_casestudy2.md`
- `experiments/fullframe/casestudy2_20260223_022102/cs2_cooling_summary.csv`
- `experiments/fullframe/casestudy2_20260223_022102/cs2_cooling_by_pair.csv`

Typical metrics:
- `mean_delta_t` (cooling between paired nights)
- `persistent_hotspots_pct_ge3`

## CS2B: Input Robustness (Monte Carlo, Current)

Research question:
- How sensitive are predictions to controlled perturbations in input channels?

Primary artifacts:
- `experiments/eval_outputs/cs2_robustness_publish/robustness_summary.csv`
- `experiments/eval_outputs/cs2_robustness_publish/cs2_rank_stability.csv`
- model-level `report_robustness.md` under each model subfolder

Typical metrics:
- `pred_mean_abs_dev_vs_clean_C`
- `pred_rmse_vs_clean_C`
- `epsilon_K`, `n_trials`

## Relationship Between CS2A and CS2B

- They are related in climate interpretation (thermal stress and resilience), but they are **not the same experiment**.
- CS2A emphasizes **spatial nocturnal persistence**.
- CS2B emphasizes **input uncertainty sensitivity**.
- In paper captions/tables, always label explicitly as `CS2A` or `CS2B`.

## Manuscript Recommendation

Use this wording pattern:
- "CS2A evaluates nocturnal cooling persistence and hotspot retention."
- "CS2B evaluates robustness to input perturbations via Monte Carlo trials."

Do not refer to both simply as "CS2" without the `A/B` suffix.
