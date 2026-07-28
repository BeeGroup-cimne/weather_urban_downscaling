# Repository split — 2026-07-28

`weather_urban_downscaling` is **v1 of Paper 1** and is now **legacy / frozen**.
It keeps only the code, configuration, and documentation needed to reproduce the
*Urban Climate* submission. Everything that was not part of that reproducibility
record has been moved to independent repositories.

## What moved out

| Was here | Now lives in | Contents |
|---|---|---|
| `future_projects/` | `~/research_roadmap` | Future research lines, publication strategy, thesis-by-compendium planning (ACME review, DestinE, LST, hybrid latent SR, Copernicus heat stress, Grad-CAM, applied-math positioning, concept notes) |
| `Paper/` | `~/paper1_urban_climate` | LaTeX sources, figures, and the full submission package for the v1 manuscript |
| `future_projects/Asim_2026/` | `~/asim2026` | ASIM 2026 conference line (moved earlier, 2026-07-07) |

Both `future_projects/` and `Paper/` were listed in `.gitignore` and were never
tracked here, so nothing was removed from this repository's git history. Each new
repository starts from a single import commit.

## What stays here

The v1 reproducibility record: `src/`, `scripts/`, `config/`, `docs/`,
`model_benchmark/`, `frontend/`, and the release metadata (`CITATION.cff`,
`.zenodo.json`, `RELEASE_NOTES.md`). See [README.md](README.md).

## Active development

Paper 1 v2 (PyTorch rewrite, `DownsrUNet`, targeting *Computational Urban
Science*) is developed in a separate repository. This one is frozen at tag
`paper1-v1-legacy`.
