# Release Notes

## Urban Climate Submission Snapshot

Date: 2026-04-16

This release snapshot supports the manuscript:

> Benchmarking Recurrent and State-Space Architectures for Reliable Spatiotemporal Downscaling of Urban Air Temperature in Heatwave Events

### Scope

- Consolidates the reproducibility code for the Urban Climate submission.
- Provides the evaluation orchestration used for Experiment 1, Experiment 2, Experiment 3, Case Study 1, Case Study 2, and robustness diagnostics.
- Includes scripts for publication bundle generation and reproducibility manifest generation.
- Adds public-release metadata through `CITATION.cff` and `.zenodo.json`.

### Included

- Model code for U-Net, ConvLSTM, Transformer, and Mamba variants.
- Evaluation scripts and configuration files.
- Lightweight reference outputs and manifests suitable for audit.
- Documentation for running the paper evaluation workflow.

### Not Included

- Restricted high-resolution UrbClim target data.
- Large trained checkpoints and regenerable experiment outputs.
- The local journal-submission folder `Paper/`, which remains ignored by Git.

### Recommended Tag

Use an annotated tag for the exact manuscript submission snapshot:

```bash
git tag -a urban-climate-submission-2026-04-16 \
  -m "Urban Climate manuscript submission snapshot"
git push origin urban-climate-submission-2026-04-16
```

### Reproducibility Notes

Full numerical replication requires access to the restricted UrbClim target fields described in the manuscript. ERA5-Land forcing data are publicly available from the Copernicus Climate Change Service Climate Data Store.
