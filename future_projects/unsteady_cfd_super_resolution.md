# New Project Concept: Space-Time Super-Resolution for Unsteady CFD

## Summary
This is a future project concept separate from the current weather downscaling work.
It targets unsteady, vortex-dominated flows with space-time super-resolution using
deep learning and optional topological control.

## Dataset & Problem
- Benchmark: Unsteady Fluid Mechanics (moving cylinder) dataset
- Fields: velocity, pressure, vorticity on a 31x31 grid
- Additional signals: drag/lift forces
- LR->HR: create low-resolution inputs via controlled downsampling
- Static context: geometry/mask, distance-to-cylinder, x/y coordinates
- Split strategy: generalize across excitation types (sine/sinesweep/multisine)

## Core Methodology
- Models: UNet, ConvLSTM, Mamba (space-time sequence modeling)
- Sequence lengths: 6 and 12 (main novelty lever)
- Metrics: MAE/RMSE/SSIM on fields + force signal consistency

## Conference Fit
Targets WCCM-ECCOMAS 2026 mini-symposia:
- Data-driven mechanics / AI for mechanics
- Data-driven CFD / aerodynamics
- Multiphysics / advanced materials

## Title (Draft)
"Space-Time Super-Resolution of Unsteady Vortex-Dominated Flows: Benchmarking UNet, ConvLSTM and Mamba on a Moving-Cylinder Dataset"

## Abstract (Draft)
We propose a space-time super-resolution framework for unsteady fluid mechanics,
using a moving-cylinder benchmark with velocity/pressure/vorticity fields.
Low-resolution inputs are downsampled and reconstructed with UNet, ConvLSTM, and
Mamba models. Temporal context is evaluated via sequence lengths 6 and 12, with
strict excitation-based splits to avoid leakage. Accuracy (MAE/RMSE/SSIM) and
force-signal consistency are reported. Results highlight improved temporal
coherence at longer sequences and competitive Mamba performance.

## Advanced Idea: Topologically Controlled Training
- Add a topological regularizer (persistent homology) to preserve vortex structure
- Start as a metric, then ablation with a low-weight topological loss
- Expected benefit: improved coherence of vortical cores and reduced artifacts
