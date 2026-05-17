---
title: "Urban wind field prediction based on sparse sensors and physics-informed graph-assisted auto-encoder"
authors: []
year: 
doi: 
tags: [urban-climate, ML, wind-field]
status: to-read
---

## Summary
Reconstructs high-resolution urban wind fields from sparse sensor measurements using a physics-informed GNN-assisted auto-encoder. The model incorporates the fluid continuity equation into the loss function and leverages GNNs to capture sensor-environment relationships. Achieves ~50% reduction in RMSE over prevalent generative DL models for multiple wind attack angles.

## Method
A graph neural network (GNN)-assisted auto-encoder is trained to map sparse sensor measurements to full high-resolution urban wind fields. The continuity equation of fluid flow is incorporated as a physics-informed term in the loss function to improve stability and physical consistency.

## Key Results
- ~50% reduction in RMSE compared to prevalent generative DL models
- Effective reconstruction of high-resolution urban wind fields from sparse sensors
- Demonstrated for multiple wind attack angles

## Limitations
- No authors or year provided in abstract
- Comparison limited to "prevalent generative DL models" — exact baselines not specified
- Generalization to different urban geometries or sensor configurations not discussed

## Relation to Thesis
Directly relevant to urban climate downscaling — addresses the core challenge of reconstructing high-resolution wind fields from sparse observations. The physics-informed (continuity equation) approach is a promising direction for integrating fluid dynamics constraints into ML-based wind field reconstruction.
