---
title: "Urban wind field prediction based on sparse sensors and physics-informed graph-assisted auto-encoder"
authors: []
year: 
doi: 
tags: [urban-climate, ML, wind-field]
status: to-read
---

## Summary
This paper addresses the challenge of reconstructing high-resolution urban wind fields from sparse sensor measurements, a critical need for urban wind disaster mitigation, wind environment assessment, and drone route planning. The authors propose a physics-informed GNN-assisted auto-encoder that outperforms existing generative deep learning models by approximately 50% in RMSE across multiple wind attack angles.

## Method
- Physics-informed graph neural network (GNN)-assisted auto-encoder architecture
- GNN component mines relationships between sparse sensors and their surrounding urban environment, enhancing utilization of limited sensor data
- Continuity equation of fluid flow incorporated into the CNN loss function to enforce physical constraints and improve model stability
- Trained to reconstruct full high-resolution urban wind fields from sparse sensor inputs

## Key Results
- ~50% reduction in RMSE compared to prevalent generative DL models for urban wind field reconstruction
- Successful reconstruction across multiple wind attack angles
- Enhanced utilization and emphasis on sparse sensor data through GNN's deep mining capabilities
- Improved stability and performance from physics-informed loss function

## Limitations
- Abstract does not specify computational cost or inference time compared to baseline models
- Generalizability to different urban morphologies beyond training configurations unclear
- Requires high-resolution training data (likely from CFD or wind tunnel experiments)
- Performance under highly transient or turbulent conditions not discussed

## Relation to Thesis
Directly relevant to urban climate downscaling as it demonstrates a physics-informed ML approach to reconstruct high-resolution flow fields from sparse observations—paralleling the challenge of downscaling coarse weather model output to urban scales. The incorporation of fluid dynamics constraints (continuity equation) into the loss function provides a template for embedding physical priors in ML downscaling models. The GNN approach for learning sensor-environment relationships could inform how to integrate urban morphology features into downscaling architectures.
