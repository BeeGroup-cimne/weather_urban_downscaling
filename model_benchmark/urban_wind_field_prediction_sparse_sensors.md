---
title: "Urban wind field prediction based on sparse sensors and physics-informed graph-assisted auto-encoder"
authors: []
year:
doi:
tags: [urban-climate, ML, wind-field]
status: to-read
---

## Summary
This paper presents a physics-informed graph neural network (GNN)-assisted auto-encoder to reconstruct high-resolution urban wind fields from sparse sensors. The method leverages GNNs to model relationships between sensors and their surrounding environment, and incorporates the continuity equation of fluid flow into the loss function to improve stability and performance.

## Method
- Uses a physics-informed graph neural network (GNN)-assisted auto-encoder architecture
- GNNs capture relationships between sparse sensors and their surrounding urban environment
- Incorporates the continuity equation of fluid flow into the convolution neural network's loss function
- Designed to work with multiple wind attack angles

## Key Results
- Approximately 50% reduction in root mean square error (RMSE) compared to prevalent generative DL models
- Successfully reconstructs high-resolution urban wind fields for multiple wind attack angles
- Enhanced utilization of sparse sensor data through deep mining capabilities of GNNs
- Improved stability through physics-informed loss function

## Limitations
- Relies on availability of some sensor data (sparse sensors)
- May require domain-specific tuning for different urban environments
- Performance may vary with sensor placement and density

## Relation to Thesis
This work directly connects to urban climate downscaling and fluid dynamics by:
- Addressing the challenge of obtaining high-resolution wind fields in urban areas (downscaling from sparse observations)
- Incorporating physical constraints (continuity equation) into the ML model, aligning with physics-informed machine learning approaches for fluid dynamics
- Demonstrating that GNNs can effectively capture spatial relationships in urban flow patterns
- Provides a framework for using limited sensor data to reconstruct full wind fields, relevant to urban climate modeling applications