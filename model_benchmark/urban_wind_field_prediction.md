---
title: "Urban wind field prediction based on sparse sensors and physics-informed graph-assisted auto-encoder"
authors: []
year: 
doi: 
tags: [urban-climate, ML, wind-field]
status: to-read
---

## Summary
This paper addresses the challenge of reconstructing high-resolution urban wind fields from sparse sensor data. It proposes a physics-informed graph neural network (GNN)-assisted auto-encoder that incorporates the continuity equation into the loss function to enhance model stability and performance. The method demonstrates approximately a 50% reduction in root mean square error compared to prevalent generative deep learning models for multiple wind attack angles.

## Method
The method uses a physics-informed graph neural network (GNN)-assisted auto-encoder. It leverages the relationship between sensors and their surrounding environment via GNNs' deep mining capabilities. The continuity equation of fluid flow is incorporated into the loss function of the convolution neural network to improve stability and performance.

## Key Results
- Approximately 50% reduction in root mean square error for reconstructing high-resolution urban wind fields
- Improvement observed for multiple wind attack angles
- Outperforms prevalent generative deep learning models in this reconstruction task

## Limitations
- Abstract does not specify the sparsity level or number of sensors required for effective reconstruction
- No details on computational complexity or training data requirements
- Limited information on generalization to diverse urban geometries or extreme wind conditions
- Potential sensitivity to sensor placement accuracy not discussed

## Relation to Thesis
This work directly relates to urban climate downscaling by providing a method to estimate high-resolution wind fields from limited sensor data, crucial for urban microclimate modeling. The integration of physics (continuity equation) with deep learning aligns with fluid dynamics-informed approaches for improving accuracy and physical consistency in downscaling models. Such techniques enhance reliability of urban wind environment assessments and support applications like natural ventilation design and pollutant dispersion modeling.