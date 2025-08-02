# Normal-Guided Pointcloud Denoiser

This repository contains the implementation of my Master's thesis on improving pointcloud denoising using normal-guided update strategies and feature-aware point classification. The pipeline builds on the work of Yadav et al. (2018) and introduces enhancements for better reconstruction, faster convergence, and increased robustness.

---

## Project Overview

The goal of this project is to denoise 3D point clouds by updating their positions using local geometric information, such as estimated normals and feature classification. Key improvements over the original pipeline include:

- Normal tensor voting for improved feature detection
- Improved updating strategy
- Tunable diffusion speeds for controlled updates
- Tuned parameters requiring fewer iterations for convergence

You can find the complete thesis in the [`/thesis`](./thesis) folder.

---

## Features

- Robust feature classification using normal voting tensors
- Improved neighborhood weighting based on normals and relative positions
- Evaluation with both Chamfer Distance (CD) and Single-sided Chamfer Distance (sCD)  
- Visualizations for interpretability and debugging  
- Modular, readable codebase with test and evaluation scripts
