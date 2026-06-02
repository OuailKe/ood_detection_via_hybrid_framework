# OpenOOD Modification: Hybrid OOD Detection Framework

## Overview
This directory contains custom modifications to the [OpenOOD](https://github.com/JianingZhu/OpenOOD) codebase to support a novel, two-stage hybrid Out-of-Distribution (OOD) detection pipeline. The framework is specifically designed for autonomous driving environments, utilizing a Vision Transformer (ViT) backbone to process high-dimensional semantic representations and identify both Near and Far OOD hazards.

## Architecture & Methodology
The pipeline introduces a dual-gate decision mechanism designed to capture both predictive uncertainty and semantic drift without altering the internal architecture of the ViT backbone.

### Stage 1: Logit-Based Gating
The first stage evaluates the predictive confidence of the 13-class BDD100K classifier. Based on configuration, this gate utilizes either **Maximum Softmax Probability (MSP)** or the **Free Energy Score** to filter out easily detectable anomalies based on raw network logits.

### Stage 2: Semantic Mahalanobis Scoring
Images that pass Stage 1 are subjected to a rigorous semantic evaluation. 
1. **Feature Extraction:** A forward pre-hook is registered on the linear classification head. This intercepts the 768-dimensional CLS token representation ($\mathbf{z}$) immediately before the linear projection is applied, capturing the raw semantic representation with negligible computational overhead.
2. **Centroid & Covariance Estimation:** Class-conditional means ($\boldsymbol{\mu}_c$) and a shared tied precision matrix ($\boldsymbol{\Sigma}^{-1}$) are fitted on the In-Distribution (ID) training features.
3. **Distance Calculation:** At inference, the semantic OOD score is computed via Mahalanobis distance:
   
   $$d_M(\mathbf{z}) = \min_{c} (\mathbf{z} - \boldsymbol{\mu}_c)^\top \boldsymbol{\Sigma}^{-1} (\mathbf{z} - \boldsymbol{\mu}_c)$$
   
   *Note: To optimize batched inference and prevent degenerate cross-sample matrix products, the per-sample distance computation is highly vectorized using PyTorch Einstein summation (`torch.einsum`).*

4. **Decision Threshold:** An image is rejected as semantically OOD if $d_M(\mathbf{z}) \geq \tau_2$, where $\tau_2$ is strictly calibrated at the 95th percentile of the ID validation set.

## Datasets
The framework is evaluated on complex driving scenes. The model is trained on BDD100K (In-Distribution), and evaluated against various semantic shifts.

| Dataset | Split | Samples | Type | OOD category |
| :--- | :--- | :--- | :--- | :--- |
| BDD100K | val | 10,000 | ID | — |
| LostAndFound | test | 1,203 | OOD | Near |
| StreetHazards | test | 1,500 | OOD | Far |
| SMIYC | test | 1,500 | OOD | Far |

## Evaluation Metrics
The framework's robustness is measured using standard threshold-independent and threshold-dependent OOD metrics:
* **AUROC:** Measures the probability that an OOD sample is assigned a higher anomaly score than an ID sample.
* **FPR@95:** Measures the False Positive Rate when the True Positive Rate is locked at 95% ($\tau_2$).

## Key Modifications to OpenOOD
* **`openood/evaluators/`**: Added custom evaluation logic for the two-stage gating mechanism.
* **`openood/postprocessors/`**: Implemented `mahalanobis_hybrid_postprocessor.py` containing the forward pre-hook extraction and `einsum` optimized matrix operations.
* **`configs/`**: Added YAML configurations for the BDD100K ViT setup and hybrid pipeline hyperparameters.

## Acknowledgments 
This modification is part of a broader master's dissertation regarding intelligent systems and OOD detection, developed during an internship at the University of Limerick.
