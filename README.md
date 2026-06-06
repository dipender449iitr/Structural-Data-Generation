# Structure-Preserving Conditional Diffusion Models for High-Fidelity Data Synthesis

## Overview

This repository contains the research implementation of **Structure-Preserving Conditional Diffusion Models for High-Fidelity Data Synthesis**, a mathematical framework for generating synthetic structured data while preserving the intrinsic relationships present in the original distribution.

Traditional synthetic data generation methods often focus on reproducing marginal distributions but fail to preserve important structural dependencies such as covariance, correlation, and geometric relationships between variables. This project addresses that challenge through a structure-aware conditional diffusion framework.

---

## Project Motivation

Many real-world domains suffer from limited, imbalanced, expensive, or sensitive datasets:

- Healthcare and rare diseases
- Industrial fault detection
- Climate and environmental modeling
- Financial risk analysis
- Scientific experiments
- Sensitive demographic studies

Although synthetic data generation can help alleviate data scarcity, generated samples often fail to preserve the underlying structural relationships that define the true data distribution.

This project aims to generate synthetic data that are not only realistic but also **structurally faithful** to the original data.

---

## Problem Statement

Let

\[
P_c = P(X|C=c)
\]

represent the true conditional distribution corresponding to a specific condition, class, regime, or subgroup \(c\).

The objective is to learn a generated distribution

\[
\hat{P}_{\theta,c}
\]

such that

\[
\hat{P}_{\theta,c} \approx P_c
\]

while simultaneously preserving important structural properties:

\[
S(\hat{P}_{\theta,c}) \approx S(P_c)
\]

where \(S(\cdot)\) denotes structural descriptors such as:

- Covariance structure
- Correlation structure
- Projected distributional geometry

---

## Key Research Contributions

### 1. Conditional Diffusion-Based Data Synthesis

The framework employs diffusion models as stochastic transport mechanisms that map a Gaussian reference distribution to a target conditional distribution.

### 2. Structure-Preserving Learning

Unlike conventional generative approaches, structural fidelity is incorporated directly into the optimization objective.

### 3. Feature-Token Operator Representation

Structured data variables are represented as feature-level tokens, enabling the model to learn interactions among variables rather than treating data as flat vectors.

### 4. Geometric Distribution Matching

The framework uses projected Wasserstein-based measures to preserve global distributional geometry.

---

## Mathematical Framework

### Conditional Measure Learning

For each condition \(c\),

\[
\hat{P}_{\theta,c} \approx P_c
\]

while ensuring

\[
(\hat{P}_{\theta,c}, S(\hat{P}_{\theta,c}))
\approx
(P_c, S(P_c))
\]

---

### Diffusion-Based Stochastic Transport

Forward corruption process:

\[
\bar{x}_{\sigma} = \bar{x} + \sigma \xi,
\qquad
\xi \sim \mathcal{N}(0,I)
\]

Reverse denoising model:

\[
D_{\theta}(\bar{x}_{\sigma}, \sigma, c)
\]

Diffusion loss:

\[
L_{diff}
=
\mathbb{E}
\left[
w(\sigma)
\|D_{\theta}(\bar{x}_{\sigma},\sigma,c)-\bar{x}\|_2^2
\right]
\]

Generated sample:

\[
\tilde{x}=T_{\theta,c}(z),
\qquad
z \sim \mathcal{N}(0,I)
\]

Generated distribution:

\[
\hat{P}_{\theta,c}
=
(T_{\theta,c})_{\#}\mathcal{N}(0,I)
\]

---

### Feature Token Representation

Each variable is represented as a token:

\[
T(x)=\{t_1,t_2,\dots,t_d\}
\]

Token embedding:

\[
t_j
=
\phi_{tok}(\bar{x}_{\sigma,j})
+
e_j
+
\phi_c(c)
+
\phi_{\sigma}(\sigma)
\]

where:

- \(e_j\) = feature identity embedding
- \(\phi_c(c)\) = condition embedding
- \(\phi_{\sigma}(\sigma)\) = noise-level embedding

---

## Structural Fidelity Functionals

### Covariance Preservation

\[
L_{cov}
=
\frac{1}{d}
\|
Cov(X_c^g)-Cov(X_c)
\|_F
\]

---

### Correlation Preservation

\[
L_{corr}
=
\frac{1}{d}
\|
Corr(X_c^g)-Corr(X_c)
\|_F
\]

---

### Projected Geometric Preservation

Using sliced Wasserstein distance:

\[
L_{sw}
=
\frac{1}{K}
\sum_{k=1}^{K}
W_1
(
\langle X_c^g,v_k\rangle,
\langle X_c,v_k\rangle
)
\]

where:

- \(v_k\) are random projection directions
- \(W_1\) denotes the Wasserstein distance

---

## Unified Objective Function

The complete training objective is

\[
L
=
L_{diff}
+
\lambda_{cov}L_{cov}
+
\lambda_{corr}L_{corr}
+
\lambda_{sw}L_{sw}
\]

This jointly optimizes:

- Distributional fidelity
- Structural preservation
- Conditional consistency

---

## Research Gap Addressed

Current generative models suffer from:

- Marginal realism without dependency preservation
- Weak performance in rare conditional regimes
- Lack of explicit structural constraints
- Limited geometric evaluation metrics

This project introduces a unified framework that directly incorporates structural fidelity into synthetic data generation.

---

## Potential Applications

### Healthcare

- Rare disease modeling
- Medical record synthesis
- Clinical decision support

### Climate Science

- Extreme weather event simulation
- Environmental risk modeling

### Industrial Systems

- Fault detection datasets
- Sensor data augmentation

### Finance

- Rare market condition simulation
- Risk assessment

### Scientific Computing

- Expensive experiment augmentation
- Simulation-assisted learning

---

## References

1. Gong et al., *A Survey on Dataset Quality in Machine Learning*, 2023.
2. Sarker, *Data Science and Analytics*, 2021.
3. Goodfellow et al., *Generative Adversarial Nets*, NeurIPS 2014.
4. Kingma & Welling, *Auto-Encoding Variational Bayes*, ICLR 2014.
5. Ho et al., *Denoising Diffusion Probabilistic Models*, NeurIPS 2020.
6. Karras et al., *Elucidating the Design Space of Diffusion-Based Generative Models*, 2022.
7. Kotelnikov et al., *TabDDPM: Modelling Tabular Data with Diffusion Models*, ICML 2023.
8. Bonneel et al., *Sliced and Radon Wasserstein Barycenters of Measures*, 2015.
9. Villani, *Optimal Transport: Old and New*, Springer, 2009.
10. Santambrogio, *Optimal Transport for Applied Mathematicians*, 2015.

---


