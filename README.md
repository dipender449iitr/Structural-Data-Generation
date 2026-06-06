# Structure-Preserving Conditional Diffusion Models for High-Fidelity Data Synthesis

---

## Table of Contents

- [Overview](#overview)
- [Motivation](#motivation)
- [Problem Statement](#problem-statement)
- [Research Gap](#research-gap)
- [Mathematical Objective](#mathematical-objective)
- [Proposed Framework](#proposed-framework)
- [Conditional Diffusion as Stochastic Transport](#conditional-diffusion-as-stochastic-transport)
- [Feature-Token Representation](#feature-token-representation)
- [Structural Fidelity Measures](#structural-fidelity-measures)
- [Unified Loss Function](#unified-loss-function)
---

## Overview

This project develops a mathematical and computational framework for **structure-preserving conditional synthetic data generation**.

The central aim is to generate high-fidelity synthetic data for a specified condition, class, subgroup, scientific state, or operational regime while preserving the structural properties of the original data distribution.

Unlike synthetic data methods that focus only on producing visually or statistically plausible samples, this framework emphasizes the preservation of relationships among variables. These relationships may include covariance, correlation, conditional dependence, and projected distributional geometry.

The proposed approach combines:

- conditional diffusion modelling;
- stochastic transport from a Gaussian reference distribution;
- feature-token operator representations;
- covariance-based structural preservation;
- correlation-based structural preservation;
- sliced Wasserstein projected geometry;
- condition-specific synthetic data generation.

The resulting framework treats synthetic data not merely as additional observations, but as a **conditional approximation of a target probability distribution** whose internal structure must remain mathematically consistent.

---

## Motivation

Data-driven modelling has become central to scientific research, engineering analysis, medical decision-making, industrial monitoring, financial modelling, environmental studies, and other structured-data domains.

However, many important data regimes are difficult to observe in sufficient quantity. Examples include:

- rare medical conditions;
- low-frequency system failures;
- extreme environmental events;
- high-risk financial states;
- sensitive population groups;
- expensive scientific experiments;
- incomplete institutional records;
- restricted-access real-world datasets.

Synthetic data generation is often used to address data scarcity. However, simply generating additional samples is not sufficient.

In structured data, the meaning of a sample does not come only from individual feature values. It also depends on the relationships among variables.

For example:

- a medical condition may be expressed through the joint behaviour of biomarkers;
- an industrial fault may appear through correlated sensor patterns;
- an environmental event may depend on interactions among temperature, pressure, humidity, and wind;
- a financial risk state may arise through dependencies among multiple indicators.

Therefore, synthetic data that reproduce individual feature ranges but distort inter-variable structure may look plausible while remaining mathematically unreliable.

This project addresses that issue by developing a **structure-preserving conditional diffusion framework**.

---

## Problem Statement

Let

```math
P_c = P(X \mid C = c)
```

denote the true conditional distribution for a condition, class, subgroup, regime, or scientific state `c`.

The task is to learn a generated conditional distribution

```math
\widehat{P}_{\theta,c}
```

such that

```math
\widehat{P}_{\theta,c} \approx P_c.
```

For structured data, closeness to the conditional distribution alone is not sufficient. The generated distribution should also preserve important structural properties of the real data.

Let

```math
S(P_c)
```

denote a family of structural functionals associated with the target conditional distribution.

The structural preservation requirement is

```math
S(\widehat{P}_{\theta,c}) \approx S(P_c).
```

Thus, the objective is not only to approximate the probability distribution, but also to preserve the mathematical structure of the data.

---

## Research Gap

Existing generative models, including GANs, VAEs, normalizing flows, and diffusion models, have shown strong performance in generating realistic samples. However, structured-data synthesis still faces several limitations.

### 1. Marginal realism without structural preservation

Generated data may reproduce individual feature distributions while failing to preserve relationships among variables.

This may result in distorted covariance patterns, weakened correlations, or incorrect dependency structures.

### 2. Weak representation of rare conditional regimes

For rare conditions, the target conditional distribution may be estimated from very few observations.

This makes the empirical conditional distribution unstable and makes conditional generation more difficult.

### 3. Lack of explicit structural constraints

Many generative objectives are based on reconstruction, likelihood, adversarial learning, or denoising loss.

These objectives improve sample realism but do not necessarily enforce preservation of structural properties.

### 4. Limited geometric evaluation

Synthetic data are often evaluated using downstream performance, reconstruction error, or visual inspection.

Such evaluations may not determine whether the geometric structure of the original distribution has been preserved.

### 5. Need for unified distributional and structural fidelity

Distributional fidelity and structural fidelity are often treated separately.

This project brings them together in one conditional diffusion and transport-based formulation.

---

## Mathematical Objective

Consider a structured dataset

```math
\mathcal{D} = \{(x_i,c_i)\}_{i=1}^{n},
```

where

```math
x_i \in \mathbb{R}^d
```

is a structured sample with `d` variables, and

```math
c_i \in \mathcal{C}
```

is a condition label.

For each condition `c`, the target object is the conditional probability measure

```math
P_c = P(X \mid C=c).
```

The generated conditional distribution is written as

```math
\widehat{P}_{\theta,c}.
```

The proposed problem is formulated as constrained conditional measure approximation:

```math
\min_{\theta}
D_0(P_c,\widehat{P}_{\theta,c})
+
\lambda
D_S(S(P_c),S(\widehat{P}_{\theta,c})).
```

Here:

- `D_0` measures distributional discrepancy;
- `D_S` measures structural discrepancy;
- `S` denotes structural functionals;
- `\lambda` controls the strength of structural preservation.

The goal is to learn synthetic data that are distributionally close and structurally faithful.

---

## Proposed Framework

The proposed framework contains four main components.

| Component | Purpose |
|---|---|
| Conditional diffusion model | Learns a denoising process conditioned on class, subgroup, or regime. |
| Stochastic transport map | Transports Gaussian noise to the target conditional distribution. |
| Feature-token representation | Represents variables as interacting tokens rather than independent scalar entries. |
| Structural fidelity losses | Enforces covariance, correlation, and projected geometric consistency. |

The framework learns a stochastic map

```math
T_{\theta,c}: \mathbb{R}^d \rightarrow \mathbb{R}^d
```

that transports a reference Gaussian distribution

```math
\gamma = \mathcal{N}(0,I)
```

to a generated conditional distribution:

```math
\widehat{P}_{\theta,c} = (T_{\theta,c})_{\#}\gamma.
```

Here, `(T_{\theta,c})_{\#}\gamma` denotes the pushforward measure of `\gamma` under the learned map.

---

## Conditional Diffusion as Stochastic Transport

The diffusion component is interpreted as a conditional stochastic transport process.

Starting from a clean standardized sample `\bar{x}`, the forward corruption process is

```math
\bar{x}_{\sigma} = \bar{x} + \sigma \xi,
\qquad
\xi \sim \mathcal{N}(0,I),
```

where `\sigma > 0` is the noise level.

The model learns a conditional denoising map

```math
D_{\theta}(\bar{x}_{\sigma},\sigma,c),
```

which estimates the clean sample from the noisy sample under condition `c`.

The denoising loss is

```math
\mathcal{L}_{\mathrm{diff}}
=
\mathbb{E}_{x,c,\sigma,\xi}
\left[
w(\sigma)
\left\|
D_{\theta}(\bar{x}_{\sigma},\sigma,c)-\bar{x}
\right\|_2^2
\right].
```

From the transport perspective, if

```math
z \sim \mathcal{N}(0,I),
```

then the generated sample is

```math
\tilde{x} = T_{\theta,c}(z).
```

The generated conditional measure is

```math
\widehat{P}_{\theta,c}
=
(T_{\theta,c})_{\#}\mathcal{N}(0,I).
```

Thus, conditional diffusion is used as a stochastic mechanism for approximating the target conditional measure.

---

## Feature-Token Representation

A structured sample

```math
x \in \mathbb{R}^d
```

contains `d` variables.

Instead of treating `x` as a flat vector, the proposed method represents it as a sequence of variable-level tokens:

```math
\mathcal{T}(x) = \{t_1,t_2,\ldots,t_d\}.
```

For a noisy standardized sample `\bar{x}_{\sigma}`, each token is defined as

```math
t_j
=
\phi_{\mathrm{tok}}(\bar{x}_{\sigma,j})
+
e_j
+
\phi_c(c)
+
\phi_{\sigma}(\sigma).
```

Here:

- `\phi_{\mathrm{tok}}` is the scalar-to-token embedding;
- `e_j` is the feature-identity embedding;
- `\phi_c(c)` is the condition embedding;
- `\phi_{\sigma}(\sigma)` is the noise-level embedding.

This formulation allows the denoising model to operate on variables as structured tokens.

Let `A_{\theta}` denote the token-interaction operator. A layer-wise representation is

```math
H^{(\ell+1)}
=
A_{\theta}^{(\ell)}
\left(
H^{(\ell)},c,\sigma
\right),
\qquad
\ell = 0,\ldots,L-1.
```

This operator-based view is useful because structured data require the model to learn relations among variables during the denoising process.

---

## Structural Fidelity Measures

Let `X_c` denote samples from the real conditional distribution and `\widehat{X}_c` denote samples from the generated conditional distribution.

The structural discrepancy is expressed through complementary functionals.

---

### 1. Covariance Functional

Covariance captures scale-dependent co-variation among variables.

The covariance discrepancy is

```math
\mathcal{L}_{\mathrm{cov}}
=
\frac{1}{d}
\left\|
\operatorname{Cov}(\widehat{X}_c)
-
\operatorname{Cov}(X_c)
\right\|_F.
```

This term penalizes mismatch between the real and generated covariance matrices.

---

### 2. Correlation Functional

Correlation captures normalized dependence and reduces sensitivity to scale.

The correlation discrepancy is

```math
\mathcal{L}_{\mathrm{corr}}
=
\frac{1}{d}
\left\|
\operatorname{Corr}(\widehat{X}_c)
-
\operatorname{Corr}(X_c)
\right\|_F.
```

This term is useful when variables have different scales or when standardized dependence patterns are important.

---

### 3. Projected Geometry Functional

Second-order structure does not fully characterize a distribution.

Therefore, projected distributional geometry is measured through sliced Wasserstein distance.

Let

```math
v_k \in \mathbb{S}^{d-1}
```

be a random unit projection direction.

The projected discrepancy is

```math
\mathcal{L}_{\mathrm{sw}}
=
\frac{1}{K}
\sum_{k=1}^{K}
W_1
\left(
\langle \widehat{X}_c,v_k\rangle,
\langle X_c,v_k\rangle
\right),
```

where `W_1` is the one-dimensional Wasserstein distance.

This term compares real and generated conditional distributions across multiple projected views.

---

## Unified Loss Function

The combined structural discrepancy is

```math
D_S
=
\lambda_{\mathrm{cov}}\mathcal{L}_{\mathrm{cov}}
+
\lambda_{\mathrm{corr}}\mathcal{L}_{\mathrm{corr}}
+
\lambda_{\mathrm{sw}}\mathcal{L}_{\mathrm{sw}}.
```

The complete objective is

```math
\mathcal{L}
=
\mathcal{L}_{\mathrm{diff}}
+
\lambda_{\mathrm{cov}}\mathcal{L}_{\mathrm{cov}}
+
\lambda_{\mathrm{corr}}\mathcal{L}_{\mathrm{corr}}
+
\lambda_{\mathrm{sw}}\mathcal{L}_{\mathrm{sw}}.
```

Equivalently, the framework can be interpreted as

```math
\min_{\theta}
\mathcal{L}_{\mathrm{diff}}(\theta)
\quad
\text{subject to}
\quad
S(\widehat{P}_{\theta,c}) \approx S(P_c).
```

Using Lagrangian relaxation gives the unified objective above.

---

