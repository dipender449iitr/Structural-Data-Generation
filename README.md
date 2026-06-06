# Structure-Preserving Conditional Diffusion Models for High-Fidelity Data Synthesis

> **Anonymous Research Repository**  
> This repository presents a structure-preserving conditional diffusion framework for high-fidelity synthetic data generation.  
> All author names, institutional affiliations, investigator details, student details, acknowledgements, and administrative identifiers have been removed to maintain anonymity.

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
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Data Format](#data-format)
- [Usage](#usage)
- [Training](#training)
- [Synthetic Data Generation](#synthetic-data-generation)
- [Evaluation](#evaluation)
- [Reproducibility](#reproducibility)
- [Privacy and Anonymity](#privacy-and-anonymity)
- [Applications](#applications)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [References](#references)
- [License](#license)

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

## Repository Structure

```text
structure-preserving-conditional-diffusion/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── pyproject.toml
├── configs/
│   ├── default.yaml
│   ├── model.yaml
│   ├── training.yaml
│   └── evaluation.yaml
├── data/
│   ├── README.md
│   └── .gitkeep
├── examples/
│   ├── quickstart_train.py
│   ├── quickstart_generate.py
│   └── quickstart_evaluate.py
├── scripts/
│   ├── train.py
│   ├── generate.py
│   ├── evaluate.py
│   └── anonymize_outputs.py
├── src/
│   └── spcdm/
│       ├── __init__.py
│       ├── data/
│       │   ├── loaders.py
│       │   ├── preprocessing.py
│       │   └── validation.py
│       ├── models/
│       │   ├── diffusion.py
│       │   ├── token_operator.py
│       │   └── embeddings.py
│       ├── losses/
│       │   ├── diffusion_loss.py
│       │   ├── covariance_loss.py
│       │   ├── correlation_loss.py
│       │   └── sliced_wasserstein.py
│       ├── training/
│       │   ├── trainer.py
│       │   └── schedulers.py
│       ├── generation/
│       │   └── sampler.py
│       ├── evaluation/
│       │   ├── statistical.py
│       │   ├── structural.py
│       │   ├── downstream.py
│       │   └── privacy.py
│       └── utils/
│           ├── seed.py
│           ├── logging.py
│           └── checkpoints.py
├── tests/
│   ├── test_losses.py
│   ├── test_data_validation.py
│   ├── test_generation.py
│   └── test_evaluation.py
└── docs/
    ├── methodology.md
    ├── evaluation_protocol.md
    ├── reproducibility.md
    └── anonymity.md
```

---

## Installation

Clone the repository:

```bash
git clone <anonymous-repository-url>
cd structure-preserving-conditional-diffusion
```

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

For Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

For editable development installation:

```bash
pip install -e .
```

Suggested dependencies:

```text
python>=3.10
numpy
pandas
scipy
scikit-learn
torch
tqdm
pyyaml
matplotlib
pytest
```

---

## Data Format

The expected input is a structured tabular dataset with a condition column and numerical feature columns.

Example:

```csv
sample_id,condition,x1,x2,x3,x4
0,A,0.12,1.40,-0.31,2.10
1,A,0.18,1.21,-0.22,2.04
2,B,-0.45,0.88,0.76,1.33
3,C,0.52,-0.13,0.91,0.67
```

Recommended conventions:

- `sample_id`: anonymous sample identifier;
- `condition`: class, subgroup, regime, or state;
- `x1,...,xd`: numerical variables;
- no personal identifiers;
- no institutional identifiers;
- no private metadata;
- no author-specific or affiliation-specific labels.

A dataset metadata file may be written as:

```yaml
dataset_name: anonymous_structured_dataset
condition_column: condition
feature_columns:
  - x1
  - x2
  - x3
  - x4
standardization: zscore
train_split: 0.70
validation_split: 0.15
test_split: 0.15
```

---

## Usage

### Train the model

```bash
python scripts/train.py \
  --config configs/default.yaml \
  --data data/anonymous_dataset.csv \
  --condition-column condition \
  --output-dir outputs/anonymous_run
```

### Generate synthetic samples

```bash
python scripts/generate.py \
  --checkpoint outputs/anonymous_run/checkpoints/best.pt \
  --condition A \
  --num-samples 1000 \
  --output outputs/anonymous_run/synthetic_A.csv
```

### Evaluate generated data

```bash
python scripts/evaluate.py \
  --real data/anonymous_dataset.csv \
  --synthetic outputs/anonymous_run/synthetic_A.csv \
  --condition-column condition \
  --report outputs/anonymous_run/evaluation_report.json
```

---

## Training

The training process consists of the following steps.

### Step 1: Data validation

Check the dataset for:

- missing values;
- non-numeric columns;
- duplicate rows;
- degenerate features;
- highly imbalanced conditions;
- insufficient sample size per condition;
- potential identifier leakage.

### Step 2: Preprocessing

The recommended preprocessing steps are:

- separate feature columns and condition labels;
- standardize numerical features;
- store the scaler for inverse transformation;
- split data into training, validation, and test sets;
- preserve condition-level stratification where possible.

### Step 3: Noise corruption

For each standardized sample, draw a noise level and Gaussian perturbation:

```math
\bar{x}_{\sigma} = \bar{x} + \sigma\xi.
```

### Step 4: Token construction

Each noisy feature is represented as a token and augmented with:

- feature identity embedding;
- condition embedding;
- noise-level embedding.

### Step 5: Denoising and structural regularization

The model minimizes a combination of:

- diffusion denoising loss;
- covariance loss;
- correlation loss;
- sliced Wasserstein loss.

### Step 6: Model selection

The best checkpoint should be selected using both distributional and structural metrics.

A low denoising loss alone is not sufficient if generated samples fail to preserve structural relationships.

---

## Synthetic Data Generation

For a target condition `c`, generation proceeds as follows:

1. sample latent Gaussian noise;
2. apply the reverse conditional diffusion process;
3. generate synthetic sample `\tilde{x}`;
4. inverse-transform standardized features;
5. evaluate structure before downstream use.

Example:

```bash
python scripts/generate.py \
  --checkpoint outputs/anonymous_run/checkpoints/best.pt \
  --conditions A B C \
  --num-samples-per-condition 1000 \
  --output-dir outputs/anonymous_run/synthetic
```

---

## Evaluation

Synthetic data should be evaluated using multiple complementary criteria.

### 1. Marginal similarity

Compare real and generated feature distributions using:

- mean error;
- variance error;
- quantile error;
- Kolmogorov-Smirnov statistic;
- histogram comparison.

### 2. Covariance fidelity

Measure whether covariance structure is preserved:

```math
\left\|
\operatorname{Cov}(\widehat{X}_c)
-
\operatorname{Cov}(X_c)
\right\|_F.
```

### 3. Correlation fidelity

Measure whether normalized dependence structure is preserved:

```math
\left\|
\operatorname{Corr}(\widehat{X}_c)
-
\operatorname{Corr}(X_c)
\right\|_F.
```

### 4. Sliced Wasserstein geometry

Measure projected distributional geometry:

```math
\frac{1}{K}
\sum_{k=1}^{K}
W_1
\left(
\langle \widehat{X}_c,v_k\rangle,
\langle X_c,v_k\rangle
\right).
```

### 5. Conditional consistency

Evaluate metrics separately for each condition.

This is important because aggregate performance may hide poor performance in rare regimes.

### 6. Downstream utility

Possible downstream protocols include:

| Protocol | Purpose |
|---|---|
| Train on real, test on real | Baseline performance |
| Train on synthetic, test on real | Synthetic-data utility |
| Train on real + synthetic, test on real | Data augmentation benefit |
| Train on real, test on synthetic | Real-to-synthetic consistency |

### 7. Privacy risk

Evaluate whether synthetic data memorizes real samples using:

- nearest-neighbour analysis;
- duplicate detection;
- membership-inference checks;
- attribute-inference checks;
- extreme-value inspection.

---

## Reproducibility

A complete experiment should report:

- random seed;
- dataset preprocessing details;
- train-validation-test split;
- model architecture;
- optimizer;
- learning rate;
- batch size;
- number of epochs;
- noise schedule;
- structural loss weights;
- number of sliced Wasserstein projection directions;
- checkpoint selection rule;
- hardware details;
- training time;
- evaluation metrics.

Example environment check:

```bash
python - <<'PY'
import platform
import sys

try:
    import torch
    torch_version = torch.__version__
    cuda_available = torch.cuda.is_available()
except Exception:
    torch_version = "not installed"
    cuda_available = False

print("Python:", sys.version)
print("Platform:", platform.platform())
print("PyTorch:", torch_version)
print("CUDA available:", cuda_available)
PY
```

---

## Privacy and Anonymity

This repository is prepared for anonymous review and privacy-aware public release.

The following information should not appear in the repository:

- author names;
- investigator names;
- student names;
- institutional affiliations;
- department names;
- personal email addresses;
- acknowledgement text revealing identity;
- grant numbers if they compromise anonymity;
- internal dataset names;
- private data links;
- metadata revealing ownership or location.

Before release, run:

```bash
grep -RniE "author|affiliation|university|institute|department|email|acknowledg|grant|funding|principal investigator|student|supervisor" . \
  --exclude-dir=.git \
  --exclude-dir=.venv
```

Also inspect:

- notebooks;
- figures;
- logs;
- checkpoint metadata;
- output reports;
- generated synthetic data files;
- configuration files.

---

## Applications

The framework can be used in domains where real data are limited, sensitive, expensive, or structurally dependent.

| Domain | Potential Use |
|---|---|
| Healthcare | Rare-condition synthesis, biomarker modelling, privacy-aware sharing |
| Climate science | Extreme-event generation and scenario analysis |
| Industrial monitoring | Fault-state augmentation and sensor-pattern preservation |
| Finance | Risk-regime simulation and stress-state modelling |
| Materials science | Limited-experiment data synthesis |
| Engineering systems | Reliability and failure-regime modelling |
| Scientific machine learning | Structure-aware data augmentation |

---

## Limitations

The framework has several limitations.

1. Covariance and correlation do not fully capture nonlinear dependence.
2. Rare-condition estimation may remain unstable when very few real samples are available.
3. Sliced Wasserstein estimates depend on the number of projection directions.
4. Structural fidelity does not automatically guarantee privacy.
5. Domain-specific constraints may be needed for scientific validity.
6. Additional structural losses may increase computational cost.
7. Evaluation must be condition-specific, not only aggregate.

---

## Future Work

Possible future extensions include:

- nonlinear dependency preservation;
- graph-based feature-interaction modelling;
- copula-based structural constraints;
- differential privacy integration;
- uncertainty quantification for rare conditions;
- domain-specific physical or scientific constraints;
- theoretical consistency analysis;
- benchmarking against GANs, VAEs, flows, and tabular diffusion models;
- extension to mixed numerical-categorical structured data.

---

## References

1. Y. Gong, G. Liu, Y. Xue, R. Li, and L. Meng, “A survey on dataset quality in machine learning,” *Information and Software Technology*, vol. 162, p. 107268, 2023.

2. I. H. Sarker, “Data science and analytics: an overview from data-driven smart computing, decision-making and applications perspective,” *SN Computer Science*, vol. 2, no. 5, p. 377, 2021.

3. I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio, “Generative adversarial nets,” in *Advances in Neural Information Processing Systems*, 2014.

4. D. P. Kingma and M. Welling, “Auto-Encoding Variational Bayes,” in *International Conference on Learning Representations*, 2014.

5. J. Ho, A. Jain, and P. Abbeel, “Denoising Diffusion Probabilistic Models,” in *Advances in Neural Information Processing Systems*, 2020.

6. T. Karras, M. Aittala, T. Aila, and S. Laine, “Elucidating the Design Space of Diffusion-Based Generative Models,” in *Advances in Neural Information Processing Systems*, 2022.

7. A. Kotelnikov, D. Baranchuk, I. Rubachev, and A. Babenko, “TabDDPM: Modelling Tabular Data with Diffusion Models,” in *Proceedings of the 40th International Conference on Machine Learning*, 2023.

8. N. Bonneel, J. Rabin, G. Peyre, and H. Pfister, “Sliced and Radon Wasserstein Barycenters of Measures,” *Journal of Mathematical Imaging and Vision*, vol. 51, pp. 22–45, 2015.

9. C. Villani, *Optimal Transport: Old and New*. Springer, 2009.

10. F. Santambrogio, *Optimal Transport for Applied Mathematicians*. Birkhauser, 2015.

---

## License

Add an appropriate open-source license before public release.

Recommended options include:

- MIT License;
- Apache License 2.0;
- BSD 3-Clause License.

The selected license should be compatible with dependencies, dataset restrictions, and intended research-use policy.

---

## Citation

For anonymous review, use:

@misc{anonymous2026spcdm,
  title  = {Structure-Preserving Conditional Diffusion Models for High-Fidelity Data Synthesis},
  author = {Anonymous Author(s)},
  year   = {2026},
  note   = {Anonymous research repository}
}
