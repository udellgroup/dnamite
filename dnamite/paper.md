---
title: 'dnamite: A Python Package for Neural Additive Models'
tags:
  - Python
  - Additive Models
  - Explainable AI
  - Survival Analysis
authors:
  - name: Mike Van Ness
    orcid: 0000-0002-7829-8164
    affiliation: 1
  - name: Madeleine Udell
    orcid: 0000-0002-3985-915X
    affiliation: 1
affiliations:
 - name: Stanford University, United States
   index: 1
date: October 5, 2025
bibliography: paper.bib
---

# Summary

Most machine learning methods work as black boxes: they make accurate predictions but offer no insight into why they reached their conclusions. 
This lack of transparency creates problems in high-stakes fields like healthcare, where practitioners need to understand and justify decisions, not just accept algorithmic outputs. 
dnamite is a Python package that addresses this challenge by implementing Neural Additive Models (NAMs), a class of models that maintain interpretability while achieving strong predictive performance. 
dnamite allows users to see exactly how each variable influences predictions through visual plots, making it possible to validate that models behave sensibly and align with domain knowledge. 
Users of dnamite can train NAMs for regression, binary classification, and survival analysis, all common prediction problems in healthcare and other applications.
Further, dnamite models can automatically identify the most important variables from high-dimensional datasets, handle common data challenges like missing values, and integrate seamlessly with scikit-learn pipelines. 
By making interpretable predictive modeling accessible with minimal code, dnamite helps practitioners incorporate transparency into their workflows without sacrificing predictive accuracy.

# Statement of Need

Interpretable machine learning is essential for high-stakes applications where understanding model decisions is as important as prediction accuracy. 
In healthcare, clinicians need to validate that risk models align with medical knowledge before trusting them with patient care [@Caruana:2015]. 
In social science and policy research, practitioners must justify algorithmic decisions and ensure models do not perpetuate biases hidden in training data [@Bolukbasi:2016].

Additive models have long been valued in statistics for their ability to capture complex nonlinear relationships while remaining interpretable.
Unlike linear models, which use only one coefficient per features, additive models allow for complex, nonlinear functions for each feature that can be customized for different use cases.
Unlike black-box machine learning models, though, additive models maintain additive structure so that the model is transparent about the role of each feature.
The statistical community has mostly focused on building additive models using splines, with mature packages like mgcv [@Wood:2001] and pyGAM [@Serven:2018].
However, such packages face limitations when scaling to large datasets, handling high-dimensional feature selection, or implementing non-parametric survival analysis methods; see Table 1.

More recently, the machine learning literature has explored additive models, mostly using neural networks to create so-called Neural Additive Models (NAMs) [@Agarwal:2021].
These NAMs can leverage the flexibility of modern neural network training to handle more complex use-cases.
However, existing NAM implementations remain fragmented and incomplete compared to mature statistical implementations.
The Python package PiML [@Sudjianto:2023] provides some NAM architectures but lacks support for survival analysis and offers only post-hoc feature selection. 
The R package neuralGAM [@Ortega:2023] implements NAMs but without feature selection or survival analysis capabilities. 
Further, no existing package handles missing values natively.
dnamite fills this gap by providing a mature implementation of NAMs that supports regression, classification, and nonparametric survival analysis with integrated feature selection capabilities and native handling of missing values.
Importantly, dnamite's models are scikit-learn estimators that can be fit with minimum code, making advanced interpretable modeling accessible to domain scientists without deep learning expertise.

<div align="center">

Table 1: Comparison of additive model packages

| Package        | Language | Models           | Feature Selection     | Survival Analysis | Missing Values |
|----------------|----------|------------------|-----------------------|------------------|----------------|
| gam            | R        | Splines          | ✗                     | Cox              | ✗              |
| mgcv           | R        | Splines          | ✗                     | Cox/AFT          | ✗              |
| mboost         | R        | Splines/Trees    | Boosting              | Cox/AFT          | ✗              |
| gamlss         | R        | Splines          | ✗                     | Parametric       | ✗              |
| gamboostLSS    | R        | Splines/Trees    | Boosting              | Parametric       | ✗              |
| bamlss         | R        | Splines          | Lasso (linear)        | Parametric       | ✗              |
| cgam           | R        | Splines          | ✗                     | ✗                | ✗              |
| spikeSlabGAM   | R        | Splines          | Spike-and-Slab        | ✗                | ✗              |
| neuralGAM      | R        | Neural Nets      | ✗                     | ✗                | ✗              |
| pyGAM          | Python   | Splines          | ✗                     | ✗                | ✗              |
| interpretml    | Python   | Trees            | Interactions only     | ✗                | ✓              |
| PiML           | Python   | Trees/Neural Nets| Post-Hoc              | ✗                | ✗              |
| dnamite        | Python   | Neural Nets      | Learnable Gates       | Nonparametric    | ✓              |

</div>

# Key Features and Design Principles

dnamite is built around several core design principles that distinguish it from existing additive modeling packages and make it particularly suitable for real-world machine learning applications.

**Scikit-learn compatibility**: All dnamite models are scikit-learn estimators, enabling seamless integration with the broader Python machine learning ecosystem. 
Users can train models, generate predictions, and create visualizations through simple function calls on pandas DataFrames, without writing custom training loops or data preprocessing pipelines. 
This design makes dnamite accessible to practitioners who may not be familiar with deep learning frameworks, while still leveraging PyTorch under the hood for computational efficiency on both CPU and GPU hardware.

**Native handling of categorical features and missing values**: The DNAMite architecture discretizes all features into bins and learns embeddings for each bin, allowing the model to handle categorical variables, continuous variables, and missing values uniformly without preprocessing. 
Critically, this approach allows dnamite to learn separate effects for missing values in each feature, enabling users to assess whether missingness patterns are informative—a capability unique among NAM implementations.
Users can visualize stratified feature importances that separate the contribution of observed versus missing values, providing insights impossible to obtain with traditional imputation approaches.

**Integrated feature selection**: dnamite implements learnable gates [@Ibrahim:2024] that perform feature selection during model training rather than as a post-hoc step. 
Each feature is associated with a continuous gate parameter that is jointly optimized with the model, with a sparsity-inducing regularizer encouraging gates to converge to zero for uninformative features. 
This approach is more principled and efficient than heuristic selection methods, and allows users to control the sparsity-accuracy tradeoff through a single regularization parameter, similar to lasso regression.

**Controllable shape function complexity**: Many NAM architectures produce overly smooth shape functions that may miss important nonlinear patterns. 
dnamite exposes the kernel smoothing parameter that controls shape function smoothness, allowing users to tune models based on their domain knowledge and interpretability needs. 
Users can also enforce monotonicity constraints on individual features when prior knowledge dictates that relationships should only increase or decrease, further enhancing model transparency and trustworthiness.

**Nonparametric survival analysis**: Unlike existing additive model packages that rely on Cox proportional hazards or parametric distributional assumptions for survival analysis, dnamite directly estimates conditional survival curves at user-specified evaluation times. 
This nonparametric approach is more flexible and produces feature importances and shape functions that describe contributions to survival probability at specific time points—interpretations that are more clinically meaningful than contributions to abstract hazard ratios or distribution parameters.

# Acknowledgements

# References