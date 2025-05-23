��#   R e s e a r c h 
# Synthetic Data Generation for Orthopaedic Oncology Research

## Abstract

### Background/Rationale
Bone and soft tissue sarcomas are rare (< 1% of cancers), leaving most single centre studies under powered and constrained by privacy rules that impede data sharing. Synthetic tabular data could enlarge cohorts while maintaining confidentiality, but the fidelity of competing generators in orthopaedic oncology is unknown.

### Research Questions
1. Can three state of the art generators—Conditional Tabular GAN (CTGAN), Tabular VAE (TVAE) and CTAB GAN Plus—reproduce the univariate distributions found in a national sarcoma registry?
2. How well do they preserve key inter variable relationships (year of diagnosis ↔ survival, histology ↔ treatment and site)?
3. Which model provides the most suitable synthetic cohort for downstream survival or health services analyses?

### Methods
A retrospective SEER*18 dataset of 39,461 sarcoma patients (13 variables) served as ground truth. Each algorithm generated seven synthetic cohorts of identical size. Fidelity was evaluated as:
- Structural validity
- Univariate similarity (mean Kolmogorov–Smirnov statistic for numeric variables and χ² divergence for categorical variables)
- Relationship preservation (ΔSpearman ρ, ΔCramér V, and Δmutual information)

Metrics were min–max normalised and averaged to yield a composite score (0–1).

### Results
- All 21 synthetic datasets were structurally valid (0/513,000 impossible values)
- CTGAN6 achieved the highest composite score (0.81) with:
  - Lowest KS distance for survival months (0.020)
  - Smallest MI gap for complex interactions (0.042)
- CTAB GAN Plus best matched categorical frequencies (mean χ² based similarity = 1.00) but showed larger numeric drift (KS 0.061)
- TVAE6 preserved the year to survival correlation almost perfectly (Δρ 0.005) yet ranked eighth overall (score 0.50) owing to categorical distortion

### Conclusions
For synthetic expansion of orthopaedic oncology registries:
- CTGAN is preferred for survival or prognostic modelling
- CTAB GAN Plus is optimal when categorical detail—such as histology specific treatment rates—is paramount
- TVAE requires further tuning before use in rare sarcoma studies

**Level of Evidence**: III (Diagnostic study)

## Repository Structure

This repository contains the following components:

### 1. Datasets
- `Total synthetic dataset/`: Contains all generated synthetic datasets
  - Real dataset (SEER*18)
  - CTGAN synthetic datasets (1-7)
  - TVAE synthetic datasets (1-7)
  - CTABGAN synthetic datasets (1-7)

### 2. Models
- `models/`: Contains trained models for each generator
  - CTGAN models
  - TVAE models
  - CTABGAN models

### 3. Scripts
- `scripts/`: Contains all analysis and evaluation scripts
  - Correlation analysis
  - Distribution analysis
  - Model training scripts
  - Evaluation metrics

## Usage

The repository contains all necessary code and data to:
1. Generate synthetic datasets using different models
2. Evaluate the quality of synthetic data
3. Compare different synthetic data generation approaches
4. Analyze the preservation of key relationships in the data

## Citation

If you use this work in your research, please cite our paper (citation to be added upon publication).
 
 
