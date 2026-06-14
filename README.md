# Wine Quality Classification

> A supervised machine learning project classifying red and white wine quality from physicochemical properties, exploring both binary and multi-class formulations.

[![Language](https://img.shields.io/badge/Language-Python%203.x-blue?style=flat-square)](https://www.python.org/)
[![Domain](https://img.shields.io/badge/Domain-Classification%20%7C%20Food%20Science-green?style=flat-square)]()
[![Dataset](https://img.shields.io/badge/Dataset-UCI%20Wine%20Quality-orange?style=flat-square)](https://archive.ics.uci.edu/ml/datasets/wine+quality)

---

## Overview

This project applies supervised machine learning to classify **wine quality** (rated 3–9 by human tasters) from 11 physicochemical properties. The dataset covers both **red wine** (1,599 samples) and **white wine** (4,898 samples) from the Vinho Verde wine region of Portugal. Two classification formulations are explored:

- **Multi-class:** Predict exact quality score (3–9)
- **Binary:** Predict high quality (score ≥ 7) vs. low quality (score < 7)

The binary formulation is more practical for production quality control — classifying a wine as "acceptable" or "premium."

---

## Dataset

| Feature | Description | Unit |
|---|---|---|
| Fixed acidity | Tartaric acid content | g/dm³ |
| Volatile acidity | Acetic acid (vinegar taste) | g/dm³ |
| Citric acid | Freshness and flavor enhancer | g/dm³ |
| Residual sugar | Sugar remaining after fermentation | g/dm³ |
| Chlorides | Salt content | g/dm³ |
| Free sulfur dioxide | Free SO₂ (antimicrobial) | mg/dm³ |
| Total sulfur dioxide | Bound + free SO₂ | mg/dm³ |
| Density | Wine density | g/cm³ |
| pH | Acidity level | 0–14 |
| Sulphates | Wine preservative / antimicrobial | g/dm³ |
| Alcohol | Alcohol percentage | % vol |
| **Quality** (target) | Human taster score | 3–9 |

---

## Methodology

### 1. Exploratory Data Analysis
- Distribution plots for all 11 features (histograms, box plots)
- **Pearson correlation heatmap** reveals:
  - **Alcohol** is the strongest positive predictor of quality (r ≈ +0.48 for red wine)
  - **Volatile acidity** is the strongest negative predictor (r ≈ −0.39)
  - Density and residual sugar are highly correlated (r ≈ +0.84 for white wine)
- Class imbalance: Quality scores 5–6 dominate; scores 3, 4, 8, 9 are rare

### 2. Feature Engineering
- Binary target creation: `quality_binary = 1 if quality >= 7 else 0`
- MinMax scaling of all 11 features to [0, 1] range
- Train/test split: 80/20 stratified by quality label

### 3. Models

| Model | Notes |
|---|---|
| Logistic Regression | Baseline; interpretable feature weights |
| Random Forest | Handles non-linearity; provides feature importance |
| Gradient Boosting | Best performance on imbalanced data |
| SVM (RBF kernel) | Strong for high-dimensional feature space |

### 4. Evaluation
- **Accuracy, Precision, Recall, F1-score** per class
- **ROC-AUC** for binary classification
- **Confusion matrix** for multi-class formulation
- Cross-validation (5-fold stratified) to prevent overfitting on imbalanced classes

---

## Key Findings

- **Alcohol content** is the single most predictive feature across both red and white wine
- **Volatile acidity** is the strongest negative predictor — high acetic acid = lower perceived quality
- The multi-class formulation suffers from extreme class imbalance (quality 3 and 9 have <50 samples)
- Binary classification achieves **>85% accuracy** on the hold-out test set with Random Forest
- White wine quality is more strongly predicted by **residual sugar and density** than red wine

---

## Getting Started

```bash
git clone https://github.com/tamer017/wine_quality.git
cd wine_quality
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook
```

---

## Skills Demonstrated

`Supervised Learning` `Binary & Multi-class Classification` `Feature Correlation Analysis` `Class Imbalance` `Random Forest` `Gradient Boosting` `ROC-AUC` `Scikit-learn` `Pandas` `Seaborn` `Python`
