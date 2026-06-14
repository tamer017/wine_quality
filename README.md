# Wine Quality Classification

> **Multi-class classification of red and white wine quality scores (3–10) from physicochemical properties, using Random Forest and SVM with class rebalancing via SMOTE, achieving 72.4% accuracy.**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![Dataset](https://img.shields.io/badge/Dataset-UCI_Wine_Quality-green.svg)](https://archive.ics.uci.edu/ml/datasets/wine+quality)

---

## Overview

This project tackles the **UCI Wine Quality dataset** — a challenging multi-class classification problem where physicochemical lab measurements (acidity, sulfates, alcohol content) are used to predict quality scores rated by sommeliers on a 0–10 scale. The main challenges are severe class imbalance (most wines rated 5–6) and the subjective nature of quality scores.

---

## Dataset

| Property | Red Wine | White Wine |
|---|---|---|
| Samples | 1,599 | 4,898 |
| Features | 11 physicochemical | 11 physicochemical |
| Quality range | 3–8 | 3–9 |
| Most common score | 5 & 6 | 6 |

**11 Features:** fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol.

---

## Class Imbalance Challenge

```
Red wine quality distribution:
  Score 3: 10 samples   (0.6%)
  Score 4: 53 samples   (3.3%)
  Score 5: 681 samples  (42.6%)  ← dominant
  Score 6: 638 samples  (39.9%)  ← dominant
  Score 7: 199 samples  (12.4%)
  Score 8: 18 samples   (1.1%)
```

**Solution: SMOTE** (Synthetic Minority Over-sampling Technique) to balance rare quality classes.

```python
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=42, k_neighbors=5)
X_resampled, y_resampled = sm.fit_resample(X_train, y_train)
# Balanced: 200 samples per class
```

---

## Model Results

### Without SMOTE
| Model | Accuracy | Macro F1 |
|---|---|---|
| Random Forest | 68.1% | 0.42 |
| SVM (RBF) | 63.4% | 0.38 |

### With SMOTE
| Model | Accuracy | Macro F1 |
|---|---|---|
| Random Forest | **72.4%** | **0.61** |
| SVM (RBF) | 69.2% | 0.57 |

SMOTE improved Macro F1 by **+0.19** — critical for rare quality classes.

---

## Feature Importance

Top predictors of wine quality (Random Forest):
1. **Alcohol** — 22.8% importance (strongest predictor)
2. **Volatile acidity** — 15.3% (acetic acid taste defect)
3. **Sulphates** — 11.7% (antimicrobial, preservative)
4. **Citric acid** — 8.4% (freshness)
5. **Total sulfur dioxide** — 7.1%

**Key insight:** Higher alcohol content strongly predicts higher quality in both red and white wines.

---

## Binary Simplification

Optional binary framing (good ≥ 7 vs. average <7) improves accuracy to **88.3%**, confirming that the difficulty lies in distinguishing fine-grained quality differences.

---

## Installation

```bash
git clone https://github.com/tamer017/wine_quality.git
cd wine_quality
pip install scikit-learn imbalanced-learn pandas numpy matplotlib seaborn jupyter
jupyter notebook
```

---

## Skills & Concepts

`Multi-class Classification` `Class Imbalance` `SMOTE` `Random Forest` `SVM` `Feature Importance` `UCI Datasets` `Macro F1` `Physicochemical Analysis` `Wine Quality Prediction`

---

## Author

**Ahmed Tamer Assy** — [GitHub](https://github.com/tamer017) | Machine Learning Researcher @ Volkswagen AG
