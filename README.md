# SVM Tumor Classifier (Benign vs Malignant)

## Overview
Trained a Support Vector Machine (SVM) to classify tumors as malignant or benign using the Breast Cancer Wisconsin (Diagnostic) dataset (available via scikit-learn).

## Dataset
- 569 samples, 30 numeric features
- Target: 0 = malignant, 1 = benign

## Method
- Train/test split (80/20) with stratification
- StandardScaler feature normalization
- SVM (RBF kernel)
- Manual hyperparameter search over C and gamma; best was:
  - C = 1
  - gamma = "scale"

## Results (test set)
- Accuracy: 0.9825
- Confusion matrix: [[41, 1], [1, 71]]
- Key takeaway: RBF kernel slightly improved performance vs linear by allowing a non-linear decision boundary.

## How to run
```bash
pip install -r requirements.txt
python src/train.py
