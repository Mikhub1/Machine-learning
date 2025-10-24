# Handwritten Digits Classification (Images, 8×8)

## Problem
Classify 8×8 grayscale digit images (0–9).

## Dataset
Uses `sklearn.datasets.load_digits()` (bundled with scikit‑learn; no internet required).

## Approach
- Flatten images (64 features) and scale.
- Train `SVC` with RBF kernel via `GridSearchCV`.
- Evaluate with accuracy, confusion matrix, classification report.
- Save a few predictions as image thumbnails under `artifacts/`.

## How to Run
```bash
python main.py
```

## Files
```
.
├── main.py
├── requirements.txt
└── README.md
```

## Extend
- Swap SVM for `KNeighborsClassifier` or `LogisticRegression`.
- Try PCA for dimensionality reduction and plot explained variance.
