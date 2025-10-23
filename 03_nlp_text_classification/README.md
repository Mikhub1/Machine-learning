# Sentiment / Intent Text Classification (NLP)

## Problem
Classify short text into 3 classes: `positive`, `neutral`, `negative` (toy dataset included).

## Dataset
A tiny in-repo dataset defined in code (no downloads). You can replace it with your CSV/JSON easily.

## Approach
- `TfidfVectorizer` for text -> sparse features.
- `LogisticRegression` (one-vs-rest) with class weights.
- 5-fold cross-validation with `GridSearchCV`.
- Evaluation with macro-F1 and per-class metrics.
- Save vectorizer + model into `artifacts/model.joblib`.

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
- Replace the toy dataset with your labeled data (CSV).
- Add `nltk` stopword removal or bigrams.
- Try `LinearSVC` for speed on large corpora.
