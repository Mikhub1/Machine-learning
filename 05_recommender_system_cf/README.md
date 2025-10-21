# User-Based Collaborative Filtering (Recommender)

## Problem
Recommend items to users based on similar users' ratings (user-user CF).

## Dataset
Synthetic user-item rating matrix created in code. You can replace it with your CSV of interactions (`user_id, item_id, rating`).

## Approach
- Compute cosine similarity between users.
- For a given target user, score unseen items as a similarity-weighted average of neighbors' ratings.
- Evaluate with leave-n-out splitting and RMSE/MAE on held-out ratings.

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
- Add item-based CF or matrix factorization (SVD).
- Add shrinkage to similarity; try mean-centering or z-score normalization.
- Implement top-N recommendation with hit-rate/precision@k/recall@k.
