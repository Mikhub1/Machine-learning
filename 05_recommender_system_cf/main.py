import os, numpy as np, pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_absolute_error, mean_squared_error

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)

def generate_ratings(n_users=50, n_items=120, density=0.1):
    # Create a low-rank ground truth preference matrix
    U = rng.normal(0, 1, size=(n_users, 5))
    V = rng.normal(0, 1, size=(n_items, 5))
    M = U @ V.T
    M = 3 + (M - M.mean()) / M.std()  # center around ~3
    M = np.clip(M, 1, 5)
    # Mask to simulate sparsity
    mask = rng.uniform(0, 1, size=(n_users, n_items)) < density
    ratings = np.where(mask, np.round(M*2)/2, np.nan)  # half-star increments
    return ratings

def train_test_split_matrix(R, holdout_per_user=5):
    R_train = R.copy()
    test_indices = []
    for u in range(R.shape[0]):
        idx = np.where(~np.isnan(R[u]))[0]
        if len(idx) <= holdout_per_user:
            continue
        test_items = rng.choice(idx, size=holdout_per_user, replace=False)
        for it in test_items:
            R_train[u, it] = np.nan
        test_indices.append((u, test_items))
    return R_train, test_indices

def predict_ratings_user_based(R_train, k=10, shrink=10.0):
    # Replace NaN with zeros for similarity computation, but keep a mask
    nan_mask = np.isnan(R_train)
    R_filled = np.nan_to_num(R_train, nan=0.0)
    # mean-center per user (only on observed entries)
    user_means = np.nanmean(R_train, axis=1)
    R_centered = (R_train - user_means[:, None])
    R_centered_filled = np.nan_to_num(R_centered, nan=0.0)
    # user-user cosine similarity on centered ratings
    S = cosine_similarity(R_centered_filled)
    # apply shrinkage: S' = (n/(n+shrink)) * S, where n is # of co-rated items
    co_counts = (~nan_mask).astype(int) @ (~nan_mask).astype(int).T
    S_shrunk = (co_counts / (co_counts + shrink)) * S
    # Predict: r̂_ui = μ_u + sum_v S_uv * (r_vi - μ_v) / sum |S_uv|
    R_hat = np.zeros_like(R_filled)
    for u in range(R_train.shape[0]):
        weights = S_shrunk[u].copy()
        np.fill_diagonal(S_shrunk, 0.0)  # ensure no self-sim
        numer = (weights[:, None] * (R_centered_filled)).sum(axis=0)
        denom = np.abs(weights).sum() + 1e-8
        R_hat[u] = user_means[u] + numer / denom
    R_hat = np.clip(R_hat, 1, 5)
    return R_hat

def evaluate(R_true, R_train, test_indices, R_hat):
    y_true, y_pred = [], []
    for u, items in test_indices:
        for it in items:
            true = R_true[u, it]
            pred = R_hat[u, it]
            if not np.isnan(true):
                y_true.append(true)
                y_pred.append(pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    print(f"MAE={mae:.3f} RMSE={rmse:.3f} on held-out ratings, n={len(y_true)}")

def main():
    os.makedirs("artifacts", exist_ok=True)
    R = generate_ratings(n_users=80, n_items=200, density=0.08)
    R_train, test_indices = train_test_split_matrix(R, holdout_per_user=5)
    R_hat = predict_ratings_user_based(R_train, k=15, shrink=20.0)
    evaluate(R, R_train, test_indices, R_hat)
    np.save("artifacts/R_true.npy", R)
    np.save("artifacts/R_hat.npy", R_hat)
    print("Saved predictions to artifacts/.")

if __name__ == "__main__":
    main()
