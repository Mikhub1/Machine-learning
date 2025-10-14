import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, RepeatedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------
# Helpers
# -----------------------------
def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def behavior_label(train_rmse, val_rmse, overfit_gap=0.10, underfit_floor=None):
    if underfit_floor is not None and train_rmse > underfit_floor and val_rmse > underfit_floor:
        return "underfitting"
    if val_rmse >= train_rmse * (1 + overfit_gap):
        return "overfitting"
    return "good generalization"

def inv_t(vec_1d, scaler_y):
    return scaler_y.inverse_transform(np.asarray(vec_1d).reshape(-1, 1))

def param_cols(p):
    keep = ["kernel","C","gamma","degree","coef0","epsilon","tol","max_iter"]
    return {k: p.get(k, None) for k in keep}

def cv_once_kfold(params, X, y_vec, k, scaler_y):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    tr_rmses, va_rmses = [], []

    p = params.copy()
    if p.get("max_iter", -1) in (1000, 5000):
        p["max_iter"] = -1
    if p.get("kernel") in ("rbf", "poly", "sigmoid"):
        p.setdefault("cache_size", 2000)

    for tr_idx, va_idx in kf.split(X):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y_vec[tr_idx], y_vec[va_idx]
        reg = SVR(**p).fit(X_tr, y_tr)

        y_tr_hat = inv_t(reg.predict(X_tr), scaler_y)
        y_va_hat = inv_t(reg.predict(X_va), scaler_y)
        y_tr_act = inv_t(y_tr, scaler_y)
        y_va_act = inv_t(y_va, scaler_y)

        tr_rmses.append(rmse(y_tr_act, y_tr_hat))
        va_rmses.append(rmse(y_va_act, y_va_hat))

    return float(np.mean(tr_rmses)), float(np.mean(va_rmses)), float(np.std(va_rmses))

def cv_once_repeated(params, X, y_vec, k, reps, scaler_y):
    rkf = RepeatedKFold(n_splits=k, n_repeats=reps, random_state=42)
    tr_rmses, va_rmses = [], []

    p = params.copy()
    if p.get("max_iter", -1) in (1000, 5000):
        p["max_iter"] = -1
    if p.get("kernel") in ("rbf", "poly", "sigmoid"):
        p.setdefault("cache_size", 2000)

    for tr_idx, va_idx in rkf.split(X):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y_vec[tr_idx], y_vec[va_idx]
        reg = SVR(**p).fit(X_tr, y_tr)

        y_tr_hat = inv_t(reg.predict(X_tr), scaler_y)
        y_va_hat = inv_t(reg.predict(X_va), scaler_y)
        y_tr_act = inv_t(y_tr, scaler_y)
        y_va_act = inv_t(y_va, scaler_y)

        tr_rmses.append(rmse(y_tr_act, y_tr_hat))
        va_rmses.append(rmse(y_va_act, y_va_hat))

    return float(np.mean(tr_rmses)), float(np.mean(va_rmses)), float(np.std(va_rmses))

# -----------------------------
# Flags
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--train", action="store_true", help="Run training, Table2 sweep, and CV.")
parser.add_argument("--test",  action="store_true", help="Run final test on best config trained on train+val.")
args = parser.parse_args()
RUN_TRAIN = args.train or (not args.train and not args.test)   # default: run both
RUN_TEST  = args.test  or (not args.train and not args.test)

# -----------------------------
# Load & preprocess
# -----------------------------
cwd = Path.cwd()
csv_file = cwd / "car_prices.csv"
df = pd.read_csv(csv_file)

X = df.iloc[:, 0:10]
y = df.iloc[:, [10]]  # 2D for y-scaler

categorical_features = ["fuel type", "seller type", "transmission"]
numerical_features = [c for c in X.columns if c not in categorical_features]

# Scale numeric
scaler_X = StandardScaler()
X_num_scaled = pd.DataFrame(
    scaler_X.fit_transform(X[numerical_features]),
    columns=numerical_features,
    index=df.index
)

# One-hot encode categoricals (use sparse=False for broader sklearn compatibility)
encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
X_cat = pd.DataFrame(
    encoder.fit_transform(X[categorical_features]),
    columns=encoder.get_feature_names_out(categorical_features),
    index=df.index
)

# Combine
X_final = pd.concat([X_num_scaled, X_cat], axis=1)

# Scale target
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y).ravel()

# 70/15/15 split
X_train, X_temp, y_train, y_temp = train_test_split(
    X_final, y_scaled, test_size=0.30, random_state=42, shuffle=True
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, shuffle=True
)

X_train_full = pd.concat([X_train, X_val], axis=0)
y_train_full = np.concatenate([y_train, y_val])

# -----------------------------
# Table 2 grid (50+ configs)
# -----------------------------
TABLE2 = [
    # linear
    dict(kernel="linear", C=0.1,  epsilon=0.05, tol=1e-3,  max_iter=-1),
    dict(kernel="linear", C=0.1,  epsilon=0.10, tol=1e-4,  max_iter=-1),
    dict(kernel="linear", C=0.5,  epsilon=0.05, tol=1e-3,  max_iter=-1),
    dict(kernel="linear", C=0.5,  epsilon=0.10, tol=1e-4,  max_iter=-1),
    dict(kernel="linear", C=1.0,  epsilon=0.05, tol=1e-4,  max_iter=5000),
    dict(kernel="linear", C=1.0,  epsilon=0.10, tol=1e-4,  max_iter=5000),
    dict(kernel="linear", C=2.0,  epsilon=0.05, tol=1e-4,  max_iter=-1),
    dict(kernel="linear", C=2.0,  epsilon=0.10, tol=1e-4,  max_iter=-1),
    dict(kernel="linear", C=5.0,  epsilon=0.05, tol=1e-4,  max_iter=-1),
    dict(kernel="linear", C=5.0,  epsilon=0.10, tol=1e-4,  max_iter=-1),
    dict(kernel="linear", C=10.0, epsilon=0.05, tol=1e-4,  max_iter=-1),
    dict(kernel="linear", C=10.0, epsilon=0.10, tol=1e-4,  max_iter=-1),

    # rbf
    dict(kernel="rbf", C=0.5,  gamma=0.01, epsilon=0.10, tol=1e-3,  max_iter=-1),
    dict(kernel="rbf", C=0.5,  gamma=0.05, epsilon=0.10, tol=1e-3,  max_iter=-1),
    dict(kernel="rbf", C=0.5,  gamma=0.10, epsilon=0.10, tol=1e-4,  max_iter=-1),
    dict(kernel="rbf", C=1.0,  gamma=0.01, epsilon=0.10, tol=1e-3,  max_iter=-1),
    dict(kernel="rbf", C=1.0,  gamma=0.05, epsilon=0.10, tol=1e-4,  max_iter=-1),
    dict(kernel="rbf", C=1.0,  gamma=0.10, epsilon=0.10, tol=1e-4,  max_iter=-1),
    dict(kernel="rbf", C=2.0,  gamma=0.01, epsilon=0.10, tol=1e-3,  max_iter=10000),
    dict(kernel="rbf", C=2.0,  gamma=0.05, epsilon=0.10, tol=1e-3,  max_iter=10000),
    dict(kernel="rbf", C=2.0,  gamma=0.10, epsilon=0.10, tol=1e-4,  max_iter=10000),
    dict(kernel="rbf", C=3.0,  gamma=0.05, epsilon=0.10, tol=1e-3,  max_iter=-1),
    dict(kernel="rbf", C=3.0,  gamma=0.10, epsilon=0.10, tol=1e-4,  max_iter=-1),
    dict(kernel="rbf", C=5.0,  gamma=0.01, epsilon=0.10, tol=1e-3,  max_iter=-1),
    dict(kernel="rbf", C=5.0,  gamma=0.05, epsilon=0.10, tol=1e-4,  max_iter=-1),
    dict(kernel="rbf", C=5.0,  gamma=0.10, epsilon=0.10, tol=1e-4,  max_iter=-1),
    dict(kernel="rbf", C=10.0, gamma=0.01, epsilon=0.10, tol=1e-4,  max_iter=-1),
    dict(kernel="rbf", C=10.0, gamma=0.10, epsilon=0.10, tol=1e-4,  max_iter=-1),

    # poly
    dict(kernel="poly", C=0.5,  degree=2, gamma=0.01, coef0=0.0,  epsilon=0.10, tol=1e-3,  max_iter=-1),
    dict(kernel="poly", C=0.5,  degree=2, gamma=0.05, coef0=0.0,  epsilon=0.10, tol=1e-3,  max_iter=-1),
    dict(kernel="poly", C=0.5,  degree=3, gamma=0.05, coef0=0.1,  epsilon=0.10, tol=1e-3,  max_iter=-1),
    dict(kernel="poly", C=1.0,  degree=2, gamma=0.05, coef0=0.0,  epsilon=0.10, tol=1e-4,  max_iter=5000),
    dict(kernel="poly", C=1.0,  degree=3, gamma=0.05, coef0=0.0,  epsilon=0.10, tol=1e-4,  max_iter=5000),
    dict(kernel="poly", C=1.0,  degree=3, gamma=0.10, coef0=0.0,  epsilon=0.10, tol=1e-4,  max_iter=5000),
    dict(kernel="poly", C=2.0,  degree=3, gamma=0.05, coef0=0.1,  epsilon=0.10, tol=1e-3,  max_iter=-1),
    dict(kernel="poly", C=2.0,  degree=4, gamma=0.01, coef0=0.0,  epsilon=0.10, tol=1e-3,  max_iter=-1),
    dict(kernel="poly", C=2.0,  degree=4, gamma=0.05, coef0=0.1,  epsilon=0.10, tol=1e-3,  max_iter=-1),
    dict(kernel="poly", C=5.0,  degree=3, gamma=0.01, coef0=0.1,  epsilon=0.10, tol=1e-4,  max_iter=-1),
    dict(kernel="poly", C=5.0,  degree=3, gamma=0.05, coef0=0.1,  epsilon=0.10, tol=1e-4,  max_iter=-1),
    dict(kernel="poly", C=5.0,  degree=4, gamma=0.05, coef0=0.0,  epsilon=0.10, tol=1e-4,  max_iter=-1),
    dict(kernel="poly", C=10.0, degree=3, gamma=0.05, coef0=0.1,  epsilon=0.10, tol=1e-4,  max_iter=-1),
    dict(kernel="poly", C=10.0, degree=3, gamma=0.10, coef0=0.1,  epsilon=0.10, tol=1e-4,  max_iter=-1),
    dict(kernel="poly", C=10.0, degree=4, gamma=0.01, coef0=0.0,  epsilon=0.10, tol=1e-4,  max_iter=-1),
    dict(kernel="poly", C=10.0, degree=4, gamma=0.05, coef0=0.1,  epsilon=0.10, tol=1e-4,  max_iter=-1),

    # sigmoid
    dict(kernel="sigmoid", C=0.5, gamma=0.01, coef0=0.1, epsilon=0.05, tol=1e-3,  max_iter=-1),
    dict(kernel="sigmoid", C=0.5, gamma=0.05, coef0=0.1, epsilon=0.05, tol=1e-3,  max_iter=-1),
    dict(kernel="sigmoid", C=1.0, gamma=0.01, coef0=0.1, epsilon=0.05, tol=1e-4,  max_iter=1000),
    dict(kernel="sigmoid", C=1.0, gamma=0.05, coef0=0.1, epsilon=0.05, tol=1e-4,  max_iter=1000),
    dict(kernel="sigmoid", C=2.0, gamma=0.01, coef0=0.1, epsilon=0.10, tol=1e-4, max_iter=-1),
    dict(kernel="sigmoid", C=2.0, gamma=0.05, coef0=0.1, epsilon=0.10, tol=1e-4, max_iter=-1),
    dict(kernel="sigmoid", C=5.0, gamma=0.01, coef0=0.0, epsilon=0.10, tol=1e-4, max_iter=-1),
    dict(kernel="sigmoid", C=5.0, gamma=0.05, coef0=0.0, epsilon=0.10, tol=1e-4, max_iter=-1),
]

def fit_eval_once(params, X_tr, y_tr, X_va, y_va, scaler_y):
    p = params.copy()
    if p.get("max_iter", -1) in (1000, 5000):
        p["max_iter"] = -1
    if p.get("kernel") in ("rbf", "poly", "sigmoid"):
        p.setdefault("cache_size", 2000)

    reg = SVR(**p).fit(X_tr, y_tr)

    ytr_hat = inv_t(reg.predict(X_tr), scaler_y)
    yva_hat = inv_t(reg.predict(X_va), scaler_y)
    ytr_act = inv_t(y_tr, scaler_y)
    yva_act = inv_t(y_va, scaler_y)

    tr_rmse = rmse(ytr_act, ytr_hat)
    va_rmse = rmse(yva_act, yva_hat)
    tr_r2   = float(r2_score(ytr_act, ytr_hat))
    va_r2   = float(r2_score(yva_act, yva_hat))
    cmt     = behavior_label(tr_rmse, va_rmse, overfit_gap=0.10)
    return tr_rmse, va_rmse, tr_r2, va_r2, cmt

best_params = None

# -----------------------------
# TRAIN BLOCK
# -----------------------------
if RUN_TRAIN:
    # Baseline sanity check
    reg0 = SVR(kernel='rbf', C=1.0, tol=1e-3, max_iter=-1).fit(X_train, y_train)
    ytr_hat = inv_t(reg0.predict(X_train), scaler_y)
    yva_hat = inv_t(reg0.predict(X_val),   scaler_y)
    ytr_act = inv_t(y_train, scaler_y)
    yva_act = inv_t(y_val,   scaler_y)
    print("--------Random Train-Validation Split (baseline)-----------")
    print(f"Training RMSE: {rmse(ytr_act, ytr_hat):.2f} | R²: {r2_score(ytr_act, ytr_hat):.2f}")
    print(f"Validation RMSE: {rmse(yva_act, yva_hat):.2f} | R²: {r2_score(yva_act, yva_hat):.2f}")

    # Sweep Table 2
    rows = []
    for i, p in enumerate(TABLE2, start=1):
        tr_rmse_i, va_rmse_i, tr_r2_i, va_r2_i, cmt = fit_eval_once(p, X_train, y_train, X_val, y_val, scaler_y)
        rows.append({
            "row": i,
            **param_cols(p),
            "Train_RMSE": round(tr_rmse_i, 4),
            "Val_RMSE":   round(va_rmse_i, 4),
            "Train_R2":   round(tr_r2_i, 4),
            "Val_R2":     round(va_r2_i, 4),
            "Comment":    cmt
        })
    table2_df = pd.DataFrame(rows).sort_values(["Val_RMSE","Train_RMSE"]).reset_index(drop=True)
    print("\n=== Table 2 – SVR Hyperparameter Results (Random Split) ===")
    print(table2_df.to_string(index=False))
    table2_df.to_csv("svr_table2_random_split.csv", index=False)

    # Best config
    best_params = table2_df.iloc[0][["kernel","C","gamma","degree","coef0","epsilon","tol","max_iter"]].to_dict()
    best_params = {k: v for k, v in best_params.items() if pd.notna(v)}
    if best_params.get("max_iter", -1) in (1000, 5000):
        best_params["max_iter"] = -1
    if best_params.get("kernel") in ("rbf", "poly", "sigmoid"):
        best_params.setdefault("cache_size", 2000)
    print("\nBest configuration (random split):", best_params)

    # ---------- K-FOLD tables (k=5,10,15) ----------
    for k_ in [5, 10, 15]:
        rows = []
        for i, p in enumerate(TABLE2, start=1):
            avg_tr, avg_va, std_va = cv_once_kfold(p, X_final, y_scaled, k_, scaler_y)  # FIX: y_scaled
            rows.append({
                "row": i,
                **param_cols(p),
                "Avg_Train_RMSE": round(avg_tr, 4),
                "Avg_Val_RMSE":   round(avg_va, 4),
                "Std_Val_RMSE":   round(std_va, 4),
            })
        k_df = pd.DataFrame(rows).sort_values(["Avg_Val_RMSE","Avg_Train_RMSE"]).reset_index(drop=True)
        out_csv = f"svr_kfold_k{k_}.csv"
        k_df.to_csv(out_csv, index=False)
        print(f"\n=== K-Fold (k={k_}) — Top 5 by Avg_Val_RMSE ===")
        print(k_df.head(5).to_string(index=False))

    # ---------- Repeated K-FOLD tables ----------
    for k_ in [5, 10, 15]:
        for r_ in [5, 10]:
            rows = []
            for i, p in enumerate(TABLE2, start=1):
                avg_tr, avg_va, std_va = cv_once_repeated(p, X_final, y_scaled, k_, r_, scaler_y)  # FIX: y_scaled
                rows.append({
                    "row": i,
                    **param_cols(p),
                    "Avg_Train_RMSE": round(avg_tr, 4),
                    "Avg_Val_RMSE":   round(avg_va, 4),
                    "Std_Val_RMSE":   round(std_va, 4),
                })
            rkf_df = pd.DataFrame(rows).sort_values(["Avg_Val_RMSE","Avg_Train_RMSE"]).reset_index(drop=True)
            out_csv = f"svr_rkf_k{k_}_r{r_}.csv"
            rkf_df.to_csv(out_csv, index=False)
            print(f"\n=== Repeated K-Fold (k={k_}, reps={r_}) — Top 5 by Avg_Val_RMSE ===")
            print(rkf_df.head(5).to_string(index=False))

# -----------------------------
# TEST BLOCK
# -----------------------------
if RUN_TEST:
    if best_params is None:
        try:
            saved = pd.read_csv("svr_table2_random_split.csv")
            best_row = saved.sort_values(["Val_RMSE","Train_RMSE"]).iloc[0]
            best_params = {k: best_row[k] for k in ["kernel","C","gamma","degree","coef0","epsilon","tol","max_iter"]
                           if (k in saved.columns and pd.notna(best_row[k]))}
        except Exception:
            best_params = dict(kernel="rbf", C=10.0, gamma=0.1, epsilon=0.1, tol=1e-4, max_iter=-1)
    if best_params.get("max_iter", -1) in (1000, 5000):
        best_params["max_iter"] = -1
    if best_params.get("kernel") in ("rbf", "poly", "sigmoid"):
        best_params.setdefault("cache_size", 2000)

    reg_final = SVR(**best_params).fit(X_train_full, y_train_full)
    y_test_hat = inv_t(reg_final.predict(X_test), scaler_y)
    y_test_act = inv_t(y_test, scaler_y)
    test_rmse = rmse(y_test_act, y_test_hat)
    test_r2   = float(r2_score(y_test_act, y_test_hat))
    print("\n=== Final Test (best config on Train+Val) ===")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Test R²:   {test_r2:.4f}")
