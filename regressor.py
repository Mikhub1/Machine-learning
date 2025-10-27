import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

# =========================================================
# Load dataset
# =========================================================
cwd = Path.cwd()
csv_file = cwd / "car_prices.csv"
df = pd.read_csv(csv_file)

# Separate features (first 10 columns) and target (11th column)
X = df.iloc[:, 0:10]
y = df.iloc[:, [10]]

categorical_features = ["fuel type", "seller type", "transmission"]
numerical_features = [c for c in X.columns if c not in categorical_features]

# --- Scale numerical features ---
scaler_X = StandardScaler()
X_num_scaled = scaler_X.fit_transform(df[numerical_features])
X_num_scaled = pd.DataFrame(X_num_scaled, columns=numerical_features, index=df.index)

# --- One-hot encode categorical features ---
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
X_cat_encoded = encoder.fit_transform(df[categorical_features])
cat_feature_names = encoder.get_feature_names_out(categorical_features)
X_cat_encoded = pd.DataFrame(X_cat_encoded, columns=cat_feature_names, index=df.index)

# --- Combine processed features ---
X_transformed = pd.concat([X_num_scaled, X_cat_encoded], axis=1)

# --- Scale the target ---
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y)
y_final = y_scaled.flatten()

# =========================================================
# Split dataset (70/15/15)
# =========================================================
X_final = X_transformed
X_train, X_temp, y_train, y_temp = train_test_split(
    X_final, y_final, test_size=0.3, random_state=42, shuffle=True
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, shuffle=True
)

# Keep names for final fits
X_train_full = pd.concat([X_train, X_val], axis=0)
y_train_full = np.concatenate((y_train, y_val))

# =========================================================
# FLAGS
# =========================================================
RUN_TABLE6_LINEAR = True
RUN_TABLE7_KNN    = True
RUN_TABLE9_RF     = True
RUN_TABLE10_SVR   = True

# ------------------- Utilities -------------------
def _effective_n_jobs_for_fit(v):
    """LinearRegression: accept None / 'None' / NaN / -1 as default (None)."""
    if v is None:
        return None
    try:
        if isinstance(v, str) and v.strip().lower() == "none":
            return None
    except Exception:
        pass
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    try:
        iv = int(v)
        return None if iv == -1 else iv
    except Exception:
        return None

def _nj_to_str(v):
    if v is None:
        return "None"
    try:
        if pd.isna(v):
            return "None"
    except Exception:
        pass
    return str(int(v))

def _to_none_or_value(x):
    """Map strings like 'None' to None; leave other values as-is."""
    if x is None:
        return None
    if isinstance(x, str) and x.strip().lower() == "none":
        return None
    return x

def _to_bool(x):
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.strip().lower() == "true"
    return bool(x)

def _to_int_or_none(x):
    """Return None for None/NaN/'None'; else cast to int (fixes 8.0 → 8)."""
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    if isinstance(x, str) and x.strip().lower() == "none":
        return None
    return int(x)

# =========================================================
# TABLE 6: Linear Regressor sweep
# =========================================================
def build_table6_param_grid_exact():
    rows = [
        dict(fit_intercept=True,  copy_X=True,  positive=False, n_jobs=None),
        dict(fit_intercept=True,  copy_X=False, positive=True,  n_jobs=None),
    ]
    n_jobs_sequence = [
        -1, 1, None, 2, 4, 6, 8, 10, -1, 12, 14, 16, 18, 20, None,
        24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80,
        84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132,
        136, 140, 144, 148, 152
    ]
    patterns = [
        dict(fit_intercept=True,  copy_X=True,  positive=True),
        dict(fit_intercept=False, copy_X=True,  positive=False),
        dict(fit_intercept=True,  copy_X=True,  positive=False),
        dict(fit_intercept=True,  copy_X=False, positive=True),
        dict(fit_intercept=True,  copy_X=True,  positive=False),
    ]
    pi = 0
    for nj in n_jobs_sequence:
        cfg = patterns[pi % len(patterns)].copy()
        cfg["n_jobs"] = nj
        rows.append(cfg)
        pi += 1
    return rows

def evaluate_linreg_combo(cfg):
    reg = LinearRegression(
        fit_intercept=cfg["fit_intercept"],
        copy_X=cfg["copy_X"],
        positive=cfg["positive"],
        n_jobs=_effective_n_jobs_for_fit(cfg["n_jobs"]),
    )
    reg.fit(X_train, y_train)

    y_tr_act  = scaler_y.inverse_transform(y_train.reshape(-1, 1))
    y_tr_pred = scaler_y.inverse_transform(reg.predict(X_train).reshape(-1, 1))
    y_v_act   = scaler_y.inverse_transform(y_val.reshape(-1, 1))
    y_v_pred  = scaler_y.inverse_transform(reg.predict(X_val).reshape(-1, 1))

    tr_mse = mean_squared_error(y_tr_act, y_tr_pred)
    v_mse  = mean_squared_error(y_v_act,  y_v_pred)
    return {
        **cfg,
        "train_rmse": round(float(np.sqrt(tr_mse)), 4),
        "val_rmse":   round(float(np.sqrt(v_mse)), 4),
        "train_mse":  round(float(tr_mse), 4),
        "val_mse":    round(float(v_mse), 4),
        "train_r2":   round(float(r2_score(y_tr_act, y_tr_pred)), 4),
        "val_r2":     round(float(r2_score(y_v_act,  y_v_pred)), 4),
    }

if RUN_TABLE6_LINEAR:
    params6 = build_table6_param_grid_exact()
    rows6_in_order = []

    print("\n=========== TABLE 6: Linear Regressor sweep ===========")
    for cfg in params6:
        row = evaluate_linreg_combo(cfg)
        rows6_in_order.append(row)
        print(f"[LR] fit_intercept={cfg['fit_intercept']}, copy_X={cfg['copy_X']}, "
              f"positive={cfg['positive']}, n_jobs={cfg['n_jobs']} | "
              f"Train RMSE={row['train_rmse']:.3f} | Val RMSE={row['val_rmse']:.3f}")

    df6_order = pd.DataFrame(rows6_in_order)
    df6_sorted = df6_order.sort_values(
        ["val_rmse", "train_rmse", "val_mse", "train_mse"], ascending=[True, True, True, True]
    )
    for d in (df6_order, df6_sorted):
        d["n_jobs"] = d["n_jobs"].apply(_nj_to_str)
    df6_order.to_csv("table6_linear_results_in_order.csv", index=False)
    df6_sorted.to_csv("table6_linear_results.csv", index=False)

    print("\nSaved:")
    print(" - table6_linear_results_in_order.csv (matches Word table order)")
    print(" - table6_linear_results.csv (sorted by Val RMSE ascending)")
    print(df6_sorted.head(10))

    # Evaluate best Linear on test
    best6 = df6_sorted.iloc[0].to_dict()
    lin_best = LinearRegression(
        fit_intercept=bool(best6["fit_intercept"]),
        copy_X=bool(best6["copy_X"]),
        positive=bool(best6["positive"]),
        n_jobs=_effective_n_jobs_for_fit(best6["n_jobs"]),
    ).fit(X_train_full, y_train_full)
    y_test_act = scaler_y.inverse_transform(y_test.reshape(-1, 1))
    y_test_pred = scaler_y.inverse_transform(lin_best.predict(X_test).reshape(-1, 1))
    print("\n----- Test Evaluation: Linear Regressor (Best from Table 6) -----")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test_act, y_test_pred)):.2f}")
    print(f"MSE : {mean_squared_error(y_test_act, y_test_pred):.2f}")
    print(f"R²  : {r2_score(y_test_act, y_test_pred):.2f}")

# =========================================================
# TABLE 7: K-Nearest Neighbor Regressor sweep
# =========================================================
def build_table7_knn_param_grid_exact():
    rows = [
        dict(n_neighbors=2,  weights="uniform",  p=1, algorithm="auto",     leaf_size=30),
        dict(n_neighbors=3,  weights="distance", p=2, algorithm="ball_tree",leaf_size=20),
        dict(n_neighbors=4,  weights="uniform",  p=3, algorithm="kd_tree",  leaf_size=35),
        dict(n_neighbors=5,  weights="distance", p=2, algorithm="brute",    leaf_size=40),
        dict(n_neighbors=6,  weights="uniform",  p=1, algorithm="auto",     leaf_size=25),
        dict(n_neighbors=7,  weights="distance", p=2, algorithm="ball_tree",leaf_size=50),
        dict(n_neighbors=8,  weights="uniform",  p=3, algorithm="kd_tree",  leaf_size=15),
        dict(n_neighbors=9,  weights="distance", p=2, algorithm="brute",    leaf_size=30),
        dict(n_neighbors=10, weights="uniform",  p=1, algorithm="auto",     leaf_size=45),
        dict(n_neighbors=12, weights="distance", p=2, algorithm="ball_tree",leaf_size=35),
        dict(n_neighbors=14, weights="uniform",  p=3, algorithm="kd_tree",  leaf_size=30),
        dict(n_neighbors=15, weights="distance", p=1, algorithm="brute",    leaf_size=20),
        dict(n_neighbors=16, weights="uniform",  p=2, algorithm="auto",     leaf_size=40),
        dict(n_neighbors=18, weights="distance", p=3, algorithm="ball_tree",leaf_size=25),
        dict(n_neighbors=20, weights="uniform",  p=2, algorithm="kd_tree",  leaf_size=50),
        dict(n_neighbors=22, weights="distance", p=1, algorithm="brute",    leaf_size=45),
        dict(n_neighbors=25, weights="uniform",  p=2, algorithm="auto",     leaf_size=30),
        dict(n_neighbors=28, weights="distance", p=3, algorithm="ball_tree",leaf_size=25),
        dict(n_neighbors=30, weights="uniform",  p=2, algorithm="kd_tree",  leaf_size=35),
        dict(n_neighbors=32, weights="distance", p=1, algorithm="brute",    leaf_size=40),
        dict(n_neighbors=35, weights="uniform",  p=2, algorithm="auto",     leaf_size=15),
        dict(n_neighbors=38, weights="distance", p=3, algorithm="ball_tree",leaf_size=50),
        dict(n_neighbors=40, weights="uniform",  p=2, algorithm="kd_tree",  leaf_size=45),
        dict(n_neighbors=45, weights="distance", p=1, algorithm="brute",    leaf_size=30),
        dict(n_neighbors=48, weights="uniform",  p=3, algorithm="auto",     leaf_size=25),
        dict(n_neighbors=50, weights="distance", p=2, algorithm="ball_tree",leaf_size=35),
        dict(n_neighbors=55, weights="uniform",  p=2, algorithm="kd_tree",  leaf_size=20),
        dict(n_neighbors=60, weights="distance", p=1, algorithm="brute",    leaf_size=40),
        dict(n_neighbors=65, weights="uniform",  p=3, algorithm="auto",     leaf_size=30),
        dict(n_neighbors=70, weights="distance", p=2, algorithm="ball_tree",leaf_size=45),
        dict(n_neighbors=75, weights="uniform",  p=1, algorithm="kd_tree",  leaf_size=25),
        dict(n_neighbors=80, weights="distance", p=3, algorithm="brute",    leaf_size=50),
        dict(n_neighbors=85, weights="uniform",  p=2, algorithm="auto",     leaf_size=35),
        dict(n_neighbors=90, weights="distance", p=1, algorithm="ball_tree",leaf_size=40),
        dict(n_neighbors=95, weights="uniform",  p=3, algorithm="kd_tree",  leaf_size=30),
        dict(n_neighbors=100,weights="distance", p=2, algorithm="brute",    leaf_size=25),
        dict(n_neighbors=110,weights="uniform",  p=2, algorithm="auto",     leaf_size=20),
        dict(n_neighbors=120,weights="distance", p=3, algorithm="ball_tree",leaf_size=35),
        dict(n_neighbors=130,weights="uniform",  p=1, algorithm="kd_tree",  leaf_size=45),
        dict(n_neighbors=140,weights="distance", p=2, algorithm="brute",    leaf_size=30),
        dict(n_neighbors=150,weights="uniform",  p=3, algorithm="auto",     leaf_size=25),
        dict(n_neighbors=160,weights="distance", p=2, algorithm="ball_tree",leaf_size=40),
        dict(n_neighbors=170,weights="uniform",  p=1, algorithm="kd_tree",  leaf_size=35),
        dict(n_neighbors=180,weights="distance", p=3, algorithm="brute",    leaf_size=20),
        dict(n_neighbors=190,weights="uniform",  p=2, algorithm="auto",     leaf_size=50),
        dict(n_neighbors=200,weights="distance", p=1, algorithm="ball_tree",leaf_size=25),
        dict(n_neighbors=220,weights="uniform",  p=3, algorithm="kd_tree",  leaf_size=30),
        dict(n_neighbors=240,weights="distance", p=2, algorithm="brute",    leaf_size=45),
        dict(n_neighbors=260,weights="uniform",  p=2, algorithm="auto",     leaf_size=35),
        dict(n_neighbors=280,weights="distance", p=3, algorithm="ball_tree",leaf_size=40),
    ]
    return rows

def evaluate_knn_combo(cfg):
    reg = KNeighborsRegressor(
        n_neighbors=int(cfg["n_neighbors"]),
        weights=cfg["weights"],
        algorithm=cfg["algorithm"],
        leaf_size=int(cfg["leaf_size"]),
        p=int(cfg["p"]),
        metric="minkowski",
    )
    reg.fit(X_train, y_train)

    y_tr_act  = scaler_y.inverse_transform(y_train.reshape(-1, 1))
    y_tr_pred = scaler_y.inverse_transform(reg.predict(X_train).reshape(-1, 1))
    y_v_act   = scaler_y.inverse_transform(y_val.reshape(-1, 1))
    y_v_pred  = scaler_y.inverse_transform(reg.predict(X_val).reshape(-1, 1))

    tr_mse = mean_squared_error(y_tr_act, y_tr_pred)
    v_mse  = mean_squared_error(y_v_act,  y_v_pred)
    return {
        **cfg,
        "train_rmse": round(float(np.sqrt(tr_mse)), 4),
        "val_rmse":   round(float(np.sqrt(v_mse)), 4),
        "train_mse":  round(float(tr_mse), 4),
        "val_mse":    round(float(v_mse), 4),
        "train_r2":   round(float(r2_score(y_tr_act, y_tr_pred)), 4),
        "val_r2":     round(float(r2_score(y_v_act,  y_v_pred)), 4),
    }

if RUN_TABLE7_KNN:
    params7 = build_table7_knn_param_grid_exact()
    rows7_in_order = []

    print("\n=========== TABLE 7: K-Nearest Neighbor Regressor sweep ===========")
    for cfg in params7:
        row = evaluate_knn_combo(cfg)
        rows7_in_order.append(row)
        print(f"[KNN] n_neighbors={cfg['n_neighbors']}, weights='{cfg['weights']}', "
              f"p={cfg['p']}, algorithm='{cfg['algorithm']}', leaf_size={cfg['leaf_size']} | "
              f"Train RMSE={row['train_rmse']:.3f} | Val RMSE={row['val_rmse']:.3f}")

    df7_order = pd.DataFrame(rows7_in_order)
    df7_sorted = df7_order.sort_values(
        ["val_rmse", "train_rmse", "val_mse", "train_mse"], ascending=[True, True, True, True]
    )
    df7_order.to_csv("table7_knn_results_in_order.csv", index=False)
    df7_sorted.to_csv("table7_knn_results.csv", index=False)

    print("\nSaved:")
    print(" - table7_knn_results_in_order.csv (matches Word table order)")
    print(" - table7_knn_results.csv (sorted by Val RMSE ascending)")
    print(df7_sorted.head(10))

    # Evaluate best KNN on the test set
    best7 = df7_sorted.iloc[0].to_dict()
    knn_best = KNeighborsRegressor(
        n_neighbors=int(best7["n_neighbors"]),
        weights=best7["weights"],
        algorithm=best7["algorithm"],
        leaf_size=int(best7["leaf_size"]),
        p=int(best7["p"]),
        metric="minkowski",
    ).fit(X_train_full, y_train_full)

    y_test_act = scaler_y.inverse_transform(y_test.reshape(-1, 1))
    y_test_pred = scaler_y.inverse_transform(knn_best.predict(X_test).reshape(-1, 1))
    print("\n----- Test Evaluation: KNN Regressor (Best from Table 7) -----")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test_act, y_test_pred)):.2f}")
    print(f"MSE : {mean_squared_error(y_test_act, y_test_pred):.2f}")
    print(f"R²  : {r2_score(y_test_act, y_test_pred):.2f}")

# =========================================================
# TABLE 9: Random Forest Regressor sweep
# =========================================================
def build_table9_rf_param_grid_exact():
    rows = [
        dict(n_estimators=50,  criterion="squared_error",  max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features="sqrt", bootstrap=True),
        dict(n_estimators=75,  criterion="squared_error",  max_depth=10,   min_samples_split=4, min_samples_leaf=2, max_features="log2", bootstrap=True),
        dict(n_estimators=100, criterion="absolute_error", max_depth=8,    min_samples_split=3, min_samples_leaf=1, max_features=None,   bootstrap=False),
        dict(n_estimators=125, criterion="squared_error",  max_depth=12,   min_samples_split=5, min_samples_leaf=2, max_features="sqrt", bootstrap=True),
        dict(n_estimators=150, criterion="friedman_mse",   max_depth=None, min_samples_split=6, min_samples_leaf=3, max_features="log2", bootstrap=True),
        dict(n_estimators=175, criterion="squared_error",  max_depth=15,   min_samples_split=4, min_samples_leaf=2, max_features=None,   bootstrap=False),
        dict(n_estimators=200, criterion="absolute_error", max_depth=10,   min_samples_split=3, min_samples_leaf=1, max_features="sqrt", bootstrap=True),

        dict(n_estimators=225, criterion="squared_error",  max_depth=None, min_samples_split=5, min_samples_leaf=2, max_features="log2", bootstrap=True),
        dict(n_estimators=250, criterion="friedman_mse",   max_depth=8,    min_samples_split=2, min_samples_leaf=1, max_features=None,   bootstrap=False),
        dict(n_estimators=275, criterion="squared_error",  max_depth=20,   min_samples_split=3, min_samples_leaf=1, max_features="sqrt", bootstrap=True),
        dict(n_estimators=300, criterion="absolute_error", max_depth=None, min_samples_split=6, min_samples_leaf=3, max_features="log2", bootstrap=True),
        dict(n_estimators=325, criterion="squared_error",  max_depth=10,   min_samples_split=2, min_samples_leaf=1, max_features=None,   bootstrap=False),
        dict(n_estimators=350, criterion="friedman_mse",   max_depth=15,   min_samples_split=4, min_samples_leaf=2, max_features="sqrt", bootstrap=True),
        dict(n_estimators=375, criterion="squared_error",  max_depth=None, min_samples_split=3, min_samples_leaf=1, max_features="log2", bootstrap=True),
        dict(n_estimators=400, criterion="absolute_error", max_depth=9,    min_samples_split=5, min_samples_leaf=2, max_features=None,   bootstrap=False),
        dict(n_estimators=425, criterion="squared_error",  max_depth=14,   min_samples_split=4, min_samples_leaf=3, max_features="sqrt", bootstrap=True),
        dict(n_estimators=450, criterion="friedman_mse",   max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features="log2", bootstrap=True),

        dict(n_estimators=475, criterion="squared_error",  max_depth=18,   min_samples_split=5, min_samples_leaf=2, max_features=None,   bootstrap=False),
        dict(n_estimators=500, criterion="absolute_error", max_depth=12,   min_samples_split=4, min_samples_leaf=1, max_features="sqrt", bootstrap=True),
        dict(n_estimators=525, criterion="squared_error",  max_depth=None, min_samples_split=3, min_samples_leaf=2, max_features="log2", bootstrap=True),
        dict(n_estimators=550, criterion="friedman_mse",   max_depth=10,   min_samples_split=6, min_samples_leaf=3, max_features=None,   bootstrap=False),
        dict(n_estimators=575, criterion="squared_error",  max_depth=16,   min_samples_split=4, min_samples_leaf=2, max_features="sqrt", bootstrap=True),
        dict(n_estimators=600, criterion="absolute_error", max_depth=None, min_samples_split=3, min_samples_leaf=1, max_features="log2", bootstrap=True),
        dict(n_estimators=625, criterion="squared_error",  max_depth=8,    min_samples_split=5, min_samples_leaf=2, max_features=None,   bootstrap=False),
        dict(n_estimators=650, criterion="friedman_mse",   max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features="sqrt", bootstrap=True),
        dict(n_estimators=675, criterion="squared_error",  max_depth=20,   min_samples_split=3, min_samples_leaf=2, max_features="log2", bootstrap=True),
        dict(n_estimators=700, criterion="absolute_error", max_depth=11,   min_samples_split=5, min_samples_leaf=3, max_features=None,   bootstrap=False),
        dict(n_estimators=725, criterion="squared_error",  max_depth=None, min_samples_split=4, min_samples_leaf=1, max_features="sqrt", bootstrap=True),
        dict(n_estimators=750, criterion="friedman_mse",   max_depth=15,   min_samples_split=2, min_samples_leaf=2, max_features="log2", bootstrap=True),
        dict(n_estimators=775, criterion="squared_error",  max_depth=None, min_samples_split=6, min_samples_leaf=3, max_features=None,   bootstrap=False),

        dict(n_estimators=800, criterion="absolute_error", max_depth=10,   min_samples_split=3, min_samples_leaf=1, max_features="sqrt", bootstrap=True),
        dict(n_estimators=825, criterion="squared_error",  max_depth=12,   min_samples_split=5, min_samples_leaf=2, max_features="log2", bootstrap=True),
        dict(n_estimators=850, criterion="friedman_mse",   max_depth=None, min_samples_split=4, min_samples_leaf=3, max_features=None,   bootstrap=False),
        dict(n_estimators=875, criterion="squared_error",  max_depth=14,   min_samples_split=3, min_samples_leaf=1, max_features="sqrt", bootstrap=True),
        dict(n_estimators=900, criterion="absolute_error", max_depth=None, min_samples_split=6, min_samples_leaf=2, max_features="log2", bootstrap=True),
        dict(n_estimators=925, criterion="squared_error",  max_depth=16,   min_samples_split=4, min_samples_leaf=3, max_features=None,   bootstrap=False),
        dict(n_estimators=950, criterion="friedman_mse",   max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features="sqrt", bootstrap=True),
        dict(n_estimators=975, criterion="squared_error",  max_depth=18,   min_samples_split=5, min_samples_leaf=2, max_features="log2", bootstrap=True),

        dict(n_estimators=1000, criterion="absolute_error", max_depth=12,  min_samples_split=4, min_samples_leaf=1, max_features=None,   bootstrap=False),
        dict(n_estimators=1050, criterion="squared_error",  max_depth=None,min_samples_split=3, min_samples_leaf=1, max_features="sqrt", bootstrap=True),
        dict(n_estimators=1100, criterion="friedman_mse",   max_depth=10,  min_samples_split=6, min_samples_leaf=3, max_features=None,   bootstrap=False),
        dict(n_estimators=1150, criterion="squared_error",  max_depth=16,  min_samples_split=4, min_samples_leaf=2, max_features="log2", bootstrap=True),
        dict(n_estimators=1200, criterion="absolute_error", max_depth=None,min_samples_split=3, min_samples_leaf=1, max_features="sqrt", bootstrap=True),
        dict(n_estimators=1250, criterion="squared_error",  max_depth=8,   min_samples_split=5, min_samples_leaf=2, max_features=None,   bootstrap=False),
        dict(n_estimators=1300, criterion="friedman_mse",   max_depth=None,min_samples_split=2, min_samples_leaf=1, max_features="log2", bootstrap=True),
        dict(n_estimators=1350, criterion="squared_error",  max_depth=20,  min_samples_split=3, min_samples_leaf=2, max_features="sqrt", bootstrap=True),
        dict(n_estimators=1400, criterion="absolute_error", max_depth=11,  min_samples_split=5, min_samples_leaf=3, max_features=None,   bootstrap=False),
        dict(n_estimators=1450, criterion="squared_error",  max_depth=None,min_samples_split=4, min_samples_leaf=1, max_features="log2", bootstrap=True),
        dict(n_estimators=1500, criterion="friedman_mse",   max_depth=15,  min_samples_split=2, min_samples_leaf=2, max_features="sqrt", bootstrap=True),
        dict(n_estimators=1550, criterion="squared_error",  max_depth=None,min_samples_split=6, min_samples_leaf=3, max_features=None,   bootstrap=False),
    ]
    return rows

def evaluate_rf_combo(cfg):
    reg = RandomForestRegressor(
        n_estimators=int(cfg["n_estimators"]),
        criterion=cfg["criterion"],
        max_depth=_to_int_or_none(cfg["max_depth"]),
        min_samples_split=int(cfg["min_samples_split"]),
        min_samples_leaf=int(cfg["min_samples_leaf"]),
        max_features=_to_none_or_value(cfg["max_features"]),
        bootstrap=_to_bool(cfg["bootstrap"]),
        random_state=42,
        n_jobs=-1,
    )
    reg.fit(X_train, y_train)

    y_tr_act  = scaler_y.inverse_transform(y_train.reshape(-1, 1))
    y_tr_pred = scaler_y.inverse_transform(reg.predict(X_train).reshape(-1, 1))
    y_v_act   = scaler_y.inverse_transform(y_val.reshape(-1, 1))
    y_v_pred  = scaler_y.inverse_transform(reg.predict(X_val).reshape(-1, 1))

    tr_mse = mean_squared_error(y_tr_act, y_tr_pred)
    v_mse  = mean_squared_error(y_v_act,  y_v_pred)
    return {
        **cfg,
        "train_rmse": round(float(np.sqrt(tr_mse)), 4),
        "val_rmse":   round(float(np.sqrt(v_mse)), 4),
        "train_mse":  round(float(tr_mse), 4),
        "val_mse":    round(float(v_mse), 4),
        "train_r2":   round(float(r2_score(y_tr_act, y_tr_pred)), 4),
        "val_r2":     round(float(r2_score(y_v_act,  y_v_pred)), 4),
    }

if RUN_TABLE9_RF:
    params9 = build_table9_rf_param_grid_exact()
    rows9_in_order = []

    print("\n=========== TABLE 9: Random Forest Regressor sweep ===========")
    for cfg in params9:
        row = evaluate_rf_combo(cfg)
        rows9_in_order.append(row)
        print(
            f"[RF] n_estimators={cfg['n_estimators']}, criterion='{cfg['criterion']}', "
            f"max_depth={cfg['max_depth']}, min_samples_split={cfg['min_samples_split']}, "
            f"min_samples_leaf={cfg['min_samples_leaf']}, max_features={cfg['max_features']}, "
            f"bootstrap={cfg['bootstrap']} | Train RMSE={row['train_rmse']:.3f} | "
            f"Val RMSE={row['val_rmse']:.3f}"
        )

    df9_order = pd.DataFrame(rows9_in_order)
    df9_sorted = df9_order.sort_values(
        ["val_rmse", "train_rmse", "val_mse", "train_mse"],
        ascending=[True, True, True, True]
    )
    df9_order.to_csv("table9_rf_results_in_order.csv", index=False)
    df9_sorted.to_csv("table9_rf_results.csv", index=False)

    print("\nSaved:")
    print(" - table9_rf_results_in_order.csv (matches Word table order)")
    print(" - table9_rf_results.csv (sorted by Val RMSE ascending)")
    print(df9_sorted.head(10))

    # Evaluate best RF on the test set
    best9 = df9_sorted.iloc[0].to_dict()
    rf_best = RandomForestRegressor(
        n_estimators=int(best9["n_estimators"]),
        criterion=best9["criterion"],
        max_depth=_to_int_or_none(best9["max_depth"]),
        min_samples_split=int(best9["min_samples_split"]),
        min_samples_leaf=int(best9["min_samples_leaf"]),
        max_features=_to_none_or_value(best9["max_features"]),
        bootstrap=_to_bool(best9["bootstrap"]),
        random_state=42,
        n_jobs=-1,
    ).fit(X_train_full, y_train_full)

    y_test_act = scaler_y.inverse_transform(y_test.reshape(-1, 1))
    y_test_pred = scaler_y.inverse_transform(rf_best.predict(X_test).reshape(-1, 1))
    print("\n----- Test Evaluation: Random Forest (Best from Table 9) -----")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test_act, y_test_pred)):.2f}")
    print(f"MSE : {mean_squared_error(y_test_act, y_test_pred):.2f}")
    print(f"R²  : {r2_score(y_test_act, y_test_pred):.2f}")

# =========================================================
# TABLE 10: Support Vector Regressor sweep
# =========================================================
def build_table10_svr_param_grid_exact():
    """
    Each row: kernel, C, degree, gamma, coef0, tol, max_iter
    (degree/coef0 ignored by some kernels; sklearn will just ignore them)
    """
    rows = []

    # ---- linear ----
    linear = [
        (0.2, 1e-4, 1000), (0.5, 5e-4, 2000), (1.0, 1e-3, 3000),
        (1.5, 2e-4, 4000), (2.0, 3e-4, 5000), (3.0, 1e-4, 6000),
        (4.0, 5e-4, 7000), (5.0, 1e-3, 8000), (6.0, 2e-4, 9000),
        (7.0, 3e-4, 10000),
    ]
    for C, tol, it in linear:
        rows.append(dict(kernel="linear", C=C, degree=3, gamma="scale", coef0=0.0, tol=tol, max_iter=it))

    # ---- poly ----
    poly = [
        (0.2, 3, 0.01, 0.05, 1e-4, 1000),
        (0.5, 3, 0.02, 0.10, 5e-4, 2000),
        (1.0, 4, 0.03, 0.00, 1e-3, 1500),
        (1.5, 4, 0.04, 0.05, 2e-4, 2500),
        (2.0, 5, 0.05, 0.10, 3e-4, 3000),
        (2.5, 5, 0.06, 0.00, 1e-4, 3500),
        (3.0, 6, 0.07, 0.05, 5e-4, 4000),
        (3.5, 6, 0.08, 0.10, 1e-3, 4500),
        (4.0, 7, 0.09, 0.00, 2e-4, 5000),
        (4.5, 7, 0.10, 0.05, 3e-4, 5500),
        (5.0, 8, 0.12, 0.10, 1e-4, 1000),
        (5.5, 8, 0.15, 0.00, 5e-4, 2000),
        (6.0, 8, 0.20, 0.05, 1e-3, 1500),
        (6.5, 6, 0.05, 0.00, 2e-4, 2500),
        (7.0, 7, 0.08, 0.10, 3e-4, 3000),
    ]
    for C, deg, g, c0, tol, it in poly:
        rows.append(dict(kernel="poly", C=C, degree=deg, gamma=g, coef0=c0, tol=tol, max_iter=it))

    # ---- rbf ----
    rbf = [
        (0.2, 0.01, 1e-4, 1000), (0.5, 0.02, 5e-4, 1500), (1.0, 0.03, 1e-3, 2000),
        (1.5, 0.04, 2e-4, 2500), (2.0, 0.05, 3e-4, 3000), (2.5, 0.06, 1e-4, 3500),
        (3.0, 0.07, 5e-4, 4000), (3.5, 0.08, 1e-3, 4500), (4.0, 0.09, 2e-4, 5000),
        (4.5, 0.10, 3e-4, 5500), (5.0, 0.12, 1e-4, 1000), (5.5, 0.15, 5e-4, 1500),
        (6.0, 0.20, 1e-3, 2000), (6.5, 0.25, 2e-4, 2500), (7.0, 0.30, 3e-4, 3000),
    ]
    for C, g, tol, it in rbf:
        rows.append(dict(kernel="rbf", C=C, degree=3, gamma=g, coef0=0.0, tol=tol, max_iter=it))

    # ---- sigmoid ----
    sigmoid = [
        (0.2, 0.01, 0.05, 1e-4, 1000), (0.5, 0.02, 0.10, 5e-4, 2000),
        (1.0, 0.03, 0.00, 1e-3, 1500), (1.5, 0.04, 0.05, 2e-4, 2500),
        (2.0, 0.05, 0.10, 3e-4, 3000), (2.5, 0.06, 0.00, 1e-4, 3500),
        (3.0, 0.07, 0.05, 5e-4, 4000),
    ]
    for C, g, c0, tol, it in sigmoid:
        rows.append(dict(kernel="sigmoid", C=C, degree=3, gamma=g, coef0=c0, tol=tol, max_iter=it))

    return rows

def evaluate_svr_combo(cfg):
    reg = SVR(
        kernel=cfg["kernel"],
        C=float(cfg["C"]),
        degree=int(cfg["degree"]),
        gamma=cfg["gamma"],     # float or "scale"
        coef0=float(cfg["coef0"]),
        tol=float(cfg["tol"]),
        max_iter=int(cfg["max_iter"]),
        epsilon=0.1,
    )
    reg.fit(X_train, y_train)

    y_tr_act  = scaler_y.inverse_transform(y_train.reshape(-1, 1))
    y_tr_pred = scaler_y.inverse_transform(reg.predict(X_train).reshape(-1, 1))
    y_v_act   = scaler_y.inverse_transform(y_val.reshape(-1, 1))
    y_v_pred  = scaler_y.inverse_transform(reg.predict(X_val).reshape(-1, 1))

    tr_mse = mean_squared_error(y_tr_act, y_tr_pred)
    v_mse  = mean_squared_error(y_v_act,  y_v_pred)
    return {
        **cfg,
        "train_rmse": round(float(np.sqrt(tr_mse)), 4),
        "val_rmse":   round(float(np.sqrt(v_mse)), 4),
        "train_mse":  round(float(tr_mse), 4),
        "val_mse":    round(float(v_mse), 4),
        "train_r2":   round(float(r2_score(y_tr_act, y_tr_pred)), 4),
        "val_r2":     round(float(r2_score(y_v_act,  y_v_pred)), 4),
    }

if RUN_TABLE10_SVR:
    params10 = build_table10_svr_param_grid_exact()
    rows10_in_order = []

    print("\n=========== TABLE 10: Support Vector Regressor sweep ===========")
    for cfg in params10:
        row = evaluate_svr_combo(cfg)
        rows10_in_order.append(row)
        print(
            f"[SVR] kernel='{cfg['kernel']}', C={cfg['C']}, degree={cfg['degree']}, "
            f"gamma={cfg['gamma']}, coef0={cfg['coef0']}, tol={cfg['tol']}, "
            f"max_iter={cfg['max_iter']} | Train RMSE={row['train_rmse']:.3f} | "
            f"Val RMSE={row['val_rmse']:.3f}"
        )

    df10_order  = pd.DataFrame(rows10_in_order)
    df10_sorted = df10_order.sort_values(
        ["val_rmse", "train_rmse", "val_mse", "train_mse"],
        ascending=[True, True, True, True]
    )
    df10_order.to_csv("table10_svr_results_in_order.csv", index=False)
    df10_sorted.to_csv("table10_svr_results.csv", index=False)

    print("\nSaved:")
    print(" - table10_svr_results_in_order.csv (matches Word table order)")
    print(" - table10_svr_results.csv (sorted by Val RMSE ascending)")
    print(df10_sorted.head(10))

    # Evaluate best SVR on the test set
    best10 = df10_sorted.iloc[0].to_dict()
    svr_best = SVR(
        kernel=best10["kernel"],
        C=float(best10["C"]),
        degree=int(best10["degree"]),
        gamma=best10["gamma"],
        coef0=float(best10["coef0"]),
        tol=float(best10["tol"]),
        max_iter=int(best10["max_iter"]),
        epsilon=0.1,
    ).fit(X_train_full, y_train_full)

    y_test_act  = scaler_y.inverse_transform(y_test.reshape(-1, 1))
    y_test_pred = scaler_y.inverse_transform(svr_best.predict(X_test).reshape(-1, 1))
    print("\n----- Test Evaluation: SVR (Best from Table 10) -----")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test_act, y_test_pred)):.2f}")
    print(f"MSE : {mean_squared_error(y_test_act, y_test_pred):.2f}")
    print(f"R²  : {r2_score(y_test_act, y_test_pred):.2f}")
