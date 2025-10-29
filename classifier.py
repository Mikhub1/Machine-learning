import glob
from pathlib import Path
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray, rgba2rgb
from skimage.transform import resize
from skimage.feature import hog

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# Extra imports for sweeps/metrics
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
import warnings

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# =========================================================
# Load dataset
# =========================================================
dataset_path = Path.cwd() / "Utensil images"

# Classes (the classes correspond to the folder names)
classes = ["FORK", "KNIFE", "SPOON"]

X, y = [], []

# Loop through each class folder to read its images
for label, cls in enumerate(classes):
    files = glob.glob(f"{dataset_path}/{cls}/*.png")
    print(f"{cls}: found {len(files)} files")  # debug

    for file in files:
        img = imread(file)

        # If RGBA (4 channels), convert to RGB
        if img.ndim == 3 and img.shape[2] == 4:
            img = rgba2rgb(img)

        # If RGB (3 channels), convert to grayscale
        if img.ndim == 3:
            img = rgb2gray(img)

        # Resize to fixed size (64x64 for consistency)
        img_resized = resize(img, (64, 64))

        # Extract HOG features
        features, _ = hog(
            img_resized,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            visualize=True,
        )

        X.append(features)
        y.append(label)

X = np.array(X)
y = np.array(y)

# Safety check: make sure we have data
print("Total samples:", len(X))

# =========================================================
# Train/Val/Test split
# =========================================================
# 70% train, 15% val, 15% test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

# For final training (train + val)
X_train_full = np.vstack((X_train, X_val))
y_train_full = np.concatenate((y_train, y_val))

# =========================================================
# ============ TABLE 1: Logistic Regression sweep =========
# =========================================================
table1_params = [
    dict(penalty="l2", solver="lbfgs", C=1.0, max_iter=100, class_weight=None),
    dict(penalty="l1", solver="liblinear", C=0.5, max_iter=200, class_weight="balanced"),
    dict(penalty="elasticnet", solver="saga", C=1.0, max_iter=500, class_weight=None, l1_ratio=0.5),
    dict(penalty="l2", solver="newton-cg", C=0.1, max_iter=300, class_weight="balanced"),
    dict(penalty="none", solver="lbfgs", C=1.0, max_iter=100, class_weight=None),

    dict(penalty="l1", solver="saga", C=2.0, max_iter=1000, class_weight=None),
    dict(penalty="elasticnet", solver="saga", C=0.8, max_iter=200, class_weight="balanced", l1_ratio=0.3),
    dict(penalty="l2", solver="liblinear", C=0.3, max_iter=400, class_weight=None),
    dict(penalty="l2", solver="saga", C=5.0, max_iter=300, class_weight="balanced"),
    dict(penalty="l1", solver="liblinear", C=0.1, max_iter=100, class_weight=None),
    dict(penalty="elasticnet", solver="saga", C=1.5, max_iter=1000, class_weight="balanced", l1_ratio=0.7),
    dict(penalty="l2", solver="lbfgs", C=0.5, max_iter=200, class_weight=None),
    dict(penalty="none", solver="saga", C=1.0, max_iter=150, class_weight="balanced"),
    dict(penalty="l1", solver="saga", C=3.0, max_iter=300, class_weight=None),
    dict(penalty="l2", solver="newton-cg", C=2.5, max_iter=500, class_weight=None),
    dict(penalty="l2", solver="saga", C=10.0, max_iter=1000, class_weight="balanced"),
    dict(penalty="elasticnet", solver="saga", C=0.2, max_iter=400, class_weight=None, l1_ratio=0.4),

    dict(penalty="l1", solver="liblinear", C=1.2, max_iter=300, class_weight="balanced"),
    dict(penalty="l2", solver="lbfgs", C=0.7, max_iter=500, class_weight=None),
    dict(penalty="none", solver="newton-cg", C=1.0, max_iter=300, class_weight="balanced"),
    dict(penalty="l1", solver="saga", C=0.05, max_iter=800, class_weight=None),
    dict(penalty="elasticnet", solver="saga", C=0.9, max_iter=600, class_weight="balanced", l1_ratio=0.6),
    dict(penalty="l2", solver="liblinear", C=4.0, max_iter=700, class_weight=None),
    dict(penalty="l2", solver="lbfgs", C=0.2, max_iter=200, class_weight="balanced"),
    dict(penalty="none", solver="saga", C=1.0, max_iter=400, class_weight=None),

    dict(penalty="l1", solver="liblinear", C=5.0, max_iter=300, class_weight="balanced"),
    dict(penalty="elasticnet", solver="saga", C=2.0, max_iter=900, class_weight=None, l1_ratio=0.2),
    dict(penalty="l2", solver="newton-cg", C=0.6, max_iter=100, class_weight="balanced"),
    dict(penalty="l2", solver="saga", C=8.0, max_iter=1000, class_weight=None),
    dict(penalty="l1", solver="saga", C=0.4, max_iter=250, class_weight="balanced"),
    dict(penalty="l2", solver="lbfgs", C=3.0, max_iter=150, class_weight=None),
    dict(penalty="none", solver="liblinear", C=1.0, max_iter=200, class_weight="balanced"),  # invalid; recorded as N/A
    dict(penalty="elasticnet", solver="saga", C=0.7, max_iter=700, class_weight=None, l1_ratio=0.8),
    dict(penalty="l1", solver="saga", C=1.0, max_iter=500, class_weight="balanced"),
    dict(penalty="l2", solver="newton-cg", C=1.5, max_iter=1000, class_weight=None),
    dict(penalty="l2", solver="saga", C=0.1, max_iter=200, class_weight="balanced"),
    dict(penalty="elasticnet", solver="saga", C=5.0, max_iter=800, class_weight=None, l1_ratio=0.9),
    dict(penalty="l1", solver="liblinear", C=2.5, max_iter=150, class_weight="balanced"),
    dict(penalty="l2", solver="lbfgs", C=6.0, max_iter=300, class_weight=None),
    dict(penalty="none", solver="saga", C=1.0, max_iter=1000, class_weight="balanced"),
    dict(penalty="elasticnet", solver="saga", C=4.0, max_iter=600, class_weight=None, l1_ratio=0.5),

    dict(penalty="l1", solver="saga", C=0.3, max_iter=200, class_weight=None),
    dict(penalty="l2", solver="liblinear", C=7.0, max_iter=1000, class_weight="balanced"),
    dict(penalty="l2", solver="newton-cg", C=0.05, max_iter=150, class_weight=None),
    dict(penalty="elasticnet", solver="saga", C=3.5, max_iter=800, class_weight="balanced", l1_ratio=0.6),
    dict(penalty="l1", solver="liblinear", C=0.2, max_iter=400, class_weight=None),
]

def _is_valid_combo(penalty, solver):
    if solver == "liblinear":
        return penalty in ("l1", "l2")
    if penalty is None:
        return solver in ("lbfgs", "newton-cg", "sag", "saga")
    if penalty == "l2":
        return solver in ("lbfgs", "newton-cg", "sag", "saga", "liblinear")
    if penalty == "l1":
        return solver in ("liblinear", "saga")
    if penalty == "elasticnet":
        return solver == "saga"
    return False

rows_in_order = []   # keep original order (for report)
rows_for_best = []   # only valid rows (for model selection)

for cfg in table1_params:
    lr_kwargs = cfg.copy()
    if isinstance(lr_kwargs.get("penalty"), str) and lr_kwargs["penalty"].lower() == "none":
        lr_kwargs["penalty"] = None

    penalty = lr_kwargs.get("penalty")
    solver = lr_kwargs.get("solver")

    l1_ratio = lr_kwargs.pop("l1_ratio", None)
    if penalty == "elasticnet":
        if l1_ratio is None:
            l1_ratio = 0.5
    else:
        l1_ratio = None

    if not _is_valid_combo(penalty, solver):
        print(f"[SKIP] Invalid combo per sklearn: penalty={penalty}, solver={solver}")
        rows_in_order.append({
            "penalty": ("None" if penalty is None else penalty),
            "solver": solver, "C": lr_kwargs.get("C"), "max_iter": lr_kwargs.get("max_iter"),
            "class_weight": str(lr_kwargs.get("class_weight")), "l1_ratio": ("NA" if l1_ratio is None else l1_ratio),
            "train_acc": "N/A", "val_acc": "N/A", "precision": "N/A", "recall": "N/A", "f1": "N/A"
        })
        continue

    pipe = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(**lr_kwargs, l1_ratio=l1_ratio, random_state=42))
    ])

    pipe.fit(X_train, y_train)
    y_val_pred = pipe.predict(X_val)

    tr_acc = pipe.score(X_train, y_train)
    v_acc  = accuracy_score(y_val, y_val_pred)
    prec   = precision_score(y_val, y_val_pred, average="weighted", zero_division=0)
    rec    = recall_score(y_val, y_val_pred, average="weighted", zero_division=0)
    f1     = f1_score(y_val, y_val_pred, average="weighted", zero_division=0)

    print(f"[LR] penalty={('None' if penalty is None else penalty):>8} | "
          f"solver={solver:>10} | C={lr_kwargs.get('C'):>4} | "
          f"train_acc={tr_acc:.4f} | val_acc={v_acc:.4f}")

    row = {
        "penalty": ("None" if penalty is None else penalty),
        "solver": solver,
        "C": lr_kwargs.get("C"),
        "max_iter": lr_kwargs.get("max_iter"),
        "class_weight": str(lr_kwargs.get("class_weight")),
        "l1_ratio": ("NA" if l1_ratio is None else l1_ratio),
        "train_acc": round(tr_acc, 4),
        "val_acc": round(v_acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
    }
    rows_in_order.append(row)
    rows_for_best.append(row)

df_sorted = pd.DataFrame(rows_for_best).sort_values(
    ["val_acc", "f1", "precision", "recall"],
    ascending=[False, False, False, False]
)
df_sorted.to_csv("table1_logreg_results.csv", index=False)

df_in_order = pd.DataFrame(rows_in_order)
df_in_order.to_csv("table1_logreg_results_in_order.csv", index=False)

print("\nSaved results to table1_logreg_results.csv (sorted)")
print(df_sorted.head(10))
print("\nSaved results to table1_logreg_results_in_order.csv (original order incl. N/A)")

# Best LR on test
if len(df_sorted) > 0:
    best_row = df_sorted.iloc[0]
    best_penalty = None if best_row["penalty"] == "None" else best_row["penalty"]
    best_solver  = best_row["solver"]
    best_kwargs = dict(
        penalty=best_penalty,
        solver=best_solver,
        C=float(best_row["C"]),
        max_iter=int(best_row["max_iter"]),
        class_weight=None if best_row["class_weight"] == "None" else "balanced",
    )
    best_l1_ratio = None if best_row["l1_ratio"] == "NA" else float(best_row["l1_ratio"])
    best_pipe = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(**best_kwargs, l1_ratio=best_l1_ratio, random_state=42))
    ])
    print("\nBest (by validation):", {**best_kwargs, "l1_ratio": best_l1_ratio})
    best_pipe.fit(X_train_full, y_train_full)
    y_test_pred = best_pipe.predict(X_test)
    print("\n----- Test Evaluation: Logistic Regression (Best from Table 1) -----")
    print(f"Accuracy : {accuracy_score(y_test, y_test_pred):.3f}")
    print(f"Precision: {precision_score(y_test, y_test_pred, average='weighted', zero_division=0):.3f}")
    print(f"Recall   : {recall_score(y_test, y_test_pred, average='weighted', zero_division=0):.3f}")
    print(f"F1-score : {f1_score(y_test, y_test_pred, average='weighted', zero_division=0):.3f}")
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_test_pred))
else:
    print("\nNo valid Logistic Regression configurations to evaluate on test set.")

# =========================================================
# ============ TABLE 2: K-Nearest Neighbors sweep =========
# =========================================================
table2_knn_params = [
    dict(n_neighbors=3,  weights="uniform",  p=2, metric="minkowski", leaf_size=30),
    dict(n_neighbors=5,  weights="distance", p=3, metric="minkowski", leaf_size=40),
    dict(n_neighbors=7,  weights="uniform",  p=4, metric="euclidean",  leaf_size=20),
    dict(n_neighbors=9,  weights="distance", p=1, metric="manhattan",  leaf_size=25),
    dict(n_neighbors=11, weights="uniform",  p=2, metric="minkowski", leaf_size=50),
    dict(n_neighbors=13, weights="distance", p=5, metric="chebyshev", leaf_size=35),
    dict(n_neighbors=15, weights="uniform",  p=3, metric="minkowski", leaf_size=45),
    dict(n_neighbors=17, weights="distance", p=2, metric="euclidean",  leaf_size=60),
    dict(n_neighbors=19, weights="uniform",  p=4, metric="manhattan",  leaf_size=70),
    dict(n_neighbors=21, weights="distance", p=1, metric="minkowski", leaf_size=80),
    dict(n_neighbors=23, weights="uniform",  p=3, metric="chebyshev", leaf_size=90),
    dict(n_neighbors=25, weights="distance", p=5, metric="euclidean",  leaf_size=100),
    dict(n_neighbors=27, weights="uniform",  p=4, metric="minkowski", leaf_size=15),
    dict(n_neighbors=29, weights="distance", p=2, metric="manhattan",  leaf_size=20),
    dict(n_neighbors=31, weights="uniform",  p=3, metric="minkowski", leaf_size=25),

    dict(n_neighbors=33, weights="distance", p=4, metric="chebyshev", leaf_size=30),
    dict(n_neighbors=35, weights="uniform",  p=5, metric="euclidean",  leaf_size=35),
    dict(n_neighbors=37, weights="distance", p=2, metric="minkowski", leaf_size=40),
    dict(n_neighbors=39, weights="uniform",  p=3, metric="manhattan",  leaf_size=45),
    dict(n_neighbors=41, weights="distance", p=4, metric="minkowski", leaf_size=50),
    dict(n_neighbors=43, weights="uniform",  p=5, metric="chebyshev", leaf_size=55),
    dict(n_neighbors=45, weights="distance", p=2, metric="euclidean",  leaf_size=60),
    dict(n_neighbors=47, weights="uniform",  p=3, metric="minkowski", leaf_size=65),
    dict(n_neighbors=49, weights="distance", p=4, metric="manhattan",  leaf_size=70),
    dict(n_neighbors=51, weights="uniform",  p=5, metric="minkowski", leaf_size=75),
    dict(n_neighbors=60, weights="distance", p=2, metric="chebyshev", leaf_size=80),
    dict(n_neighbors=70, weights="uniform",  p=3, metric="euclidean",  leaf_size=85),
    dict(n_neighbors=80, weights="distance", p=4, metric="minkowski", leaf_size=90),
    dict(n_neighbors=90, weights="uniform",  p=5, metric="manhattan",  leaf_size=95),
    dict(n_neighbors=100,weights="distance", p=3, metric="minkowski", leaf_size=100),

    dict(n_neighbors=120,weights="uniform",  p=2, metric="minkowski", leaf_size=10),
    dict(n_neighbors=140,weights="distance", p=4, metric="euclidean",  leaf_size=15),
    dict(n_neighbors=160,weights="uniform",  p=5, metric="manhattan",  leaf_size=20),
    dict(n_neighbors=180,weights="distance", p=3, metric="minkowski", leaf_size=25),
    dict(n_neighbors=200,weights="uniform",  p=2, metric="chebyshev", leaf_size=30),
    dict(n_neighbors=100,weights="distance", p=4, metric="minkowski", leaf_size=35),
    dict(n_neighbors=150,weights="uniform",  p=5, metric="euclidean",  leaf_size=40),
    dict(n_neighbors=200,weights="distance", p=3, metric="manhattan",  leaf_size=45),
    dict(n_neighbors=175,weights="uniform",  p=2, metric="minkowski", leaf_size=50),

    dict(n_neighbors=125,weights="distance", p=4, metric="chebyshev", leaf_size=55),
    dict(n_neighbors=80, weights="uniform",  p=3, metric="minkowski", leaf_size=60),
    dict(n_neighbors=60, weights="distance", p=5, metric="euclidean",  leaf_size=65),
    dict(n_neighbors=40, weights="uniform",  p=2, metric="minkowski", leaf_size=70),
    dict(n_neighbors=20, weights="distance", p=3, metric="manhattan",  leaf_size=75),
    dict(n_neighbors=10, weights="uniform",  p=4, metric="minkowski", leaf_size=80),
    dict(n_neighbors=5,  weights="distance", p=5, metric="chebyshev", leaf_size=85),
]

def _can_fit_knn(n_neighbors, n_train):
    return n_neighbors <= n_train

knn_rows_in_order, knn_rows_for_best = [], []
n_train = len(X_train)

for cfg in table2_knn_params:
    n_neighbors = cfg["n_neighbors"]
    record = {
        "n_neighbors": n_neighbors,
        "weights": cfg["weights"],
        "p": cfg["p"],
        "metric": cfg["metric"],
        "leaf_size": cfg["leaf_size"],
    }
    if not _can_fit_knn(n_neighbors, n_train):
        print(f"[SKIP] K too large for training set: n_neighbors={n_neighbors} > n_train={n_train}")
        record.update({"train_acc": "N/A", "val_acc": "N/A", "precision": "N/A", "recall": "N/A", "f1": "N/A"})
        knn_rows_in_order.append(record)
        continue

    pipe = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=cfg["weights"],
            p=cfg["p"],
            metric=cfg["metric"],
            leaf_size=cfg["leaf_size"],
            n_jobs=-1,
            algorithm="auto"
        ))
    ])
    pipe.fit(X_train, y_train)
    y_tr_pred  = pipe.predict(X_train)
    y_val_pred = pipe.predict(X_val)

    tr_acc = accuracy_score(y_train, y_tr_pred)
    v_acc  = accuracy_score(y_val, y_val_pred)
    prec   = precision_score(y_val, y_val_pred, average="weighted", zero_division=0)
    rec    = recall_score(y_val, y_val_pred, average="weighted", zero_division=0)
    f1     = f1_score(y_val, y_val_pred, average="weighted", zero_division=0)

    print(f"[KNN] k={n_neighbors:>3}, weights={cfg['weights']:>8}, p={cfg['p']}, metric={cfg['metric']:>10}, "
          f"leaf={cfg['leaf_size']:>3} | train_acc={tr_acc:.4f} | val_acc={v_acc:.4f}")

    record.update({
        "train_acc": round(tr_acc, 4),
        "val_acc": round(v_acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
    })
    knn_rows_in_order.append(record)
    knn_rows_for_best.append(record)

if len(knn_rows_for_best) > 0:
    df_knn_sorted = pd.DataFrame(knn_rows_for_best).sort_values(
        ["val_acc", "f1", "precision", "recall"],
        ascending=[False, False, False, False]
    )
else:
    df_knn_sorted = pd.DataFrame(columns=["n_neighbors","weights","p","metric","leaf_size",
                                          "train_acc","val_acc","precision","recall","f1"])

df_knn_sorted.to_csv("table2_knn_results.csv", index=False)
pd.DataFrame(knn_rows_in_order).to_csv("table2_knn_results_in_order.csv", index=False)

print("\nSaved results to table2_knn_results.csv (sorted)")
print(df_knn_sorted.head(10))
print("\nSaved results to table2_knn_results_in_order.csv (original order incl. N/A)")

# Best KNN on test
if len(df_knn_sorted) > 0:
    best_knn = df_knn_sorted.iloc[0]
    best_cfg = dict(
        n_neighbors=int(best_knn["n_neighbors"]),
        weights=best_knn["weights"],
        p=int(best_knn["p"]),
        metric=best_knn["metric"],
        leaf_size=int(best_knn["leaf_size"])
    )
    best_knn_pipe = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(**best_cfg, n_jobs=-1, algorithm="auto"))
    ])
    print("\nBest KNN (by validation):", best_cfg)
    best_knn_pipe.fit(X_train_full, y_train_full)
    y_test_pred = best_knn_pipe.predict(X_test)
    print("\n----- Test Evaluation: KNN (Best from Table 2) -----")
    print(f"Accuracy : {accuracy_score(y_test, y_test_pred):.3f}")
    print(f"Precision: {precision_score(y_test, y_test_pred, average='weighted', zero_division=0):.3f}")
    print(f"Recall   : {recall_score(y_test, y_test_pred, average='weighted', zero_division=0):.3f}")
    print(f"F1-score : {f1_score(y_test, y_test_pred, average='weighted', zero_division=0):.3f}")
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_test_pred))
else:
    print("\nNo valid KNN configurations to evaluate on test set.")

# =========================================================
# ============ TABLE 3: Decision Tree sweep ===============
# =========================================================
table3_dt_params = [
    dict(criterion="gini", splitter="best",   max_depth=10, min_samples_split=2, min_samples_leaf=1, max_features="sqrt", max_leaf_nodes=None, min_impurity_decrease=0.0),
    dict(criterion="entropy", splitter="random", max_depth=15, min_samples_split=3, min_samples_leaf=2, max_features="log2", max_leaf_nodes=50, min_impurity_decrease=0.01),
    dict(criterion="log_loss", splitter="best",  max_depth=20, min_samples_split=4, min_samples_leaf=3, max_features=None, max_leaf_nodes=40, min_impurity_decrease=0.02),
    dict(criterion="gini", splitter="random",  max_depth=None, min_samples_split=5, min_samples_leaf=2, max_features="sqrt", max_leaf_nodes=60, min_impurity_decrease=0.0),
    dict(criterion="entropy", splitter="best", max_depth=8,  min_samples_split=6, min_samples_leaf=1, max_features="log2", max_leaf_nodes=None, min_impurity_decrease=0.03),

    dict(criterion="gini", splitter="best",   max_depth=12, min_samples_split=2, min_samples_leaf=4, max_features=None, max_leaf_nodes=30, min_impurity_decrease=0.01),
    dict(criterion="log_loss", splitter="random", max_depth=25, min_samples_split=3, min_samples_leaf=1, max_features="sqrt", max_leaf_nodes=70, min_impurity_decrease=0.0),
    dict(criterion="entropy", splitter="best", max_depth=18, min_samples_split=4, min_samples_leaf=5, max_features="log2", max_leaf_nodes=90, min_impurity_decrease=0.02),
    dict(criterion="gini", splitter="random",  max_depth=30, min_samples_split=5, min_samples_leaf=3, max_features=None, max_leaf_nodes=50, min_impurity_decrease=0.04),
    dict(criterion="entropy", splitter="best", max_depth=None, min_samples_split=2, min_samples_leaf=2, max_features="sqrt", max_leaf_nodes=80, min_impurity_decrease=0.01),

    dict(criterion="gini", splitter="best",   max_depth=7,  min_samples_split=3, min_samples_leaf=1, max_features="log2", max_leaf_nodes=None, min_impurity_decrease=0.0),
    dict(criterion="log_loss", splitter="random", max_depth=35, min_samples_split=4, min_samples_leaf=2, max_features=None, max_leaf_nodes=100, min_impurity_decrease=0.02),
    dict(criterion="entropy", splitter="best", max_depth=40, min_samples_split=5, min_samples_leaf=4, max_features="sqrt", max_leaf_nodes=70, min_impurity_decrease=0.03),
    dict(criterion="gini", splitter="random",  max_depth=22, min_samples_split=6, min_samples_leaf=3, max_features="log2", max_leaf_nodes=60, min_impurity_decrease=0.0),
    dict(criterion="entropy", splitter="best", max_depth=16, min_samples_split=8, min_samples_leaf=1, max_features=None, max_leaf_nodes=40, min_impurity_decrease=0.05),
    dict(criterion="log_loss", splitter="random", max_depth=None, min_samples_split=9, min_samples_leaf=2, max_features="sqrt", max_leaf_nodes=None, min_impurity_decrease=0.01),

    dict(criterion="gini", splitter="best",   max_depth=13, min_samples_split=10, min_samples_leaf=3, max_features="log2", max_leaf_nodes=30, min_impurity_decrease=0.0),
    dict(criterion="entropy", splitter="random", max_depth=25, min_samples_split=2,  min_samples_leaf=4, max_features=None, max_leaf_nodes=50, min_impurity_decrease=0.02),
    dict(criterion="gini", splitter="best",   max_depth=28, min_samples_split=3,  min_samples_leaf=5, max_features="sqrt", max_leaf_nodes=None, min_impurity_decrease=0.03),
    dict(criterion="log_loss", splitter="random", max_depth=32, min_samples_split=4,  min_samples_leaf=2, max_features="log2", max_leaf_nodes=90, min_impurity_decrease=0.0),
    dict(criterion="entropy", splitter="best", max_depth=45, min_samples_split=5,  min_samples_leaf=3, max_features=None, max_leaf_nodes=60, min_impurity_decrease=0.01),
    dict(criterion="gini", splitter="random",  max_depth=None, min_samples_split=6,  min_samples_leaf=1, max_features="sqrt", max_leaf_nodes=80, min_impurity_decrease=0.02),
    dict(criterion="log_loss", splitter="best",  max_depth=10, min_samples_split=8,  min_samples_leaf=4, max_features="log2", max_leaf_nodes=None, min_impurity_decrease=0.03),

    dict(criterion="entropy", splitter="random", max_depth=14, min_samples_split=9,  min_samples_leaf=2, max_features=None, max_leaf_nodes=50, min_impurity_decrease=0.04),
    dict(criterion="gini", splitter="best",   max_depth=17, min_samples_split=10, min_samples_leaf=1, max_features="sqrt", max_leaf_nodes=70, min_impurity_decrease=0.01),
    dict(criterion="log_loss", splitter="random", max_depth=19, min_samples_split=2,  min_samples_leaf=2, max_features="log2", max_leaf_nodes=100, min_impurity_decrease=0.0),
    dict(criterion="entropy", splitter="best", max_depth=21, min_samples_split=3,  min_samples_leaf=5, max_features=None, max_leaf_nodes=90, min_impurity_decrease=0.05),

    dict(criterion="gini", splitter="random",  max_depth=24, min_samples_split=4,  min_samples_leaf=3, max_features="sqrt", max_leaf_nodes=80, min_impurity_decrease=0.02),
    dict(criterion="entropy", splitter="best", max_depth=27, min_samples_split=5,  min_samples_leaf=4, max_features="log2", max_leaf_nodes=60, min_impurity_decrease=0.0),
    dict(criterion="log_loss", splitter="random", max_depth=29, min_samples_split=6,  min_samples_leaf=1, max_features=None, max_leaf_nodes=None, min_impurity_decrease=0.03),
    dict(criterion="gini", splitter="best",   max_depth=31, min_samples_split=7,  min_samples_leaf=2, max_features="sqrt", max_leaf_nodes=40, min_impurity_decrease=0.01),
    dict(criterion="entropy", splitter="random", max_depth=33, min_samples_split=8,  min_samples_leaf=5, max_features="log2", max_leaf_nodes=50, min_impurity_decrease=0.02),
    dict(criterion="gini", splitter="best",   max_depth=36, min_samples_split=9,  min_samples_leaf=3, max_features=None, max_leaf_nodes=70, min_impurity_decrease=0.04),
    dict(criterion="entropy", splitter="random", max_depth=38, min_samples_split=10, min_samples_leaf=1, max_features="sqrt", max_leaf_nodes=80, min_impurity_decrease=0.0),
    dict(criterion="log_loss", splitter="best",  max_depth=41, min_samples_split=2,  min_samples_leaf=3, max_features="log2", max_leaf_nodes=100, min_impurity_decrease=0.01),
    dict(criterion="gini", splitter="random",  max_depth=None, min_samples_split=3,  min_samples_leaf=4, max_features=None, max_leaf_nodes=None, min_impurity_decrease=0.05),

    dict(criterion="entropy", splitter="best", max_depth=43, min_samples_split=4,  min_samples_leaf=3, max_features="sqrt", max_leaf_nodes=90, min_impurity_decrease=0.02),
    dict(criterion="gini", splitter="random",  max_depth=46, min_samples_split=5,  min_samples_leaf=1, max_features="log2", max_leaf_nodes=80, min_impurity_decrease=0.03),
    dict(criterion="log_loss", splitter="best",  max_depth=48, min_samples_split=6,  min_samples_leaf=5, max_features=None, max_leaf_nodes=60, min_impurity_decrease=0.0),
    dict(criterion="entropy", splitter="random", max_depth=50, min_samples_split=7,  min_samples_leaf=2, max_features="sqrt", max_leaf_nodes=70, min_impurity_decrease=0.04),
    dict(criterion="gini", splitter="best",   max_depth=9,  min_samples_split=8,  min_samples_leaf=1, max_features="log2", max_leaf_nodes=90, min_impurity_decrease=0.02),
    dict(criterion="entropy", splitter="random", max_depth=11, min_samples_split=9,  min_samples_leaf=2, max_features=None, max_leaf_nodes=None, min_impurity_decrease=0.01),
    dict(criterion="log_loss", splitter="best",  max_depth=13, min_samples_split=10, min_samples_leaf=3, max_features="sqrt", max_leaf_nodes=40, min_impurity_decrease=0.0),
    dict(criterion="gini", splitter="random",  max_depth=15, min_samples_split=2,  min_samples_leaf=4, max_features="log2", max_leaf_nodes=50, min_impurity_decrease=0.03),
    dict(criterion="entropy", splitter="best", max_depth=18, min_samples_split=3,  min_samples_leaf=5, max_features=None, max_leaf_nodes=70, min_impurity_decrease=0.05),
    dict(criterion="gini", splitter="random",  max_depth=20, min_samples_split=4,  min_samples_leaf=2, max_features="sqrt", max_leaf_nodes=80, min_impurity_decrease=0.02),
]

dt_rows_in_order, dt_rows_for_best = [], []

for cfg in table3_dt_params:
    # Build and train (trees don't need scaling)
    dt = DecisionTreeClassifier(
        criterion=cfg["criterion"],
        splitter=cfg["splitter"],
        max_depth=cfg["max_depth"],
        min_samples_split=cfg["min_samples_split"],
        min_samples_leaf=cfg["min_samples_leaf"],
        max_features=cfg["max_features"],
        max_leaf_nodes=cfg["max_leaf_nodes"],
        min_impurity_decrease=cfg["min_impurity_decrease"],
        random_state=42,
    )
    dt.fit(X_train, y_train)

    # Predictions
    y_tr_pred  = dt.predict(X_train)
    y_val_pred = dt.predict(X_val)

    # Metrics
    tr_acc = accuracy_score(y_train, y_tr_pred)
    v_acc  = accuracy_score(y_val, y_val_pred)
    prec   = precision_score(y_val, y_val_pred, average="weighted", zero_division=0)
    rec    = recall_score(y_val, y_val_pred, average="weighted", zero_division=0)
    f1     = f1_score(y_val, y_val_pred, average="weighted", zero_division=0)

    # Console quick view
    print(f"[DT] crit={cfg['criterion']:>8}, split={cfg['splitter']:>6}, "
          f"depth={str(cfg['max_depth']):>4}, mss={cfg['min_samples_split']}, "
          f"msl={cfg['min_samples_leaf']}, feat={str(cfg['max_features']):>5}, "
          f"leafs={str(cfg['max_leaf_nodes']):>4} | train_acc={tr_acc:.4f} | val_acc={v_acc:.4f}")

    row = {
        **cfg,
        "train_acc": round(tr_acc, 4),
        "val_acc": round(v_acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
    }
    dt_rows_in_order.append(row)
    dt_rows_for_best.append(row)

# Save Decision Tree results
df_dt_sorted = pd.DataFrame(dt_rows_for_best).sort_values(
    ["val_acc", "f1", "precision", "recall"],
    ascending=[False, False, False, False]
)
df_dt_sorted.to_csv("table3_decisiontree_results.csv", index=False)
pd.DataFrame(dt_rows_in_order).to_csv("table3_decisiontree_results_in_order.csv", index=False)

print("\nSaved results to table3_decisiontree_results.csv (sorted)")
print(df_dt_sorted.head(10))
print("\nSaved results to table3_decisiontree_results_in_order.csv (original order)")

# Best Decision Tree on test
if len(df_dt_sorted) > 0:
    best_dt_row = df_dt_sorted.iloc[0]
    best_dt = DecisionTreeClassifier(
        criterion=best_dt_row["criterion"],
        splitter=best_dt_row["splitter"],
        max_depth=None if str(best_dt_row["max_depth"])=="None" else int(best_dt_row["max_depth"]),
        min_samples_split=int(best_dt_row["min_samples_split"]),
        min_samples_leaf=int(best_dt_row["min_samples_leaf"]),
        max_features=None if str(best_dt_row["max_features"])=="None" else best_dt_row["max_features"],
        max_leaf_nodes=None if str(best_dt_row["max_leaf_nodes"])=="None" else int(best_dt_row["max_leaf_nodes"]),
        min_impurity_decrease=float(best_dt_row["min_impurity_decrease"]),
        random_state=42,
    )
    print("\nBest Decision Tree (by validation):", dict(best_dt_row))
    best_dt.fit(X_train_full, y_train_full)
    y_test_pred = best_dt.predict(X_test)

    print("\n----- Test Evaluation: Decision Tree (Best from Table 3) -----")
    print(f"Accuracy : {accuracy_score(y_test, y_test_pred):.3f}")
    print(f"Precision: {precision_score(y_test, y_test_pred, average='weighted', zero_division=0):.3f}")
    print(f"Recall   : {recall_score(y_test, y_test_pred, average='weighted', zero_division=0):.3f}")
    print(f"F1-score : {f1_score(y_test, y_test_pred, average='weighted', zero_division=0):.3f}")
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_test_pred))

# =========================================================
# ============ TABLE 4: Random Forest sweep ===============
# =========================================================
table4_rf_params = [
    dict(n_estimators=10,  criterion="gini",    max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features="sqrt",  bootstrap=True),
    dict(n_estimators=20,  criterion="entropy", max_depth=5,    min_samples_split=3, min_samples_leaf=2, max_features="log2",  bootstrap=True),
    dict(n_estimators=30,  criterion="gini",    max_depth=10,   min_samples_split=4, min_samples_leaf=1, max_features=None,    bootstrap=False),
    dict(n_estimators=40,  criterion="entropy", max_depth=8,    min_samples_split=2, min_samples_leaf=3, max_features="sqrt",  bootstrap=True),
    dict(n_estimators=50,  criterion="gini",    max_depth=15,   min_samples_split=5, min_samples_leaf=2, max_features="log2",  bootstrap=True),
    dict(n_estimators=60,  criterion="entropy", max_depth=None, min_samples_split=6, min_samples_leaf=1, max_features=None,    bootstrap=True),
    dict(n_estimators=70,  criterion="gini",    max_depth=12,   min_samples_split=3, min_samples_leaf=2, max_features="sqrt",  bootstrap=False),
    dict(n_estimators=80,  criterion="entropy", max_depth=20,   min_samples_split=4, min_samples_leaf=3, max_features="log2",  bootstrap=True),
    dict(n_estimators=90,  criterion="gini",    max_depth=25,   min_samples_split=5, min_samples_leaf=2, max_features=None,    bootstrap=True),
    dict(n_estimators=100, criterion="entropy", max_depth=30,   min_samples_split=10,min_samples_leaf=4, max_features="sqrt",  bootstrap=True),

    dict(n_estimators=120, criterion="gini",    max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features="log2",  bootstrap=False),
    dict(n_estimators=140, criterion="entropy", max_depth=18,   min_samples_split=3, min_samples_leaf=2, max_features="sqrt",  bootstrap=True),
    dict(n_estimators=160, criterion="gini",    max_depth=22,   min_samples_split=4, min_samples_leaf=3, max_features=None,    bootstrap=True),
    dict(n_estimators=180, criterion="entropy", max_depth=25,   min_samples_split=5, min_samples_leaf=2, max_features="sqrt",  bootstrap=False),
    dict(n_estimators=200, criterion="gini",    max_depth=28,   min_samples_split=6, min_samples_leaf=1, max_features="log2",  bootstrap=True),
    dict(n_estimators=220, criterion="entropy", max_depth=None, min_samples_split=8, min_samples_leaf=2, max_features=None,    bootstrap=True),
    dict(n_estimators=240, criterion="gini",    max_depth=14,   min_samples_split=3, min_samples_leaf=3, max_features="sqrt",  bootstrap=True),
    dict(n_estimators=260, criterion="entropy", max_depth=16,   min_samples_split=2, min_samples_leaf=2, max_features="log2",  bootstrap=False),

    dict(n_estimators=280, criterion="gini",    max_depth=19,   min_samples_split=7, min_samples_leaf=2, max_features=None,    bootstrap=True),
    dict(n_estimators=300, criterion="entropy", max_depth=21,   min_samples_split=9, min_samples_leaf=3, max_features="sqrt",  bootstrap=True),
    dict(n_estimators=350, criterion="gini",    max_depth=24,   min_samples_split=10,min_samples_leaf=4, max_features="log2",  bootstrap=True),
    dict(n_estimators=400, criterion="entropy", max_depth=None, min_samples_split=5, min_samples_leaf=1, max_features=None,    bootstrap=False),
    dict(n_estimators=450, criterion="gini",    max_depth=26,   min_samples_split=4, min_samples_leaf=2, max_features="sqrt",  bootstrap=True),
    dict(n_estimators=500, criterion="entropy", max_depth=18,   min_samples_split=2, min_samples_leaf=2, max_features="log2",  bootstrap=True),

    dict(n_estimators=550, criterion="gini",    max_depth=30,   min_samples_split=7, min_samples_leaf=3, max_features=None,    bootstrap=True),
    dict(n_estimators=600, criterion="entropy", max_depth=None, min_samples_split=6, min_samples_leaf=2, max_features="sqrt",  bootstrap=True),
    dict(n_estimators=650, criterion="gini",    max_depth=22,   min_samples_split=8, min_samples_leaf=4, max_features="log2",  bootstrap=False),
    dict(n_estimators=700, criterion="entropy", max_depth=25,   min_samples_split=5, min_samples_leaf=2, max_features=None,    bootstrap=True),
    dict(n_estimators=750, criterion="gini",    max_depth=None, min_samples_split=9, min_samples_leaf=1, max_features="sqrt",  bootstrap=True),
    dict(n_estimators=800, criterion="entropy", max_depth=32,   min_samples_split=3, min_samples_leaf=2, max_features="log2",  bootstrap=True),
    dict(n_estimators=850, criterion="gini",    max_depth=28,   min_samples_split=4, min_samples_leaf=2, max_features=None,    bootstrap=False),
    dict(n_estimators=900, criterion="entropy", max_depth=None, min_samples_split=7, min_samples_leaf=3, max_features="sqrt",  bootstrap=True),
    dict(n_estimators=950, criterion="gini",    max_depth=20,   min_samples_split=5, min_samples_leaf=1, max_features="log2",  bootstrap=True),
    dict(n_estimators=1000,criterion="entropy", max_depth=25,   min_samples_split=10,min_samples_leaf=3, max_features=None,    bootstrap=True),

    dict(n_estimators=60,  criterion="gini",    max_depth=8,    min_samples_split=2, min_samples_leaf=2, max_features="sqrt",  bootstrap=True),
    dict(n_estimators=90,  criterion="entropy", max_depth=12,   min_samples_split=4, min_samples_leaf=1, max_features="log2",  bootstrap=False),
    dict(n_estimators=150, criterion="gini",    max_depth=16,   min_samples_split=5, min_samples_leaf=2, max_features=None,    bootstrap=True),
    dict(n_estimators=250, criterion="entropy", max_depth=18,   min_samples_split=6, min_samples_leaf=2, max_features="sqrt",  bootstrap=True),
    dict(n_estimators=350, criterion="gini",    max_depth=None, min_samples_split=3, min_samples_leaf=2, max_features="log2",  bootstrap=True),
    dict(n_estimators=450, criterion="entropy", max_depth=20,   min_samples_split=5, min_samples_leaf=3, max_features=None,    bootstrap=False),
    dict(n_estimators=550, criterion="gini",    max_depth=30,   min_samples_split=2, min_samples_leaf=2, max_features="sqrt",  bootstrap=True),
    dict(n_estimators=650, criterion="entropy", max_depth=28,   min_samples_split=7, min_samples_leaf=3, max_features="log2",  bootstrap=True),
    dict(n_estimators=750, criterion="gini",    max_depth=None, min_samples_split=6, min_samples_leaf=1, max_features=None,    bootstrap=True),
    dict(n_estimators=850, criterion="entropy", max_depth=35,   min_samples_split=3, min_samples_leaf=2, max_features="sqrt",  bootstrap=True),
    dict(n_estimators=950, criterion="gini",    max_depth=40,   min_samples_split=9, min_samples_leaf=4, max_features="log2",  bootstrap=True),
    dict(n_estimators=1000,criterion="entropy", max_depth=None, min_samples_split=8, min_samples_leaf=2, max_features=None,    bootstrap=True),
    

]

rf_rows_in_order, rf_rows_for_best = [], []

for cfg in table4_rf_params:
    rf = RandomForestClassifier(
        n_estimators=cfg["n_estimators"],
        criterion=cfg["criterion"],
        max_depth=cfg["max_depth"],
        min_samples_split=cfg["min_samples_split"],
        min_samples_leaf=cfg["min_samples_leaf"],
        max_features=cfg["max_features"],
        bootstrap=cfg["bootstrap"],
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X_train, y_train)
    y_tr_pred  = rf.predict(X_train)
    y_val_pred = rf.predict(X_val)

    tr_acc = accuracy_score(y_train, y_tr_pred)
    v_acc  = accuracy_score(y_val, y_val_pred)
    prec   = precision_score(y_val, y_val_pred, average="weighted", zero_division=0)
    rec    = recall_score(y_val, y_val_pred, average="weighted", zero_division=0)
    f1     = f1_score(y_val, y_val_pred, average="weighted", zero_division=0)

    print(f"[RF] n={cfg['n_estimators']:>4}, crit={cfg['criterion']:>7}, depth={str(cfg['max_depth']):>4}, "
          f"mss={cfg['min_samples_split']}, msl={cfg['min_samples_leaf']}, feat={str(cfg['max_features']):>5}, "
          f"boot={cfg['bootstrap']} | train_acc={tr_acc:.4f} | val_acc={v_acc:.4f}")

    row = {**cfg, "train_acc": round(tr_acc,4), "val_acc": round(v_acc,4),
           "precision": round(prec,4), "recall": round(rec,4), "f1": round(f1,4)}
    rf_rows_in_order.append(row)
    rf_rows_for_best.append(row)

# Save Table 4
df_rf_sorted = pd.DataFrame(rf_rows_for_best).sort_values(
    ["val_acc","f1","precision","recall"], ascending=[False,False,False,False]
)
df_rf_sorted.to_csv("table4_randomforest_results.csv", index=False)
pd.DataFrame(rf_rows_in_order).to_csv("table4_randomforest_results_in_order.csv", index=False)

print("\nSaved results to table4_randomforest_results.csv (sorted)")
print(df_rf_sorted.head(10))

# Best RF on test
if len(df_rf_sorted) > 0:
    best_rf_cfg = df_rf_sorted.iloc[0].to_dict()
    best_rf = RandomForestClassifier(
        n_estimators=int(best_rf_cfg["n_estimators"]),
        criterion=best_rf_cfg["criterion"],
        max_depth=None if str(best_rf_cfg["max_depth"])=="None" else int(best_rf_cfg["max_depth"]),
        min_samples_split=int(best_rf_cfg["min_samples_split"]),
        min_samples_leaf=int(best_rf_cfg["min_samples_leaf"]),
        max_features=None if str(best_rf_cfg["max_features"])=="None" else best_rf_cfg["max_features"],
        bootstrap=bool(best_rf_cfg["bootstrap"]),
        n_jobs=-1,
        random_state=42,
    )
    print("\nBest RF (by validation):", best_rf_cfg)
    best_rf.fit(X_train_full, y_train_full)
    y_test_pred = best_rf.predict(X_test)

    print("\n----- Test Evaluation: Random Forest (Best from Table 4) -----")
    print(f"Accuracy : {accuracy_score(y_test, y_test_pred):.3f}")
    print(f"Precision: {precision_score(y_test, y_test_pred, average='weighted', zero_division=0):.3f}")
    print(f"Recall   : {recall_score(y_test, y_test_pred, average='weighted', zero_division=0):.3f}")
    print(f"F1-score : {f1_score(y_test, y_test_pred, average='weighted', zero_division=0):.3f}")
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_test_pred))

# ============ TABLE 5: Support Vector Classifier =========
# =========================================================
svc_rows_in_order, svc_rows_for_best = [], []

# Linear kernel block
linear_grid = [
    dict(kernel="linear", C=0.1, tol=1e-4,  max_iter=1000),
    dict(kernel="linear", C=0.2, tol=5e-4,  max_iter=2000),
    dict(kernel="linear", C=0.5, tol=1e-3,  max_iter=3000),
    dict(kernel="linear", C=1.0, tol=2e-4,  max_iter=4000),
    dict(kernel="linear", C=1.5, tol=3e-4,  max_iter=5000),
    dict(kernel="linear", C=2.0, tol=1e-4,  max_iter=6000),
    dict(kernel="linear", C=3.0, tol=5e-4,  max_iter=7000),
    dict(kernel="linear", C=4.0, tol=1e-3,  max_iter=8000),
    dict(kernel="linear", C=5.0, tol=2e-4,  max_iter=9000),
    dict(kernel="linear", C=6.0, tol=3e-4,  max_iter=10000),
]

# Polynomial kernel block
poly_grid = [
    dict(kernel="poly", C=0.1, gamma=0.01, degree=2, coef0=0.0,  tol=1e-4,  max_iter=1000),
    dict(kernel="poly", C=0.2, gamma=0.02, degree=2, coef0=0.05, tol=5e-4,  max_iter=2000),
    dict(kernel="poly", C=0.5, gamma=0.03, degree=3, coef0=0.10, tol=1e-3,  max_iter=1500),
    dict(kernel="poly", C=1.0, gamma=0.04, degree=3, coef0=0.0,  tol=2e-4,  max_iter=2500),
    dict(kernel="poly", C=1.5, gamma=0.05, degree=4, coef0=0.05, tol=3e-4,  max_iter=3000),
    dict(kernel="poly", C=2.0, gamma=0.06, degree=4, coef0=0.10, tol=1e-4,  max_iter=3500),
    dict(kernel="poly", C=2.5, gamma=0.07, degree=5, coef0=0.0,  tol=5e-4,  max_iter=4000),
    dict(kernel="poly", C=3.0, gamma=0.08, degree=5, coef0=0.05, tol=1e-3,  max_iter=4500),
    dict(kernel="poly", C=3.5, gamma=0.09, degree=6, coef0=0.10, tol=2e-4,  max_iter=5000),
    dict(kernel="poly", C=4.0, gamma=0.10, degree=6, coef0=0.0,  tol=3e-4,  max_iter=5500),
    dict(kernel="poly", C=4.5, gamma=0.12, degree=7, coef0=0.05, tol=1e-4,  max_iter=1000),
    dict(kernel="poly", C=5.0, gamma=0.15, degree=7, coef0=0.10, tol=5e-4,  max_iter=2000),
    dict(kernel="poly", C=5.5, gamma=0.20, degree=8, coef0=0.0,  tol=1e-3,  max_iter=1500),
    dict(kernel="poly", C=6.0, gamma=0.25, degree=8, coef0=0.05, tol=2e-4,  max_iter=2500),
    dict(kernel="poly", C=6.5, gamma=0.30, degree=8, coef0=0.10, tol=3e-4, max_iter=3000),
]

# RBF kernel block
rbf_grid = [
    dict(kernel="rbf", C=0.1, gamma=0.01, tol=1e-4,  max_iter=1000),
    dict(kernel="rbf", C=0.2, gamma=0.02, tol=5e-4,  max_iter=1500),
    dict(kernel="rbf", C=0.5, gamma=0.03, tol=1e-3,  max_iter=2000),
    dict(kernel="rbf", C=1.0, gamma=0.04, tol=2e-4,  max_iter=2500),
    dict(kernel="rbf", C=1.5, gamma=0.05, tol=3e-4,  max_iter=3000),
    dict(kernel="rbf", C=2.0, gamma=0.06, tol=1e-4,  max_iter=3500),
    dict(kernel="rbf", C=2.5, gamma=0.07, tol=5e-4,  max_iter=4000),
    dict(kernel="rbf", C=3.0, gamma=0.08, tol=1e-3,  max_iter=4500),
    dict(kernel="rbf", C=3.5, gamma=0.09, tol=2e-4,  max_iter=5000),
    dict(kernel="rbf", C=4.0, gamma=0.10, tol=3e-4, max_iter=5500),
    dict(kernel="rbf", C=4.5, gamma=0.12, tol=1e-4, max_iter=1000),
    dict(kernel="rbf", C=5.0, gamma=0.15, tol=5e-4, max_iter=1500),
    dict(kernel="rbf", C=5.5, gamma=0.20, tol=1e-3, max_iter=2000),
    dict(kernel="rbf", C=6.0, gamma=0.25, tol=2e-4, max_iter=2500),
    dict(kernel="rbf", C=6.5, gamma=0.30, tol=3e-4, max_iter=3000),
]

# Sigmoid kernel block
sigmoid_grid = [
    dict(kernel="sigmoid", C=0.1, gamma=0.01, coef0=0.0,  tol=1e-4,  max_iter=1000),
    dict(kernel="sigmoid", C=0.2, gamma=0.02, coef0=0.05, tol=5e-4,  max_iter=1500),
    dict(kernel="sigmoid", C=0.5, gamma=0.03, coef0=0.10, tol=1e-3,  max_iter=2000),
    dict(kernel="sigmoid", C=1.0, gamma=0.04, coef0=0.0,  tol=2e-4,  max_iter=2500),
    dict(kernel="sigmoid", C=1.5, gamma=0.05, coef0=0.05, tol=3e-4,  max_iter=3000),
    dict(kernel="sigmoid", C=2.0, gamma=0.06, coef0=0.10, tol=1e-4,  max_iter=3500),
    dict(kernel="sigmoid", C=2.5, gamma=0.07, coef0=0.0,  tol=5e-4,  max_iter=4000),
]

svc_grid = linear_grid + poly_grid + rbf_grid + sigmoid_grid

def _rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

for cfg in svc_grid:
    # Build SVC with scaling
    svc_kwargs = cfg.copy()
    kernel = svc_kwargs.pop("kernel")
    degree = svc_kwargs.pop("degree", 3)
    gamma  = svc_kwargs.pop("gamma", "scale")
    coef0  = svc_kwargs.pop("coef0", 0.0)

    pipe = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, **svc_kwargs, probability=False))
    ])

    pipe.fit(X_train, y_train)
    y_tr_pred  = pipe.predict(X_train)
    y_val_pred = pipe.predict(X_val)

    tr_rmse = _rmse(y_train, y_tr_pred)
    v_rmse  = _rmse(y_val,   y_val_pred)

    print(f"[SVC] k={kernel:>7}, C={cfg.get('C'):>3}, "
          f"deg={cfg.get('degree','-')}, gamma={cfg.get('gamma','-')}, "
          f"coef0={cfg.get('coef0','-')} | train_RMSE={tr_rmse:.4f} | val_RMSE={v_rmse:.4f}")

    row = {**cfg, "train_rmse": round(tr_rmse,4), "val_rmse": round(v_rmse,4)}
    svc_rows_in_order.append(row)
    svc_rows_for_best.append(row)

# Save Table 5 (sorted by lowest Val. RMSE)
df_svc_sorted = pd.DataFrame(svc_rows_for_best).sort_values(
    ["val_rmse", "train_rmse"], ascending=[True, True]
)
df_svc_sorted.to_csv("table5_svc_results.csv", index=False)
pd.DataFrame(svc_rows_in_order).to_csv("table5_svc_results_in_order.csv", index=False)

print("\nSaved results to table5_svc_results.csv (sorted)")
print(df_svc_sorted.head(10))

# Best SVC on test (we'll still report standard classification metrics on the held-out test set)
if len(df_svc_sorted) > 0:
    best = df_svc_sorted.iloc[0].to_dict()
    best_kernel = best["kernel"]
    best_kwargs = dict(C=float(best["C"]), tol=float(best["tol"]), max_iter=int(best["max_iter"]))
    if best_kernel == "linear":
        degree, gamma, coef0 = 3, "scale", 0.0
    elif best_kernel == "poly":
        degree, gamma, coef0 = int(best["degree"]), float(best["gamma"]), float(best["coef0"])
    elif best_kernel == "rbf":
        degree, gamma, coef0 = 3, float(best["gamma"]), 0.0
    else:  # sigmoid
        degree, gamma, coef0 = 3, float(best["gamma"]), float(best["coef0"])

    best_pipe = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel=best_kernel, degree=degree, gamma=gamma, coef0=coef0, **best_kwargs))
    ])

    print("\nBest SVC (by lowest Val. RMSE):", best)
    best_pipe.fit(X_train_full, y_train_full)
    y_test_pred = best_pipe.predict(X_test)

    print("\n----- Test Evaluation: SVC (Best from Table 5) -----")
    print(f"Accuracy : {accuracy_score(y_test, y_test_pred):.3f}")
    print(f"Precision: {precision_score(y_test, y_test_pred, average='weighted', zero_division=0):.3f}")
    print(f"Recall   : {recall_score(y_test, y_test_pred, average='weighted', zero_division=0):.3f}")
    print(f"F1-score : {f1_score(y_test, y_test_pred, average='weighted', zero_division=0):.3f}")
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_test_pred))
else:
    print("\nNo valid SVC configurations to evaluate on test set.")