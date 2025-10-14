import glob
from pathlib import Path
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray, rgba2rgb
from skimage.transform import resize
from skimage.feature import hog
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

#Load dataset into the compiler
dataset_path = Path.cwd() / "Utensil images"

# Classes (the classes correspond to the folder names)
classes = ["FORK", "KNIFE", "SPOON"]

X, y = [], []

# Loop through each class folder to read its images
for label, cls in enumerate(classes):
    # load all PNG images
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
        features, _ = hog(img_resized, 
                          pixels_per_cell=(8, 8), 
                          cells_per_block=(2, 2), 
                          visualize=True)

        X.append(features)
        y.append(label)

X = np.array(X)
y = np.array(y)

# Safety check: make sure we have data
print("Total samples:", len(X))

# Train/test split
# First split into train (70%) and temp (30%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# Split equally into validation (15%) and test (15%)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

X_train_full = np.vstack((X_train, X_val))
y_train_full = np.concatenate((y_train, y_val))

# Train SVM for model selection
clf = svm.SVC(kernel='rbf', C=1.0, max_iter = -1, coef0 = 0.0, tol=1e-3, degree=3)
clf.fit(X_train, y_train)

Train_flag = True
Test_flag = False

if Train_flag:
    # Evaluate
    y_pred_rs_val = clf.predict(X_val)
    
    train_acc_rs = clf.score(X_train, y_train)
    val_acc_rs = clf.score(X_val,y_val)
    
    print("--------Random Split-----------")
    print(f"Training Accuracy: {train_acc_rs:.2f}")
    print(f"Validation Accuracy: {val_acc_rs:.2f}")
    # Confusion Matrix
    cm_val_rand_split = confusion_matrix(y_test, y_pred_rs_val)
    print("\nValidation Confusion Matrix:\n", cm_val_rand_split)
    print("\nValidation Classification Report:\n", classification_report(y_val, y_pred_rs_val, target_names=classes))
    
    #### Uncomment (Ctrl +1) lines 92 to 99 once you determine the best hyperparameters
    ## Evaluating the performance on the test data 
    clf_final = clf.fit(X_train_full, y_train_full)
    
    y_pred = clf_final.predict(X_test)
    test_acc = clf_final.score(X_test, y_test)
    print(f"Test Accuracy: {test_acc:.2f}")
    print("\nTest Classification Report:\n", classification_report(y_test, y_pred, target_names=classes))
    cm_test_rand_split = confusion_matrix(y_test, y_pred)
    print("\Test Confusion Matrix:\n", cm_test_rand_split)
    
    #---------------------------K-fold-------------------------------------------
    print("\n--------K-fold Cross Validation-----------")
    kfold = KFold(n_splits=15, shuffle=True, random_state=42)
    
    # Run K-fold CV
    train_scores_kf = []
    val_scores_kf = []
    all_y_true_kf = []
    all_y_pred_kf = []
    
    fold = 1
    for train_idx, val_idx in kfold.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        
        clf.fit(X_train, y_train)
        
        # Training accuracy
        train_acc_kf = clf.score(X_train, y_train)
        train_scores_kf.append(train_acc_kf)
        
        # Test accuracy (on validation fold)
        val_acc_kf = clf.score(X_val, y_val)
        val_scores_kf.append(val_acc_kf)
        
        # Store predictions for overall metrics
        y_pred_val_kf = clf.predict(X_val)
        all_y_true_kf.extend(y_val)
        all_y_pred_kf.extend(y_pred_val_kf)
        
        print(f"Fold {fold} -> Train Acc: {train_acc_kf:.2f}, Validation Acc: {val_acc_kf:.2f}")
        fold += 1
    
    print("\nAverage Training Accuracy:", round(np.mean(train_scores_kf),2))
    print("Average Validation Accuracy:", round(np.mean(val_scores_kf),2))
    print("\nCross-validation Confusion Matrix:\n", confusion_matrix(all_y_true_kf, all_y_pred_kf))
    print("\nCross-validation Classification Report:\n", classification_report(all_y_true_kf, all_y_pred_kf, target_names=classes))
    
    
    ###-------------------------Repeated K-fold-------------------------------------
    print("\n--------Repeated K-fold Cross Validation-----------")
    rkf = RepeatedKFold(n_splits=15, n_repeats=5, random_state=42)
    
    train_scores_rkf = []
    val_scores_rkf = []
    all_y_true_rkf = []
    all_y_pred_rkf = []
    
    fold = 1
    for train_idx, val_idx in rkf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        #Using the same model from random splitting (line 59)
        clf_rkf = svm.SVC(kernel='rbf', C=1.0, max_iter = -1, coef0 = 0.0, tol=1e-3, degree=3)
        clf.fit(X_train, y_train)
        
        train_acc_rkf = clf.score(X_train, y_train)
        val_acc_rkf = clf.score(X_val, y_val)
        
        train_scores_rkf.append(train_acc_rkf)
        val_scores_rkf.append(val_acc_rkf)
        
        # Collect predictions for overall report
        y_pred_rkf = clf.predict(X_val)
        all_y_true_rkf.extend(y_val)
        all_y_pred_rkf.extend(y_pred_rkf)
        
        print(f"Split {fold} -> Train Acc: {train_acc_rkf:.2f}, Validation Acc: {val_acc_rkf:.2f}")
        fold += 1
    
    print("\nAverage Training Accuracy:", round(np.mean(train_scores_rkf),2))
    print("Average Validation Accuracy:", round(np.mean(val_scores_rkf),2))
    print("\nCross-validation Confusion Matrix:\n", confusion_matrix(all_y_true_rkf, all_y_pred_rkf))
    print("\nCross-validation Classification Report:\n", classification_report(all_y_true_rkf, all_y_pred_rkf, target_names=classes))

##------------------------%Final testing
#### Uncomment (Ctrl +1) lines 181 to 187 once you determine the best hyperparameters
## Evaluating the performance on the test data 
if Test_flag:
     clf_final = clf.fit(X_train_full, y_train_full)
     y_pred = clf_final.predict(X_test)
     test_acc = clf_final.score(X_test, y_test)
     print(f"Test Accuracy: {test_acc:.2f}")
     print("\nTest Classification Report:\n", classification_report(y_test, y_pred, target_names=classes))
     cm_test_rand_split = confusion_matrix(y_test, y_pred)
     print("\Test Confusion Matrix:\n", cm_test_rand_split)
     
     
# ==== 0) Small fixes to your current code ====
# - confusion_matrix mismatch (you used y_test with y_pred_rs_val); fix to y_val
# - repeated KFold block created clf_rkf but used clf; use the same variable consistently
# - optional: prefer Stratified splits for classification; the lab says K-Fold, but stratified is safer for class balance

# ==== 1) Utility: comment labeler for over/underfitting ====
def generalization_comment(train_acc: float, val_acc: float) -> str:
    gap = train_acc - val_acc
    # thresholds you can tune a bit
    HIGH = 0.85          # "high accuracy" bar
    LOW  = 0.70          # "low accuracy" bar
    GAP_OVERFIT = 0.05   # train >> val beyond this => overfit
    GAP_CLOSE   = 0.03   # train ~ val within this => close

    # Clear underfitting: both low
    if train_acc < LOW and val_acc < LOW:
        return "underfitting"

    # Overfitting: training much higher than validation
    if gap >= GAP_OVERFIT and train_acc >= HIGH:
        return "overfitting"

    # Good generalization: both decent and close
    if val_acc >= HIGH and abs(gap) <= GAP_CLOSE:
        return "good generalization"

    # Borderline/ambiguous cases
    if val_acc < LOW and abs(gap) <= GAP_CLOSE:
        return "underfitting"
    if gap > 0 and val_acc >= LOW:
        return "mild overfitting"
    if gap < 0 and (train_acc >= LOW or val_acc >= LOW):
        return "possible underfit (val>train)"

    return "adequate / needs tuning"

# ==== 2) Table 1 hyperparameter combinations ====
# Encode each row of Table 1 as a dict of SVC(**params). Only include keys that SVC actually uses for the kernel.
# degree and coef0 are ignored by SVC for kernels that don’t use them (safe to pass anyway).
TABLE1 = [
    # ---- linear ----
    dict(kernel="linear", C=0.1,  max_iter=-1, coef0=0.0, tol=1e-3, degree=3),
    dict(kernel="linear", C=1.0,  max_iter=-1, coef0=0.0, tol=1e-4, degree=3),
    dict(kernel="linear", C=10.0, max_iter=-1, coef0=0.0, tol=1e-3, degree=3),
    dict(kernel="linear", C=0.5,  max_iter=-1, coef0=0.0, tol=1e-4, degree=3),
    dict(kernel="linear", C=5.0,  max_iter=-1, coef0=0.0, tol=1e-2, degree=3),
    dict(kernel="linear", C=1.0,  max_iter=5000, coef0=0.0, tol=1e-3, degree=3),
    dict(kernel="linear", C=0.1,  max_iter=10000, coef0=0.0, tol=1e-3, degree=3),
    dict(kernel="linear", C=2.0,  max_iter=2000,  coef0=0.0, tol=1e-4, degree=3),
    dict(kernel="linear", C=0.5,  max_iter=-1,    coef0=0.0, tol=1e-2, degree=3),
    dict(kernel="linear", C=3.0,  max_iter=-1,    coef0=0.0, tol=1e-3, degree=3),
    dict(kernel="linear", C=1.0,  max_iter=1000,  coef0=0.0, tol=1e-2, degree=3),
    dict(kernel="linear", C=0.05, max_iter=-1,    coef0=0.0, tol=1e-4, degree=3),

    # ---- sigmoid ----
    dict(kernel="sigmoid", C=0.1, gamma=0.01, coef0=0.0,  tol=1e-3,  max_iter=-1),
    dict(kernel="sigmoid", C=1.0, gamma=0.01, coef0=0.0,  tol=1e-3,  max_iter=-1),
    dict(kernel="sigmoid", C=10.0,gamma=0.01, coef0=0.0,  tol=1e-3,  max_iter=-1),
    dict(kernel="sigmoid", C=0.5, gamma=0.05, coef0=0.0,  tol=1e-3,  max_iter=-1),
    dict(kernel="sigmoid", C=1.0, gamma=0.05, coef0=0.0,  tol=1e-4,  max_iter=-1),
    dict(kernel="sigmoid", C=5.0, gamma=0.05, coef0=0.1,  tol=1e-3,  max_iter=-1),
    dict(kernel="sigmoid", C=1.0, gamma=0.1,  coef0=0.0,  tol=1e-3,  max_iter=5000),
    dict(kernel="sigmoid", C=0.1, gamma=0.1,  coef0=0.1,  tol=1e-4,  max_iter=10000),
    dict(kernel="sigmoid", C=2.0, gamma=0.1,  coef0=0.0,  tol=1e-3,  max_iter=2000),
    dict(kernel="sigmoid", C=0.5, gamma=0.2,  coef0=0.1,  tol=1e-2,  max_iter=-1),
    dict(kernel="sigmoid", C=3.0, gamma=0.2,  coef0=0.0,  tol=1e-3,  max_iter=-1),
    dict(kernel="sigmoid", C=1.0, gamma=0.05, coef0=0.05, tol=1e-4,  max_iter=1000),

    # ---- poly ----
    dict(kernel="poly", C=0.1, degree=2, gamma=0.01, coef0=0.0, tol=1e-3,  max_iter=-1),
    dict(kernel="poly", C=0.5, degree=2, gamma=0.05, coef0=0.0, tol=1e-3,  max_iter=-1),
    dict(kernel="poly", C=1.0, degree=2, gamma=0.1,  coef0=0.0, tol=1e-3,  max_iter=-1),
    dict(kernel="poly", C=5.0, degree=3, gamma=0.01, coef0=0.1, tol=1e-4,  max_iter=-1),
    dict(kernel="poly", C=10.0,degree=3, gamma=0.05, coef0=0.1, tol=1e-3,  max_iter=-1),
    dict(kernel="poly", C=1.0, degree=3, gamma=0.1,  coef0=0.0, tol=1e-4,  max_iter=5000),
    dict(kernel="poly", C=0.1, degree=3, gamma=0.1,  coef0=0.05,tol=1e-3,  max_iter=10000),
    dict(kernel="poly", C=2.0, degree=4, gamma=0.01, coef0=0.0, tol=1e-3,  max_iter=2000),
    dict(kernel="poly", C=0.5, degree=4, gamma=0.05, coef0=0.1, tol=1e-2,  max_iter=-1),
    dict(kernel="poly", C=3.0, degree=4, gamma=0.1,  coef0=0.05,tol=1e-3,  max_iter=-1),
    dict(kernel="poly", C=1.0, degree=5, gamma=0.05, coef0=0.0, tol=1e-4,  max_iter=1000),
    dict(kernel="poly", C=5.0, degree=5, gamma=0.1,  coef0=0.1, tol=1e-3,  max_iter=-1),
    dict(kernel="poly", C=10.0,degree=5, gamma=0.2,  coef0=0.0, tol=1e-3,  max_iter=-1),

    # ---- rbf ----
    dict(kernel="rbf", C=0.1,  gamma=0.01, tol=1e-3, max_iter=-1,  coef0=0.0, degree=3),
    dict(kernel="rbf", C=0.5,  gamma=0.01, tol=1e-4, max_iter=-1,  coef0=0.0, degree=3),
    dict(kernel="rbf", C=1.0,  gamma=0.01, tol=1e-3, max_iter=-1,  coef0=0.0, degree=3),
    dict(kernel="rbf", C=5.0,  gamma=0.01, tol=1e-3, max_iter=-1,  coef0=0.0, degree=3),
    dict(kernel="rbf", C=10.0, gamma=0.01, tol=1e-4, max_iter=-1,  coef0=0.0, degree=3),
    dict(kernel="rbf", C=1.0,  gamma=0.05, tol=1e-3, max_iter=5000,coef0=0.0, degree=3),
    dict(kernel="rbf", C=0.1,  gamma=0.05, tol=1e-4, max_iter=10000,coef0=0.0, degree=3),
    dict(kernel="rbf", C=2.0,  gamma=0.05, tol=1e-3, max_iter=2000,coef0=0.0, degree=3),
    dict(kernel="rbf", C=0.5,  gamma=0.1,  tol=1e-2, max_iter=-1,  coef0=0.0, degree=3),
    dict(kernel="rbf", C=3.0,  gamma=0.1,  tol=1e-3, max_iter=-1,  coef0=0.0, degree=3),
    dict(kernel="rbf", C=1.0,  gamma=0.2,  tol=1e-4, max_iter=1000,coef0=0.0, degree=3),
    dict(kernel="rbf", C=5.0,  gamma=0.2,  tol=1e-3, max_iter=-1,  coef0=0.0, degree=3),
    dict(kernel="rbf", C=10.0, gamma=0.2,  tol=1e-3, max_iter=-1,  coef0=0.0, degree=3),
]

# ==== 3) Core evaluator for a single random split (your existing X_train/X_val etc.) ====

def evaluate_table1_random_split(X_train, y_train, X_val, y_val, classes, table=TABLE1):
    rows = []
    for i, params in enumerate(table, start=1):
        clf = svm.SVC(**params)
        clf.fit(X_train, y_train)
        train_acc = clf.score(X_train, y_train)
        val_acc   = clf.score(X_val, y_val)
        rows.append({
            "row": i,
            "kernel": params.get("kernel"),
            "C": params.get("C"),
            "gamma": params.get("gamma", None),
            "degree": params.get("degree", None),
            "coef0": params.get("coef0", None),
            "tol": params.get("tol"),
            "max_iter": params.get("max_iter"),
            "Training accuracy": round(train_acc, 4),
            "Validation accuracy": round(val_acc, 4),
            "Comments": generalization_comment(train_acc, val_acc)
        })
    df = pd.DataFrame(rows)
    return df.sort_values(["kernel","C","gamma","degree","tol","max_iter"], na_position="last")

# Example usage on your random split:
rand_df = evaluate_table1_random_split(X_train, y_train, X_val, y_val, classes)
print("\n=== Random split results (Table 1) ===")
print(rand_df.to_string(index=False))

rand_df.to_csv("svc_table1_random_split.csv", index=False)

def evaluate_table1_kfold(X, y, n_splits: int, table=TABLE1):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rows = []
    for i, params in enumerate(table, start=1):
        tr_scores, va_scores = [], []
        for tr_idx, va_idx in kfold.split(X):
            X_tr, X_va = X[tr_idx], X[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]

            clf = svm.SVC(**params)
            clf.fit(X_tr, y_tr)
            tr_scores.append(clf.score(X_tr, y_tr))
            va_scores.append(clf.score(X_va, y_va))

        rows.append({
            "row": i, **{k:v for k,v in params.items() if k in ["kernel","C","gamma","degree","coef0","tol","max_iter"]},
            "Avg_Train_Acc": float(np.mean(tr_scores)),
            "Avg_Val_Acc": float(np.mean(va_scores)),
            "Std_Val_Acc": float(np.std(va_scores)),
        })
    return pd.DataFrame(rows).sort_values(["Avg_Val_Acc","Avg_Train_Acc"], ascending=[False, False])

def evaluate_table1_repeated_kfold(X, y, n_splits: int, n_repeats: int, table=TABLE1):
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    rows = []
    for i, params in enumerate(table, start=1):
        tr_scores, va_scores = [], []
        for tr_idx, va_idx in rkf.split(X):
            X_tr, X_va = X[tr_idx], X[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]

            clf = svm.SVC(**params)
            clf.fit(X_tr, y_tr)
            tr_scores.append(clf.score(X_tr, y_tr))
            va_scores.append(clf.score(X_va, y_va))

        rows.append({
            "row": i, **{k:v for k,v in params.items() if k in ["kernel","C","gamma","degree","coef0","tol","max_iter"]},
            "Avg_Train_Acc": float(np.mean(tr_scores)),
            "Avg_Val_Acc": float(np.mean(va_scores)),
            "Std_Val_Acc": float(np.std(va_scores)),
        })
    return pd.DataFrame(rows).sort_values(["Avg_Val_Acc","Avg_Train_Acc"], ascending=[False, False])

# --- Produce the 3 K-fold tables (K=5,10,15) ---
kf5  = evaluate_table1_kfold(X, y, n_splits=5)
kf10 = evaluate_table1_kfold(X, y, n_splits=10)
kf15 = evaluate_table1_kfold(X, y, n_splits=15)
kf5.to_csv("svc_table1_kfold_5.csv", index=False)
kf10.to_csv("svc_table1_kfold_10.csv", index=False)
kf15.to_csv("svc_table1_kfold_15.csv", index=False)
print("Top K=5:\n", kf5.head(5).to_string(index=False))
print("Top K=10:\n", kf10.head(5).to_string(index=False))
print("Top K=15:\n", kf15.head(5).to_string(index=False))

# --- Produce the 6 Repeated K-fold tables (5/10/15 folds × 5/10 reps) ---
rkf_5x5   = evaluate_table1_repeated_kfold(X, y, n_splits=5,  n_repeats=5)
rkf_5x10  = evaluate_table1_repeated_kfold(X, y, n_splits=5,  n_repeats=10)
rkf_10x5  = evaluate_table1_repeated_kfold(X, y, n_splits=10, n_repeats=5)
rkf_10x10 = evaluate_table1_repeated_kfold(X, y, n_splits=10, n_repeats=10)
rkf_15x5  = evaluate_table1_repeated_kfold(X, y, n_splits=15, n_repeats=5)
rkf_15x10 = evaluate_table1_repeated_kfold(X, y, n_splits=15, n_repeats=10)
rkf_5x5.to_csv("svc_table1_rkf_5x5.csv", index=False)
rkf_5x10.to_csv("svc_table1_rkf_5x10.csv", index=False)
rkf_10x5.to_csv("svc_table1_rkf_10x5.csv", index=False)
rkf_10x10.to_csv("svc_table1_rkf_10x10.csv", index=False)
rkf_15x5.to_csv("svc_table1_rkf_15x5.csv", index=False)
rkf_15x10.to_csv("svc_table1_rkf_15x10.csv", index=False)
print("Top 5x5:\n", rkf_5x5.head(5).to_string(index=False))
print("Top 5x10:\n", rkf_5x10.head(5).to_string(index=False))
print("Top 10x5:\n", rkf_10x5.head(5).to_string(index=False))
print("Top 10x10:\n", rkf_10x10.head(5).to_string(index=False))
print("Top 15x5:\n", rkf_15x5.head(5).to_string(index=False))
print("Top 15x10:\n", rkf_15x10.head(5).to_string(index=False))
