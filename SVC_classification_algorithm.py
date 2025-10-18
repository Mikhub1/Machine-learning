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


# #---------------------------K-fold-------------------------------------------
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
### Evaluating the performance on the test data 
clf_final = clf.fit(X_train_full, y_train_full)

y_pred = clf_final.predict(X_test)
test_acc = clf_final.score(X_test, y_test)
print(f"Test Accuracy: {test_acc:.2f}")
print("\nTest Classification Report:\n", classification_report(y_test, y_pred, target_names=classes))
cm_test_rand_split = confusion_matrix(y_test, y_pred)
print("\Test Confusion Matrix:\n", cm_test_rand_split)