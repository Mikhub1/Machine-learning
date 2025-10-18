import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MaxAbsScaler, MinMaxScaler
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
# Current working directory
cwd = Path.cwd()
# Path to the CSV file 
csv_file = cwd / "car_prices.csv"
# Load the CSV
df = pd.read_csv(csv_file)

# Separate features (first 10 columns) and target (11th column)
X = df.iloc[:, 0:10]
y = df.iloc[:,[10]]

categorical_features = ["fuel type", "seller type", "transmission"]  # adjust names
numerical_features = [col for col in X.columns if col not in categorical_features]

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

# Split dataset into train and test
X_final = X_transformed  # only input features

# Perform train-test split
X_train, X_temp, y_train, y_temp = train_test_split(
    X_final, y_final, test_size=0.3, random_state=42, shuffle = True
)

# Split equally into validation (15%) and test (15%)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, shuffle = True
)

X_train_full = np.concatenate((X_train, X_val))
y_train_full = np.concatenate((y_train, y_val))

# Fit the model
reg = SVR(kernel='rbf', C=1, max_iter = -1, coef0 = 0.0, tol=1e-3, degree=3)  # try 'rbf' too
reg.fit(X_train, y_train)

# Predictions
y_train_actual = scaler_y.inverse_transform(y_train.reshape(-1,1))
y_train_pred_scaled = reg.predict(X_train)
y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1,1))

y_val_actual = scaler_y.inverse_transform(y_val.reshape(-1,1))
y_val_pred_scaled = reg.predict(X_val)
y_val_pred = scaler_y.inverse_transform(y_val_pred_scaled.reshape(-1,1))

# Evaluate model
# --- Training Errors ---
train_mse = mean_squared_error(y_train_actual, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_r2 = r2_score(y_train_actual, y_train_pred)

# --- Val Errors ---
val_mse = mean_squared_error(y_val_actual, y_val_pred)
val_rmse = np.sqrt(val_mse)
val_r2 = r2_score(y_val_actual, y_val_pred)


print("--------Random Train-Test Split-----------")
print("Training Results:")
print(f"  RMSE: {train_rmse:.2f}")
print(f"  MSE:  {train_mse:.2f}")
print(f"  R²:   {train_r2:.2f}")

print("\nValidation Results:")
print(f"  RMSE: {val_rmse:.2f}")
print(f"  MSE:  {val_mse:.2f}")
print(f"  R²:   {val_r2:.2f}")



# ----------------K-Fold ----------------
# --- K-Fold setup ---
k = 5  # number of folds (you can change this)
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Containers for storing metrics
train_rmse_list, val_rmse_list = [], []
train_r2_list, val_r2_list = [], []

# --- Perform K-Fold Cross Validation ---
for fold, (train_idx, val_idx) in enumerate(kf.split(X_final)):
    X_train, X_val = X_final.iloc[train_idx], X_final.iloc[val_idx]
    y_train, y_val = y_final[train_idx], y_final[val_idx]

    # Fit model using the same regressor from 62
    reg.fit(X_train, y_train)

    # Predictions (rescale back to original target space)
    y_train_actual = scaler_y.inverse_transform(y_train.reshape(-1, 1))
    y_train_pred = scaler_y.inverse_transform(reg.predict(X_train).reshape(-1, 1))

    y_val_actual = scaler_y.inverse_transform(y_val.reshape(-1, 1))
    y_val_pred = scaler_y.inverse_transform(reg.predict(X_val).reshape(-1, 1))

    # --- Compute metrics ---
    train_mse = mean_squared_error(y_train_actual, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_r2 = r2_score(y_train_actual, y_train_pred)

    val_mse = mean_squared_error(y_val_actual, y_val_pred)
    val_rmse = np.sqrt(val_mse)
    val_r2 = r2_score(y_val_actual, y_val_pred)

    # Store results
    train_rmse_list.append(train_rmse)
    val_rmse_list.append(val_rmse)
    train_r2_list.append(train_r2)
    val_r2_list.append(val_r2)

    print(f"Fold {fold+1}:")
    print(f"  Train RMSE: {train_rmse:.2f}, R²: {train_r2:.2f}")
    print(f"  Validation  RMSE: {val_rmse:.2f}, R²: {val_r2:.2f}")
    print("-"*40)

# --- Average results across folds ---
print("\n-------- K-Fold Cross Validation Results --------")
print(f"Average Train RMSE: {np.mean(train_rmse_list):.2f}")
print(f"Average Train R²:   {np.mean(train_r2_list):.2f}")
print(f"Average Validation RMSE:  {np.mean(val_rmse_list):.2f}")
print(f"Average Validation R²:    {np.mean(val_r2_list):.2f}")


# ----------------Repeated K-Fold ----------------
# --- Setup ---
k = 5           # number of folds
n_repeats = 3   # number of repetitions

rkf = RepeatedKFold(n_splits=k, n_repeats=n_repeats, random_state=42)

# Containers for metrics
train_rmse_list, val_rmse_list = [], []
train_r2_list, val_r2_list = [], []

# --- Perform Repeated K-Fold Cross Validation ---
for fold, (train_idx, val_idx) in enumerate(rkf.split(X_final)):
    X_train, X_val = X_final.iloc[train_idx], X_final.iloc[val_idx]
    y_train, y_val = y_final[train_idx], y_final[val_idx]

    # Fit model (assume 'reg' is your regressor)
    reg.fit(X_train, y_train)

    # Predictions (rescale back to original target space)
    y_train_actual = scaler_y.inverse_transform(y_train.reshape(-1,1))
    y_train_pred   = scaler_y.inverse_transform(reg.predict(X_train).reshape(-1,1))

    y_val_actual = scaler_y.inverse_transform(y_val.reshape(-1,1))
    y_val_pred   = scaler_y.inverse_transform(reg.predict(X_val).reshape(-1,1))

    # --- Compute metrics ---
    train_mse = mean_squared_error(y_train_actual, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_r2 = r2_score(y_train_actual, y_train_pred)

    val_mse = mean_squared_error(y_val_actual, y_val_pred)
    val_rmse = np.sqrt(val_mse)
    val_r2 = r2_score(y_val_actual, y_val_pred)

    # Store results
    train_rmse_list.append(train_rmse)
    val_rmse_list.append(val_rmse)
    train_r2_list.append(train_r2)
    val_r2_list.append(val_r2)

    print(f"Fold {fold+1}:")
    print(f"  Train RMSE: {train_rmse:.2f}, R²: {train_r2:.2f}")
    print(f"  Validation RMSE: {val_rmse:.2f}, R²: {val_r2:.2f}")
    print("-"*40)

# --- Average results across all folds and repeats ---
print("\n-------- Repeated K-Fold Cross Validation Results --------")
print(f"Average Train RMSE: {np.mean(train_rmse_list):.2f}")
print(f"Average Train R²:   {np.mean(train_r2_list):.2f}")
print(f"Average Validation RMSE:  {np.mean(val_rmse_list):.2f}")
print(f"Average Validation R²:    {np.mean(val_r2_list):.2f}")


###----------------Evaluating the performance (test errors)-------------------
# reg_final = reg.fit(X_train_full, y_train_full)
# y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1,1))
# y_test_pred_scaled = reg_final.predict(X_test)
# y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled.reshape(-1,1))

# test_mse = mean_squared_error(y_test_actual, y_test_pred)
# test_rmse = np.sqrt(test_mse)
# test_r2 = r2_score(y_test_actual, y_test_pred)

# print("\nTest Results:")
# print(f"  RMSE: {test_rmse:.2f}")
# print(f"  MSE:  {test_mse:.2f}")
# print(f"  R²:   {test_r2:.2f}")

