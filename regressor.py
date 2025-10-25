import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MaxAbsScaler, MinMaxScaler
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
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

# Models ---- uncomment the algorithm you would like to run and comment the others
reg = LinearRegression(fit_intercept=True)
# reg = KNeighborsRegressor(n_neighbors=3,weights="distance",p=2,metric="minkowski",leaf_size=50)
# reg = DecisionTreeRegressor(criterion="friedman_mse", splitter="random", max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features="log2")
# reg = RandomForestRegressor(n_estimators=5000,max_depth=None,criterion="squared_error",bootstrap=True)
# reg = SVR(kernel='rbf',epsilon=0.1,C=1.0) 

# ------------------- FLAGS -------------------
TRAIN_FLAG = True   # Set to True to run training/CV stages
TEST_FLAG  = False  # Set to True to run final testing stage

# ------------------- TRAINING -------------------
if TRAIN_FLAG:
    
# Train the model
    reg.fit(X_train, y_train)
    
    # Predictions
    y_train_actual = scaler_y.inverse_transform(y_train.reshape(-1,1))
    y_train_pred_scaled = reg.predict(X_train)
    y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1,1)) #inverse scaling
    
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
    
    
    print("Training Results:")
    print(f"  RMSE: {train_rmse:.2f}")
    print(f"  MSE:  {train_mse:.2f}")
    print(f"  R²:   {train_r2:.2f}")
    
    print("\nValidation Results:")
    print(f"  RMSE: {val_rmse:.2f}")
    print(f"  MSE:  {val_mse:.2f}")
    print(f"  R²:   {val_r2:.2f}")
    
    
###----------------TESTING-------------------
if TEST_FLAG:
    reg_final = reg.fit(X_train_full, y_train_full)
    y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1,1))
    y_test_pred_scaled = reg_final.predict(X_test)
    y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled.reshape(-1,1))
    
    test_mse = mean_squared_error(y_test_actual, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(y_test_actual, y_test_pred)
    
    print("\nTest Results:")
    print(f"  RMSE: {test_rmse:.2f}")
    print(f"  MSE:  {test_mse:.2f}")
    print(f"  R²:   {test_r2:.2f}")

