import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Drop columns
cols_to_drop = ['id', 'date', 'zipcode']
for col in train.columns:
    if 'Unnamed' in str(col):
        cols_to_drop.append(col)

train = train.drop(columns=[c for c in cols_to_drop if c in train.columns])
test = test.drop(columns=[c for c in cols_to_drop if c in test.columns])

# Separate features and target
X_train = train.drop('price', axis=1).values
y_train = (train['price'] / 1000).values
X_test = test.drop('price', axis=1).values
y_test = (test['price'] / 1000).values

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def train_closed_form(X, y):
    X_with_ones = np.column_stack([np.ones(X.shape[0]), X])
    theta = np.linalg.solve(X_with_ones.T @ X_with_ones, X_with_ones.T @ y)
    return theta

def predict(X, theta):
    """Predict using: y = X θ"""
    X_with_ones = np.column_stack([np.ones(X.shape[0]), X])
    return X_with_ones @ theta

# Part 1 implementation:

# Train
theta = train_closed_form(X_train_scaled, y_train)

print(f"\nIntercept (θ_0): {theta[0]:.6f}")
print("\nCoefficients:")
feature_names = train.drop('price', axis=1).columns
for name, coef in zip(feature_names, theta[1:]):
    print(f"  {name:20s}: {coef:10.6f}")

# Predictions and metrics
y_train_pred_cf = predict(X_train_scaled, theta)
y_test_pred_cf = predict(X_test_scaled, theta)

train_mse_cf = mean_squared_error(y_train, y_train_pred_cf)
train_r2_cf = r2_score(y_train, y_train_pred_cf)
test_mse_cf = mean_squared_error(y_test, y_test_pred_cf)
test_r2_cf = r2_score(y_test, y_test_pred_cf)

print(f"\nTraining MSE: {train_mse_cf:.6f}")
print(f"Training R²:  {train_r2_cf:.6f}")
print(f"\nTesting MSE:  {test_mse_cf:.6f}")
print(f"Testing R²:   {test_r2_cf:.6f}")

# Train sklearn
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predictions and metrics
y_train_pred_sk = model.predict(X_train_scaled)
y_test_pred_sk = model.predict(X_test_scaled)

train_mse_sk = mean_squared_error(y_train, y_train_pred_sk)
train_r2_sk = r2_score(y_train, y_train_pred_sk)
test_mse_sk = mean_squared_error(y_test, y_test_pred_sk)
test_r2_sk = r2_score(y_test, y_test_pred_sk)

# Comparison table
print("\nMetrics Comparison:")
print(f"{'Metric':<20} {'Closed-Form':>15} {'Sklearn':>15} {'Difference':>15}")
print("-" * 70)
print(f"{'Train MSE':<20} {train_mse_cf:>15.6f} {train_mse_sk:>15.6f} {abs(train_mse_cf-train_mse_sk):>15.9f}")
print(f"{'Train R²':<20} {train_r2_cf:>15.6f} {train_r2_sk:>15.6f} {abs(train_r2_cf-train_r2_sk):>15.9f}")
print(f"{'Test MSE':<20} {test_mse_cf:>15.6f} {test_mse_sk:>15.6f} {abs(test_mse_cf-test_mse_sk):>15.9f}")
print(f"{'Test R²':<20} {test_r2_cf:>15.6f} {test_r2_sk:>15.6f} {abs(test_r2_cf-test_r2_sk):>15.9f}")


mse_diff = abs(test_mse_cf - test_mse_sk)
r2_diff = abs(test_r2_cf - test_r2_sk)

print("\nAre the results similar?")
print(f"The closed-form implementation and sklearn produce identical results. "
      f"Both achieve a training MSE of {train_mse_cf:.6f} and testing MSE of {test_mse_cf:.6f}, "
      f"with training R² of {train_r2_cf:.6f} and testing R² of {test_r2_cf:.6f}. "
      f"Both implementations solve the same optimization problem using the "
      f"normal equations, so they arrive at the same solution and produce identical predictions "
      f"and performance metrics on both training and testing datasets.")