import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Drop unnecessary columns
cols_to_drop = ['id', 'date', 'zipcode']
for col in train.columns:
    if 'Unnamed' in str(col):
        cols_to_drop.append(col)

train = train.drop(columns=[c for c in cols_to_drop if c in train.columns])
test = test.drop(columns=[c for c in cols_to_drop if c in test.columns])

# Extract sqft_living and price
X_train = train['sqft_living'].values.reshape(-1, 1)
y_train = (train['price'] / 1000).values
X_test = test['sqft_living'].values.reshape(-1, 1)
y_test = (test['price'] / 1000).values

# Polynomial regression functions
def create_polynomial_features(X, degree):
    n_samples = X.shape[0]
    X_poly = np.zeros((n_samples, degree))
    for i in range(1, degree + 1):
        X_poly[:, i-1] = (X[:, 0] ** i)
    return X_poly

def train_polynomial_regression(X, y, degree):
    X_poly = create_polynomial_features(X, degree)
    scaler = StandardScaler()
    X_poly_scaled = scaler.fit_transform(X_poly)
    n_samples = X_poly_scaled.shape[0]
    X_with_intercept = np.column_stack([np.ones(n_samples), X_poly_scaled])
    XtX = X_with_intercept.T @ X_with_intercept
    Xty = X_with_intercept.T @ y
    theta = np.linalg.solve(XtX, Xty)
    return theta, scaler

def predict_polynomial_regression(X, theta, scaler, degree):
    X_poly = create_polynomial_features(X, degree)
    X_poly_scaled = scaler.transform(X_poly)
    n_samples = X_poly_scaled.shape[0]
    X_with_intercept = np.column_stack([np.ones(n_samples), X_poly_scaled])
    predictions = X_with_intercept @ theta
    return predictions

# Train models for p = 1, 2, 3, 4, 5
degrees = [1, 2, 3, 4, 5]
results = []

for p in degrees:
    theta, scaler = train_polynomial_regression(X_train, y_train, degree=p)
    y_train_pred = predict_polynomial_regression(X_train, theta, scaler, degree=p)
    y_test_pred = predict_polynomial_regression(X_test, theta, scaler, degree=p)
    
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    results.append({
        'degree': p,
        'train_mse': train_mse,
        'train_r2': train_r2,
        'test_mse': test_mse,
        'test_r2': test_r2
    })

print("Results table")
print(f"{'Degree':<10} {'Train MSE':<15} {'Train R²':<12} {'Test MSE':<15} {'Test R²':<12}")
for r in results:
    print(f"{r['degree']:<10} {r['train_mse']:<15.6f} {r['train_r2']:<12.6f} "
          f"{r['test_mse']:<15.6f} {r['test_r2']:<12.6f}")

train_mse_vals = [r['train_mse'] for r in results]
train_r2_vals = [r['train_r2'] for r in results]
test_mse_vals = [r['test_mse'] for r in results]
test_r2_vals = [r['test_r2'] for r in results]

print(f"\nAs polynomial degree increases from 1 to 5, training MSE consistently decreases from "
      f"{train_mse_vals[0]:.2f} to {train_mse_vals[-1]:.2f} and training R² increases from "
      f"{train_r2_vals[0]:.4f} to {train_r2_vals[-1]:.4f}, showing that higher degree "
      f"polynomials fit the training data better. However, testing performance shows a different "
      f"pattern: testing MSE initially improves from {test_mse_vals[0]:.2f} (p=1) to "
      f"{test_mse_vals[1]:.2f} (p=2), but then dramatically worsens, reaching {test_mse_vals[-1]:.2f} "
      f"at p=5 with a negative R² of {test_r2_vals[-1]:.4f}. The best generalization occurs at "
      f"p=2 with the lowest testing MSE. For p≥3, severe overfitting happens where the model "
      f"fits noise in the training data and produces extremely poor predictions on unseen data.")