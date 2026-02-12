import pandas as pd
import numpy as np
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

feature_names = train.drop('price', axis=1).columns

# Gradient descent implementation
def gradient_descent(X, y, alpha, num_iterations):
    # Add intercept term
    n_samples, n_features = X.shape
    X_with_intercept = np.column_stack([np.ones(n_samples), X])
    
    # Initialize theta to zeros
    theta = np.zeros(n_features + 1)
    
    # Gradient descent loop
    for i in range(num_iterations):
        predictions = X_with_intercept @ theta
        
        error = predictions - y
        
        # Gradient: (1/m) * X^T * error
        gradient = (1/n_samples) * (X_with_intercept.T @ error)

        theta = theta - alpha * gradient
    
    return theta

def predict(X, theta):
    n_samples = X.shape[0]
    X_with_intercept = np.column_stack([np.ones(n_samples), X])
    return X_with_intercept @ theta

# Part 2:
learning_rates = [0.01, 0.1, 0.5]
iterations_list = [10, 50, 100]

results = []

for alpha in learning_rates:
    for num_iter in iterations_list:
        # Train model
        theta = gradient_descent(X_train_scaled, y_train, alpha, num_iter)
        
        # Make predictions
        y_train_pred = predict(X_train_scaled, theta)
        y_test_pred = predict(X_test_scaled, theta)
        
        # Calculate metrics
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        results.append({
            'alpha': alpha,
            'iterations': num_iter,
            'theta': theta,
            'train_mse': train_mse,
            'train_r2': train_r2,
            'test_mse': test_mse,
            'test_r2': test_r2
        })

print(f"{'Learning Rate':<15} {'Iterations':<12} {'Train MSE':<15} {'Train R²':<12} {'Test MSE':<15} {'Test R²':<12}")

for r in results:
    print(f"{r['alpha']:<15} {r['iterations']:<12} {r['train_mse']:<15.6f} {r['train_r2']:<12.6f} "
          f"{r['test_mse']:<15.6f} {r['test_r2']:<12.6f}")

# Show theta values for selected configurations
print("\n\nTheta values")

for alpha in learning_rates:
    print(f"\nLearning Rate alpha = {alpha}")
    
    for num_iter in iterations_list:
        r = [x for x in results if x['alpha'] == alpha and x['iterations'] == num_iter][0]
        theta = r['theta']
        
        print(f"\nAfter {num_iter} iterations:")
        print(f"  Intercept: {theta[0]:.6f}")
        print(f"  Coefficients: {theta[1:][:5]}... (showing first 5)")

# Compare with closed-form solution
print("\n\nComparison:")

# Closed-form solution
n_samples = X_train_scaled.shape[0]
X_with_intercept = np.column_stack([np.ones(n_samples), X_train_scaled])
XtX = X_with_intercept.T @ X_with_intercept
Xty = X_with_intercept.T @ y_train
theta_optimal = np.linalg.solve(XtX, Xty)

y_train_pred_optimal = predict(X_train_scaled, theta_optimal)
y_test_pred_optimal = predict(X_test_scaled, theta_optimal)
train_mse_optimal = mean_squared_error(y_train, y_train_pred_optimal)
train_r2_optimal = r2_score(y_train, y_train_pred_optimal)
test_mse_optimal = mean_squared_error(y_test, y_test_pred_optimal)
test_r2_optimal = r2_score(y_test, y_test_pred_optimal)

print(f"\nOptimal (Closed-form) Solution:")
print(f"  Train MSE: {train_mse_optimal:.6f}, Train R²: {train_r2_optimal:.6f}")
print(f"  Test MSE:  {test_mse_optimal:.6f}, Test R²:  {test_r2_optimal:.6f}")

# Find best gradient descent result
best_result = min(results, key=lambda x: x['train_mse'])
print(f"\nBest Gradient Descent Result (α={best_result['alpha']}, iter={best_result['iterations']}):")
print(f"  Train MSE: {best_result['train_mse']:.6f}, Train R²: {best_result['train_r2']:.6f}")
print(f"  Test MSE:  {best_result['test_mse']:.6f}, Test R²:  {best_result['test_r2']:.6f}")

# Group by learning rate
for alpha in learning_rates:
    alpha_results = [r for r in results if r['alpha'] == alpha]
    mse_values = [r['train_mse'] for r in alpha_results]
    
    print(f"\nLearning Rate α = {alpha}:")
    print(f"  Train MSE progression: ", end="")
    for i, r in enumerate(alpha_results):
        print(f"{r['train_mse']:.2f} ({r['iterations']} iter)", end="")
        if i < len(alpha_results) - 1:
            print(" → ", end="")
    print()
    
    # Check convergence
    if len(alpha_results) >= 2:
        improvement = alpha_results[-1]['train_mse'] - alpha_results[0]['train_mse']
        if abs(improvement) < 100:
            convergence = "Converged"
        else:
            convergence = "Still improving"
        print(f"  Status: {convergence}")

print(f"\nGeneral Observations:")
print(f"The algorithm behavior varies significantly with learning rate. With alpha=0.01, the algorithm "
      f"converges very slowly, starting with high MSE (294798.73) at 10 iterations and gradually "
      f"improving to 70118.99 at 100 iterations, but still far from optimal and requiring many more "
      f"iterations. With alpha=0.1, convergence is much faster and more stable, achieving MSE of 31497.69 "
      f"at 100 iterations, which is very close to the optimal closed-form solution (MSE = 31486.17). "
      f"With alpha=0.5, the learning rate is too high and causes divergence. The algorithm becomes "
      f"unstable with coefficients increasing to very high levels (10^21 and beyond) and MSE "
      f"also increasing high instead of decreasing. For this problem, alpha=0.1 with 100 iterations "
      f"is sufficient to nearly converge to the optimal solution, while alpha=0.01 needs significantly more "
      f"iterations and alpha=0.5 diverges completely and never converges.")