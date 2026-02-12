import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# Gradient descent for ridge regression
def ridge_gradient_descent(X, y, lamda, alpha=0.01, num_iterations=10000):
    N, d = X.shape
    theta = np.zeros(d)
    
    # Adjust learning rate based on lambda to prevent divergence
    adjusted_alpha = alpha / (1 + lamda)
    
    for i in range(num_iterations):
        predictions = X @ theta
        error = predictions - y
        gradient = (2/N) * (X.T @ error) + 2 * lamda * theta
        theta = theta - adjusted_alpha * gradient
    
    return theta

# Simulate data
np.random.seed(42)
N = 1000

# X uniformly distributed on [-2, 2]
X = np.random.uniform(-2, 2, N)

# Y = 1 + 2X + e, where e ~ N(0, 2)
e = np.random.normal(0, 2, N)
Y = 1 + 2*X + e

# Reshape X for regression
X_reg = X.reshape(-1, 1)

# Test different lambda values
lambdas = [1, 10, 100, 1000, 10000]

print(f"\nData: N = {N}, X ~ Uniform[-2, 2], Y = 1 + 2X + e, e ~ N(0, 2)")
print(f"True parameters: intercept = 1, slope = 2")

print(f"\n{'Lambda':<10} {'Slope':<15} {'MSE':<15} {'R²':<15}")
print("-" * 70)

for lamda in lambdas:
    # Train ridge regression
    theta = ridge_gradient_descent(X_reg, Y, lamda, alpha=0.01, num_iterations=10000)
    
    # Predictions
    Y_pred = X_reg @ theta
    
    # Metrics
    mse = mean_squared_error(Y, Y_pred)
    r2 = r2_score(Y, Y_pred)
    
    print(f"{lamda:<10} {theta[0]:<15.6f} {mse:<15.6f} {r2:<15.6f}")

print("As the regularization parameter λ increases, the slope coefficient shrinks toward zero. "
      "At λ=1, the slope is 1.09 "
      "with MSE=6.20 and R²=0.31, which is reasonably close to the true slope of 2. However, as "
      "λ increases to 10, 100, 1000, and 10000, the slope shrinks dramatically to 0.23, 0.03, "
      "0.003, and 0.0003 respectively, approaching zero. Correspondingly, the MSE increases from "
      "6.20 to over 10, and R² becomes negative (indicating performance worse than simply predicting "
      "the mean). This means severe underfitting: high λ values over-penalize the coefficients, "
      "preventing the model from learning the actual relationship of Y = 1 + 2X. ")