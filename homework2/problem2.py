import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Drop col id, date, zipcode, and any unnamed index columns
cols_to_drop = ['id', 'date', 'zipcode']
for col in train.columns:
    if 'Unnamed' in str(col):
        cols_to_drop.append(col)

train = train.drop(columns=[c for c in cols_to_drop if c in train.columns])
test = test.drop(columns=[c for c in cols_to_drop if c in test.columns])

# Separate features and target
X_train = train.drop('price', axis=1)
y_train = train['price'] / 1000

X_test = test.drop('price', axis=1)
y_test = test['price'] / 1000

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Part 1:
print("Part 1 results:")
print(f"\nIntercept: {model.intercept_:.6f}")
print("\nCoefficients:")
for feature, coef in zip(X_train.columns, model.coef_):
    print(f"  {feature:20s}: {coef:10.6f}")

y_train_pred = model.predict(X_train_scaled)
train_mse = mean_squared_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

print(f"\nTraining MSE: {train_mse:.6f}")
print(f"Training R2:  {train_r2:.6f}")

# Part 2: 
print("\n\nPart 2 results:")

y_test_pred = model.predict(X_test_scaled)
test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"\nTesting MSE: {test_mse:.6f}")
print(f"Testing R2:  {test_r2:.6f}")

# Part 3:
print("\n\nPart 3 interpreatation:")

# Most important features
coef_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': model.coef_,
    'Abs_Coef': np.abs(model.coef_)
}).sort_values('Abs_Coef', ascending=False)

print("\nTop 5 most important features:")
for i, row in coef_df.head(5).iterrows():
    print(f"  {row['Feature']:20s}: {row['Coefficient']:10.6f}")

print(f"\nModel fit: R² = {test_r2:.4f} means the model explains {test_r2*100:.1f}% of variance")
print(f"Average error (RMSE): ${np.sqrt(test_mse)*1000:,.2f}")
print(f"Training MSE = {train_mse:.6f}, Testing MSE = {test_mse:.6f}")
print(f"MSE ratio (test/train): {test_mse/train_mse:.4f}")

print("\n\nPart 3 response:")
print("\nWhich features contribute most?")
print(f"The features that contribute most to the model are {coef_df.iloc[0]['Feature']} "
      f"({coef_df.iloc[0]['Coefficient']:.2f}), {coef_df.iloc[1]['Feature']} "
      f"({coef_df.iloc[1]['Coefficient']:.2f}), and {coef_df.iloc[2]['Feature']} "
      f"({coef_df.iloc[2]['Coefficient']:.2f}), indicating that construction quality, "
      f"geographic location, and building age are the strongest predictors of house price. "
      f"Features like waterfront ({coef_df.iloc[3]['Coefficient']:.2f}) and living space "
      f"({coef_df.iloc[4]['Coefficient']:.2f}) also have substantial positive impacts on price.")

print("\nIs the model fitting well?")
print(f"The model shows moderately good fit, with a training R² of {train_r2:.4f} "
      f"and testing R² of {test_r2:.4f}, meaning it explains approximately {test_r2*100:.1f}% "
      f"of price variance in unseen data. However, while the model does show a large portion of "
      f"patterns, about {(1-test_r2)*100:.1f}% of variance is leftover/unexplained.")

print("\nHow large is the model error?")
print(f"The model error is moderate, with an RMSE of ${np.sqrt(test_mse)*1000:,.2f} on the "
      f"test set, representing the average prediction error. Given typical house prices in "
      f"this dataset range from hundreds of thousands to over a million dollars, this error "
      f"represents a noticeable but reasonable margin of uncertainty.")

print("\nHow do training and testing MSE relate?")
print(f"The testing MSE ({test_mse:.2f}) is approximately {test_mse/train_mse:.2f}x larger "
      f"than the training MSE ({train_mse:.2f}), showing some overfitting where the model "
      f"performs better on training data. However, the relatively close R² values "
      f"({train_r2:.4f} vs {test_r2:.4f}) suggest the overfitting is not severe and the model "
      f"still generalizes reasonably well to unseen data.")
