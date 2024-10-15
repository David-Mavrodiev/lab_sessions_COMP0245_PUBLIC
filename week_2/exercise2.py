import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Generate synthetic data
x1 = np.arange(0, 10, 0.1)
x2 = np.arange(0, 10, 0.1)
x1, x2 = np.meshgrid(x1, x2)
y = np.sin(x1) * np.cos(x2) + np.random.normal(scale=0.1, size=x1.shape)

# Flatten the arrays
x1 = x1.flatten()
x2 = x2.flatten()
y = y.flatten()
X = np.vstack((x1, x2)).T

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train Bagging Regressor with Decision Tree as base estimator
base_tree = DecisionTreeRegressor(max_depth=5)
bagging_model = BaggingRegressor(estimator=base_tree, n_estimators=50, random_state=42)
bagging_model.fit(X_train, y_train)

# Predict on the test set
y_pred_bagging = bagging_model.predict(X_test)

# Evaluate the performance
mse_bagging = mean_squared_error(y_test, y_pred_bagging)
r2_bagging = r2_score(y_test, y_pred_bagging)

print(f"Bagging Regressor MSE: {mse_bagging}")
print(f"Bagging Regressor R²: {r2_bagging}")

# Plot the true vs predicted values
plt.figure(figsize=(14, 6))

# Plot the true signal
plt.subplot(1, 2, 1)
plt.scatter(x1, y, color='blue', alpha=0.5, label="True values")
plt.title("True Signal")

# Plot Bagging Regressor predictions
plt.subplot(1, 2, 2)
plt.scatter(x1, bagging_model.predict(X), color='green', alpha=0.5, label="Predicted (Bagging Regressor)")
plt.title("Bagging Regressor Predictions")

plt.show()
