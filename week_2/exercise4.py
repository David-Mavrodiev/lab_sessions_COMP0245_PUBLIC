import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
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

# Initialize and train AdaBoost Regressor
base_tree = DecisionTreeRegressor(max_depth=4)  # Weak learner
ada_model = AdaBoostRegressor(estimator=base_tree, n_estimators=50, random_state=42)
ada_model.fit(X_train, y_train)

# Predict on the training set
y_train_pred = ada_model.predict(X_train)

# Compute the prediction error on the training set
mse_train = mean_squared_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)

print(f"AdaBoost Regressor Training MSE: {mse_train}")
print(f"AdaBoost Regressor Training R²: {r2_train}")

# Plot the true vs predicted values for training set
plt.figure(figsize=(14, 6))

# Plot the true signal
plt.subplot(1, 2, 1)
plt.scatter(x1, y, color='blue', alpha=0.5, label="True values")
plt.title("True Signal")

# Plot AdaBoost Regressor predictions (on training set)
plt.subplot(1, 2, 2)
plt.scatter(x1, ada_model.predict(X), color='green', alpha=0.5, label="Predicted (AdaBoost)")
plt.title("AdaBoost Regressor Predictions (Training Set)")

plt.show()
