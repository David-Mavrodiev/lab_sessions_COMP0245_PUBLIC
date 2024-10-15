import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
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

# Initialize and train Decision Tree Regressor
tree = DecisionTreeRegressor(max_depth=5, splitter='best')
tree.fit(X_train, y_train)

# Predict on test set
y_pred_tree = tree.predict(X_test)

# Calculate MSE and R² for Decision Tree
mse_tree = mean_squared_error(y_test, y_pred_tree)
r2_tree = r2_score(y_test, y_pred_tree)

print(f"Decision Tree Regressor MSE: {mse_tree}")
print(f"Decision Tree Regressor R²: {r2_tree}")

# Polynomial Regression
poly = PolynomialFeatures(degree=5)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)

# Train Linear Regression on polynomial features
poly_reg = LinearRegression()
poly_reg.fit(X_poly_train, y_train)

# Predict on test set
y_pred_poly = poly_reg.predict(X_poly_test)

# Calculate MSE and R² for Polynomial Regression
mse_poly = mean_squared_error(y_test, y_pred_poly)
r2_poly = r2_score(y_test, y_pred_poly)

print(f"Polynomial Regression MSE: {mse_poly}")
print(f"Polynomial Regression R²: {r2_poly}")

# Plot comparison
plt.figure(figsize=(14, 6))

# Plot the true signal
plt.subplot(1, 2, 1)
plt.scatter(x1, y, color='blue', alpha=0.5, label="True values")
plt.title("True Signal")

# Plot Decision Tree predictions
plt.subplot(1, 2, 2)
plt.scatter(x1, tree.predict(X), color='green', alpha=0.5, label="Predicted (Decision Tree)")
plt.title("Decision Tree Predictions")

plt.show()