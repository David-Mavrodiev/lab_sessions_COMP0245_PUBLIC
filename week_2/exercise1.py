import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import plot_tree

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

# Initialize the model
tree = DecisionTreeRegressor(max_depth=5, splitter='best')
pol_reg = LinearRegression()

tree.fit(X_train, y_train)
pol_reg.fit(X_train, y_train)

y_pred_tree = tree.predict(X_test)
y_pred_pol = pol_reg.predict(X_test)

mse_tree = mean_squared_error(y_test, y_pred_tree)
r2_tree = r2_score(y_test, y_pred_tree)

mse_pol = mean_squared_error(y_test, y_pred_pol)
r2_pol = r2_score(y_test, y_pred_pol)

print(f"Decision Tree MSE: {mse_tree}, R2: {r2_tree}")
print(f"Polynomial Regression MSE: {mse_pol}, R2: {r2_pol}")