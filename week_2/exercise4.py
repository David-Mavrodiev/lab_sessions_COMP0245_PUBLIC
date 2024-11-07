import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Generate or reuse synthetic data
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
ada_regressor = AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=8), n_estimators=50, random_state=42, loss='linear')

# Train the model
ada_regressor.fit(X_train, y_train)

# Predictions
y_pred_ada = ada_regressor.predict(X_test)

# Evaluate the model
mse_ada = mean_squared_error(y_test, y_pred_ada)
r2_ada = r2_score(y_test, y_pred_ada)

# results = open("week_2/results.txt", "a")
# results.write(f"\n\n\nAdaBoost Regressor:\nMean Squared Error: {mse_ada}\nR-squared: {r2_ada}")
print(f"AdaBoost Regressor MSE: {mse_ada}, R2: {r2_ada}")
