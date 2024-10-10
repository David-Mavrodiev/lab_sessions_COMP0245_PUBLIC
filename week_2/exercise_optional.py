from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


data = fetch_california_housing()
X = data.data
y = data.target
feature_names = data.feature_names

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the model
random_forest = RandomForestRegressor(n_estimators=50, random_state=42)

# Train the model
random_forest.fit(X_train, y_train)

# Predictions
y_pred_forest = random_forest.predict(X_test)

# Evaluate the model
mse_forest = mean_squared_error(y_test, y_pred_forest)
r2_forest = r2_score(y_test, y_pred_forest)

print(f"Random Forest Regressor MSE: {mse_forest}, R2: {r2_forest}")

plt.figure(figsize=(10, 7))
plt.scatter(y_test, y_pred_forest, alpha=0.5)
plt.plot([0, 5], [0, 5], '--k')
plt.axis('equal')
plt.xlabel('True Value')
plt.ylabel('Predicted Value')
plt.title('Random Forest Regressor')
plt.show()
