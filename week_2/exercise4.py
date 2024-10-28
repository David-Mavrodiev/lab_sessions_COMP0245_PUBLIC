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

ada_regressor = AdaBoostRegressor(estimator=(DecisionTreeRegressor(max_depth=8)), n_estimators=50, random_state=42, loss='linear')
ada_regressor.fit(X_train, y_train)
y_pred = ada_regressor.predict(X_test)

mse_boosting = mean_squared_error(y_test, y_pred)
print(f"Boosting MSE: {mse_boosting}")
r2_boosting = r2_score(y_test, y_pred)
print(f"Boosting Score: {r2_boosting}")

# 绘图
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# 绘制真实值的2D图像
sc1 = ax[0].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis')
ax[0].set_title("True Values")
ax[0].set_xlabel("x1")
ax[0].set_ylabel("x2")
fig.colorbar(sc1, ax=ax[0], label="y")

# Decsion Tree预测值
sc2 = ax[1].scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='viridis')
ax[1].set_title("Boositng Prediction (max_depth=8)")
ax[1].set_xlabel("x1")
ax[1].set_ylabel("x2")
fig.colorbar(sc2, ax=ax[1], label="y")

plt.tight_layout()
plt.show()




