from sklearn.datasets import fetch_california_housing
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

data = fetch_california_housing()
X = data.data
y = data.target
feature_names = data.feature_names

# Perform data preprocessing
# Calculate IQR to remove outliers
Q1 = np.percentile(X, 25, axis=0)
Q3 = np.percentile(X, 75, axis=0)
IQR = Q3 - Q1
lower_bound = Q1 - 2.3 * IQR
upper_bound = Q3 + 2.3 * IQR

# 过Filtering Outliers
mask = np.all((X >= lower_bound) & (X <= upper_bound), axis=1)
X = X[mask]
y = y[mask]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# # Split data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# # 决策树 1
# tree = DecisionTreeRegressor(max_depth = 13)
# tree.fit(X_train, y_train)
# y_pred_decision_tree = tree.predict(X_test)

# # bagging 2
# bagging_regressor = BaggingRegressor(estimator=DecisionTreeRegressor(max_depth=13), n_estimators=50, random_state=42) # skilearn已更新，base_estimator => estimator
# bagging_regressor.fit(X_train, y_train)
# y_pred_bagging = bagging_regressor.predict(X_test)

# # 随机森林 3
# random_forest_regressor = RandomForestRegressor(n_estimators=50, random_state=42) # skilearn已更新，base_estimator => estimator
# random_forest_regressor.fit(X_train, y_train)
# y_pred_random_forest = random_forest_regressor.predict(X_test)

# adaboosting 4
ada_regressor = AdaBoostRegressor(estimator=(DecisionTreeRegressor(max_depth=13)), n_estimators=50, random_state=42, loss='linear')
ada_regressor.fit(X_train, y_train)
y_pred_boosting = ada_regressor.predict(X_test)

y_pred_boosting = np.clip(y_pred_boosting, None, 5) # Clip values to 5

# y_pred_all = [y_pred_decision_tree, y_pred_bagging, y_pred_random_forest, y_pred_boosting]
# y_pred = y_pred_all[0]
# for i, y_pred_all in enumerate(y_pred_all):
#     mse_temp = 0
#     mse = mean_squared_error(y_test, y_pred_all)
#     r2 = r2_score(y_test, y_pred_all)
#     print(f"Model {i+1} MSE: {mse}")
#     print(f"Model {i+1} R^2 Score: {r2}")
#     mse_temp = mse
#     if mse_temp < mse: # 选择MSE最小的模型
#         mse_temp = mse
#         y_pred = y_pred_all
#     else:
#         continue

mse_ran_for = mean_squared_error(y_test, y_pred_boosting)
print(f"random forest MSE: {mse_ran_for}")
r2_ran_for = r2_score(y_test, y_pred_boosting)
print(f"random forest R^2 Score: {r2_ran_for}")

# 绘制真实值与预测值对比 越接近红线 效果越好
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_boosting, color='blue', alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title("True vs Predicted House Prices")
plt.xlabel("True House Price")
plt.ylabel("Predicted House Price")
plt.show()

##
## 这是效果最好的！！
##