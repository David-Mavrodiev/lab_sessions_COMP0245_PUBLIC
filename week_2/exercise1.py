import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.tree import plot_tree
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


# Decision tree 
tree = DecisionTreeRegressor(max_depth = 14)
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)

mse_1 = mean_squared_error(y_test, y_pred)
print(f"Decision Tree MSE: {mse_1}")
r2= r2_score(y_test, y_pred)
print(f"Decision Tree R^2 Score: {r2}")


# polynomial 
# degree = 5
poly = PolynomialFeatures(degree=5)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.fit_transform(X_test)

# Fit the model
poly_reg = LinearRegression()
poly_reg.fit(X_train_poly, y_train)
y_pred_poly = poly_reg.predict(X_test_poly)

mse_poly = mean_squared_error(y_test, y_pred_poly)
print(f"Poly MSE: {mse_poly}")
r2_poly = r2_score(y_test, y_pred_poly)
print(f"Poly R^2 Score: {r2_poly}")


# test different max_depth and splitter
depths = [3, 5, 7, 10]  # max_depth
splitters = ['best', 'random']  # splitter

# save the error
print("Test diff depth of decision tree:")
for depth in depths:
    for splitter in splitters:
        tree_1 = DecisionTreeRegressor(max_depth=depth, splitter=splitter, random_state=42)
        tree_1.fit(X_train, y_train)
        y_pred_1 = tree_1.predict(X_test)
        mse_1 = mean_squared_error(y_test, y_pred_1)  # 计算均方误差
        r2_1 = r2_score(y_test, y_pred_1)
        print(f"max_depth: {depth}, splitter: {splitter}, Test MSE: {mse_1}, Test R^2: {r2_1}")
        # best one:
        # max_depth: 10, splitter: best, Test MSE: 0.020857446079726286, Test R^2: 0.9199109127735708


fig, ax = plt.subplots(1, 3, figsize=(12, 6))

# true values
sc1 = ax[0].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis')
ax[0].set_title("True Values")
ax[0].set_xlabel("x1")
ax[0].set_ylabel("x2")
fig.colorbar(sc1, ax=ax[0], label="y")

# Decsion Tree predict values
sc2 = ax[1].scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='viridis')
ax[1].set_title("Decision Tree (max_depth=14)")
ax[1].set_xlabel("x1")
ax[1].set_ylabel("x2")
fig.colorbar(sc2, ax=ax[1], label="y")

# Poly predict values
sc3 = ax[2].scatter(X_test[:, 0], X_test[:, 1], c=y_pred_poly, cmap='viridis')
ax[2].set_title("Polynomial Regression")
ax[2].set_xlabel("x1")
ax[2].set_ylabel("x2")
fig.colorbar(sc3, ax=ax[2], label="y")

plt.tight_layout()
plt.show()


# 绘制Decision Tree
# plt.figure(figsize=(16, 10))  # Set the size of the figure
# plot_tree(tree)
# plt.show()  # Show the plot

# # 绘制图像
# fig = plt.figure(figsize=(10, 8))

# # 绘制真实值的3D图像
# ax1 = fig.add_subplot(121, projection='3d')
# ax1.scatter(X_test[:, 0], X_test[:, 1], y_test, c='blue', label='True Values', marker='o')
# ax1.set_title("True Values")
# ax1.set_xlabel("x1")
# ax1.set_ylabel("x2")
# ax1.set_zlabel("y")
# ax1.legend()

# # 绘制预测值的3D图像
# ax2 = fig.add_subplot(122, projection='3d')
# ax2.scatter(X_test[:, 0], X_test[:, 1], y_pred, c='red', label='Predicted Values', marker='^')
# ax2.set_title("Predicted Values")
# ax2.set_xlabel("x1")
# ax2.set_ylabel("x2")
# ax2.set_zlabel("y")
# ax2.legend()

# plt.tight_layout()
# plt.show()



