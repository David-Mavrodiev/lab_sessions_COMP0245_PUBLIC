import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib  # For saving and loading models

# Save Data
base_dir = "C:/Users/23943/Desktop/RF_depth_diff_train"

# max_depth
max_depth_values = [2]

# save MSE
# mse_summary = {joint_idx: [] for joint_idx in range(7)}

# Load the saved data
script_dir = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(script_dir, 'data.pkl')

# Check if the file exists
if not os.path.isfile(filename):
    print(f"Error: File {filename} not found in {script_dir}")
else:
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    # Extract data
    time_array = np.array(data['time'])            # Shape: (N,)
    q_mes_all = np.array(data['q_mes_all'])        # Shape: (N, 7)
    goal_positions = np.array(data['goal_positions'])  # Shape: (N, 3)

    # Optional: Normalize time data for better performance
    # time_array = (time_array - time_array.min()) / (time_array.max() - time_array.min())

    # Combine time and goal data to form the input features
    X = np.hstack((time_array.reshape(-1, 1), goal_positions))  # Shape: (N, 4)

    # Split ratio
    split_ratio = 0.8

    # Initialize lists to hold training and test data for all joints
    x_train_list = []
    x_test_list = []
    y_train_list = []
    y_test_list = []

    for max_depth in max_depth_values:
        depth_dir = os.path.join(base_dir, str(max_depth))
        os.makedirs(depth_dir, exist_ok=True)

        mse_values = []  # store the MSE of each joint at this depth

        for joint_idx in range(7):
            y = q_mes_all[:, joint_idx]  # Shape: (N,)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, train_size=split_ratio, shuffle=True, random_state=42
            )

            # Store split data
            x_train_list.append(X_train)
            x_test_list.append(X_test)
            y_train_list.append(y_train)
            y_test_list.append(y_test)

            # Initialize the Random Forest regressor
            rf_model = RandomForestRegressor(
                n_estimators=100,    # Number of trees
                max_depth=max_depth, # Maximum depth of the tree
                random_state=42,
                n_jobs=-1            # Use all available cores
            )

            # TRAIN
            rf_model.fit(X_train, y_train)

            # EVALUATION
            y_test_pred = rf_model.predict(X_test)
            test_mse = np.mean((y_test - y_test_pred) ** 2)
            mse_values.append(test_mse)

            # save the model
            model_filename = os.path.join(depth_dir, f'rf_joint{joint_idx+1}_depth_{max_depth}.joblib')
            joblib.dump(rf_model, model_filename)
            print(f'Model for Joint {joint_idx+1} saved as {model_filename}')

            # predict trajectory plt
            sorted_indices = np.argsort(X_test[:, 0])
            X_test_sorted = X_test[sorted_indices]
            y_test_sorted = y_test[sorted_indices]
            y_test_pred_sorted = y_test_pred[sorted_indices]

            plt.figure(figsize=(10, 5))
            plt.plot(X_test_sorted[:, 0], y_test_sorted, label='True Joint Positions')
            plt.plot(X_test_sorted[:, 0], y_test_pred_sorted, label='Predicted Joint Positions', linestyle='--')
            plt.xlabel('Time (s)')
            plt.ylabel('Joint Position (rad)')
            plt.title(f'Joint {joint_idx+1} Position Prediction on Test Set')
            plt.legend()
            plt.grid(True)
            # plt.savefig(os.path.join(depth_dir, f'joint_{joint_idx+1}_depth_{max_depth}.png'))
            # plt.show()
            plt.close()
        
        # mse_summary[max_depth] = mse_values
        # # save MSE log
        # with open(os.path.join(depth_dir, f'mse_log_depth_{max_depth}.txt'), 'w') as f:
        #     f.write(f'max_depth={max_depth} 的MSE:\n')
        #     f.write('\n'.join([f'关节 {i+1}: {mse:.6f}' for i, mse in enumerate(mse_values)]))

# 生成并保存 max_depth 对平均MSE的影响图
# for joint_idx in range(7):
#     plt.figure()
#     plt.plot(max_depth_values, mse_summary[joint_idx], marker='o')
#     plt.xlabel('Max Depth')
#     plt.ylabel('MSE')
#     plt.title(f'joint {joint_idx+1} - max_depth vs MSE')
#     plt.grid()
#     plt.savefig(os.path.join(base_dir, f'joint_{joint_idx+1}_depth_vs_mse.png'))
#     plt.close()

# # 将MSE汇总保存为表格 用于后续数据分析
# import pandas as pd
# mse_table = pd.DataFrame(mse_summary, index=[f'Joint {i+1}' for i in range(7)])
# mse_table.to_csv(os.path.join(base_dir, 'mse_summary.csv'))
