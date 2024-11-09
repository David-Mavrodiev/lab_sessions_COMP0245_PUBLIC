import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import glob

# 初始化空列表以保存所有结果
all_results = []

# 遍历并加载每个文件
for file_name in glob.glob("C:/Users/23943/Desktop/2.2/test/0.6138_0.0596_0.12/results_depth_*.pkl"):
    with open(file_name, "rb") as f:
        results = pickle.load(f)
        all_results.extend(results)  # 将每个文件中的结果添加到 all_results 列表中

# 创建3D图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 设置一次图例避免重复
ax.scatter([], [], color='red', label='Goal Position')
ax.scatter([], [], color='blue', label='Final Predicted Position')

# 检查每个结果的范围
for result in all_results:
    depth = result["depth"]
    trajectory = np.array(result["trajectory"])
    goal_position = result["goal_position"]
    final_predicted_position = result["final_predicted_position"]
    position_error = result["position_error"]

    # 打印轨迹范围
    if trajectory.size > 0:
        print(f"Depth {depth} - Trajectory range: "
              f"X: {trajectory[:, 0].min()} to {trajectory[:, 0].max()}, "
              f"Y: {trajectory[:, 1].min()} to {trajectory[:, 1].max()}, "
              f"Z: {trajectory[:, 2].min()} to {trajectory[:, 2].max()}")

    # 绘制轨迹
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], label=f"Trajectory (depth={depth})")

    # 绘制目标位置和最终预测位置
    ax.scatter(*goal_position, color='red')
    ax.scatter(*final_predicted_position, color='blue')

    # 显示误差信息
    plt.figtext(0.85, 0.8 - depth * 0.05, f"Depth {depth} - Error: {position_error:.4f} m", bbox=dict(facecolor='white'))

# 设置坐标轴范围
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([0, 1])  # 根据您的数据范围进行调整

# 设置标签和标题
ax.set_xlabel("X Position (m)")
ax.set_ylabel("Y Position (m)")
ax.set_zlabel("Z Position (m)")
ax.set_title("Predicted Cartesian Trajectories with Different Depths")
plt.legend()
plt.show()
