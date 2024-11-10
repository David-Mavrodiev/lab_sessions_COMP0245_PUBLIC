import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import glob

# Initialise an empty list to save all results
all_results = []

# load each depth file
for file_name in glob.glob("C:/Users/23943/Desktop/2.2/test/0.6138_0.0596_0.12/results_depth_*.pkl"):
    with open(file_name, "rb") as f:
        results = pickle.load(f)
        all_results.extend(results) 

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter([], [], color='red', label='Goal Position')
ax.scatter([], [], color='blue', label='Final Predicted Position')

# for each depth
for result in all_results:
    depth = result["depth"]
    trajectory = np.array(result["trajectory"])
    goal_position = result["goal_position"]
    final_predicted_position = result["final_predicted_position"]
    position_error = result["position_error"]

    # check traj range
    if trajectory.size > 0:
        print(f"Depth {depth} - Trajectory range: "
              f"X: {trajectory[:, 0].min()} to {trajectory[:, 0].max()}, "
              f"Y: {trajectory[:, 1].min()} to {trajectory[:, 1].max()}, "
              f"Z: {trajectory[:, 2].min()} to {trajectory[:, 2].max()}")

    # plt traj
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], label=f"Trajectory (depth={depth})")

    # plt goal position and final predicted position
    ax.scatter(*goal_position, color='red')
    ax.scatter(*final_predicted_position, color='blue')

    # plt error
    plt.figtext(0.85, 0.8 - depth * 0.05, f"Depth {depth} - Error: {position_error:.4f} m", bbox=dict(facecolor='white'))

ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([0, 1]) 

ax.set_xlabel("X Position (m)")
ax.set_ylabel("Y Position (m)")
ax.set_zlabel("Z Position (m)")
ax.set_title("Predicted Cartesian Trajectories with Different Depths")
plt.legend()
plt.show()
