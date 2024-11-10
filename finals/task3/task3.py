import numpy as np
import time
import os
import matplotlib.pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference, CartesianDiffKin
import threading
import pickle
import torch.nn as nn
import torch
from sklearn.ensemble import RandomForestRegressor
import joblib  # For saving and loading models
import plotly.graph_objects as go

# # Set the model type: "neural_network" or "random_forest"
# neural_network_or_random_forest = "neural_network"  # Change to "random_forest" to use Random Forest models

# MLP Model Definition
class MLP(nn.Module):
    def __init__(self, hidden_size=512):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(4, hidden_size),  # Input layer to hidden layer (4 inputs: time + goal positions)
            nn.ReLU(),
            nn.Linear(hidden_size, 1)   # Hidden layer to output layer
        )

    def forward(self, x):
        return self.model(x)
    
def exponential_moving_average(data, alpha=0.3):
    smoothed_data = np.zeros_like(data)
    smoothed_data[0] = data[0]  # Initial condition
    for t in range(1, len(data)):
        smoothed_data[t] = alpha * data[t] + (1 - alpha) * smoothed_data[t - 1]
    return smoothed_data

def plot_actual_joint_trajectories(pos, vel, des_pos=None, des_vel=None):
    for joint_idx in range(7):
        for model_name in pos.keys():
            fig, axs = plt.subplots(2, 1, figsize=(10, 8))  # 2x1 grid for position and velocity
            fig.suptitle(f'Joint {joint_idx+1} - {model_name} Trajectory Tracking')
            
            # Position plot for the model
            axs[0].plot(pos[model_name][:, joint_idx], label=f'{model_name} Position')
            if des_pos is not None and model_name in des_pos:
                axs[0].plot(des_pos[model_name][:, joint_idx], label='Desired Position', linestyle='--', color='orange', zorder=10)
            axs[0].set_title(f'{model_name} - Position')
            axs[0].set_xlabel('Time steps')
            axs[0].set_ylabel('Position (rad)')
            axs[0].legend()
            axs[0].grid(True)

            # Velocity plot for the model
            axs[1].plot(vel[model_name][:, joint_idx], label=f'{model_name} Velocity')
            if des_vel is not None and model_name in des_vel:
                axs[1].plot(des_vel[model_name][:, joint_idx], label='Desired Velocity', linestyle='--', color='orange', zorder=10)
            axs[1].set_title(f'{model_name} - Velocity')
            axs[1].set_xlabel('Time steps')
            axs[1].set_ylabel('Velocity (rad/s)')
            axs[1].legend()
            axs[1].grid(True)

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            # if des_pos is not None and model_name in des_pos:
            #     plt.savefig(f"/Users/joefarah/Desktop/Figures/ML_Final/Task_3/joint{joint_idx+1}_{model_name}_with_actual.png", dpi=300)
            # else:
            #     plt.savefig(f"/Users/joefarah/Desktop/Figures/ML_Final/Task_3/joint{joint_idx+1}_{model_name}_desired_only.png", dpi=300)
            plt.show()
            # plt.close("all")

def plot_tracking_error_and_torque(tracking_errors, control_torques):
    num_joints = 7

    # Plot tracking errors for each joint with subplots for each model
    for joint_idx in range(num_joints):
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))  # 2x2 grid for up to 4 models
        fig.suptitle(f'Tracking Error for Joint {joint_idx + 1}')
        
        for i, (model_name, error) in enumerate(tracking_errors.items()):
            row, col = divmod(i, 2)
            axs[row, col].plot(error[:, joint_idx], label=f'{model_name}')
            axs[row, col].set_title(model_name)
            axs[row, col].set_xlabel('Time steps')
            axs[row, col].set_ylabel('Tracking Error (rad)')
            axs[row, col].grid(True)
            axs[row, col].legend()
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f"/Users/joefarah/Desktop/Figures/ML_Final/Task_3/joint{joint_idx+1}_error.png", dpi=300)
        # plt.show()
        plt.close("all")

    # Plot control torques for each joint with subplots for each model
    for joint_idx in range(num_joints):
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))  # 2x2 grid for up to 4 models
        fig.suptitle(f'Control Torque for Joint {joint_idx + 1}')
        
        for i, (model_name, torques) in enumerate(control_torques.items()):
            row, col = divmod(i, 2)
            axs[row, col].plot(torques[:, joint_idx], label=f'{model_name}')
            axs[row, col].set_title(model_name)
            axs[row, col].set_xlabel('Time steps')
            axs[row, col].set_ylabel('Torque (Nm)')
            axs[row, col].grid(True)
            axs[row, col].legend()
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f"/Users/joefarah/Desktop/Figures/ML_Final/Task_3/joint{joint_idx+1}_torque.png", dpi=300)
        # plt.show()
        # plt.close("all")

def plot_with_smoothed_data(pos, vel):
    model_names = list(pos.keys())
    
    # Iterate through the dictionary in pairs
    for joint_idx in range(7):
        for i in range(0, len(model_names), 2):
            original_model = model_names[i]
            smoothed_model = model_names[i + 1] if i + 1 < len(model_names) else None  # Check if there's an original model
            
            fig, axs = plt.subplots(2, 1, figsize=(10, 8))  # 2x1 grid for position and velocity
            fig.suptitle(f'Joint {joint_idx+1} - {smoothed_model.split("with")[0].strip()} Trajectory Tracking')

            # Plot smoothed position and velocity data
            axs[0].plot(pos[smoothed_model][:, joint_idx], label=f'Position (Smoothed)')
            axs[1].plot(vel[smoothed_model][:, joint_idx], label=f'Velocity (Smoothed)', zorder=10)

            # Plot original position and velocity data if available
            if original_model:
                axs[0].plot(pos[original_model][:, joint_idx], label=f'Position (Original)', linestyle='--')
                axs[1].plot(vel[original_model][:, joint_idx], label=f'Velocity (Original)', linestyle='--')

            # Configure plot labels and legends
            axs[0].set_title(f'{smoothed_model} - Position')
            axs[0].set_xlabel('Time steps')
            axs[0].set_ylabel('Position (rad)')
            axs[0].legend()
            axs[0].grid(True)

            axs[1].set_title(f'{smoothed_model} - Velocity')
            axs[1].set_xlabel('Time steps')
            axs[1].set_ylabel('Velocity (rad/s)')
            axs[1].legend()
            axs[1].grid(True)

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            # plt.savefig(f"/Users/joefarah/Desktop/Figures/ML_Final/Task_3/joint{joint_idx+1}_{smoothed_model}_smoothed_with_original.png", dpi=300)
            plt.show()
            # plt.close("all")

def plot_goal_vs_final_position_with_models(goal_position, final_positions):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    fig.suptitle('Goal vs Final Position for All Models')

    # Plot the goal position
    ax.scatter(goal_position[0], goal_position[1], goal_position[2], label='Goal Position', color='red', marker='x')

    # Plot the final position for each model and draw lines with error text
    for model_name, final_position in final_positions.items():
        ax.scatter(final_position[0], final_position[1], final_position[2], label=f'{model_name} Final Position')
        
        # Draw a line from the goal position to the final position
        ax.plot([goal_position[0], final_position[0]], 
                [goal_position[1], final_position[1]], 
                [goal_position[2], final_position[2]], 
                linestyle='--', color='gray')
        
        # Calculate the error
        error = np.linalg.norm(np.array(goal_position) - np.array(final_position))
        
        # Add text along the line with the error
        mid_point = [(goal_position[0] + final_position[0]) / 2,
                     (goal_position[1] + final_position[1]) / 2,
                     (goal_position[2] + final_position[2]) / 2]
        ax.text(mid_point[0], mid_point[1], mid_point[2], f'Error: {error:.3f}', color='black', fontsize=8)

    # Configure plot labels and legends
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Z Position (m)')
    ax.legend(loc='upper left')
    ax.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    # plt.savefig("/Users/joefarah/Desktop/Figures/ML_Final/Task_3/3dplot.png", dpi=300)
    plt.show()
    plt.close("all")

    return ax.get_xlim(), ax.get_ylim(), ax.get_zlim()


def plot_goal_vs_final_position_with_models_plotly(goal_position, final_positions, x_limits, y_limits, z_limits, output_filename='/Users/joefarah/Desktop/Figures/ML_Final/Task_3/3dplot.html', show_plot=True):
    # Create the figure
    fig = go.Figure()

    # Plot the goal position (red marker, 'x' shape)
    fig.add_trace(go.Scatter3d(
        x=[goal_position[0]],
        y=[goal_position[1]],
        z=[goal_position[2]],
        mode='markers',
        marker=dict(size=3, color='red', symbol='x'),
        name='Goal Position',
        hovertext='Goal Position'
    ))

    # Plot each final position with corresponding labels and colors
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink']  # Match colors to your Matplotlib plot
    for i, (model_name, final_position) in enumerate(final_positions.items()):
        fig.add_trace(go.Scatter3d(
            x=[final_position[0]],
            y=[final_position[1]],
            z=[final_position[2]],
            mode='markers',
            marker=dict(size=5, color=colors[i % len(colors)]),  # Rotate through colors
            hovertext=f"{model_name} Final Position",
            name=f'{model_name} Final Position'
        ))

        # Add dashed line from the goal position to each final position
        fig.add_trace(go.Scatter3d(
            x=[goal_position[0], final_position[0]],
            y=[goal_position[1], final_position[1]],
            z=[goal_position[2], final_position[2]],
            mode='lines',
            line=dict(color='gray', dash='dash'),
            showlegend=False
        ))

        # Calculate and display the error at the midpoint
        error = np.linalg.norm(np.array(goal_position) - np.array(final_position))
        mid_point = [(goal_position[0] + final_position[0]) / 2,
                     (goal_position[1] + final_position[1]) / 2,
                     (goal_position[2] + final_position[2]) / 2]
        fig.add_trace(go.Scatter3d(
            x=[mid_point[0]],
            y=[mid_point[1]],
            z=[mid_point[2]],
            mode='text',
            text=[f'Error: {error:.3f}'],
            textposition='top center',
            showlegend=False
        ))

    # Configure the layout to match Matplotlib's look, using the provided axis limits
    fig.update_layout(
        title="Goal vs Final Position for All Models",
        scene=dict(
            xaxis=dict(title="X Position (m)", range=x_limits, backgroundcolor="white", gridcolor="lightgray", zerolinecolor="gray"),
            yaxis=dict(title="Y Position (m)", range=y_limits, backgroundcolor="white", gridcolor="lightgray", zerolinecolor="gray"),
            zaxis=dict(title="Z Position (m)", range=z_limits, backgroundcolor="white", gridcolor="lightgray", zerolinecolor="gray"),
        ),
        paper_bgcolor="white",  # Set the entire plot background to white
        plot_bgcolor="white",   # Set the 3D scene's surrounding area background to white
        legend=dict(x=0.01, y=0.99)  # Position the legend to the upper left
    )

    # Show plot interactively
    if show_plot:
        fig.show()

    # Save plot as HTML
    # fig.write_html(output_filename)
    # print(f"Interactive 3D plot saved as {output_filename}")

def main(neural_network_or_random_forest="neural_network", depth=None, goal_position=None, smoothing_alpha=None):
    # Load the saved data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(script_dir, 'data.pkl')  # Replace with your actual filename
    if not os.path.isfile(filename):
        print(f"Error: File {filename} not found in {script_dir}")
        return
    else:
        with open(filename, 'rb') as f:
            data = pickle.load(f)

        # Extract data
        time_array = np.array(data['time'])            # Shape: (N,)
        # Optional: Normalize time data for better performance
        # time_array = (time_array - time_array.min()) / (time_array.max() - time_array.min())

    # Load all the models in a list
    models = []
    if neural_network_or_random_forest == "neural_network":
        for joint_idx in range(7):
            # Instantiate the model
            model = MLP()
            # Load the saved model
            model_filename = os.path.join(script_dir, f'mlp_models/neuralq{joint_idx+1}.pt')
            model.load_state_dict(torch.load(model_filename))
            model.eval()
            models.append(model)
    elif neural_network_or_random_forest == "random_forest":
        for joint_idx in range(7):
            # Load the saved Random Forest model
            model_filename = os.path.join(script_dir, f'tree_models/rf{"_" + str(depth) if depth != None else ""}_joint{joint_idx+1}.joblib')
            model = joblib.load(model_filename)
            models.append(model)
    else:
        print("Invalid model type specified. Please set neural_network_or_random_forest to 'neural_network' or 'random_forest'")
        return

    # Generate a new goal position
    goal_position_bounds = {
        'x': (0.6, 0.8),
        'y': (-0.1, 0.1),
        'z': (0.12, 0.12)
    }
    # Create a set of goal positions

    goal_positions = []
    if goal_position is None:
        number_of_goal_positions_to_test = 10
        for i in range(number_of_goal_positions_to_test):
            goal_positions.append([
                np.random.uniform(*goal_position_bounds['x']),
                np.random.uniform(*goal_position_bounds['y']),
                np.random.uniform(*goal_position_bounds['z'])
            ])
    else:
        goal_positions.append(goal_position)

    conf_file_name = "pandaconfig.json"  # Configuration file for the robot
    root_dir = os.path.dirname(os.path.abspath(__file__))

    # Configuration for the simulation
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=root_dir, use_gui=False)  # Initialize simulation interface

    # Get active joint names from the simulation
    ext_names = sim.getNameActiveJoints()
    ext_names = np.expand_dims(np.array(ext_names), axis=0)  # Adjust the shape for compatibility

    source_names = ["pybullet"]  # Define the source for dynamic modeling

    # Create a dynamic model of the robot
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False, 0, root_dir)
    num_joints = dyn_model.getNumberofActuatedJoints()

    controlled_frame_name = "panda_link8"
    init_joint_angles = sim.GetInitMotorAngles()
    init_cartesian_pos, init_R = dyn_model.ComputeFK(init_joint_angles, controlled_frame_name)
    print(f"Initial joint angles: {init_joint_angles}")

    # Command and control loop
    cmd = MotorCommands()  # Initialize command structure for motors

    # PD controller gains
    kp = 1000  # Proportional gain
    kd = 100   # Derivative gain

    # Get joint velocity limits
    joint_vel_limits = sim.GetBotJointsVelLimit()

    time_step = sim.GetTimeStep()
    # Generate test time array
    test_time_array = np.arange(time_array.min(), time_array.max(), time_step)

    for goal_position in goal_positions:
        print("Testing new goal position------------------------------------")
        print(f"Goal position: {goal_position}")

        # Initialize the simulation
        sim.ResetPose()
        current_time = 0  # Initialize current time

        # Create test input features
        test_goal_positions = np.tile(goal_position, (len(test_time_array), 1))  # Shape: (num_points, 3)
        test_input = np.hstack((test_time_array.reshape(-1, 1), test_goal_positions))  # Shape: (num_points, 4)

        # Predict joint positions for the new goal position
        predicted_joint_positions_over_time = np.zeros((len(test_time_array), 7))  # Shape: (num_points, 7)

        for joint_idx in range(7):
            if neural_network_or_random_forest == "neural_network":
                # Prepare the test input
                test_input_tensor = torch.from_numpy(test_input).float()  # Shape: (num_points, 4)

                # Predict joint positions using the neural network
                with torch.no_grad():
                    predictions = models[joint_idx](test_input_tensor).numpy().flatten()  # Shape: (num_points,)
            elif neural_network_or_random_forest == "random_forest":
                # Predict joint positions using the Random Forest
                predictions = models[joint_idx].predict(test_input)  # Shape: (num_points,)

            # Store the predicted joint positions
            if smoothing_alpha is not None:
                # For smoothing the data
                predicted_joint_positions_over_time[:, joint_idx] = exponential_moving_average(predictions, alpha=smoothing_alpha)
            else:
                predicted_joint_positions_over_time[:, joint_idx] = predictions

            

        # Compute qd_des_over_time by numerically differentiating the predicted joint positions
        qd_des_over_time = np.gradient(predicted_joint_positions_over_time, axis=0, edge_order=2) / time_step
        # Clip the joint velocities to the joint limits
        qd_des_over_time_clipped = np.clip(qd_des_over_time, -np.array(joint_vel_limits), np.array(joint_vel_limits))

        q_des_all, qd_des_all, q_mes_all, qd_mes_all = [], [], [], []
        tracking_error, tau_cmd_all = [], []

        # Data collection loop
        while current_time < test_time_array.max() + time_step:
            # Measure current state
            q_mes = sim.GetMotorAngles(0)
            qd_mes = sim.GetMotorVelocities(0)
            qdd_est = sim.ComputeMotorAccelerationTMinusOne(0)

            # Get the index corresponding to the current time
            current_index = int(current_time / time_step)
            if current_index >= len(test_time_array):
                current_index = len(test_time_array) - 1

            # Get q_des and qd_des_clip from predicted data
            q_des = predicted_joint_positions_over_time[current_index, :]  # Shape: (7,)
            qd_des_clip = qd_des_over_time_clipped[current_index, :]      # Shape: (7,)

            # Control command
            tau_cmd = feedback_lin_ctrl(dyn_model, q_mes, qd_mes, q_des, qd_des_clip, kp, kd)
            cmd.SetControlCmd(tau_cmd, ["torque"] * 7)  # Set the torque command
            sim.Step(cmd, "torque")  # Simulation step with torque command

            # Keyboard event handling
            keys = sim.GetPyBulletClient().getKeyboardEvents()
            qKey = ord('q')

            # Exit logic with 'q' key
            if qKey in keys and keys[qKey] & sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
                print("Exiting simulation.")
                break

            # Store data
            tracking_error.append(q_des - q_mes)
            tau_cmd_all.append(tau_cmd)
            q_mes_all.append(q_mes)
            qd_mes_all.append(qd_mes)
            q_des_all.append(q_des)
            qd_des_all.append(qd_des_clip)

            # Time management
            # time.sleep(time_step)  # Control loop timing
            current_time += time_step

        # After the trajectory, compute the final cartesian position
        final_predicted_joint_positions = predicted_joint_positions_over_time[-1, :]  # Shape: (7,)
        final_cartesian_pos, final_R = dyn_model.ComputeFK(final_predicted_joint_positions, controlled_frame_name)
        print(f"Final computed cartesian position: {final_cartesian_pos}")
        # Compute position error
        position_error = np.linalg.norm(final_cartesian_pos - goal_position)
        print(f"Position error between computed position and goal: {position_error}")

        # Convert the lists to numpy arrays
        tracking_error = np.array(tracking_error)   # Shape: (num_points, 7)
        tau_cmd_all = np.array(tau_cmd_all)         # Shape: (num_points, 7)
        q_mes_all = np.array(q_mes_all)             # Shape: (num_points, 7)
        qd_mes_all = np.array(qd_mes_all)           # Shape: (num_points, 7)
        q_des_all = np.array(q_des_all)             # Shape: (num_points, 7)
        qd_des_all = np.array(qd_des_all)           # Shape: (num_points, 7)

        # Compute actual final cartesian position
        final_q_mes = q_mes_all[-1, :]
        final_actual_cartesian_pos, final_R = dyn_model.ComputeFK(final_q_mes, controlled_frame_name)

        return q_des_all, tracking_error, tau_cmd_all, goal_positions, qd_des_all, q_mes_all, qd_mes_all, final_actual_cartesian_pos



if __name__ == '__main__':
    positions_actual = {}
    velocity_actual = {}
    desired_poss = {}
    desired_vels = {}
    tracking_errors = {}
    control_torques = {}
    goal_positions = []
    final_positions = {}

    print(f"MLP")
    desired_pos, tracking_error_nn, tau_cmd_all_nn, goal_positions, desired_vel, q_mes_all, qd_mes_all, final_pos = main("neural_network")
    positions_actual["Neural Network"] = q_mes_all
    velocity_actual["Neural Network"] = qd_mes_all
    desired_poss["Neural Network"] = desired_pos
    desired_vels["Neural Network"] = desired_vel
    tracking_errors["Neural Network"] = tracking_error_nn
    control_torques["Neural Network"] = tau_cmd_all_nn
    final_positions["Neural Network"] = final_pos
    print(f"Model type: MLP completed")

    depths = [None, 2, 10]
    
    for depth in depths:
        for i in range(2):
            alpha = 0.1 if depth != 2 else 0.001
            print(f"Random Forest with depth: {depth}")
            goal_positions = None if len(goal_positions) == 0 else goal_positions[0] # Use the same goal position for all models
            desired_pos, tracking_error_rf, tau_cmd_all_rf, goal_positions, desired_vel, q_mes_all, qd_mes_all, final_pos = main("random_forest", depth, goal_positions, smoothing_alpha=alpha if i == 1 else None)
            positions_actual[f"Random Forest {'smoothed ' if i == 1 else ''}with depth: {depth}"] = q_mes_all
            velocity_actual[f"Random Forest {'smoothed ' if i == 1 else ''}with depth: {depth}"] = qd_mes_all
            desired_poss[f"Random Forest {'smoothed ' if i == 1 else ''}with depth: {depth}"] = desired_pos
            desired_vels[f"Random Forest {'smoothed ' if i == 1 else ''}with depth: {depth}"] = desired_vel
            if i == 0:
                tracking_errors[f"Random Forest with depth: {depth}"] = tracking_error_rf
                control_torques[f"Random Forest with depth: {depth}"] = tau_cmd_all_rf
            final_positions[f"Random Forest {'smoothed ' if i == 1 else ''}with depth: {depth}"] = final_pos
            print(f"Random Forest {'smoothed ' if i == 1 else ''}with depth: {depth} completed")

    plot_actual_joint_trajectories(desired_poss, desired_vels)
    plot_actual_joint_trajectories(positions_actual, velocity_actual, desired_poss, desired_vels)
    plot_tracking_error_and_torque(tracking_errors, control_torques)
    plot_with_smoothed_data(dict(list(desired_poss.items())[1:]), dict(list(desired_vels.items())[1:])) # Skip the first model as it is the Neural Network
    x_lim, y_lim, z_lim = plot_goal_vs_final_position_with_models(goal_positions[0], final_positions)
    plot_goal_vs_final_position_with_models_plotly(goal_positions[0], final_positions, x_lim, y_lim, z_lim, show_plot=True)



