import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib  # For saving and loading models

# Set the visualization flag
visualize = True  # Set to True to enable visualization, False to disable
training_flag = False  # Set to True to train the models, False to skip training
test_cartesian_accuracy_flag = True  # Set to True to test the model with a new goal position, False to skip testing

# Save Data
save_dir = "C:/Users/23943/Desktop/2.2/test/0.6138_0.0596_0.12"
depth_values = [8]

if not training_flag:
    # Load the saved data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(script_dir, 'data.pkl')  # Replace with your actual filename
    if not os.path.isfile(filename):
        print(f"Error: File {filename} not found in {script_dir}")
    else:
        with open(filename, 'rb') as f:
            data = pickle.load(f)

        # Extract data
        time_array = np.array(data['time'])  

# 存储轨迹、预测目标位置和误差信息
results = []

for depth in depth_values:
    # Testing with new goal positions
    print("\nTesting the model with new goal positions...")

    # Load all the models into a list
    models = []
    for joint_idx in range(7):
        # Load the saved model
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # The name of the saved model
        model_filename = os.path.join(script_dir, f'rf_joint{joint_idx+1}_depth_{depth}.joblib') # 8个depth

        try:
            rf_model = joblib.load(model_filename)

        except FileNotFoundError:
            print(f"Cannot find file {model_filename}")
            print("task_22_goal_pos needs to be run at least once with training_flag=True")
            quit()

        models.append(rf_model)

    # Generate new goal positions
    goal_positions = [[0.6138, 0.0596, 0.12]]  # 设置一个固定的goal position

    # Generate test time array
    test_time_array = np.linspace(time_array.min(), time_array.max(), 100)  # For example, 100 time steps

    # Initialize the dynamic model
    from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, CartesianDiffKin

    conf_file_name = "pandaconfig.json"  # Configuration file for the robot
    root_dir = os.path.dirname(os.path.abspath(__file__))
    # Adjust root directory if necessary
    name_current_directory = "tests"
    root_dir = root_dir.replace(name_current_directory, "")
    # Initialize simulation interface
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=root_dir)

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

    for goal_position in goal_positions:
        print("\nTesting new goal position------------------------------------")
        print(f"Goal position: {goal_position}")

        # Create test input features
        test_goal_positions = np.tile(goal_position, (len(test_time_array), 1))  # Shape: (100, 3)
        test_input = np.hstack((test_time_array.reshape(-1, 1), test_goal_positions))  # Shape: (100, 4)

        # Predict joint positions for the new goal position
        predicted_joint_positions_over_time = np.zeros((len(test_time_array), 7))  # Shape: (num_points, 7)

        for joint_idx in range(7):
            # Predict joint positions
            y_pred = models[joint_idx].predict(test_input)  # Shape: (num_points,)
            # Store the predicted joint positions
            predicted_joint_positions_over_time[:, joint_idx] = y_pred

        # Get the final predicted joint positions (at the last time step)
        final_predicted_joint_positions = predicted_joint_positions_over_time[-1, :]  # Shape: (7,)

        # Compute forward kinematics
        final_cartesian_pos, final_R = dyn_model.ComputeFK(final_predicted_joint_positions, controlled_frame_name)

        print(f"Computed cartesian position: {final_cartesian_pos}")
        print(f"Predicted joint positions at final time step: {final_predicted_joint_positions}")

        # Compute position error
        position_error = np.linalg.norm(final_cartesian_pos - goal_position)
        print(f"Position error between computed position and goal: {position_error}")

        # Save data for later visualization
        cartesian_positions_over_time = []
        for i in range(len(test_time_array)):
            joint_positions = predicted_joint_positions_over_time[i, :]
            cartesian_pos, _ = dyn_model.ComputeFK(joint_positions, controlled_frame_name)
            cartesian_positions_over_time.append(cartesian_pos.copy())
            # print(f"Time step {i}: Cartesian position = {cartesian_pos}")  # 检查每个时间步的位置

        # cartesian_positions_over_time = np.array(cartesian_positions_over_time)  # Shape: (num_points, 3)
        
        # Append all results for this depth and goal position
        results.append({
            "depth": str(depth)+'_normalized',
            "trajectory": cartesian_positions_over_time,
            "goal_position": goal_position,
            "final_predicted_position": final_cartesian_pos.tolist(),
            "position_error": position_error
        })
   
        # print("Results to be saved:")
        # for res in results:
        #     print(f"Depth: {res['depth']}")
        #     print(f"Trajectory shape: {np.array(res['trajectory']).shape}")
        #     print(f"Trajectory data: {np.array(res['trajectory'])}")

# 存储所有的results
results_filename = os.path.join(save_dir, f'results_depth_{depth}.pkl')
with open(results_filename, 'wb') as f:
    pickle.dump(results, f)
print(f"Results saved to {results_filename}")
