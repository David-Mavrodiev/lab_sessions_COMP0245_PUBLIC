import numpy as np
import time
import os
import json
import matplotlib.pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference 
from sklearn.metrics import r2_score, mean_squared_error

def update_noise_config(config_file_path, noise_level):
    """Update the noise levels in the configuration file."""
    with open(config_file_path, 'r') as file:
        config = json.load(file)

    # Update the noise covariances in the "robot_pybullet" -> "robot_noise"
    config["robot_pybullet"]["robot_noise"][0]["joint_cov"] = noise_level
    config["robot_pybullet"]["robot_noise"][0]["joint_vel_cov"] = noise_level
    config["robot_pybullet"]["robot_noise"][0]["joint_acc_cov"] = noise_level

    with open(config_file_path, 'w') as file:
        json.dump(config, file, indent=4)
    print(f"Updated noise level to {noise_level} for joint_cov, joint_vel_cov, and joint_acc_cov in the config file.")


def main():
    # Configuration for the simulation
    conf_file_name = "pandaconfig.json"  # Configuration file for the robot
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=cur_dir, use_gui=False)  # Initialize simulation interface

    # Get active joint names from the simulation
    ext_names = sim.getNameActiveJoints()
    ext_names = np.expand_dims(np.array(ext_names), axis=0)  # Adjust the shape for compatibility

    source_names = ["pybullet"]  # Define the source for dynamic modeling

    # Create a dynamic model of the robot
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False, 0, cur_dir)
    num_joints = dyn_model.getNumberofActuatedJoints()

    # Print initial joint angles
    print(f"Initial joint angles: {sim.GetInitMotorAngles()}")

    # Sinusoidal reference
    # Specify different amplitude values for each joint
    amplitudes = [np.pi/4, np.pi/6, np.pi/4, np.pi/4, np.pi/4, np.pi/4, np.pi/4]  # Example amplitudes for joints
    # Specify different frequency values for each joint
    frequencies = [0.4, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4]  # Example frequencies for joints

    # Convert lists to NumPy arrays for easier manipulation in computations
    amplitude = np.array(amplitudes)
    frequency = np.array(frequencies)
    ref = SinusoidalReference(amplitude, frequency, sim.GetInitMotorAngles())  # Initialize the reference
    
    
    # Simulation parameters
    time_step = sim.GetTimeStep()
    current_time = 0
    max_time = 5  # seconds
    
    # Command and control loop
    cmd = MotorCommands()  # Initialize command structure for motors
    # PD controller gains
    kp = 1000
    kd = 100

    # Initialize data storage
    tau_mes_all = []
    regressor_all = []


    # Data collection loop
    while current_time < max_time:
        # Measure current state
        q_mes = sim.GetMotorAngles(0)
        qd_mes = sim.GetMotorVelocities(0)
        qdd_mes = sim.ComputeMotorAccelerationTMinusOne(0)
        
        # Compute sinusoidal reference trajectory
        q_d, qd_d = ref.get_values(current_time)  # Desired position and velocity
        
        # Control command
        tau_cmd = feedback_lin_ctrl(dyn_model, q_mes, qd_mes, q_d, qd_d, kp, kd)
        cmd.SetControlCmd(tau_cmd, ["torque"]*7)  # Set the control command for the torque
        sim.Step(cmd, "torque")

        # Get measured torque
        tau_mes = sim.GetMotorTorques(0)

        if dyn_model.visualizer: 
            for index in range(len(sim.bot)):  # Conditionally display the robot model
                q = sim.GetMotorAngles(index)
                dyn_model.DisplayModel(q)  # Update the display of the robot model

        # Exit logic with 'q' key
        keys = sim.GetPyBulletClient().getKeyboardEvents()
        qKey = ord('q')
        if qKey in keys and keys[qKey] and sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
            break

        
        # TODO Compute regressor and store it
        if current_time > 0.75:
        # Store the regressor and the measured torque
        
            regressor_all.append(dyn_model.ComputeDynamicRegressor(q_mes,qd_mes, qdd_mes))
            tau_mes_all.append(tau_mes)
        
        
        current_time += time_step
        # Optional: print current time
        print(f"Current time in seconds: {current_time:.2f}")

    # TODO After data collection, stack all the regressor and all the torquen and compute the parameters 'a'  using pseudoinverse
    regressor_all = np.vstack(regressor_all)
    tau_mes_all = np.hstack(tau_mes_all)
   
    a = np.linalg.pinv(regressor_all) @ tau_mes_all

    print(f"Parameters 'a' for the linear model: {a}")

    # TODO validate the model by comparing the simulated output of the manipulator with the model learned of the estimated parameters 

    tau_pred_all = regressor_all @ a
    torque_error = tau_mes_all - tau_pred_all
    plt.plot(torque_error)
    plt.title("Torque prediction error")
    plt.xlabel("Time")
    plt.ylabel("Torque error")
    # plt.show()

    # Adjusted R-squared
    r2 = r2_score(tau_mes_all, tau_pred_all)
    n = regressor_all.shape[0]
    p = regressor_all.shape[1]
    adjusted_r2 = 1 - (((1 - r2) * (n - 1)) / (n - p - 1))

    residuals = tau_mes_all - tau_pred_all

    rss = np.sum(residuals**2)
    tss = np.sum((tau_mes_all - np.mean(tau_mes_all))**2)

    f_statistic = ((tss - rss) / p) / (rss / (n - p))

    mse = mean_squared_error(tau_mes_all, tau_pred_all)
    mae = np.mean(np.abs(tau_mes_all - tau_pred_all))
    se = np.sqrt(np.diagonal(mse * np.linalg.pinv(np.dot(regressor_all.T, regressor_all))))
    conf_intervals = [a - 1.96 * se, a + 1.96 * se]
    
    results = open("results.txt", "w")

    # # TODO compute the metrics for the linear model
    print(f"Adjusted R-squared: {adjusted_r2}")
    print(f"F-statistic: {f_statistic}")
    print(f"MAE: {mae}")
    # results.write(f"\n\n\nAdjusted R-squared: {adjusted_r2}")
    # results.write(f"\n\nF-statistic: {f_statistic}")
    # results.write(f"\n\nConfidence intervals: {conf_intervals}")

    # TODO plot the torque prediction error for each joint (optional)
    # Assuming each joint's torque is measured and stored separately, reshape the arrays
    # Reshape the torque arrays for easier plotting per joint
    num_joints = len(tau_mes_all) // len(amplitudes)  # Calculate number of samples per joint
    tau_mes_all = np.reshape(tau_mes_all, (num_joints, len(amplitudes)))
    tau_pred_all = np.reshape(tau_pred_all, (num_joints, len(amplitudes)))

    # Create a 2x4 grid layout for the subplots
    fig, axs = plt.subplots(2, 4, figsize=(20, 5))  # Adjust figure size for better visualization

    for i in range(len(amplitudes)):
        row, col = divmod(i, 4)  # Calculate row and column index
        axs[row, col].plot(tau_mes_all[:, i], label="Actual Torque")
        axs[row, col].plot(tau_pred_all[:, i], label="Predicted Torque", linestyle='--')
        axs[row, col].set_title(f"Joint {i + 1} Torque Comparison")
        axs[row, col].set_xlabel("Time Step")
        axs[row, col].set_ylabel("Torque (Nm)")
        axs[row, col].legend()

    # Hide the last empty subplot if any (for a 2x4 grid with 7 joints)
    for j in range(len(amplitudes), 8):
        fig.delaxes(axs.flatten()[j])

    # Show the combined plot
    plt.tight_layout()
    plt.show()

    

if __name__ == '__main__':
    main()
