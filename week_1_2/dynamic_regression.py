import numpy as np
import time
import os
import matplotlib.pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference 
import scipy.stats as stats

def plot_results(n):
    # Load arrays from file
    conf_intervals_pred = np.load('conf_intervals_pred.npy', allow_pickle=True)  # `allow_pickle=True` is needed when saving multiple arrays
    measurements = np.load('tau_mes.npy', allow_pickle=True)

    # Split confidence intervals into lower and upper bounds
    lower_bounds = [ci[0] for ci in conf_intervals_pred]
    upper_bounds = [ci[1] for ci in conf_intervals_pred]

    # Downsample the data, e.g., select every 100th point
    downsample_factor = 1000
    indices = np.arange(0, len(measurements[0]), downsample_factor)

    # Downsampled data
    downsampled_measured = measurements[0][indices]
    downsampled_predicted = measurements[1][indices]
    downsampled_lower_bounds = [lower_bounds[i] for i in indices]
    downsampled_upper_bounds = [upper_bounds[i] for i in indices]

    # Create figure for plotting
    plt.figure(figsize=(10, 6))

    # Plot downsampled data
    plt.plot(indices, downsampled_measured, label="Measured Torque", color="blue", marker="o", linestyle="None", markersize=4)
    plt.plot(indices, downsampled_predicted, label="Predicted Torque", color="red", linestyle="--")

    # Plot confidence intervals as grey shaded area (downsampled)
    plt.fill_between(indices, downsampled_lower_bounds, downsampled_upper_bounds, color="gray", alpha=0.3, label="95% Confidence Interval")

    plt.title("Measured vs Predicted Torque with Confidence Intervals (Downsampled)")
    plt.xlabel("Data Points (Downsampled)")
    plt.ylabel("Torque")
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    # Configuration for the simulation
    conf_file_name = "pandaconfig.json"  # Configuration file for the robot
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=cur_dir)  # Initialize simulation interface

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
    max_time = 10  # seconds
    
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
        cmd.tau_cmd = feedback_lin_ctrl(dyn_model, q_mes, qd_mes, q_d, qd_d, kp, kd)
        sim.Step(cmd, "torque")

        # Get measured torque
        tau_mes = sim.GetMotorTorques(0)

        tau_mes_all.append(tau_mes)
        
        if dyn_model.visualizer: 
            for index in range(len(sim.bot)):  # Conditionally display the robot model
                q = sim.GetMotorAngles(index)
                dyn_model.DisplayModel(q)  # Update the display of the robot model

        # Exit logic with 'q' key
        keys = sim.GetPyBulletClient().getKeyboardEvents()
        qKey = ord('q')
        if qKey in keys and keys[qKey] and sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
            break

        cur_regressor = dyn_model.ComputeDynamicRegressor(q_mes, qd_mes, qdd_mes)
        regressor_all.append(cur_regressor)
        
        current_time += time_step
        print(f"Current time in seconds: {current_time:.2f}")

    regressor_all = np.vstack(regressor_all)
    #regressor_all = np.hstack([np.ones((regressor_all.shape[0], 1)), regressor_all])
    tau_mes_all = np.hstack(tau_mes_all)

    #a = np.linalg.inv(np.dot(regressor_all.T, regressor_all)).dot(regressor_all.T).dot(tau_mes_all)

    a = np.linalg.pinv(regressor_all) @ tau_mes_all

    print("Estimated parameters 'a':", a)
    tau_pred = regressor_all @ a
    
    mse = np.mean((tau_mes_all - tau_pred)**2)
    print(f"Mean Squared Error: {mse}")

    ss_residual = np.sum((tau_mes_all - tau_pred) ** 2)
    ss_total = np.sum((tau_mes_all - np.mean(tau_mes_all)) ** 2)
    r_squared = 1 - (ss_residual / ss_total)
    print(f"R^2: {r_squared}")

    n = regressor_all.shape[0]
    p = regressor_all.shape[1]
    adjusted_r_squared = 1 - ((1 - r_squared) * (n - 1)) / (n - p)
    print(f"Adjusted R^2: {adjusted_r_squared}")

    f_statistic = (r_squared / p) / ((1 - r_squared) / (n - p))
    print(f"F-statistic: {f_statistic}")

    residuals = tau_mes_all - tau_pred
    residual_variance = np.sum(residuals ** 2) / (n - p)
    cov_matrix = residual_variance * np.linalg.pinv(regressor_all.T @ regressor_all)
    std_errors = np.sqrt(np.diag(cov_matrix))
    confidence_level = 0.95
    alpha = 1 - confidence_level
    t_value = stats.t.ppf(1 - alpha / 2, df=(n - p - 1))

    # Now plot the parameters and their confidence intervals
    plt.figure(figsize=(10, 6))

    # Plot the parameter estimates as points
    plt.errorbar(range(p), a, yerr=t_value * std_errors, fmt='o', label='Parameter Estimate', color='blue', capsize=5)

    # Add labels and title
    plt.xlabel('Parameter Index')
    plt.ylabel('Parameter Value')
    plt.title('Estimated Parameters with 95% Confidence Intervals (1 noise)')
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)  # Optional: add horizontal line at 0
    plt.grid(True)
    plt.legend()
    plt.show()

    #conf_intervals = [(a[i] - t_value * std_errors[i], a[i] + t_value * std_errors[i]) for i in range(p)]
    #print("Confidence Intervals for the Parameters:")
    #for i, conf in enumerate(conf_intervals):
    #    print(f"Parameter {i}: {conf}")

    # Confidence intervals for predictions
    #pred_std_errors = np.sqrt(np.diag(regressor_all @ cov_matrix @ regressor_all.T))
    #conf_intervals_pred = [(tau_pred[i] - t_value * pred_std_errors[i], tau_pred[i] + t_value * pred_std_errors[i]) for i in range(n)]

    #print("Confidence Intervals for Predictions:")
    #for i, conf_pred in enumerate(conf_intervals_pred):
    #    print(f"Prediction {i}: {conf_pred}")

    #np.save('conf_intervals_pred.npy', conf_intervals_pred, allow_pickle=True)
    #np.save('tau_mes.npy', [tau_mes_all, tau_pred], allow_pickle=True)

if __name__ == '__main__':
    main()
    #plot_results(100)
