import numpy as np
import time
import os
import matplotlib.pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference 
from sklearn.metrics import mean_squared_error, r2_score

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
    time_all = []  # 用于存储当前时间

    # Data collection loop
    # while current_time < max_time - time_step: # 确保只运行10000个time
    while current_time < max_time:

        # 存储时间 用于后续做图
        time_all.append(current_time)

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
        tau_mes = sim.GetMotorTorques(0) # 这里获得的数据应该是7*1

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
        # 此处改动，通过q qd qdd计算regressor 即Y
        current_regressor = dyn_model.ComputeDynamicRegressor(q_mes, qd_mes, qdd_mes) # 此处的regression matrix应该是 7*70

        # print(f"current_regressor shape: {current_regressor.shape}")
        # print(f"tau_mes shape: {tau_mes.shape}")

        regressor_all.append(current_regressor) # 这是当前的真实值regressor Y 此处应该是7*70
        tau_mes_all.append(tau_mes) # 这是当前t的真实值力矩,一共七个joint 所以需要不断循环收集 此处应该是7*1

        # print(f"regressor_all shape: {regressor_all.shape}")
        # print(f"tau_mes_all shape: {tau_mes_all.shape}")
        
        current_time += time_step
        # Optional: print current time
        print(f"Current time in seconds: {current_time:.2f}")

    # TODO After data collection, stack all the regressor and all the torquen and compute the parameters 'a'  using pseudoinverse
    # 机器人每个t下的力矩torque即regressor_all 为Y  tau_mes_all就是u
    regression_all = []
    regression_all = np.vstack(regressor_all) # 此处应该10000个regressor_all vertical堆叠（乘前面） 是70000*70
    tau_real = []
    tau_real = np.hstack(tau_mes_all) #  u应该是70000*1  !!!IMPORTANT!!! horizontal stack!!

    # print(f"regressor_all shape: {regressor_all.shape}")
    # print(f"tau_real shape: {tau_real.shape}")

    # 计算伪逆矩阵 a = Y.u
    # a = np.linalg.pinv(regressor_all).dot(tau_real)
    a = np.linalg.pinv(regression_all) @ (tau_real) # a必须是一个（n*1）的矩阵 此处应为70*1   故regression matrix的伪逆应该是70*70000 u应该是70000*1
    # print(f"a: {a}")
    # print(f"a shape: {a.shape}")
    
    # TODO compute the metrics for the linear model
    # 预测力矩 预测下一步的行动
    tau_pred = regression_all @ (a) # 70*1
    # print(f"tau_pred shape: {tau_pred.shape}")
   
    # TODO plot the  torque prediction error for each joint (optional)
    #mse = (mean_squared_error(tau_real, tau_pred)) / 10000
    #r2 = r2_score(tau_real, tau_pred)
    #print("MSE:", mse)
    #print("R^2:", r2)

    # 将tau_real和tau_pred重塑为(N, 7)，每列对应一个关节 !!!IMPORTANT!!!
    N = len(tau_mes_all)  # 时间步数
    tau_real = np.array(tau_mes_all).reshape(N, 7)
    tau_pred = tau_pred.reshape(N, 7)
    time_all = np.array(time_all)

    # 初始化列表来存储每个关节的MSE和R²
    mse_per_joint = []
    r2_per_joint = []

    # 对每个关节计算MSE和R²
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()  # 将轴数组展平成一维

    time_filter = time_all >= 0.5
    time_all_filtered = time_all[time_filter]
    tau_real_filtered = tau_real[time_filter]
    tau_pred_filtered = tau_pred[time_filter]

    # **计算F统计量**
    # 首先，展开筛选后的实际值和预测值为一维数组
    y = tau_real_filtered.flatten()     # 实际值，形状：(N,)
    y_pred = tau_pred_filtered.flatten()  # 预测值，形状：(N,)
    # 总的观测数
    n = len(y)
    # 参数数量（包括截距项，如果有）
    # 这里假设模型中没有截距项，所以参数数量为回归矩阵的列数
    p = regression_all.shape[1]
    # 计算y的平均值
    y_mean = np.mean(y)
    # 计算总平方和（TSS）
    TSS = np.sum((y - y_mean) ** 2)
    # 计算残差平方和（SSE）
    SSE = np.sum((y - y_pred) ** 2)
    # 计算回归平方和（SSR）
    SSR = TSS - SSE
    # 计算F统计量
    F_statistic = (SSR / (p - 1)) / (SSE / (n - p))
    print(f"F-statistic: {F_statistic}")

    # 置信区间计算
    # 标准误差
    SE = np.sqrt(SSE / (n - p))  # 整体的标准误差
    # 设置置信区间的因子，95%的置信区间对应1.96倍的标准误差
    z = 1.96
    confidence_interval = z * SE

    for i in range(7):
        mse = (mean_squared_error(tau_real[:, i], tau_pred[:, i])) / 10000 #  !!!IMPORTANT!!! 别忘了/10000
        r2 = r2_score(tau_real[:, i], tau_pred[:, i])
        mse_per_joint.append(mse)
        r2_per_joint.append(r2)
        print(f"joint: {i+1}")
        print(f"MSE: {mse}")
        print(f"R²: {r2}")

        ax = axes[i]
        ax.plot(time_all_filtered, tau_real_filtered[:, i], label='real')
        ax.plot(time_all_filtered, tau_pred_filtered[:, i], label='pred', linestyle='--')

        # 计算置信区间
        tau_real_joint = tau_real_filtered[:, i]
        tau_pred_joint = tau_pred_filtered[:, i]
        #upper_bound = tau_pred_joint + confidence_interval  # 上置信区间
        #lower_bound = tau_pred_joint - confidence_interval  # 下置信区间
        #ax.fill_between(time_all_filtered, lower_bound, upper_bound, color='gray', alpha=0.3, label='95% confidence interval')

        ax.set_title(f'joint {i+1} torque')
        ax.set_xlabel('time')
        ax.set_ylabel('torque')
        # ax.legend()
        ax.grid(True)

        # 调整子图之间的间距
    fig.delaxes(axes[-1])  # 删除最后一个子图，因为只有7个关节
    plt.subplots_adjust(hspace=0.5)  # 调整子图之间的高度间距

    # 添加统一的图例
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3)  # 在图的顶部居中放置图例

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
