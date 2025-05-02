import numpy as np
import casadi as cs
import matplotlib.pyplot as plt
import os
import robosuite.demos.optimizing_max_jacobian as omj
import h5py
import pathlib

# current_dir = pathlib.Path.cwd()
current_dir = os.path.dirname(os.path.abspath(__file__))
# Define the path to the HDF5 file
results_dir = "../results/2025-04-20_19-10-20"
hdf5_file = "collected_dataset_only_final_states.hdf5"
hdf5_path = os.path.join(current_dir, results_dir, hdf5_file)

# Read the v_f_des from run_729
run_id = 729
run_str = f"run_{run_id}"
param_str = "parameters"
with h5py.File(hdf5_path, 'r') as f:
    v_f_des = f[run_str]['v_f_des'][:]
    print(f"Desired final velocity from {run_str}: {v_f_des}")
    v_f_act = f[run_str]['v_f_act'][:]
    print(f"Actual impact velocity from {run_str}: {v_f_act}")

    p_f_des = f[run_str]['p_f_des'][:]
    print(f"Desired final position from {run_str}: {p_f_des}") 
    p_f_act = f[run_str]['p_f_act'][:]
    print(f"Actual final position from {run_str}: {p_f_act}")

    # Read the parameters from the HDF5 file
    m = f[param_str]['m'][:]
    l = f[param_str]['l'][:]
    r = f[param_str]['r'][:]
    q_lower = f[param_str]['q_lower'][:]
    q_upper = f[param_str]['q_upper'][:]
    dq_lower = f[param_str]['dq_lower'][:]
    dq_upper = f[param_str]['dq_upper'][:]
    tau_lower = f[param_str]['tau_lower'][:]
    tau_upper = f[param_str]['tau_upper'][:]
    T = f[param_str]['T'][()]
    N = f[param_str]['N'][()]
    weight_v = f[param_str]['weight_v'][()]
    weight_xdd = f[param_str]['weight_xdd'][()]
    weight_tau_smooth = f[param_str]['weight_tau_smooth'][()]
    weight_terminal = f[param_str]['weight_terminal'][()]
    boundary_epsilon = f[param_str]['boundary_epsilon'][()]
    

# Optimize the trajectory
solution = omj.optimize_trajectory_cartesian_accel_flex_pose(p_f_des, v_f_des, m, l, r,
                        q_lower, q_upper, dq_lower, dq_upper,
                        tau_lower, tau_upper,
                        T, N,
                        weight_v, weight_xdd,
                        weight_tau_smooth, weight_terminal,
                        boundary_epsilon)
omj.display_and_save_solution(solution, 'opt_trajectory_max_cartesian_accel_flex_pose.npy')
# check the offscreen simulation
from robosuite.demos.planar_arm_contact_ctrl import robot_xml, RobotSystem
import mujoco
# === Initialize the robot system === #
model = mujoco.MjModel.from_xml_string(robot_xml)
data = mujoco.MjData(model)
robot = RobotSystem(model, data, gui=False)
sim_data = robot.run_simulation_offscreen(solution, sim_dt = 0.001)
print("p_f_des:", sim_data["p_f_des"], "v_f_des:", sim_data["v_f_des"])
print("p_f_opt:", sim_data["final_position"], "v_f_opt:", sim_data["final_velocity"])
print("p_f_sim:", sim_data["impact_pos"], "v_f_sim:", sim_data["impact_vel"])