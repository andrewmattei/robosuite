import os
import numpy as np
import datetime
import h5py
import mujoco

from robosuite.projects.impact_control.planar_arm_contact_ctrl import robot_xml, RobotSystem
import robosuite.projects.impact_control.optimizing_max_jacobian as opt

# smaller sample data size
ONLY_FINAL_STATES = True
# === Initialize the robot system === #
model = mujoco.MjModel.from_xml_string(robot_xml)
data = mujoco.MjData(model)
robot = RobotSystem(model, data, gui=False)

# === Define parameters === #
m = robot.m
l = robot.l
r = robot.r

# Bounds
q_lower = -4/5 * np.pi * np.ones(3)
q_upper =  4/5 * np.pi * np.ones(3)
dq_lower = -2 * np.pi * np.ones(3)
dq_upper =  2 * np.pi * np.ones(3)
tau_lower = -1 * np.ones(3)
tau_upper =  1 * np.ones(3)

# Optimization params
T = 0.3
N = 60
weight_v=1e-1, 
weight_xdd=1e-2,
weight_tau_smooth = 0, 
weight_terminal = 1e-3,
boundary_epsilon=1e-3

# Sampling parameters
n_px = 10
n_vx = 10
n_vy = 10

# Define ranges
l_px, r_px = 0.3, 0.7    # position x range around 0
l_vx, r_vx = -0.1, 0.1    # velocity x range
l_vy, r_vy = -3.0, -0.5   # velocity y range (negative for downward motion)


# Define the list of final states (p_f, v_f) for the robot
samples = opt.sample_pf_vf_grid(l_px, r_px, n_px,
                               l_vx, r_vx, n_vx,
                               l_vy, r_vy, n_vy)

# Output directory
# date_str
date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
save_dir = os.path.join(os.path.dirname(__file__), '..', 'results', date_str)
os.makedirs(save_dir, exist_ok=True)
if ONLY_FINAL_STATES:
    hdf5_path = os.path.join(save_dir, 'collected_dataset_only_final_states.hdf5')
else:
    hdf5_path = os.path.join(save_dir, 'collected_dataset.hdf5')


# HDF5 file write
with h5py.File(hdf5_path, 'w') as f:
    params_dict = {
        'weight_v': weight_v,
        'weight_xdd': weight_xdd,
        'weight_tau_smooth': weight_tau_smooth,
        'weight_terminal': weight_terminal,
        'boundary_epsilon': boundary_epsilon,
        'T': T,
        'N': N,
        'm': m,
        'l': l,
        'r': r,
        'q_lower': q_lower,
        'q_upper': q_upper,
        'dq_lower': dq_lower,
        'dq_upper': dq_upper,
        'tau_lower': tau_lower,
        'tau_upper': tau_upper,
        'l_px': l_px,
        'r_px': r_px,
        'l_vx': l_vx,
        'r_vx': r_vx,
        'l_vy': l_vy,
        'r_vy': r_vy,
        'n_px': n_px,
        'n_vx': n_vx,
        'n_vy': n_vy,
        'samples': samples
    }
    # Save the weights dictionary to the HDF5 file
    weights_grp = f.create_group('parameters')
    for key, value in params_dict.items():
        weights_grp.create_dataset(key, data=value)

    # Determine number of digits needed for padding based on total samples
    n_digits = len(str(len(samples)))
    # Save the parameters to the HDF5 file
    for i, (p_f, v_f) in enumerate(samples):
        run_id = str(i).zfill(n_digits)  # Zero-pad the index
        print(f"\n--- Running for condition {i+1}/{len(samples)} ---")
        try:
            sol = opt.optimize_trajectory_cartesian_accel_flex_pose(
                p_f, v_f, m, l, r,
                q_lower, q_upper, dq_lower, dq_upper,
                tau_lower, tau_upper,
                T, N,
                weight_v, weight_xdd, weight_tau_smooth, weight_terminal,
                boundary_epsilon
            )

            npy_path = os.path.join(save_dir, f'sol_{run_id}.npy')
            opt.save_solution_to_npy(sol, npy_path)

            sim_data = robot.run_simulation_offscreen(sol, sim_dt = 0.001)

            grp = f.create_group(f'run_{run_id}')
            grp.create_dataset('p_f_des', data=p_f)
            grp.create_dataset('v_f_des', data=v_f)
            grp.create_dataset('opt_cost', data=sol['cost'])
            grp.create_dataset('p_f_opt', data=sim_data['final_position'])
            grp.create_dataset('v_f_opt', data=sim_data['final_velocity'])
            grp.create_dataset('p_f_sim', data=sim_data['impact_pos'])
            grp.create_dataset('v_f_sim', data=sim_data['impact_vel'])
            grp.create_dataset('t_f_sim', data=sim_data['impact_time'])
            grp.create_dataset('traj_duration', data=sim_data['times'][-1] - sim_data['times'][0])
            grp.create_dataset('npy_path', data=npy_path)
            if not ONLY_FINAL_STATES:
                grp.create_dataset('q_opt', data=sim_data['q_opt'])
                grp.create_dataset('dq_opt', data=sim_data['dq_opt'])
                grp.create_dataset('tau_opt', data=sim_data['tau_opt'])
                grp.create_dataset('times_opt', data=sim_data['times_opt'])
                grp.create_dataset('ee_pos_opt', data=sim_data['ee_pos_opt'])
                grp.create_dataset('ee_vel_opt', data=sim_data['ee_vel_opt'])
                grp.create_dataset('q_sim', data=sim_data['q'])
                grp.create_dataset('dq_sim', data=sim_data['dq'])
                grp.create_dataset('tau_sim', data=sim_data['tau'])
                grp.create_dataset('ee_pos_sim', data=sim_data['ee_pos'])
                grp.create_dataset('ee_vel_sim', data=sim_data['ee_vel'])
                grp.create_dataset('time', data=sim_data['times'])

            # print(f"Saved run_{run_id} to: {hdf5_path}")

        except Exception as e:
            print(f"[ERROR] Failed for condition {i+1}: {e}")

if robot.gui_on:
    robot.viewer.close()
