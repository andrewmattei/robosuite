import os
import numpy as np
import datetime
import h5py
import mujoco

from robosuite.demos.planar_arm_contact_ctrl import robot_xml, RobotSystem
import robosuite.demos.optimizing_max_jacobian as opt


# === Define parameters === #
# === Initialize the robot system === #
model = mujoco.MjModel.from_xml_string(robot_xml)
data = mujoco.MjData(model)
robot = RobotSystem(model, data, gui=False)

m = robot.m
l = robot.l
r = robot.r

# Bounds
q_lower = -4/5 * np.pi * np.ones(3)
q_upper =  4/5 * np.pi * np.ones(3)
dq_lower = -2 * np.pi * np.ones(3)
dq_upper =  2 * np.pi * np.ones(3)
tau_lower = -10 * np.ones(3)
tau_upper =  10 * np.ones(3)

# Optimization params
T = 0.3
N = 60

# Define the list of final states (q_f, v_f) for the robot
q_ref = np.array([np.pi*0.5, -np.pi*0.5, -np.pi*0.5])
# create a list vel with x 0, y varies from -3 to -0.5 with step 0.5
v_f_list = [np.array([0.0, y]) for y in np.arange(-3.0, -0.4, 0.5)]


# the end-effector sampled will lie on the table with coordinate x coordinate 
# [fk[q_ref][0]-l_ext, fk[q_ref[0]+r_ext] and same y coordinate as fk[q_ref][1]
samples = opt.sample_qf_given_y_line_casadi(robot, q_ref,
                                            l_ext=0.2, r_ext=0.2, n_pose = 5,
                                            v_f_list=v_f_list
                                            )

# Output directory
# date_str
date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
save_dir = os.path.join(os.path.dirname(__file__), '..', 'results', date_str)
os.makedirs(save_dir, exist_ok=True)
hdf5_path = os.path.join(save_dir, 'collected_dataset.hdf5')


# HDF5 file write
with h5py.File(hdf5_path, 'w') as f:
    weights_dict = {
        'weight_v': 1e-1,
        'weight_xdd': 1e-2,
        'weight_tau_smooth': 0.0,
        'weight_terminal': 1e-3
    }
    # Save the weights dictionary to the HDF5 file
    weights_grp = f.create_group('weights')
    for key, value in weights_dict.items():
        weights_grp.create_dataset(key, data=value)
    # Save the parameters to the HDF5 file
    for i, (q_f, v_f) in enumerate(samples):
        print(f"\n--- Running for condition {i+1}/{len(samples)} ---")
        try:
            sol = opt.optimize_trajectory_cartesian_accel(
                q_f, v_f, m, l, r,
                q_lower, q_upper, dq_lower, dq_upper,
                tau_lower, tau_upper,
                T, N,
                weight_v=1e-1, weight_xdd=1e-2,
                weight_tau_smooth=0.0, weight_terminal=1e-3
            )

            npy_path = os.path.join(save_dir, f'sol_{i}.npy')
            opt.save_solution_to_npy(sol, npy_path)

            sim_data = robot.run_simulation_offscreen(sol, sim_dt = 0.001)

            grp = f.create_group(f'run_{i}')
            grp.create_dataset('q', data=sim_data['q'])
            grp.create_dataset('dq', data=sim_data['dq'])
            grp.create_dataset('tau', data=sim_data['tau'])
            grp.create_dataset('ee_pos', data=sim_data['ee_pos'])
            grp.create_dataset('ee_vel', data=sim_data['ee_vel'])
            grp.create_dataset('time', data=sim_data['times'])
            grp.create_dataset('q_f_des', data=q_f)
            grp.create_dataset('v_f_des', data=v_f)
            grp.create_dataset('opt_cost', data=sol['cost'])
            grp.create_dataset('q_f_sim', data=sim_data['final_pose'])
            grp.create_dataset('v_f_sim', data=sim_data['final_velocity'])
            grp.create_dataset('traj_duration', data=sim_data['times'][-1] - sim_data['times'][0])
            grp.create_dataset('npy_path', data=npy_path)

            print(f"Saved run_{i} to: {hdf5_path}")

        except Exception as e:
            print(f"[ERROR] Failed for condition {i+1}: {e}")

if robot.gui_on:
    robot.viewer.close()
