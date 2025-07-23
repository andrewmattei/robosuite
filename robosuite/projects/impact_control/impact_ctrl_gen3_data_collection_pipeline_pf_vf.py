import os
import numpy as np
import datetime
import h5py
import robosuite as suite
import robosuite.projects.shared_scripts.optimizing_gen3_arm as opt
import robosuite.projects.impact_control.gen3_contact_ctrl as gen3_ctrl

# === Initialize robot environment === #

env = suite.make(
    env_name="Kinova3ContactControl",
    robots="Kinova3SRL",
    has_renderer=False,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    control_freq=20, # doesn't matter here
    horizon=10000, # doesn't matter here
)

# Reset the environment
env.reset()
env.gui_on = False
active_robot = env.robots[0]

# === Define parameters === #
# Bounds from environment
q_lower = env.q_lower
q_upper = env.q_upper
dq_lower = env.dq_lower
dq_upper = env.dq_upper
tau_lower = env.tau_lower
tau_upper = env.tau_upper

# Optimization params
T = 1.0  # Time horizon
N = 100  # Number of discretization points
v_p_mag = 1.5  # Peak velocity magnitude
sim_dt = 1e-3  # sec
record_dt = 1e-3  # sec


# Sampling parameters
n_px = 10  # number of x positions to sample
n_py = 10  # number of z positions to sample
n_vz = 10  # number of z velocities to sample

# Define ranges for sampling
# Position ranges (in meters)
p_ee_to_ball_bottom = 0.05 + 0.025  # EE to ball bottom distance
l_px, r_px = 0.3, 0.6    # x position range
l_py, r_py = -0.4, 0.4    # z position range relative to table
# Velocity ranges (in m/s)
l_vz, r_vz = -0.2, -0.05  # z velocity range (negative for downward motion)

# Function to generate target poses and velocities
def sample_pf_vf_grid():
    """Generate grid of final poses and velocities"""
    px = np.linspace(l_px, r_px, n_px)
    py = np.linspace(l_py, r_py, n_py)
    vz = np.linspace(l_vz, r_vz, n_vz)
    
    samples = []
    for x in px:
        for y in py:
            for vz_val in vz:
                p_f = np.array([x, y, p_ee_to_ball_bottom])
                v_f = np.array([0.0, 0.0, vz_val])
                samples.append((p_f, v_f))
    
    return samples

# Generate samples
samples = sample_pf_vf_grid()

# Output directory setup
date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
save_dir = os.path.join(os.path.dirname(__file__), '..', 'results', date_str)
os.makedirs(save_dir, exist_ok=True)
hdf5_path = os.path.join(save_dir, 'collected_dataset_gen3.hdf5')

# HDF5 file write
with h5py.File(hdf5_path, 'w') as f:
    # Save parameters
    params_dict = {
        'T': T,
        'N': N,
        'v_p_mag': v_p_mag,
        'sim_dt': sim_dt,
        'record_dt': record_dt,
        'q_lower': q_lower,
        'q_upper': q_upper,
        'dq_lower': dq_lower,
        'dq_upper': dq_upper,
        'tau_lower': tau_lower,
        'tau_upper': tau_upper,
        'l_px': l_px,
        'r_px': r_px,
        'l_py': l_py,
        'r_py': r_py,
        'l_vz': l_vz,
        'r_vz': r_vz,
        'n_px': n_px,
        'n_py': n_py,
        'n_vz': n_vz,
        'samples': samples
    }
    
    weights_grp = f.create_group('parameters')
    for key, value in params_dict.items():
        weights_grp.create_dataset(key, data=value)

    # Padding for run IDs
    n_digits = len(str(len(samples)))
    
    # Run optimization and simulation for each sample
    for i, (p_f, v_f) in enumerate(samples):
        run_id = str(i).zfill(n_digits)
        print(f"\n--- Running for condition {i+1}/{len(samples)} ---")
        
        try:
            # Get IK solution for target pose
            target_pose = env.fk_fun(active_robot.init_qpos).full()
            target_pose[:3, 3] = p_f
            q_sol = opt.inverse_kinematics_casadi(
                target_pose, 
                env.fk_fun,
                active_robot.init_qpos, 
                env.q_lower, 
                env.q_upper
            ).full().flatten()

            # Generate initial trajectory using manipulability ellipsoid
            traj = opt.back_propagate_traj_using_manip_ellipsoid(
                v_f, q_sol, env.fk_fun, env.jac_fun,
                N=N, dt=T/N, v_p_mag=v_p_mag
            )

            # Precompute LQR linearization
            Z_init = traj['Z']
            T_init = traj['T']
            U_init = traj['U']
            env.linearization_cache = opt.linearize_dynamics_along_trajectory(
                T_init, U_init, Z_init, 
                env.M_fun, env.C_fun, env.G_fun
            )

            # Run simulation
            sim_data = env.run_simulation_offscreen(
                T_init, U_init, Z_init,
                slow_factor=1.0,
                sim_dt=sim_dt,
                record_dt=record_dt
            )

            # Save optimization and simulation results
            npy_path = os.path.join(save_dir, f'sol_{run_id}.npy')
            np.save(npy_path, traj)

            # Create dataset for this run
            grp = f.create_group(f'run_{run_id}')
            grp.create_dataset('p_f_des', data=p_f)
            grp.create_dataset('v_f_des', data=v_f)
            grp.create_dataset('p_f_opt', data=sim_data['ee_pos_opt'][-1])
            grp.create_dataset('v_f_opt', data=sim_data['ee_vel_opt'][-1])
            grp.create_dataset('p_f_sim', data=sim_data['impact_pos'])
            grp.create_dataset('v_f_sim', data=sim_data['impact_vel'])
            grp.create_dataset('t_f_sim', data=sim_data['impact_time'])
            grp.create_dataset('traj_duration', data=sim_data['times'][-1] - sim_data['times'][0])
            grp.create_dataset('npy_path', data=npy_path)
            
            # Save trajectory data
            grp.create_dataset('q_opt', data=sim_data['Z_opt'][:active_robot.dof,:])
            grp.create_dataset('dq_opt', data=sim_data['Z_opt'][active_robot.dof:,:])
            grp.create_dataset('tau_opt', data=sim_data['tau_opt'])
            grp.create_dataset('times_opt', data=sim_data['times_opt'])
            grp.create_dataset('ee_pos_opt', data=sim_data['ee_pos_opt'])
            grp.create_dataset('ee_vel_opt', data=sim_data['ee_vel_opt'])
            grp.create_dataset('q_sim', data=sim_data['q_pos'])
            grp.create_dataset('dq_sim', data=sim_data['q_vel'])
            grp.create_dataset('tau_sim', data=sim_data['tau'])
            grp.create_dataset('ee_pos_sim', data=sim_data['ee_pos'])
            grp.create_dataset('ee_vel_sim', data=sim_data['ee_vel'])
            grp.create_dataset('time', data=sim_data['times'])

        except Exception as e:
            print(f"[ERROR] Failed for condition {i+1}: {e}")

print(f"Data collection complete. Results saved to: {hdf5_path}")
