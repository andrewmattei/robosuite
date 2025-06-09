import os
import sys
import numpy as np
import datetime
import h5py
import time
import traceback

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'demos'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

import robosuite.demos.optimizing_gen3_arm as opt
import robosuite.demos.gen3_contact_ctrl_on_hardware as hw_ctrl
import robosuite.utils.tool_box_no_ros as tb
import robosuite.utils.kortex_utilities as kortex_utils

from kortex_api.autogen.messages import BaseCyclic_pb2


class TCPArguments:
    def __init__(self):
        self.ip = "192.168.0.10"
        self.username = "admin"
        self.password = "admin"

def find_last_completed_run(hdf5_path):
    """Find the last successfully completed run in the HDF5 file"""
    if not os.path.exists(hdf5_path):
        return -1
    
    try:
        with h5py.File(hdf5_path, 'r') as f:
            run_keys = [key for key in f.keys() if key.startswith('run_')]
            if not run_keys:
                return -1
            
            # Extract run numbers and find the highest
            run_numbers = [int(key.split('_')[1]) for key in run_keys]
            return max(run_numbers)
    except Exception as e:
        print(f"Error reading existing file: {e}")
        return -1

def load_existing_parameters(hdf5_path):
    """Load existing parameters from HDF5 file"""
    try:
        with h5py.File(hdf5_path, 'r') as f:
            params_grp = f['parameters']
            params = {}
            for key in params_grp.keys():
                params[key] = params_grp[key][()]
            return params
    except Exception as e:
        print(f"Error loading parameters: {e}")
        return None

def ask_user_action(failed_run_id, total_runs):
    """Ask user what to do after a failed run"""
    print(f"\n{'='*60}")
    print(f"❌ Failed to collect data for run {failed_run_id}")
    print(f"Progress: {failed_run_id}/{total_runs}")
    print(f"{'='*60}")
    print("Options:")
    print("1. Continue to next trajectory (c)")
    print("2. Retry current trajectory (r)")
    print("3. Exit data collection (e)")
    print("4. Exit and save resume point (s)")
    print("5. Rerun specific trajectory and exit (o)")  # New option
    
    while True:
        choice = input("Choose option [c/r/e/s/o]: ").lower().strip()
        if choice in ['c', 'r', 'e', 's', 'o']:
            return choice
        print("Invalid choice. Please enter 'c', 'r', 'e', 's', or 'o'")


def rerun_single_trajectory():
    """Handle rerunning a single specific trajectory"""
    print("\n🎯 RERUN SINGLE TRAJECTORY")
    print("="*60)
    
    # Look for existing data directories
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    if not os.path.exists(results_dir):
        print("No previous data collection sessions found.")
        return None
    
    # Find hardware data directories
    hw_dirs = [d for d in os.listdir(results_dir) if d.startswith('hardware_data_')]
    if not hw_dirs:
        print("No previous hardware data collection sessions found.")
        return None
    
    hw_dirs.sort(reverse=True)  # Most recent first
    
    print("Found previous data collection sessions:")
    for i, dirname in enumerate(hw_dirs[:5]):  # Show last 5
        dir_path = os.path.join(results_dir, dirname)
        hdf5_path = os.path.join(dir_path, 'hardware_collected_dataset_gen3.hdf5')
        if os.path.exists(hdf5_path):
            last_run = find_last_completed_run(hdf5_path)
            print(f"{i+1}. {dirname} - Last run: {last_run}")
    
    # Select session
    choice = input("\nEnter session number: ").strip()
    try:
        session_idx = int(choice) - 1
        if not (0 <= session_idx < len(hw_dirs)):
            print("Invalid session selection.")
            return None
    except ValueError:
        print("Invalid session selection.")
        return None
    
    selected_dir = os.path.join(results_dir, hw_dirs[session_idx])
    hdf5_path = os.path.join(selected_dir, 'hardware_collected_dataset_gen3.hdf5')
    
    if not os.path.exists(hdf5_path):
        print("HDF5 file not found.")
        return None
    
    # Load existing parameters and find available runs
    existing_params = load_existing_parameters(hdf5_path)
    if existing_params is None:
        print("Failed to load existing parameters.")
        return None
    
    # Get total number of samples
    total_samples = int(existing_params['total_samples'])
    
    # Show available runs to rerun
    print(f"\nTotal possible runs: 0 to {total_samples-1}")
    
    # Check which runs exist
    try:
        with h5py.File(hdf5_path, 'r') as f:
            existing_runs = [key for key in f.keys() if key.startswith('run_')]
            existing_run_ids = sorted([int(key.split('_')[1]) for key in existing_runs])
            print(f"Existing runs: {existing_run_ids[:10]}{'...' if len(existing_run_ids) > 10 else ''}")
    except Exception as e:
        print(f"Error reading existing runs: {e}")
        return None
    
    # Get specific run to rerun
    run_input = input(f"\nEnter specific run number to rerun (0-{total_samples-1}): ").strip()
    try:
        target_run_id = int(run_input)
        if not (0 <= target_run_id < total_samples):
            print(f"Run ID must be between 0 and {total_samples-1}")
            return None
    except ValueError:
        print("Invalid run ID.")
        return None
    
    return {
        'session_dir': selected_dir,
        'hdf5_path': hdf5_path,
        'target_run_id': target_run_id,
        'params': existing_params
    }


def create_resume_file(save_dir, last_completed_run, total_samples):
    """Create a resume file with current progress"""
    resume_file = os.path.join(save_dir, 'resume_info.txt')
    with open(resume_file, 'w') as f:
        f.write(f"Last completed run: {last_completed_run}\n")
        f.write(f"Total samples: {total_samples}\n")
        f.write(f"Next run to execute: {last_completed_run + 1}\n")
        f.write(f"Remaining runs: {total_samples - (last_completed_run + 1)}\n")
    print(f"Resume information saved to: {resume_file}")

def check_for_resume():
    """Check if user wants to resume from previous session"""
    print("\n" + "="*60)
    print("🔄 RESUME DATA COLLECTION")
    print("="*60)
    
    # Look for existing data directories
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    if not os.path.exists(results_dir):
        print("No previous data collection sessions found.")
        return None, None
    
    # Find hardware data directories
    hw_dirs = [d for d in os.listdir(results_dir) if d.startswith('hardware_data_')]
    if not hw_dirs:
        print("No previous hardware data collection sessions found.")
        return None, None
    
    hw_dirs.sort(reverse=True)  # Most recent first
    
    print("Found previous data collection sessions:")
    for i, dirname in enumerate(hw_dirs[:5]):  # Show last 5
        dir_path = os.path.join(results_dir, dirname)
        hdf5_path = os.path.join(dir_path, 'hardware_collected_dataset_gen3.hdf5')
        if os.path.exists(hdf5_path):
            last_run = find_last_completed_run(hdf5_path)
            resume_file = os.path.join(dir_path, 'resume_info.txt')
            status = "✅ Complete" if not os.path.exists(resume_file) else f"📊 Last run: {last_run}"
            print(f"{i+1}. {dirname} - {status}")
    
    choice = input("\nEnter session number to resume (or 'n' for new session): ").strip()
    
    if choice.lower() == 'n':
        return None, None
    
    try:
        session_idx = int(choice) - 1
        if 0 <= session_idx < len(hw_dirs):
            selected_dir = os.path.join(results_dir, hw_dirs[session_idx])
            hdf5_path = os.path.join(selected_dir, 'hardware_collected_dataset_gen3.hdf5')
            if os.path.exists(hdf5_path):
                return selected_dir, hdf5_path
    except ValueError:
        pass
    
    print("Invalid selection. Starting new session.")
    return None, None


def sample_pf_vf_grid(n_px=5, n_py=5, n_vz=5,
                      l_px=0.4, r_px=0.6,    # x position range
                      l_py=-0.12, r_py=0.16,   # y position range
                      z_impact=0.1,
                      l_vz=-0.5, r_vz=-0.1):

    # Generate grid of final poses and velocities
    # Position ranges (in meters)
    # Velocity ranges (in m/s)
    # z velocity range (negative for downward motion)
    
    px = np.linspace(l_px, r_px, n_px)
    py = np.linspace(l_py, r_py, n_py)
    vz = np.linspace(l_vz, r_vz, n_vz)
    
    samples = []
    for x in px:
        for y in py:
            for vz_val in vz:
                p_f = np.array([x, y, z_impact])
                v_f = np.array([0.0, 0.0, vz_val])
                samples.append((p_f, v_f))
    
    return samples

def collect_single_trajectory(gen3, p_f, v_f, run_id, save_dir):
    """Collect data for a single trajectory"""
    try:
        print(f"\nRun {run_id}: Target position: {p_f}, Target velocity: {v_f}")
        
        
        # Initial configuration
        q_init = np.radians([0, 15, 180, -130, 0, -35, 90])
        target_pose = gen3.fk_fun(q_init).full()
        target_pose[0:3, 3] = p_f
        
        # Solve inverse kinematics
        q_sol = opt.inverse_kinematics_casadi(
            target_pose,
            gen3.fk_fun,
            q_init, gen3.q_lower, gen3.q_upper
        ).full().flatten()
        
        # Generate trajectory
        T_horizon = 1.0  # seconds
        N = 150  # number of points
        dt = T_horizon / N  # time step
        # v_p_mag = 1.0  # m/s
        v_offset = 0.5  # m/s offset for velocity magnitude
        v_p_mag = v_offset + np.linalg.norm(v_f) 
        
        traj = opt.back_propagate_traj_using_manip_ellipsoid(
            v_f, q_sol, gen3.fk_fun, gen3.jac_fun, N=N, dt=dt, v_p_mag=v_p_mag
        )
        
        T_opt = traj['T']
        U_opt = traj['U']
        Z_opt = traj['Z']

        # import goal to gen3
        gen3.p_f = p_f
        gen3.v_f = v_f
        gen3.v_p = v_p_mag
        gen3.R_f = target_pose[:3, :3]

        # Obtain linearization cache for LQR controller
        gen3.linearization_cache = opt.linearize_dynamics_along_trajectory(
            T_opt, U_opt, Z_opt, gen3.M_fun, gen3.C_fun, gen3.G_fun
        )
        
        print(f"Generated trajectory with {len(T_opt)} points, duration: {T_opt[-1]:.2f}s")
        ## lyapunov retreat parameters
        gen3.retreat_distance = 0.15  # meters
        retreat_target_pose = target_pose.copy()
        retreat_target_pose[0:3, 3] += gen3.retreat_distance * (-v_f) / np.linalg.norm(v_f)
        q_retreat = opt.inverse_kinematics_casadi(
            retreat_target_pose,
            gen3.fk_fun,
            q_sol, gen3.q_lower, gen3.q_upper
        ).full().flatten()
        gen3.retreat_q = q_retreat
        
        # Move to start position
        q_start_360 = tb.to_kinova_joints(Z_opt[:7, 0])
        
        joint_speed = None  # degrees per second
        gen3.action_aborted = False
        finished = tb.move_joints(gen3.base, q_start_360, 
                                joint_speed,
                                gen3.check_for_end_or_abort)

        if gen3.action_aborted or not finished:
            print("Failed to move to start position")
            return None
            
        print("Moved to start position. Starting trajectory execution...")
        
        # Execute trajectory
        success = gen3.init_low_level_control(
            sampling_time=0.001, t_end=60,
            target_func=lambda dt: gen3.run_trajectory_execution(T_opt, U_opt, Z_opt, dt)
        )
        
        if success:
            print("Control initialized successfully. Running...")
            while gen3.cyclic_running:
                try:
                    time.sleep(0.1)
                except KeyboardInterrupt:
                    print("Keyboard interrupt received")
                    break
            
            # Stop control
            gen3.stop_low_level_control()
            
            # Prepare reference data for comparison
            num = len(T_opt)
            ee_pos_opt = np.zeros((num, 3))
            ee_vel_opt = np.zeros((num, 3))
            for k in range(num):
                qk = Z_opt[:7, k]
                dqk = Z_opt[7:, k]
                ee_pos_opt[k] = gen3.pos_fun(qk).full().flatten()
                jac_pos = gen3.jac_fun(qk)[0:3, :]
                ee_vel_opt[k] = (jac_pos @ dqk).full().flatten()
            
            # Convert logged data to numpy arrays
            times = np.array(gen3.times)
            q_pos = np.array(gen3.q_pos)
            q_vel = np.array(gen3.q_vel)
            ee_pos = np.array(gen3.ee_pos)
            ee_vel = np.array(gen3.ee_vel)
            tau_log = np.array(gen3.tau_log)
            tau_measured = np.array(gen3.tau_measured)
            
            # Package results
            results = {
                'T_horizon': T_horizon,
                'N': N,

                'p_f_des': p_f,
                'v_f_des': v_f,
                'p_f_opt': ee_pos_opt[-1],
                'v_f_opt': ee_vel_opt[-1],
                'impact_time': gen3.impact_time,
                'impact_pos': gen3.impact_pos,
                'impact_vel': gen3.impact_vel,
                'traj_duration': times[-1] - times[0] if len(times) > 0 else 0,
                
                # Optimal trajectory data
                'T_opt': T_opt,
                'Z_opt': Z_opt,
                'U_opt': U_opt,
                'ee_pos_opt': ee_pos_opt,
                'ee_vel_opt': ee_vel_opt,
                
                # Actual execution data
                'times': times,
                'q_pos': q_pos,
                'q_vel': q_vel,
                'ee_pos': ee_pos,
                'ee_vel': ee_vel,
                'tau_log': tau_log,
                'tau_measured': tau_measured,
                'tau_friction': np.array(gen3.tau_friction) if gen3.tau_friction else np.array([]),
            }

            # Save trajectory as .npy file
            npy_path = os.path.join(save_dir, f'traj_{run_id}.npy')
            np.save(npy_path, traj)
            results['npy_path'] = npy_path

            print(f"Run {run_id} completed successfully")
            return results
            
        else:
            print("Failed to initialize control")
            return None
            
    except Exception as e:
        print(f"Error in run {run_id}: {e}")
        traceback.print_exc()
        try:
            gen3.stop_low_level_control()
        except:
            pass
        return None

def main():
    """Main data collection function"""
    # Setup arguments
    args = TCPArguments()

    # Check if user wants to rerun a single trajectory
    print("\n" + "="*60)
    print("KINOVA GEN3 DATA COLLECTION")
    print("="*60)
    print("1. Resume existing collection (r)")
    print("2. Start new collection (n)")
    print("3. Rerun single trajectory (o)")
    
    mode_choice = input("Choose mode [r/n/o]: ").lower().strip()
    
    if mode_choice == 'o':
        # Single trajectory rerun mode
        rerun_info = rerun_single_trajectory()
        if rerun_info is None:
            print("Failed to setup single trajectory rerun.")
            return
        
        # Extract rerun information
        save_dir = rerun_info['session_dir']
        hdf5_path = rerun_info['hdf5_path']
        target_run_id = rerun_info['target_run_id']
        existing_params = rerun_info['params']
        
        # Extract parameters
        n_px = int(existing_params['n_px'])
        n_py = int(existing_params['n_py'])
        n_vz = int(existing_params['n_vz'])
        l_px = float(existing_params['l_px'])
        r_px = float(existing_params['r_px'])
        l_py = float(existing_params['l_py'])
        r_py = float(existing_params['r_py'])
        z_impact = float(existing_params['z_impact'])
        l_vz = float(existing_params['l_vz'])
        r_vz = float(existing_params['r_vz'])
        
        # Generate samples to get the specific target
        samples = sample_pf_vf_grid(n_px, n_py, n_vz,
                                     l_px=l_px, r_px=r_px,
                                     l_py=l_py, r_py=r_py,
                                     z_impact=z_impact,
                                     l_vz=l_vz, r_vz=r_vz)
        
        if target_run_id >= len(samples):
            print(f"Target run ID {target_run_id} exceeds available samples {len(samples)}")
            return
        
        target_p_f, target_v_f = samples[target_run_id]
        print(f"\nTarget trajectory: Run {target_run_id}")
        print(f"p_f: {target_p_f}")
        print(f"v_f: {target_v_f}")
        
        # Connect to robot and execute single trajectory
        with kortex_utils.DeviceConnection.createTcpConnection(args) as router, \
             kortex_utils.DeviceConnection.createUdpConnection(args) as router_real_time:
            
            # Create controller
            gen3 = hw_ctrl.Kinova3HardwareController(
                router, router_real_time, 
                home_pose="Mujoco_Home",
                use_friction_compensation=False
            )
            
            # Move to home position initially
            print("Moving to home position...")
            finished = gen3.move_to_home_position()
            if gen3.action_aborted or not finished:
                print("Failed to move to home position. Exiting.")
                return
            
            # Reset controller state
            gen3.kill_the_thread = False
            gen3.already_stopped = False
            gen3.cyclic_running = False
            gen3.action_aborted = False
            
            # Clear previous data
            gen3.times = []
            gen3.q_pos = []
            gen3.q_vel = []
            gen3.ee_pos = []
            gen3.ee_vel = []
            gen3.tau_log = []
            gen3.tau_measured = []
            gen3.tau_friction = []
            
            # Collect the specific trajectory
            n_digits = len(str(len(samples)))
            run_id_str = str(target_run_id).zfill(n_digits)
            
            print(f"\n{'='*50}")
            print(f"Rerunning trajectory {target_run_id}")
            print(f"{'='*50}")
            
            results = collect_single_trajectory(gen3, target_p_f, target_v_f, run_id_str, save_dir)
            
            if results is not None:
                # Save to HDF5 (overwrite existing if present)
                with h5py.File(hdf5_path, 'a') as f:
                    run_key = f'run_{run_id_str}'
                    
                    # Remove existing run if it exists
                    if run_key in f:
                        del f[run_key]
                        print(f"Removed existing {run_key}")
                    
                    # Save new data
                    grp = f.create_group(run_key)
                    for key, value in results.items():
                        if value is not None:
                            grp.create_dataset(key, data=value)
                
                print(f"✅ Successfully reran and saved run {target_run_id}")
                print(f"Results saved to: {hdf5_path}")
            else:
                print(f"❌ Failed to rerun trajectory {target_run_id}")
        
        return
    
    # Check for resume
    resume_dir, resume_hdf5 = check_for_resume()

    if resume_dir and resume_hdf5:
        # Resume mode
        print(f"\n🔄 Resuming data collection from: {resume_dir}")
        
        # Load existing parameters
        existing_params = load_existing_parameters(resume_hdf5)
        if existing_params is None:
            print("Failed to load existing parameters. Starting new session.")
            resume_dir, resume_hdf5 = None, None
        else:
            # Extract parameters
            n_px = int(existing_params['n_px'])
            n_py = int(existing_params['n_py'])
            n_vz = int(existing_params['n_vz'])
            l_px = float(existing_params['l_px'])
            r_px = float(existing_params['r_px'])
            l_py = float(existing_params['l_py'])
            r_py = float(existing_params['r_py'])
            z_impact = float(existing_params['z_impact'])
            l_vz = float(existing_params['l_vz'])
            r_vz = float(existing_params['r_vz'])
            date_str = existing_params.get('collection_date', datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
            
            # Find last completed run
            last_completed = find_last_completed_run(resume_hdf5)
            print(f"Last completed run: {last_completed}")
            
            save_dir = resume_dir
            hdf5_path = resume_hdf5

    if not resume_dir:
        # New session mode
        print("\n🆕 Starting new data collection session")
        
        # Sampling parameters
        n_px = 10  # Start with smaller grid for hardware testing
        n_py = 10
        n_vz = 10

        l_px, r_px = 0.5, 0.6    # x position range
        l_py, r_py = -0.05, 0.05   # y position range
        z_impact = 0.1
        l_vz, r_vz = -0.5, -0.1    # z velocity range (negative for downward motion)
        
        last_completed = -1
        
        # Output directory setup
        date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_dir = os.path.join(os.path.dirname(__file__), '..', 'results', f'hardware_data_{date_str}')
        os.makedirs(save_dir, exist_ok=True)
        hdf5_path = os.path.join(save_dir, 'hardware_collected_dataset_gen3.hdf5')

    # Generate samples
    samples = sample_pf_vf_grid(n_px, n_py, n_vz,
                                 l_px=l_px, r_px=r_px,
                                 l_py=l_py, r_py=r_py,
                                 z_impact=z_impact,
                                 l_vz=l_vz, r_vz=r_vz)

    start_idx = last_completed + 1
    remaining_samples = samples[start_idx:]

    print(f"Total trajectory conditions: {len(samples)}")
    print(f"Starting from run: {start_idx}")
    print(f"Remaining to collect: {len(remaining_samples)}")
    print(f"Results will be saved to: {save_dir}")
    
    # Connect to robot and execute
    with kortex_utils.DeviceConnection.createTcpConnection(args) as router, \
         kortex_utils.DeviceConnection.createUdpConnection(args) as router_real_time:
        
        # Create controller
        gen3 = hw_ctrl.Kinova3HardwareController(
            router, router_real_time, 
            home_pose="Mujoco_Home",
            use_friction_compensation=False
        )
        
        # Move to home position initially
        print("Moving to home position...")
        finished = gen3.move_to_home_position()
        if gen3.action_aborted or not finished:
            print("Failed to move to home position. Exiting.")
            return
        
        # HDF5 file setup - CRITICAL FIX: Use correct mode based on resume status
        file_mode = 'a' if resume_dir else 'w'  # Append if resuming, write if new
        print(f"Opening HDF5 file in {'append' if resume_dir else 'write'} mode")
        
        with h5py.File(hdf5_path, file_mode) as f:
            # Save parameters (only for new sessions)
            if not resume_dir:
                print("Creating new parameters group...")
                params = {
                    'n_px': n_px,
                    'n_py': n_py,
                    'n_vz': n_vz,
                    'l_px': l_px,
                    'r_px': r_px,
                    'l_py': l_py,
                    'r_py': r_py,
                    'z_impact': z_impact,
                    'l_vz': l_vz,
                    'r_vz': r_vz,
                    'total_samples': len(samples),
                    'q_lower': gen3.q_lower,
                    'q_upper': gen3.q_upper,
                    'dq_lower': gen3.dq_lower,
                    'dq_upper': gen3.dq_upper,
                    'tau_lower': gen3.tau_lower,
                    'tau_upper': gen3.tau_upper,
                    'collection_date': date_str,
                    'samples': samples
                }
                
                params_grp = f.create_group('parameters')
                for key, value in params.items():
                    params_grp.create_dataset(key, data=value)
            else:
                print("Using existing parameters from resumed session")

            # Load existing successful runs count
            if resume_dir and 'summary' in f:
                successful_runs = int(f['summary']['successful_runs'][()])
                print(f"Resuming with {successful_runs} previously successful runs")
            else:
                successful_runs = 0
            
            # Padding for run IDs
            n_digits = len(str(len(samples)))

            # Collect data for each remaining sample
            for local_i, (p_f, v_f) in enumerate(remaining_samples):
                actual_i = start_idx + local_i
                run_id = str(actual_i).zfill(n_digits)
                
                print(f"\n{'='*50}")
                print(f"Running trajectory {actual_i+1}/{len(samples)}")
                print(f"Progress: {successful_runs}/{len(remaining_samples)} successful")  # Fixed denominator
                print(f"{'='*50}")

                # Reset controller state between trajectories
                gen3.kill_the_thread = False
                gen3.already_stopped = False
                gen3.cyclic_running = False
                gen3.action_aborted = False
                
                # Clear previous data
                gen3.times = []
                gen3.q_pos = []
                gen3.q_vel = []
                gen3.ee_pos = []
                gen3.ee_vel = []
                gen3.tau_log = []
                gen3.tau_measured = []
                gen3.tau_friction = []
                
                # Retry mechanism for current trajectory
                retry_current = True
                while retry_current:
                    # Collect single trajectory
                    results = collect_single_trajectory(gen3, p_f, v_f, run_id, save_dir)
                    
                    if results is not None:
                        # Save to HDF5
                        grp = f.create_group(f'run_{run_id}')
                        for key, value in results.items():
                            if value is not None:
                                grp.create_dataset(key, data=value)
                        successful_runs += 1
                        
                        print(f"✅ Successfully saved run {run_id}")
                        retry_current = False
                        
                        # Optional: Plot results for this run
                        # gen3.plot_results(results['T_opt'], results['Z_opt'], 
                        #                   results['ee_pos_opt'], results['ee_vel_opt'])
                    else:
                        # Handle failed run
                        action = ask_user_action(actual_i+1, len(samples))
                        
                        if action == 'c':  # Continue to next
                            print(f"⏭️ Skipping run {run_id} and continuing...")
                            retry_current = False
                        elif action == 'r':  # Retry current
                            print(f"🔄 Retrying run {run_id}...")
                            # Clear data and retry
                            gen3.times = []
                            gen3.q_pos = []
                            gen3.q_vel = []
                            gen3.ee_pos = []
                            gen3.ee_vel = []
                            gen3.tau_log = []
                            gen3.tau_measured = []
                            gen3.tau_friction = []
                            continue
                        elif action == 'o':  # Rerun specific and exit
                            print("🎯 Switching to single trajectory rerun mode...")
                            # Update summary before exiting
                            if 'summary' in f:
                                del f['summary']
                            summary_grp = f.create_group('summary')
                            summary_grp.create_dataset('total_attempted', data=actual_i)
                            summary_grp.create_dataset('successful_runs', data=successful_runs)
                            summary_grp.create_dataset('success_rate', data=successful_runs/actual_i if actual_i > 0 else 0)
                            
                            print("Current session saved. Please restart the script and choose option 'o' for single trajectory rerun.")
                            return
                        elif action == 'e':  # Exit without saving resume
                            print("❌ Exiting data collection...")
                            return
                        elif action == 's':  # Exit and save resume
                            print("💾 Saving progress and exiting...")
                            create_resume_file(save_dir, actual_i-1, len(samples))
                            # Update summary before exiting
                            if 'summary' in f:
                                del f['summary']
                            summary_grp = f.create_group('summary')
                            summary_grp.create_dataset('total_attempted', data=actual_i)
                            summary_grp.create_dataset('successful_runs', data=successful_runs)
                            summary_grp.create_dataset('success_rate', data=successful_runs/actual_i if actual_i > 0 else 0)
                            return
                
                # Safety pause between runs
                time.sleep(1.0)
            
            # Save final summary
            if 'summary' in f:
                del f['summary']
            summary_grp = f.create_group('summary')
            summary_grp.create_dataset('total_attempted', data=len(samples))
            summary_grp.create_dataset('successful_runs', data=successful_runs)
            summary_grp.create_dataset('success_rate', data=successful_runs/len(samples))  # Fixed denominator
    
    print(f"\n🎉 Data collection complete!")
    print(f"Successful runs: {successful_runs}/{len(samples)}")  # Fixed denominator
    print(f"Success rate: {successful_runs/len(samples)*100:.1f}%")  # Fixed denominator
    print(f"Results saved to: {hdf5_path}")
    
    # Remove resume file if exists (collection completed)
    resume_file = os.path.join(save_dir, 'resume_info.txt')
    if os.path.exists(resume_file):
        os.remove(resume_file)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nData collection interrupted by user")
    except Exception as e:
        print(f"Error in main: {e}")
        traceback.print_exc()
