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

import robosuite.projects.shared_scripts.optimizing_gen3_arm as opt
import robosuite.projects.impact_control.gen3_contact_ctrl_on_hardware as hw_ctrl
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

#################################################
# BRS VALIDATION MODULE
#################################################

def validate_brs_samples():
    """
    Validate BRS samples by running experiments for each validation sample
    and recording the results.
    """
    print("\n" + "="*60)
    print("🔍 BRS SAMPLE VALIDATION")
    print("="*60)
    
    # Setup arguments for connection
    args = TCPArguments()
    
    # First, select the BRS sample file
    brs_file = select_brs_sample_file()
    if not brs_file:
        print("No BRS sample file selected. Exiting validation.")
        return
    
    # Load BRS sample data
    samples, goal_set = load_brs_samples(brs_file)
    if not samples or len(samples) == 0:
        print("Failed to load samples from file or no samples found.")
        return
    
    # Create output directory for validation results
    date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    validation_dir = os.path.join(os.path.dirname(__file__), '..', 'results', f'brs_validation_{date_str}')
    os.makedirs(validation_dir, exist_ok=True)
    
    # Create HDF5 file for validation results
    validation_hdf5 = os.path.join(validation_dir, 'brs_validation_results.hdf5')
    
    print(f"Found {len(samples)} validation samples in BRS file")
    print(f"Results will be saved to: {validation_dir}")
    
    # Connect to robot and execute validation
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
        
        # Initialize validation results file
        with h5py.File(validation_hdf5, 'w') as f:
            # Save metadata
            meta = f.create_group('metadata')
            meta.create_dataset('validation_date', data=date_str)
            meta.create_dataset('brs_file', data=os.path.basename(brs_file))
            meta.create_dataset('total_samples', data=len(samples))
            
            # Save goal set if available
            if len(goal_set) > 0:
                goal_grp = f.create_group('goal_set')
                goal_grp.create_dataset('bounds', data=goal_set)
                print(f"Saved goal set bounds to validation file")
            
            # Create parameters group
            params = f.create_group('parameters')
            params.create_dataset('q_lower', data=gen3.q_lower)
            params.create_dataset('q_upper', data=gen3.q_upper)
            params.create_dataset('dq_lower', data=gen3.dq_lower)
            params.create_dataset('dq_upper', data=gen3.dq_upper)
            params.create_dataset('tau_lower', data=gen3.tau_lower)
            params.create_dataset('tau_upper', data=gen3.tau_upper)
            
            # Track success metrics
            successful_runs = 0
            pos_achieved_count = 0
            vel_achieved_count = 0
            
            # Run validation for each sample
            for i, (p_f_des, v_f_des) in enumerate(samples):
                p_f_des = np.array(p_f_des).flatten()
                v_f_des = np.array(v_f_des).flatten()
                print(f"\n{'='*50}")
                print(f"Validating Sample {i+1}/{len(samples)}")
                print(f"p_f_des: {p_f_des}")
                print(f"v_f_des: {v_f_des}")
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
                
                # Format run ID with proper padding
                n_digits = len(str(len(samples)))
                run_id = str(i).zfill(n_digits)
                
                # Run validation for this sample
                results = collect_single_trajectory(gen3, p_f_des, v_f_des, run_id, validation_dir)
                
                if results is not None:
                    # Extract actual impact values
                    p_f_act = results['impact_pos']
                    v_f_act = results['impact_vel']
                    
                    # Calculate error metrics
                    pos_error = np.linalg.norm(p_f_act - p_f_des) if p_f_act is not None else None
                    vel_error = np.linalg.norm(v_f_act - v_f_des) if v_f_act is not None else None
                    
                    # Check goal set achievement if goal_set is available
                    pos_achieved = False
                    vel_achieved = False
                    
                    if len(goal_set) > 0 and p_f_act is not None and v_f_act is not None:
                        # Check if it's in position goal_set[0,:2] <= p_f_act[:2] <= goal_set[1,:2]
                        pos_achieved = np.all(goal_set[0,:2] <= p_f_act[:2]) and np.all(p_f_act[:2] <= goal_set[1,:2])
                        if pos_achieved:
                            print(f"✅ Position goal achieved for sample {i+1}")
                            pos_achieved_count += 1
                        else:
                            print(f"❌ Position goal NOT achieved for sample {i+1}")
                            print(f"   Target: {p_f_act[:2]}")
                            print(f"   Bounds: [{goal_set[0,:2]}, {goal_set[1,:2]}]")

                        vel_achieved = np.all(goal_set[0,2] <= v_f_act[2]) and np.all(v_f_act[2] <= goal_set[1,2])
                        if vel_achieved:
                            print(f"✅ Velocity goal achieved for sample {i+1}")
                            vel_achieved_count += 1
                        else:
                            print(f"❌ Velocity goal NOT achieved for sample {i+1}")
                            print(f"   Target: {v_f_act[2]}")
                            print(f"   Bounds: [{goal_set[0,2]}, {goal_set[1,2]}]")
                    else:
                        # Fallback to error-based success criteria
                        pos_threshold = 0.02  # 2cm position error threshold
                        vel_threshold = 0.05  # 0.05 m/s velocity error threshold
                        
                        pos_achieved = pos_error is not None and pos_error < pos_threshold
                        vel_achieved = vel_error is not None and vel_error < vel_threshold
                        
                        if pos_achieved:
                            pos_achieved_count += 1
                        if vel_achieved:
                            vel_achieved_count += 1
                    
                    # Overall validation success
                    validation_success = pos_achieved and vel_achieved
                    if validation_success:
                        successful_runs += 1
                        print(f"✅ Overall validation SUCCESS for sample {i+1}")
                    else:
                        print(f"❌ Overall validation FAILED for sample {i+1}")
                    
                    # Add validation-specific metrics to results
                    validation_results = {
                        'pos_error': pos_error,
                        'vel_error': vel_error,
                        'pos_achieved': pos_achieved,
                        'vel_achieved': vel_achieved,
                        'validation_success': validation_success,
                    }
                    
                    # Save all data to HDF5 (both trajectory data and validation results)
                    validation_grp = f.create_group(f'validation_{run_id}')
                    
                    # Save all trajectory results from collect_single_trajectory
                    for key, value in results.items():
                        if value is not None:
                            validation_grp.create_dataset(key, data=value)
                    
                    # Save validation-specific results
                    for key, value in validation_results.items():
                        if value is not None:
                            validation_grp.create_dataset(key, data=value)
                    
                    print(f"💾 Saved complete validation data for sample {i+1}")
                    
                    if pos_error is not None and vel_error is not None:
                        print(f"   Position error: {pos_error:.4f} m")
                        print(f"   Velocity error: {vel_error:.4f} m/s")
                
                else:
                    print(f"❌ Execution failed for sample {i+1}")
                    
                    # Save failure record with minimal info
                    validation_grp = f.create_group(f'validation_{run_id}')
                    validation_grp.create_dataset('p_f_des', data=p_f_des)
                    validation_grp.create_dataset('v_f_des', data=v_f_des)
                    validation_grp.create_dataset('validation_success', data=False)
                    validation_grp.create_dataset('pos_achieved', data=False)
                    validation_grp.create_dataset('vel_achieved', data=False)
                
                # Ask if user wants to continue after each sample
                # if i < len(samples) - 1:  # If not the last sample
                #     try:
                #         choice = input("\nContinue to next sample? [y/n]: ").lower().strip()
                #         if choice != 'y':
                #             print("Validation interrupted by user.")
                #             break
                #     except KeyboardInterrupt:
                #         print("\nValidation interrupted by keyboard interrupt.")
                #         break
                
                # Safety pause between runs
                time.sleep(1.0)
            
            # Save final summary
            summary = f.create_group('summary')
            summary.create_dataset('total_samples', data=len(samples))
            summary.create_dataset('completed_samples', data=i+1)
            summary.create_dataset('successful_runs', data=successful_runs)
            summary.create_dataset('pos_achieved_count', data=pos_achieved_count)
            summary.create_dataset('vel_achieved_count', data=vel_achieved_count)
            summary.create_dataset('success_rate', data=successful_runs/(i+1) if i >= 0 else 0)
            summary.create_dataset('pos_achievement_rate', data=pos_achieved_count/(i+1) if i >= 0 else 0)
            summary.create_dataset('vel_achievement_rate', data=vel_achieved_count/(i+1) if i >= 0 else 0)
    
    # Generate validation report
    generate_validation_report(validation_hdf5, validation_dir)
    
    print(f"\n🎉 BRS Validation complete!")
    print(f"Successful validations: {successful_runs}/{i+1}")
    print(f"Position goals achieved: {pos_achieved_count}/{i+1}")
    print(f"Velocity goals achieved: {vel_achieved_count}/{i+1}")
    print(f"Overall success rate: {successful_runs/(i+1)*100:.1f}%" if i >= 0 else "No samples processed")
    print(f"Results saved to: {validation_hdf5}")
    print(f"Report saved to: {os.path.join(validation_dir, 'validation_report.md')}")


def select_brs_sample_file():
    """Select a BRS sample file for validation - simplified version"""
    print("\nSelecting BRS sample file for validation...")
    
    # Look for hardware data collection sessions
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    if not os.path.exists(results_dir):
        print("No results directory found.")
        return None
    
    # Find hardware data directories
    hw_dirs = [d for d in os.listdir(results_dir) if d.startswith('hardware_data_')]
    if not hw_dirs:
        print("No hardware data sessions found.")
        return None
    
    hw_dirs.sort(reverse=True)  # Most recent first
    
    print("\nAvailable hardware data sessions:")
    for i, dirname in enumerate(hw_dirs):
        dir_path = os.path.join(results_dir, dirname)
        brs_file = os.path.join(dir_path, 'brs_samples.h5')
        status = "✅ has brs_samples.h5" if os.path.exists(brs_file) else "❌ no brs_samples.h5"
        print(f"{i+1}. {dirname} - {status}")
    
    try:
        choice = int(input(f"\nSelect session (1-{len(hw_dirs)}): ")) - 1
        if 0 <= choice < len(hw_dirs):
            selected_dir = os.path.join(results_dir, hw_dirs[choice])
            brs_file = os.path.join(selected_dir, 'brs_samples.h5')
            
            if os.path.exists(brs_file):
                print(f"Selected: {brs_file}")
                return brs_file
            else:
                print("No brs_samples.h5 found in selected session.")
                return None
        else:
            print("Invalid selection.")
            return None
    except ValueError:
        print("Invalid input. Please enter a number.")
        return None

def load_brs_samples(brs_file):
    """Load BRS samples from HDF5 file"""
    try:
        samples = []
        goal_set = []
        with h5py.File(brs_file, 'r') as f:
            # Check if file has the expected structure
            if 'goal_set' in f:
                lb = f['goal_set']['lb'][()]
                ub = f['goal_set']['ub'][()]
                goal_set = np.array([lb, ub]).squeeze()
                print(f"Loaded goal set with bounds: \n{goal_set}")
            else:
                print("No goal set found.")

            validation_groups = [k for k in f.keys() if k.startswith('validation_')]

            if validation_groups:
                # New format with validation_XXX groups
                print(f"Loading {len(validation_groups)} samples from validation groups")
                for group_name in validation_groups:
                    group = f[group_name]
                    if 'p_f_des' in group and 'v_f_des' in group:
                        p_f_des = group['p_f_des'][()]
                        v_f_des = group['v_f_des'][()]
                        samples.append((p_f_des, v_f_des))
            elif 'all_samples' in f:
                # Old format with single all_samples dataset
                print("Loading samples from 'all_samples' dataset")
                all_samples = f['all_samples'][()]
                for i in range(len(all_samples)):
                    # Convert samples into the required format
                    p_f_des = np.array([all_samples[i,0], all_samples[i,1], 0.1])
                    v_f_des = np.array([0, 0, all_samples[i,2]])
                    samples.append((p_f_des, v_f_des))
            else:
                # Try to find any groups that might contain samples
                for group_name in f.keys():
                    if isinstance(f[group_name], h5py.Group):
                        group = f[group_name]
                        if 'state' in group:
                            sample = group['state'][()]
                            p_f_des = np.array([sample[0], sample[1], 0.1]).flatten()
                            v_f_des = np.array([0, 0, sample[2]]).flatten()
                            samples.append((p_f_des, v_f_des))
            
            # Limit number of samples if needed
            max_samples = 1000  # Maximum number of samples to validate
            if len(samples) > max_samples:
                print(f"Found {len(samples)} samples, limiting to first {max_samples} for validation")
                samples = samples[:max_samples]
            
            return samples, goal_set
    except Exception as e:
        print(f"Error loading BRS samples: {e}")
        traceback.print_exc()
        return None

def generate_validation_report(validation_hdf5, validation_dir):
    """Generate a validation report in Markdown format"""
    try:
        report_path = os.path.join(validation_dir, 'validation_report.md')
        
        with h5py.File(validation_hdf5, 'r') as f:
            # Extract summary data
            total_samples = f['summary/total_samples'][()]
            completed_samples = f['summary/completed_samples'][()]
            successful_runs = f['summary/successful_runs'][()]
            success_rate = f['summary/success_rate'][()]
            
            # Extract validation data
            validation_results = []
            for i in range(completed_samples):
                run_id = str(i).zfill(len(str(total_samples)))
                group_name = f'validation_{run_id}'
                
                if group_name in f:
                    group = f[group_name]
                    result = {
                        'sample_id': i,
                        'p_f_des': group['p_f_des'][()] if 'p_f_des' in group else None,
                        'v_f_des': group['v_f_des'][()] if 'v_f_des' in group else None,
                        'impact_pos': group['impact_pos'][()] if 'impact_pos' in group else None,
                        'impact_vel': group['impact_vel'][()] if 'impact_vel' in group else None,
                        'pos_error': group['pos_error'][()] if 'pos_error' in group else None,
                        'vel_error': group['vel_error'][()] if 'vel_error' in group else None,
                        'success': group['validation_success'][()] if 'validation_success' in group else False
                    }
                    validation_results.append(result)
        
        # Generate Markdown report
        with open(report_path, 'w') as report:
            # Title and summary
            date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            report.write(f"# BRS Validation Report\n\n")
            report.write(f"**Generated:** {date_str}\n\n")
            
            # Summary section
            report.write("## Summary\n\n")
            report.write(f"- **Total Samples:** {total_samples}\n")
            report.write(f"- **Completed Samples:** {completed_samples}\n")
            report.write(f"- **Successful Validations:** {successful_runs}\n")
            report.write(f"- **Success Rate:** {success_rate*100:.1f}%\n\n")
            
            # Results table
            report.write("## Validation Results\n\n")
            report.write("| Sample | Target Position | Target Velocity | Actual Position | Actual Velocity | Position Error | Velocity Error | Success |\n")
            report.write("|--------|----------------|----------------|----------------|----------------|---------------|---------------|--------|\n")
            
            for result in validation_results:
                p_f_des = result['p_f_des']
                v_f_des = result['v_f_des']
                impact_pos = result['impact_pos']
                impact_vel = result['impact_vel']
                pos_error = result['pos_error']
                vel_error = result['vel_error']
                success = result['success']
                
                p_f_des_str = f"[{p_f_des[0]:.3f}, {p_f_des[1]:.3f}, {p_f_des[2]:.3f}]" if p_f_des is not None else "N/A"
                v_f_des_str = f"[{v_f_des[0]:.3f}, {v_f_des[1]:.3f}, {v_f_des[2]:.3f}]" if v_f_des is not None else "N/A"
                impact_pos_str = f"[{impact_pos[0]:.3f}, {impact_pos[1]:.3f}, {impact_pos[2]:.3f}]" if impact_pos is not None else "N/A"
                impact_vel_str = f"[{impact_vel[0]:.3f}, {impact_vel[1]:.3f}, {impact_vel[2]:.3f}]" if impact_vel is not None else "N/A"
                pos_error_str = f"{pos_error:.3f} m" if pos_error is not None else "N/A"
                vel_error_str = f"{vel_error:.3f} m/s" if vel_error is not None else "N/A"
                success_str = "✅" if success else "❌"
                
                report.write(f"| {result['sample_id']} | {p_f_des_str} | {v_f_des_str} | {impact_pos_str} | {impact_vel_str} | {pos_error_str} | {vel_error_str} | {success_str} |\n")
            
            # Error Analysis
            report.write("\n## Error Analysis\n\n")
            
            # Calculate statistics for position and velocity errors
            pos_errors = [r['pos_error'] for r in validation_results if r['pos_error'] is not None]
            vel_errors = [r['vel_error'] for r in validation_results if r['vel_error'] is not None]
            
            if pos_errors:
                avg_pos_error = sum(pos_errors) / len(pos_errors)
                max_pos_error = max(pos_errors)
                min_pos_error = min(pos_errors)
                report.write(f"### Position Error Statistics\n\n")
                report.write(f"- **Average Position Error:** {avg_pos_error:.3f} m\n")
                report.write(f"- **Maximum Position Error:** {max_pos_error:.3f} m\n")
                report.write(f"- **Minimum Position Error:** {min_pos_error:.3f} m\n\n")
            
            if vel_errors:
                avg_vel_error = sum(vel_errors) / len(vel_errors)
                max_vel_error = max(vel_errors)
                min_vel_error = min(vel_errors)
                report.write(f"### Velocity Error Statistics\n\n")
                report.write(f"- **Average Velocity Error:** {avg_vel_error:.3f} m/s\n")
                report.write(f"- **Maximum Velocity Error:** {max_vel_error:.3f} m/s\n")
                report.write(f"- **Minimum Velocity Error:** {min_vel_error:.3f} m/s\n\n")
            
            # Conclusion
            report.write("## Conclusion\n\n")
            if successful_runs / completed_samples > 0.8:
                report.write("The BRS validation shows excellent accuracy, with most samples executing successfully. ")
                report.write("The controller achieves reliable tracking of target position and velocity at impact.\n\n")
            elif successful_runs / completed_samples > 0.5:
                report.write("The BRS validation shows good performance, with the majority of samples executing successfully. ")
                report.write("Some improvements may be needed to enhance the controller's accuracy.\n\n")
            else:
                report.write("The BRS validation indicates performance issues, with a significant number of samples failing. ")
                report.write("Further investigation and controller tuning are recommended.\n\n")

        print(f"Validation report generated: {report_path}")
        
    except Exception as e:
        print(f"Error generating validation report: {e}")
        traceback.print_exc()

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
    print("4. Run BRS validation (v)")
    
    mode_choice = input("Choose mode [r/n/o/v]: ").lower().strip()
    
    if mode_choice == 'v':
        # Call validation module
        validate_brs_samples()
        return
    
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