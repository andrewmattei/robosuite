import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import impact_ctrl_gen3_hardware_data_collection_pipeline_pf_vf as data_collection

def plot_hardware_impact_errors(hdf5_path):
    """
    Load HDF5 file and plot position/velocity errors between desired and hardware-measured 
    values for the Gen3 robot (3D data).
    Handle missing fields by marking them as NaN.
    """
    with h5py.File(hdf5_path, 'r') as f:
        run_keys = [key for key in f.keys() if key.startswith('run_')]
        n_runs = len(run_keys)
        
        # Initialize arrays with NaN for 3D data
        px_errors = np.full(n_runs, np.nan)
        py_errors = np.full(n_runs, np.nan)
        pz_errors = np.full(n_runs, np.nan)
        vx_errors = np.full(n_runs, np.nan)
        vy_errors = np.full(n_runs, np.nan)
        vz_errors = np.full(n_runs, np.nan)
        
        # Hardware-specific metrics
        impact_times = np.full(n_runs, np.nan)
        traj_durations = np.full(n_runs, np.nan)
        
        # Track failed runs
        failed_runs = []
        
        # Collect errors for each run
        for i, run_key in enumerate(run_keys):
            run_data = f[run_key]
            try:
                # Get desired values
                p_f_des = run_data['p_f_des'][:]
                v_f_des = run_data['v_f_des'][:]
                
                # Get hardware-measured impact values
                if 'impact_pos' in run_data and 'impact_vel' in run_data:
                    p_f_hw = run_data['impact_pos'][:]
                    v_f_hw = run_data['impact_vel'][:]
                else:
                    # Fall back to optimal trajectory final values if impact not detected
                    p_f_hw = run_data['p_f_opt'][:]
                    v_f_hw = run_data['v_f_opt'][:]
                
                # Calculate errors
                px_errors[i] = p_f_hw[0] - p_f_des[0]
                py_errors[i] = p_f_hw[1] - p_f_des[1]
                pz_errors[i] = p_f_hw[2] - p_f_des[2]
                vx_errors[i] = v_f_hw[0] - v_f_des[0]
                vy_errors[i] = v_f_hw[1] - v_f_des[1]
                vz_errors[i] = v_f_hw[2] - v_f_des[2]
                
                # Hardware-specific data
                if 'impact_time' in run_data:
                    impact_times[i] = run_data['impact_time'][()]
                if 'traj_duration' in run_data:
                    traj_durations[i] = run_data['traj_duration'][()]
                    
            except KeyError as e:
                failed_runs.append((run_key, str(e)))
                continue
    
    # Create figure with 8 subplots (6 errors + 2 hardware metrics)
    fig, axes = plt.subplots(8, 1, figsize=(12, 20))
    run_indices = np.arange(n_runs)
    
    # Plot each error type
    error_data = [px_errors, py_errors, pz_errors, vx_errors, vy_errors, vz_errors]
    error_titles = ['Position X Error', 'Position Y Error', 'Position Z Error',
                   'Velocity X Error', 'Velocity Y Error', 'Velocity Z Error']
    error_units = ['(m)', '(m)', '(m)', '(m/s)', '(m/s)', '(m/s)']
    
    for ax, errors, title, unit in zip(axes[:6], error_data, error_titles, error_units):
        # Plot successful runs
        valid_mask = ~np.isnan(errors)
        ax.plot(run_indices[valid_mask], errors[valid_mask], 'b.-', label='Successful')
        
        # Mark failed runs
        failed_mask = np.isnan(errors)
        if np.any(failed_mask):
            ax.plot(run_indices[failed_mask], np.zeros_like(run_indices[failed_mask]), 
                   'rx', label='Failed', markersize=15)
        
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax.set_title(f'Kinova Gen3 Hardware {title} (hw - des)')
        ax.set_ylabel(f'Error {unit}')
        ax.grid(True, alpha=0.3)
        
        if np.any(valid_mask):
            mean_err = np.mean(errors[valid_mask])
            std_err = np.std(errors[valid_mask])
            ax.text(0.02, 0.95, f'Mean: {mean_err:.4f}\nStd: {std_err:.4f}', 
                   transform=ax.transAxes, fontsize=8, 
                   bbox=dict(facecolor='white', alpha=0.8))
        ax.legend()
    
    # Hardware-specific plots
    # Impact timing
    ax = axes[6]
    valid_mask = ~np.isnan(impact_times)
    ax.plot(run_indices[valid_mask], impact_times[valid_mask], 'g.-', label='Impact Time')
    ax.set_title('Impact Detection Time')
    ax.set_ylabel('Time (s)')
    ax.grid(True, alpha=0.3)
    if np.any(valid_mask):
        mean_time = np.mean(impact_times[valid_mask])
        std_time = np.std(impact_times[valid_mask])
        ax.text(0.02, 0.95, f'Mean: {mean_time:.3f}s\nStd: {std_time:.3f}s', 
               transform=ax.transAxes, fontsize=8, 
               bbox=dict(facecolor='white', alpha=0.8))
    ax.legend()
    
    # Trajectory duration
    ax = axes[7]
    valid_mask = ~np.isnan(traj_durations)
    ax.plot(run_indices[valid_mask], traj_durations[valid_mask], 'm.-', label='Trajectory Duration')
    ax.set_title('Trajectory Execution Duration')
    ax.set_ylabel('Duration (s)')
    ax.set_xlabel('Run Index')
    ax.grid(True, alpha=0.3)
    if np.any(valid_mask):
        mean_dur = np.mean(traj_durations[valid_mask])
        std_dur = np.std(traj_durations[valid_mask])
        ax.text(0.02, 0.95, f'Mean: {mean_dur:.3f}s\nStd: {std_dur:.3f}s', 
               transform=ax.transAxes, fontsize=8, 
               bbox=dict(facecolor='white', alpha=0.8))
    ax.legend()
    
    # Print statistics for successful runs
    print("\nHardware Statistics (excluding failed runs):")
    for errors, name in zip(error_data, error_titles):
        valid_errors = errors[~np.isnan(errors)]
        if len(valid_errors) > 0:
            print(f"{name} - Mean: {np.mean(valid_errors):.4f}, "
                  f"Std: {np.std(valid_errors):.4f}")
    
    # Hardware timing statistics
    valid_impact_times = impact_times[~np.isnan(impact_times)]
    valid_durations = traj_durations[~np.isnan(traj_durations)]
    if len(valid_impact_times) > 0:
        print(f"Impact Time - Mean: {np.mean(valid_impact_times):.3f}s, "
              f"Std: {np.std(valid_impact_times):.3f}s")
    if len(valid_durations) > 0:
        print(f"Trajectory Duration - Mean: {np.mean(valid_durations):.3f}s, "
              f"Std: {np.std(valid_durations):.3f}s")
    
    # Print failed runs
    if failed_runs:
        print("\nFailed runs:")
        for run_key, error in failed_runs:
            print(f"{run_key}: {error}")
    
    plt.tight_layout()
    
    # Save plot
    plot_dir = os.path.dirname(hdf5_path)
    plot_path = os.path.join(plot_dir, 'hardware_gen3_impact_errors.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved hardware error plot to: {plot_path}")
    
    plt.show()

def plot_hardware_3d_errors(hdf5_path):
    """
    Create 3D scatter plots of position and velocity errors for hardware data.
    """
    with h5py.File(hdf5_path, 'r') as f:
        run_keys = [key for key in f.keys() if key.startswith('run_')]
        n_runs = len(run_keys)
        
        # Initialize arrays
        p_errors = np.full((n_runs, 3), np.nan)
        v_errors = np.full((n_runs, 3), np.nan)
        
        # Collect errors
        for i, run_key in enumerate(run_keys):
            run_data = f[run_key]
            try:
                p_f_des = run_data['p_f_des'][:]
                v_f_des = run_data['v_f_des'][:]
                
                # Use impact data if available, otherwise use optimal trajectory
                if 'impact_pos' in run_data and 'impact_vel' in run_data:
                    p_f_hw = run_data['impact_pos'][:]
                    v_f_hw = run_data['impact_vel'][:]
                else:
                    p_f_hw = run_data['p_f_opt'][:]
                    v_f_hw = run_data['v_f_opt'][:]
                
                p_errors[i] = p_f_hw - p_f_des
                v_errors[i] = v_f_hw - v_f_des
            except KeyError:
                continue

    # Create 3D plots
    fig = plt.figure(figsize=(15, 7))
    
    # Position errors
    ax1 = fig.add_subplot(121, projection='3d')
    valid_mask = ~np.isnan(p_errors).any(axis=1)
    if np.any(valid_mask):
        scatter1 = ax1.scatter(p_errors[valid_mask, 0], 
                              p_errors[valid_mask, 1], 
                              p_errors[valid_mask, 2],
                              c=np.linalg.norm(p_errors[valid_mask], axis=1),
                              cmap='viridis')
        plt.colorbar(scatter1, ax=ax1, label='Error Magnitude (m)')
    ax1.set_title('Hardware Position Errors')
    ax1.set_xlabel('X Error (m)')
    ax1.set_ylabel('Y Error (m)')
    ax1.set_zlabel('Z Error (m)')
    
    # Velocity errors
    ax2 = fig.add_subplot(122, projection='3d')
    valid_mask = ~np.isnan(v_errors).any(axis=1)
    if np.any(valid_mask):
        scatter2 = ax2.scatter(v_errors[valid_mask, 0], 
                              v_errors[valid_mask, 1], 
                              v_errors[valid_mask, 2],
                              c=np.linalg.norm(v_errors[valid_mask], axis=1),
                              cmap='viridis')
        plt.colorbar(scatter2, ax=ax2, label='Error Magnitude (m/s)')
    ax2.set_title('Hardware Velocity Errors')
    ax2.set_xlabel('X Error (m/s)')
    ax2.set_ylabel('Y Error (m/s)')
    ax2.set_zlabel('Z Error (m/s)')
    
    plt.suptitle('Gen3 Hardware 3D Error Distribution', fontsize=14)
    plt.tight_layout()
    
    # Save plot
    plot_dir = os.path.dirname(hdf5_path)
    plot_path = os.path.join(plot_dir, 'hardware_gen3_3d_errors.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved hardware 3D error plot to: {plot_path}")
    
    plt.show()

def plot_hardware_torque_analysis(hdf5_path):
    """
    Plot torque-related analysis for hardware data.
    """
    with h5py.File(hdf5_path, 'r') as f:
        run_keys = [key for key in f.keys() if key.startswith('run_')]
        
        # Select a few successful runs for detailed torque analysis
        successful_runs = []
        for run_key in run_keys[:5]:  # Look at first 5 runs
            run_data = f[run_key]
            if 'tau_log' in run_data and 'tau_measured' in run_data:
                successful_runs.append((run_key, run_data))
        
        if not successful_runs:
            print("No runs with torque data found.")
            return
        
        fig, axes = plt.subplots(len(successful_runs), 1, figsize=(12, 4*len(successful_runs)))
        if len(successful_runs) == 1:
            axes = [axes]
        
        for i, (run_key, run_data) in enumerate(successful_runs):
            try:
                times = run_data['times'][:]
                tau_log = run_data['tau_log'][:]
                tau_measured = run_data['tau_measured'][:]
                
                ax = axes[i]
                
                # Plot commanded vs measured torques for first 3 joints
                for j in range(min(3, tau_log.shape[1])):
                    ax.plot(times, tau_log[:, j], '--', label=f'Commanded Joint {j+1}', alpha=0.7)
                    ax.plot(times, tau_measured[:, j], '-', label=f'Measured Joint {j+1}')
                
                ax.set_title(f'{run_key}: Commanded vs Measured Torques')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Torque (Nm)')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
            except Exception as e:
                print(f"Error plotting torques for {run_key}: {e}")
        
        plt.tight_layout()
        
        # Save plot
        plot_dir = os.path.dirname(hdf5_path)
        plot_path = os.path.join(plot_dir, 'hardware_gen3_torque_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved torque analysis plot to: {plot_path}")
        
        plt.show()

def plot_trajectory_comparison(hdf5_path, run_index=0):
    """
    Compare planned vs executed trajectory for a specific run.
    """
    with h5py.File(hdf5_path, 'r') as f:
        run_keys = [key for key in f.keys() if key.startswith('run_')]
        
        if run_index >= len(run_keys):
            print(f"Run index {run_index} not found. Available runs: {len(run_keys)}")
            return
        
        run_key = run_keys[run_index]
        run_data = f[run_key]
        
        try:
            # Planned trajectory
            T_opt = run_data['T_opt'][:]
            ee_pos_opt = run_data['ee_pos_opt'][:]
            ee_vel_opt = run_data['ee_vel_opt'][:]
            
            # Executed trajectory
            times = run_data['times'][:]
            ee_pos = run_data['ee_pos'][:]
            ee_vel = run_data['ee_vel'][:]
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # Position comparison
            for i in range(3):
                ax = axes[0, i]
                ax.plot(T_opt, ee_pos_opt[:, i], 'b--', label='Planned', linewidth=2)
                ax.plot(times, ee_pos[:, i], 'r-', label='Executed', alpha=0.8)
                ax.set_title(f'End-Effector Position {["X", "Y", "Z"][i]}')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Position (m)')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # Velocity comparison
            for i in range(3):
                ax = axes[1, i]
                ax.plot(T_opt, ee_vel_opt[:, i], 'b--', label='Planned', linewidth=2)
                ax.plot(times, ee_vel[:, i], 'r-', label='Executed', alpha=0.8)
                ax.set_title(f'End-Effector Velocity {["X", "Y", "Z"][i]}')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Velocity (m/s)')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.suptitle(f'{run_key}: Planned vs Executed Trajectory', fontsize=14)
            plt.tight_layout()
            
            # Save plot
            plot_dir = os.path.dirname(hdf5_path)
            plot_path = os.path.join(plot_dir, f'hardware_trajectory_comparison_{run_key}.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Saved trajectory comparison plot to: {plot_path}")
            
            plt.show()
            
        except KeyError as e:
            print(f"Missing data for trajectory comparison: {e}")

def generate_hardware_summary_report(hdf5_path):
    """
    Generate a comprehensive summary report of the hardware data collection.
    """
    with h5py.File(hdf5_path, 'r') as f:
        print("="*60)
        print("HARDWARE DATA COLLECTION SUMMARY REPORT")
        print("="*60)
        
        # Collection parameters
        if 'parameters' in f:
            params = f['parameters']
            print("\nCollection Parameters:")
            for key in params.keys():
                value = params[key][()]
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value}")
                elif key == 'collection_date':
                    print(f"  {key}: {value}")
        
        # Summary statistics
        if 'summary' in f:
            summary = f['summary']
            total_attempted = summary['total_attempted'][()]
            successful_runs = summary['successful_runs'][()]
            success_rate = summary['success_rate'][()]
            
            print(f"\nExecution Summary:")
            print(f"  Total Attempted: {total_attempted}")
            print(f"  Successful Runs: {successful_runs}")
            print(f"  Success Rate: {success_rate:.1%}")
        
        # Analyze run data
        run_keys = [key for key in f.keys() if key.startswith('run_')]
        if run_keys:
            print(f"\nDetailed Analysis ({len(run_keys)} runs):")
            
            # Error statistics
            all_pos_errors = []
            all_vel_errors = []
            impact_detections = 0
            
            for run_key in run_keys:
                run_data = f[run_key]
                try:
                    p_f_des = run_data['p_f_des'][:]
                    v_f_des = run_data['v_f_des'][:]
                    
                    if 'impact_pos' in run_data:
                        p_f_hw = run_data['impact_pos'][:]
                        v_f_hw = run_data['impact_vel'][:]
                        impact_detections += 1
                    else:
                        p_f_hw = run_data['p_f_opt'][:]
                        v_f_hw = run_data['v_f_opt'][:]
                    
                    pos_error = np.linalg.norm(p_f_hw - p_f_des)
                    vel_error = np.linalg.norm(v_f_hw - v_f_des)
                    
                    all_pos_errors.append(pos_error)
                    all_vel_errors.append(vel_error)
                    
                except KeyError:
                    continue
            
            if all_pos_errors:
                print(f"  Position Error (RMS):")
                print(f"    Mean: {np.mean(all_pos_errors):.4f} m")
                print(f"    Std:  {np.std(all_pos_errors):.4f} m")
                print(f"    Max:  {np.max(all_pos_errors):.4f} m")
                
                print(f"  Velocity Error (RMS):")
                print(f"    Mean: {np.mean(all_vel_errors):.4f} m/s")
                print(f"    Std:  {np.std(all_vel_errors):.4f} m/s")
                print(f"    Max:  {np.max(all_vel_errors):.4f} m/s")
                
                print(f"  Impact Detections: {impact_detections}/{len(run_keys)} ({impact_detections/len(run_keys):.1%})")
        
        print("="*60)

if __name__ == "__main__":
    # Load the HDF5 file - Update this path to match your hardware data
    _, hdf5_path = data_collection.check_for_resume()
    # results_dir = "../results/hardware_data_2025-06-05_12-29-56"  # Replace with actual directory
    # hdf5_file = "hardware_collected_dataset_gen3.hdf5"
    # hdf5_path = os.path.join(os.path.dirname(__file__), results_dir, hdf5_file)
    
    if not os.path.exists(hdf5_path):
        print(f"HDF5 file not found: {hdf5_path}")
        print("Please update the results_dir variable with the correct path.")
        exit(1)
    
    print(f"Loading hardware data from: {hdf5_path}")
    
    # Generate comprehensive analysis
    generate_hardware_summary_report(hdf5_path)
    
    # Create visualizations
    plot_hardware_impact_errors(hdf5_path)
    # plot_hardware_3d_errors(hdf5_path)
    # plot_hardware_torque_analysis(hdf5_path)
    
    # Compare a specific trajectory (first successful run)
    plot_trajectory_comparison(hdf5_path, run_index=823)
