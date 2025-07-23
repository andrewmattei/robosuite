import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

def plot_impact_errors(hdf5_path):
    """
    Load HDF5 file and plot position/velocity errors between desired and simulated 
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
        
        # Track failed runs
        failed_runs = []
        
        # Collect errors for each run
        for i, run_key in enumerate(run_keys):
            run_data = f[run_key]
            try:
                # Get desired values
                p_f_des = run_data['p_f_des'][:]
                v_f_des = run_data['v_f_des'][:]
                
                # Get simulated values
                p_f_sim = run_data['p_f_sim'][:]
                v_f_sim = run_data['v_f_sim'][:]
                
                # Calculate errors
                px_errors[i] = p_f_sim[0] - p_f_des[0]
                py_errors[i] = p_f_sim[1] - p_f_des[1]
                pz_errors[i] = p_f_sim[2] - p_f_des[2]
                vx_errors[i] = v_f_sim[0] - v_f_des[0]
                vy_errors[i] = v_f_sim[1] - v_f_des[1]
                vz_errors[i] = v_f_sim[2] - v_f_des[2]
            except KeyError as e:
                failed_runs.append((run_key, str(e)))
                continue
    
    # Create figure with 6 subplots (3D data)
    fig, axes = plt.subplots(6, 1, figsize=(10, 15))
    run_indices = np.arange(n_runs)
    
    # Plot each error type
    for ax, errors, title in zip(axes, 
                               [px_errors, py_errors, pz_errors,
                                vx_errors, vy_errors, vz_errors],
                               ['Kinova Gen3 Position X', 'Position Y', 'Position Z',
                                'Velocity X', 'Velocity Y', 'Velocity Z']):
        # Plot successful runs
        valid_mask = ~np.isnan(errors)
        ax.plot(run_indices[valid_mask], errors[valid_mask], 'b.-', label='Successful')
        
        # Mark failed runs
        failed_mask = np.isnan(errors)
        if np.any(failed_mask):
            ax.plot(run_indices[failed_mask], np.zeros_like(run_indices[failed_mask]), 
                   'rx', label='Failed', markersize=15)
        
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax.set_title(f'{title} Error (sim - des)')
        ax.set_ylabel('Error (m)' if 'Position' in title else 'Error (m/s)')
        ax.grid(True, alpha=0.3)
        if np.any(valid_mask):
            mean_err = np.mean(errors[valid_mask])
            std_err = np.std(errors[valid_mask])
            ax.text(0.02, 0.95, f'Mean: {mean_err:.4f}\nStd: {std_err:.4f}', 
                   transform=ax.transAxes, fontsize=8, 
                   bbox=dict(facecolor='white', alpha=0.8))
        ax.legend()
    
    axes[-1].set_xlabel('Run Index')
    # plt.suptitle('Gen3 Impact Control Results', fontsize=14)
    
    # Print statistics for successful runs
    print("\nStatistics (excluding failed runs):")
    for errors, name in zip([px_errors, py_errors, pz_errors,
                           vx_errors, vy_errors, vz_errors],
                          ['Position X', 'Position Y', 'Position Z',
                           'Velocity X', 'Velocity Y', 'Velocity Z']):
        valid_errors = errors[~np.isnan(errors)]
        if len(valid_errors) > 0:
            print(f"{name} Error - Mean: {np.mean(valid_errors):.4f}, "
                  f"Std: {np.std(valid_errors):.4f}")
    
    # Print failed runs
    if failed_runs:
        print("\nFailed runs:")
        for run_key, error in failed_runs:
            print(f"{run_key}: {error}")
    
    plt.tight_layout()
    
    # Save plot
    plot_dir = os.path.dirname(hdf5_path)
    plot_path = os.path.join(plot_dir, 'gen3_impact_errors.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved error plot to: {plot_path}")
    
    plt.show()

def plot_3d_errors(hdf5_path):
    """
    Create 3D scatter plots of position and velocity errors.
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
                p_f_sim = run_data['p_f_sim'][:]
                v_f_sim = run_data['v_f_sim'][:]
                
                p_errors[i] = p_f_sim - p_f_des
                v_errors[i] = v_f_sim - v_f_des
            except KeyError:
                continue

    # Create 3D plots
    fig = plt.figure(figsize=(15, 7))
    
    # Position errors
    ax1 = fig.add_subplot(121, projection='3d')
    valid_mask = ~np.isnan(p_errors).any(axis=1)
    scatter1 = ax1.scatter(p_errors[valid_mask, 0], 
                          p_errors[valid_mask, 1], 
                          p_errors[valid_mask, 2],
                          c=np.linalg.norm(p_errors[valid_mask], axis=1),
                          cmap='viridis')
    plt.colorbar(scatter1, label='Error Magnitude (m)')
    ax1.set_title('Position Errors')
    ax1.set_xlabel('X Error (m)')
    ax1.set_ylabel('Y Error (m)')
    ax1.set_zlabel('Z Error (m)')
    
    # Velocity errors
    ax2 = fig.add_subplot(122, projection='3d')
    valid_mask = ~np.isnan(v_errors).any(axis=1)
    scatter2 = ax2.scatter(v_errors[valid_mask, 0], 
                          v_errors[valid_mask, 1], 
                          v_errors[valid_mask, 2],
                          c=np.linalg.norm(v_errors[valid_mask], axis=1),
                          cmap='viridis')
    plt.colorbar(scatter2, label='Error Magnitude (m/s)')
    ax2.set_title('Velocity Errors')
    ax2.set_xlabel('X Error (m/s)')
    ax2.set_ylabel('Y Error (m/s)')
    ax2.set_zlabel('Z Error (m/s)')
    
    plt.suptitle('Gen3 3D Error Distribution', fontsize=14)
    plt.tight_layout()
    
    # Save plot
    plot_dir = os.path.dirname(hdf5_path)
    plot_path = os.path.join(plot_dir, 'gen3_3d_errors.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved 3D error plot to: {plot_path}")
    
    plt.show()

if __name__ == "__main__":
    # Load the HDF5 file
    results_dir = "../results/2025-06-02_15-15-52"  # Replace with actual directory
    hdf5_file = "collected_dataset_gen3.hdf5"
    hdf5_path = os.path.join(os.path.dirname(__file__), results_dir, hdf5_file)
    
    plot_impact_errors(hdf5_path)
    plot_3d_errors(hdf5_path)
