import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

def plot_impact_errors(hdf5_path):
    """
    Load HDF5 file and plot position/velocity errors between desired and simulated values.
    Handle missing fields by marking them as NaN.
    """
    with h5py.File(hdf5_path, 'r') as f:
        run_keys = [key for key in f.keys() if key.startswith('run_')]
        n_runs = len(run_keys)
        
        # Initialize arrays with NaN
        px_errors = np.full(n_runs, np.nan)
        py_errors = np.full(n_runs, np.nan)
        vx_errors = np.full(n_runs, np.nan)
        vy_errors = np.full(n_runs, np.nan)
        
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
                vx_errors[i] = v_f_sim[0] - v_f_des[0]
                vy_errors[i] = v_f_sim[1] - v_f_des[1]
            except KeyError as e:
                failed_runs.append((run_key, str(e)))
                continue
    
    # Create figure
    fig, axes = plt.subplots(4, 1, figsize=(10, 12))
    run_indices = np.arange(n_runs)
    
    # Plot each error type
    for ax, errors, title in zip(axes, 
                               [px_errors, py_errors, vx_errors, vy_errors],
                               ['Position X', 'Position Y', 'Velocity X', 'Velocity Y']):
        # Plot successful runs
        valid_mask = ~np.isnan(errors)
        ax.plot(run_indices[valid_mask], errors[valid_mask], 'b.-', label='Successful')
        
        # Mark failed runs
        failed_mask = np.isnan(errors)
        if np.any(failed_mask):
            ax.plot(run_indices[failed_mask], np.zeros_like(run_indices[failed_mask]), 
                   'rx', label='Failed', markersize=15)
        
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_title(f'{title} Error (sim - des)')
        ax.set_ylabel('Error (m)' if 'Position' in title else 'Error (m/s)')
        ax.grid(True)
        ax.legend()
    
    axes[-1].set_xlabel('Run Index')
    
    # Print statistics for successful runs only
    print("\nStatistics (excluding failed runs):")
    for errors, name in zip([px_errors, py_errors, vx_errors, vy_errors],
                          ['Position X', 'Position Y', 'Velocity X', 'Velocity Y']):
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
    plot_path = os.path.join(plot_dir, 'impact_errors.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved error plot to: {plot_path}")
    
    plt.show()

if __name__ == "__main__":
    # Load the HDF5 file
    # results_dir = "../results/2025-04-28_19-00-57" 
    results_dir = "../results/2025-04-29_21-42-32"
    hdf5_file = "collected_dataset_only_final_states.hdf5"
    hdf5_path = os.path.join(os.path.dirname(__file__), results_dir, hdf5_file)
    
    plot_impact_errors(hdf5_path)