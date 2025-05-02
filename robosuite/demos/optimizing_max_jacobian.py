import numpy as np
import casadi as cs
import matplotlib.pyplot as plt
import os

# ============================== Helper Functions ==============================
def forward_kinematics_homogeneous(q, l, site=3):
    """
    Computes the homogeneous transform from base to a specified site
    for a planar 3DOF arm using rotation matrices.

    Parameters:
        q : SX [3x1] joint angles
        l : [3] list of link lengths
        site : int, which link to compute transform to (1, 2, or 3)
               1: base to link 1
               2: base to link 2
               3: base to end-effector (default)
    Returns:
        T_0_site : 3x3 homogeneous transformation matrix (planar SE(2))
    """
    def rot(theta):
        return cs.vertcat(
            cs.horzcat(cs.cos(theta), -cs.sin(theta), 0),
            cs.horzcat(cs.sin(theta),  cs.cos(theta), 0),
            cs.horzcat(0,              0,             1)
        )

    def trans(x, y):
        return cs.vertcat(
            cs.horzcat(1, 0, x),
            cs.horzcat(0, 1, y),
            cs.horzcat(0, 0, 1)
        )

    # Base to link 1
    T1 = rot(q[0]) @ trans(l[0], 0)
    if site == 1:
        return T1
        
    # Link 1 to link 2
    T2 = rot(q[1]) @ trans(l[1], 0)
    if site == 2:
        return T1 @ T2
        
    # Link 2 to end-effector
    T3 = rot(q[2]) @ trans(l[2], 0)
    if site == 3:
        return T1 @ T2 @ T3
        
    raise ValueError(f"Invalid site {site}. Must be 1, 2, or 3.")

def end_effector_position(q, l):
    """
    Extracts the (x, y) position of the end-effector from the homogeneous transform

    Parameters:
        q : SX [3x1] joint angles
        l : [3] list of link lengths
    Returns:
        pos : SX [2x1] end-effector position
    """
    T = forward_kinematics_homogeneous(q, l)
    return T[:2, 2]


def inverse_kinematics_casadi(target_pose, l, q_init=None):
    """
    Solve inverse kinematics using CasADi and symbolic homogeneous FK.

    Args:
        target_pose: np.array shape (2,) desired EE position
        l: list of link lengths [l1, l2, l3]
        q_init: Optional np.array shape (3,) initial guess

    Returns:
        np.array shape (3,) or None if solve fails
    """
    q_sym = cs.SX.sym("q", 3)
    ee_pos = end_effector_position(q_sym, l)

    cost = cs.sumsqr(ee_pos - target_pose)
    nlp = {'x': q_sym, 'f': cost}
    solver = cs.nlpsol('ik_solver', 'ipopt', nlp, {'ipopt.print_level': 0, 'print_time': 0})

    if q_init is None:
        q_init = np.zeros(3)

    try:
        sol = solver(x0=q_init, lbx=-np.pi, ubx=np.pi)
        return np.array(sol['x']).flatten()
    except RuntimeError:
        print("[IK FAIL] CasADi solver failed.")
        return None


def compute_jacobian(q, l):
    """Compute Jacobian for 3DOF planar arm"""
    # Assuming q is a CasADi SX matrix
    theta1 = q[0]
    theta2 = q[1]
    theta3 = q[2]

    # Create a 2x3 zero matrix for the Jacobian using SX instead of SX
    J = cs.SX.zeros(2, 3)

    # Fill in the Jacobian matrix with symbolic expressions for 2D position
    J[0, 0] = -l[0] * cs.sin(theta1) - l[1] * cs.sin(theta1 + theta2) - l[2] * cs.sin(theta1 + theta2 + theta3)
    J[0, 1] = -l[1] * cs.sin(theta1 + theta2) - l[2] * cs.sin(theta1 + theta2 + theta3)
    J[0, 2] = -l[2] * cs.sin(theta1 + theta2 + theta3)
    J[1, 0] = l[0] * cs.cos(theta1) + l[1] * cs.cos(theta1 + theta2) + l[2] * cs.cos(theta1 + theta2 + theta3)
    J[1, 1] = l[1] * cs.cos(theta1 + theta2) + l[2] * cs.cos(theta1 + theta2 + theta3)
    J[1, 2] = l[2] * cs.cos(theta1 + theta2 + theta3)

    return J


def formulate_symbolic_dynamic_matrices(m, l, r, q_sym=None, dq_sym=None):
    """Symbolic representation of M(q) and C(q, dq) for a 3DOF planar arm"""
    if q_sym is None or dq_sym is None:
        # Create symbolic variables
        q_sym = cs.SX.sym('q', 3, 1)   # Column vector (3x1)
        dq_sym = cs.SX.sym('dq', 3, 1) # Column vector (3x1)
    else:
        assert q_sym.shape == (3, 1) and dq_sym.shape == (3, 1), "Invalid shape for q_sym or dq_sym"

    # Link parameters
    m1, m2, m3 = m[0], m[1], m[2]
    l1, l2, l3 = l[0], l[1], l[2]
    r1, r2, r3 = r[0], r[1], r[2]

    # Position of each link's center of mass
    p1 = cs.vertcat(0.5 * l1 * cs.cos(q_sym[0]), 0.5 * l1 * cs.sin(q_sym[0]))
    p2 = cs.vertcat(
        l1 * cs.cos(q_sym[0]) + 0.5 * l2 * cs.cos(q_sym[0] + q_sym[1]),
        l1 * cs.sin(q_sym[0]) + 0.5 * l2 * cs.sin(q_sym[0] + q_sym[1])
    )
    p3 = cs.vertcat(
        l1 * cs.cos(q_sym[0]) + l2 * cs.cos(q_sym[0] + q_sym[1]) + 0.5 * l3 * cs.cos(q_sym[0] + q_sym[1] + q_sym[2]),
        l1 * cs.sin(q_sym[0]) + l2 * cs.sin(q_sym[0] + q_sym[1]) + 0.5 * l3 * cs.sin(q_sym[0] + q_sym[1] + q_sym[2])
    )

    # Compute Jacobians of the COM positions
    J1 = cs.jacobian(p1, q_sym)
    J2 = cs.jacobian(p2, q_sym)
    J3 = cs.jacobian(p3, q_sym)

    # Velocities of the COMs (using dq_sym)
    v1 = J1 @ dq_sym
    v2 = J2 @ dq_sym
    v3 = J3 @ dq_sym

    # Kinetic energy terms, including rotational KE for each link
    I = lambda m, l, r: m * (l**2 + 3*r**2) / 12
    # Kinetic energy terms, including translational and rotational kinetic energies
    KE = (0.5 * m1 * cs.dot(v1, v1) + 0.5 * I(m1, l1, r1) * dq_sym[0]**2 +
          0.5 * m2 * cs.dot(v2, v2) + 0.5 * I(m2, l2, r2) * (dq_sym[0] + dq_sym[1])**2 +
          0.5 * m3 * cs.dot(v3, v3) + 0.5 * I(m3, l3, r3) * (dq_sym[0] + dq_sym[1] + dq_sym[2])**2)

    # Compute the inertia matrix M(q) using the Hessian (second-order differentiation)
    M = cs.jacobian(cs.jacobian(KE, dq_sym), dq_sym)

    # Compute the Coriolis matrix C(q, dq)
    C = cs.SX.zeros(3, 3)
    for k in range(3):
        for j in range(3):
            C_sum = 0
            for i in range(3):
                # Instead of differentiating with respect to dq_sym[i,0],
                # differentiate with respect to the full q_sym and extract the ith element.
                dMkj_dq = cs.jacobian(M[k, j], q_sym)  # 1x3 row vector
                dMij_dq = cs.jacobian(M[i, j], q_sym)  # 1x3 row vector
                # Extract the corresponding partial derivatives
                partial1 = dMkj_dq[0, i]
                partial2 = dMij_dq[0, k]
                C_sum += (partial1 - 0.5 * partial2) * dq_sym[i, 0]
            C[k, j] = C_sum

    # Create CasADi functions for numerical evaluation
    M_fun = cs.Function('M', [q_sym], [M])
    C_fun = cs.Function('C', [q_sym, dq_sym], [C])
    return M, C, M_fun, C_fun


def compute_dynamics_matrices(M_fun, C_fun, q, dq):
    """computation of M(q) and C(q, dq) for a 3DOF planar arm"""
    M_eval = M_fun(q)
    C_eval = C_fun(q, dq)
    return M_eval, C_eval

def compute_singular_values(J):
    """Compute singular values of a 2x3 Jacobian without SVD"""
    JJT = J @ J.T
    trace = cs.trace(JJT)
    det = cs.det(JJT)
    
    # Eigenvalues of JJT: λ = [trace ± sqrt(trace² - 4*det)]/2
    sqrt_term = cs.sqrt(trace**2 - 4*det)
    sigma1_sq = (trace + sqrt_term)/2
    sigma2_sq = (trace - sqrt_term)/2
    
    return cs.sqrt(sigma1_sq), cs.sqrt(sigma2_sq)


def damping_pseudoinverse(J, damping=1e-4):
    """
    Computes the damped pseudoinverse of a Jacobian matrix J using the formula:
    J⁺ = Jᵀ (JJᵀ + λ²I)⁻¹
    where λ is the damping factor and I is the identity matrix.
    
    Parameters:
        J : SX (2x3) symbolic Jacobian matrix
        damping : float, damping factor for pseudoinverse
    
    Returns:
        J_pinv : SX (3x2) symbolic damped pseudoinverse of J
    """
    JT = J.T
    JJ_T = J @ JT
    J_pinv = JT @ cs.inv(JJ_T + damping * cs.SX.eye(2))
    
    return J_pinv


def compute_symbolic_cartesian_acceleration(M, C, J, q, dq, tau, G=None):
    """
    Computes symbolic Cartesian acceleration:
    xdd = J * M⁻¹ (tau - C*dq - G + M*J⁺*Jdot*dq)
    
    Parameters:
        M : SX (3x3) symbolic mass matrix
        C : SX (3x3) symbolic Coriolis matrix
        J : SX (2x3) symbolic Jacobian of end-effector
        q : SX (3x1) symbolic joint angles
        dq: SX (3x1) symbolic joint velocities
        tau: SX (3x1) symbolic joint torques
        damping : float, damping factor for damped pseudoinverse
        G : SX (3x1) gravity vector (optional, default is 0)
    
    Returns:
        xdd_sym : SX (2x1) symbolic expression for Cartesian acceleration
    """
    # Gravity term (default to zero if not given)
    if G is None:
        G = cs.SX.zeros(3, 1)

    # Compute Jdot symbolically: Jdot = sum_i (∂J/∂q_i * dq_i)
    Jdot = cs.SX.zeros(2, 3)
    for i in range(3):
        dJ_dqi = cs.jacobian(J[:, i], q)  # (2x3)
        Jdot += dJ_dqi @ dq[i]

    # Damped least-squares pseudoinverse of J
    J_pinv = damping_pseudoinverse(J)

    # Compute symbolic expression for Cartesian acceleration
    acc_dyn = tau - C @ dq - G + M @ (J_pinv @ (Jdot @ dq))
    xdd_sym = J @ cs.solve(M, acc_dyn)

    return xdd_sym


def linearize_dynamics_along_trajectory(T_opt, U_opt, Z_opt, m, l, r):
    """
    Linearizes the dynamics around a given state-control trajectory.

    Args:
        T_opt: (N,) time vector.
        U_opt: (3, N) optimal torque inputs.
        Z_opt: (6, N+1) optimal [q; dq] trajectory.
        m, l, r: robot parameters.
        dt: time step.

    Returns:
        A_list: list of discrete-time A matrices.
        B_list: list of discrete-time B matrices.
    """
    
    dt_list = np.diff(T_opt)
    dt_list = np.append(dt_list, dt_list[-1])  # Append last value to match U_opt length
    N = U_opt.shape[1]
    A_list, B_list = [], []

    q_sym = cs.SX.sym("q", 3)
    dq_sym = cs.SX.sym("dq", 3)
    tau_sym = cs.SX.sym("tau", 3)
    z_sym = cs.vertcat(q_sym, dq_sym)

    # Get dynamics
    M_sym, C_sym, _, _ = formulate_symbolic_dynamic_matrices(m, l, r, q_sym, dq_sym)
    ddq_sym = cs.solve(M_sym, tau_sym - C_sym @ dq_sym)

    z_next = cs.vertcat(dq_sym, ddq_sym)
    f_cont = cs.Function("f_cont", [z_sym, tau_sym], [z_next])

    for k in range(N):
        dt_k = dt_list[k]
        z_k = Z_opt[:, k]
        u_k = U_opt[:, k]

        A_k = cs.jacobian(f_cont(z_sym, tau_sym), z_sym)
        B_k = cs.jacobian(f_cont(z_sym, tau_sym), tau_sym)

        A_k_fun = cs.Function("A_k_fun", [z_sym, tau_sym], [A_k])
        B_k_fun = cs.Function("B_k_fun", [z_sym, tau_sym], [B_k])

        A_d = np.eye(6) + dt_k * A_k_fun(z_k, u_k).full()
        B_d = dt_k * B_k_fun(z_k, u_k).full()

        A_list.append(A_d)
        B_list.append(B_d)

    return A_list, B_list



def sample_qf_given_y_line_casadi(robot, q_ref, l_ext, r_ext, n_pose, v_f_list):
    """
    Samples joint configurations with same y as FK(q_ref), and different x across a range.
    Each pose is paired with all velocities in v_f_list.

    Args:
        robot: RobotSystem with .l and compatible FK
        q_ref: np.array shape (3,), reference joint configuration
        l_ext, r_ext: float, x-range around FK(q_ref)
        n_pose: int, number of pose samples along x-axis
        v_f_list: list of 2D velocities (np.array shape (2,))

    Returns:
        List of (q_f, v_f) tuples of length n_pose * len(v_f_list)
    """

    # Convert q_ref to CasADi SX for symbolic FK computation
    q_sym = cs.SX.sym("q", 3)
    ee_pos_fun = cs.Function("ee_pos_fun", [q_sym], [end_effector_position(q_sym, robot.l)])
    p_ref = ee_pos_fun(q_ref).full().flatten()
    y_ref = p_ref[1]

    x_range = np.linspace(p_ref[0] - l_ext, p_ref[0] + r_ext, n_pose)
    qf_vf_pairs = []

    for x_i in x_range:
        target_pose = np.array([x_i, y_ref])
        q_f = inverse_kinematics_casadi(target_pose, robot.l, q_init=q_ref)
        if q_f is not None:
            for v_f in v_f_list:
                qf_vf_pairs.append((q_f, v_f))
        else:
            print(f"[WARNING] IK failed for x = {x_i}, skipping this pose.")

    return qf_vf_pairs


def sample_pf_vf_grid(l_px, r_px, n_px,      # position x range and samples
                      l_vx, r_vx, n_vx,       # velocity x range and samples
                      l_vy, r_vy, n_vy):      # velocity y range and samples
    """
    Samples end-effector position and velocity pairs in a grid pattern.
    Positions are sampled along x-axis with y=0 (table surface).
    Velocities are sampled in a 2D grid within specified ranges.
    
    Args:
        l_px, r_px: left/right bounds for position x sampling
        n_px: number of position samples
        l_vx, r_vx: left/right bounds for velocity x sampling
        n_vx: number of velocity x samples
        l_vy, r_vy: lower/upper bounds for velocity y sampling
        n_vy: number of velocity y samples
        
    Returns:
        List of (p_f, v_f) tuples where:
            p_f: position [x, 0]
            v_f: velocity [vx, vy]
    """
    # Sample positions along x-axis (y is always 0)
    x_range = np.linspace(l_px, r_px, n_px)
    
    # Sample velocities in 2D grid
    vx_range = np.linspace(l_vx, r_vx, n_vx)
    vy_range = np.linspace(l_vy, r_vy, n_vy)
    
    # Generate all combinations
    pf_vf_pairs = []
    
    for px in x_range:
        p_f = np.array([px, 0.0])  # position on table surface
        for vx in vx_range:
            for vy in vy_range:
                v_f = np.array([vx, vy])
                pf_vf_pairs.append((p_f, v_f))
            
    return pf_vf_pairs


from scipy.interpolate import interp1d

def match_trajectories(T_des, *args):
    """
    A translation of the MATLAB function match_trajectories to Python.
    Original MATLAB code in: 
    https://github.com/skousik/simulator/blob/master/src/utility/interpolation/match_trajectories.m
    match_trajectories(T_des, T1, Z1, T2, Z2, ..., interp_type)
    
    Given desired sample times T_des (1D array) and any number of time vectors (T_i)
    with associated trajectories (Z_i, shape: n_states x n_t), this function reinterpolates
    each trajectory linearly at the desired times.
    
    If T_des[0] < T_i[0], then the output Z_i is pre-padded with the first column of Z_i,
    and similarly if T_des[-1] > T_i[-1].
    
    Parameters:
        T_des : 1D array-like
            The desired sample times.
        *args:
            A sequence of time vectors and trajectories: (T1, Z1, T2, Z2, ..., interp_type)
            where interp_type is an optional string at the end (default 'linear').
            
    Returns:
        list of np.ndarray
            A list containing the reinterpolated trajectories corresponding to each input pair.
    """
    # Determine the interpolation type. If the last argument is a string, use it.
    if isinstance(args[-1], str):
        interp_type = args[-1]
        traj_args = args[:-1]
    else:
        interp_type = 'linear'
        traj_args = args

    results = []
    T_des = np.atleast_1d(T_des)

    # Process each pair: (T, Z)
    for i in range(0, len(traj_args), 2):
        T = np.array(traj_args[i])
        Z = np.array(traj_args[i+1])
        # Ensure T is a 1D array and Z is 2D (n_states x n_t)
        T = T.flatten()

        # Pad T and Z if T_des extends outside the original T range.
        if T_des[0] < T[0]:
            T = np.concatenate(([T_des[0]], T))
            # Pre-pad Z with its first column.
            first_col = Z[:, [0]]
            Z = np.concatenate((first_col, Z), axis=1)

        if T_des[-1] > T[-1]:
            T = np.concatenate((T, [T_des[-1]]))
            # Append the last column of Z.
            last_col = Z[:, [-1]]
            Z = np.concatenate((Z, last_col), axis=1)

        # If no interpolation is needed (single time point that matches T_des)
        if len(T) == 1 and np.allclose(T_des, T):
            result = Z
        else:
            # Create an interpolator.
            # Note: We transpose Z so that interpolation is performed for each state.
            f = interp1d(T, Z.T, kind=interp_type, axis=0, bounds_error=False, fill_value="extrapolate")
            # Evaluate at T_des and then transpose back.
            result = f(T_des).T

        results.append(result)

    return results


def save_solution_to_npy(solution, filename):
    """
    Save the solution to a .npy file.
    
    Parameters:
        solution : dict
            The solution dictionary containing the trajectory data.
        filename : str
            The name of the file to save the solution.
        upsample : int
            The number of points to upsample the trajectory.
    """
    T = solution['t_f']
    l = solution['l']
    num_steps = solution['q'].shape[1]
    
    # Upsample the trajectory
    Z_opt = np.vstack((solution['q'], solution['dq']))
    # U_opt = np.hstack((solution['tau'], np.zeros((3,1)))) # !!bad shouldn't pad with zero
    U_opt = np.hstack((solution['tau'], solution['tau'][:, -1:]))  # pad with last value
    T_opt = np.linspace(0, T, num_steps)
    
    
    # Save the trajectory data
    trajectory_data = {
        'T_opt': T_opt,
        'Z_opt': Z_opt,
        'U_opt': U_opt,
        'T': T,
        'l': l,
        'm': solution['m'],
        'r': solution['r'],
    }
    
    # Save to .npy file
    np.save(filename, trajectory_data)


def display_and_save_solution(solution, filename):
    # Display the results
    T = solution['t_f']
    l = solution['l']
    num_steps = solution['q'].shape[1]
    v_ee = np.zeros((2, num_steps))
    q_sym = cs.SX.sym('q', 3)
    J_p3 = compute_jacobian(q_sym, l)
    J_p3_fun = cs.Function('J_p3', [q_sym], [J_p3])
    for k in range(num_steps):
        qk = solution['q'][:, k]
        dqk = solution['dq'][:, k]
        v_ee[:, k] = np.array(J_p3_fun(qk)) @ dqk

    v_ee_mag = np.linalg.norm(v_ee, axis=0)
    time = np.linspace(0, T, num_steps)

    t_dense = np.linspace(0, T, 1000)
    Z_opt = np.vstack((solution['q'], solution['dq']))
    # U_opt = np.hstack((solution['tau'], np.zeros((3,1)))) # !!bad shouldn't pad with zero
    U_opt = np.hstack((solution['tau'], solution['tau'][:, -1:]))  # pad with last value
    T_opt = np.linspace(0, T, num_steps)
    [Z_dense] = match_trajectories(t_dense, T_opt, Z_opt)

    # Save the trajectory data
    trajectory_data = {
        'T_opt': T_opt,
        'Z_opt': Z_opt,
        'U_opt': U_opt,
    }
    # get current file location
    # notebook_path = os.path.abspath(globals()['_dh'][0])
    current_path = os.path.dirname(os.path.realpath(__file__))
    save_path = os.path.join(current_path, filename)
    np.save(save_path, trajectory_data)

    # plot torque, v_ee, and v_ee_mag
    plt.figure(figsize=(6, 8))
    plt.subplot(4, 1, 1)
    plt.plot(time[:-1], solution['tau'].T)
    plt.ylabel('Torque (Nm)')
    plt.subplot(4, 1, 2)
    plt.plot(time, v_ee_mag)
    plt.ylabel('EEF velocity (m/s)')
    plt.subplot(4, 1, 3)
    plt.plot(time, v_ee[0, :], label='x')
    plt.plot(time, v_ee[1, :], label='y')
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('EEF components (m/s)')
    plt.subplot(4, 1, 4)
    plt.plot(t_dense, Z_dense[0:3, :].T)
    plt.plot(time, solution['q'].T,'o')
    plt.xlabel('Time (s)')
    plt.ylabel('Joint angles (rad)')
    plt.legend(['q1', 'q2', 'q3'])
    plt.tight_layout()
    plt.show()


# ========================== Trajectory Optimization ==========================


def optimize_trajectory(q_start, dq_start, v_des, m, l, r,
                        q_lower, q_upper, dq_lower, dq_upper,
                        tau_lower, tau_upper,
                        T, N,
                        weight_v=0.001, weight_M00=10, weight_tau=0.001,
                        weight_tau_smooth=1e-3):
    """
    Optimize a trajectory for the 3DOF planar arm.
    
    The trajectory starts at q_start and dq_start, and over a time horizon T with N steps,
    the optimizer aims to drive the end-effector (p3) velocity toward v_des.
    
    Constraints:
        - Joint positions and velocities are kept within provided bounds.
        - Joint torques (tau) are kept within provided bounds.
        - System dynamics are enforced via a discretized model:
              q[k+1] = q[k] + dt * dq[k]
              dq[k+1] = dq[k] + dt * ddq[k],
          where ddq[k] is computed from the dynamics:
              M(q) ddq + C(q, dq)dq = tau.
              
    Additionally, a running cost includes:
        - A term penalizing the squared error between the current end-effector velocity 
          (computed from forward kinematics of the third link) and the desired velocity v_des.
        - A term that *rewards* a high norm of the gradient dM00/dq (by subtracting its squared norm).
        - A small penalty on torque effort.
        
    Parameters:
      q_start    : (3x1 np.array) initial joint angles.
      dq_start   : (3x1 np.array) initial joint velocities.
      v_des      : (1d np.array) desired end-effector velocity.
      m, l, r    : lists/arrays of link masses, lengths, and center-of-mass distances.
      q_lower, q_upper: (3x1 arrays) joint limits.
      dq_lower, dq_upper: (3x1 arrays) joint velocity limits.
      tau_lower, tau_upper: (3x1 arrays) joint torque limits.
      T          : total time horizon.
      N          : number of discretization steps.
      weight_v   : weight for the end-effector velocity tracking cost.
      weight_dM  : weight for the dM00/dq cost (note: it is subtracted so that maximizing dM00/dq is encouraged).
      weight_tau : weight for a small torque effort penalty.
      
    Returns:
      sol : dictionary with optimal trajectories for q, dq, and tau.
    """
    dt = T / N
    # v_des = cs.DM(v_des)

    # Get the symbolic dynamics functions from your provided formulation.
    # These functions assume a 3DOF arm.
    q_sym = cs.SX.sym('q', 3)
    dq_sym = cs.SX.sym('dq', 3)
    M, C, M_fun, C_fun = formulate_symbolic_dynamic_matrices(m, l, r, q_sym, dq_sym)

    # Try #1 for stretching constraint
    dM00_dq = cs.jacobian(M[0, 0], q_sym)
    dM00_dq_fun = cs.Function('dM00_dq', [q_sym], [dM00_dq])

    # Try #2 for stretching constraint
    det_M = cs.det(M)  # Symbolic determinant of the mass matrix
    det_M_fun = cs.Function('det_M', [q_sym], [det_M])  # Function to evaluate det(M) at specific q


    J_p3 = compute_jacobian(q_sym, l)
    J_p3_fun = cs.Function('J_p3', [q_sym], [J_p3])

    # Create an Opti instance
    opti = cs.Opti()

    # Decision variables: state and control trajectories
    Q = opti.variable(3, N+1)   # joint angles trajectory
    dQ = opti.variable(3, N+1)  # joint velocities trajectory
    Tau = opti.variable(3, N)   # joint torques (control input)

    # Set initial conditions
    opti.subject_to(Q[:, 0] == q_start)
    opti.subject_to(dQ[:, 0] == dq_start)

    p_opts = {"print_time": True}
    s_opts = {"print_level": 5}  # Increase to a higher level for more detailed output
    opti.solver('ipopt', p_opts, s_opts)

    # Cost accumulator
    total_cost = 0


    # Loop over time steps to enforce dynamics and accumulate cost
    for k in range(N):
        # Evaluate dynamics functions at the current state
        qk = Q[:, k]
        dqk = dQ[:, k]
        tau_k = Tau[:, k]
        M_k = M_fun(qk)
        C_k = C_fun(qk, dqk)
        
        
        # Compute acceleration: ddq = M^{-1} (tau - C*dq)
        ddq_k = cs.mtimes(cs.inv(M_k), (Tau[:, k] - cs.mtimes(C_k, dqk)))
        
        # Dynamics constraints (Euler integration)
        opti.subject_to(Q[:, k+1] == qk + dt * dqk)
        opti.subject_to(dQ[:, k+1] == dqk + dt * ddq_k)
        
        # State bounds constraints
        opti.subject_to(q_lower <= qk)
        opti.subject_to(qk <= q_upper)
        opti.subject_to(dq_lower <= dqk)
        opti.subject_to(dqk <= dq_upper)
        
        # Control (torque) bounds constraints
        opti.subject_to(tau_lower <= Tau[:, k])
        opti.subject_to(Tau[:, k] <= tau_upper)
        
        # Compute end-effector velocity using forward kinematics:
        # p3 = end_effector_pos(qk)
        J_p3 = J_p3_fun(qk)  # 2x3 Jacobian
        v_ee = cs.mtimes(J_p3, dqk) # end-effector velocity (2x1)

        # Compute the magnitude of the end-effector velocity
        epsilon = 1e-6
        v_ee_mag = cs.sqrt(cs.sumsqr(v_ee) + epsilon)

        
        # Cost for velocity tracking (squared error)
        cost_v = cs.sumsqr(v_ee_mag - v_des)
        # cost_v = -cs.sumsqr(v_ee)

        # TODO change the cost_v to pure velocity
        # TODO set a specific inertia value instead of just maximizing
        
        
        # M00 = M_k[0, 0]
        # want higher M00 towards the end of the trajectory
        # cost_M00 = - M00 * (k/N)**4
        
        # Optional cost on torque effort (to avoid unrealistic inputs)
        # cost_tau = cs.sumsqr(Tau[:, k])
        cost_tau = 0
        
        # Accumulate cost for this time step
        total_cost = total_cost + weight_v * cost_v + weight_tau * cost_tau
        if k < N - 1:
            tau_k_next = Tau[:, k+1]
            tau_diff = tau_k_next - tau_k
            cost_tau_smooth = cs.sumsqr(tau_diff)
            total_cost += weight_tau_smooth * cost_tau_smooth

    # Add a terminal cost on the final inertia value
    # Try #1
    # M00_N = M_fun(Q[:, -1])[0, 0]
    # cost_M00_N = - M00_N
    # total_cost = total_cost + weight_M00 * cost_M00_N

    # Try #2
    det_M_N = det_M_fun(Q[:, -1])
    cost_det_M_N = -det_M_N
    total_cost = total_cost + weight_M00 * cost_det_M_N

    # Also enforce bounds on the final state
    opti.subject_to(q_lower <= Q[:, N])
    opti.subject_to(Q[:, N] <= q_upper)
    opti.subject_to(dq_lower <= dQ[:, N])
    opti.subject_to(dQ[:, N] <= dq_upper)
    
    # Set the objective: minimize total cost over the trajectory
    opti.minimize(total_cost)
    
    # Set up the solver (e.g. IPOPT)
    p_opts = {"print_time": False}
    s_opts = {"print_level": 0}
    opti.solver('ipopt', p_opts, s_opts)
    
    # Solve the NLP
    sol = opti.solve()
    
    # Extract the solution trajectories
    q_opt = sol.value(Q)
    dq_opt = sol.value(dQ)
    tau_opt = sol.value(Tau)
    
    return {'q': q_opt, 'dq': dq_opt, 'tau': tau_opt, 'cost': sol.value(total_cost),
            't_f': T, 'm': m, 'l': l, 'r': r,}


def optimize_trajectory_cartesian_accel(q_f, v_f, m, l, r,
                        q_lower, q_upper, dq_lower, dq_upper,
                        tau_lower, tau_upper,
                        T, N,
                        weight_v=1e-3, weight_xdd=1e-3,
                        weight_tau_smooth = 1e-3, weight_terminal = 1e-3):
    """
    Optimize a trajectory for the 3DOF planar arm.
    
    The trajectory ends q_f and dq_f, and over a time horizon T with N steps,
    the optimizer aims to drive the end-effector (p3) velocity toward v_f from zero
    
              
        
    Parameters:
      q_f    : (3x1 np.array) final joint angles.
      v_f      : (2d np.array) desired end-effector velocity.
      m, l, r    : lists/arrays of link masses, lengths, and center-of-mass distances.
      q_lower, q_upper: (3x1 arrays) joint limits.
      dq_lower, dq_upper: (3x1 arrays) joint velocity limits.
      tau_lower, tau_upper: (3x1 arrays) joint torque limits.
      T          : total time horizon.
      N          : number of discretization steps.

    Returns:
      sol : dictionary with optimal trajectories for q, dq, and tau.
    """
    dt = T / N

    # Define symbolic variables
    q_sym = cs.SX.sym('q', 3)
    dq_sym = cs.SX.sym('dq', 3)
    tau_sym = cs.SX.sym('tau', 3)

    # Symbolic dynamics
    M, C, _, _ = formulate_symbolic_dynamic_matrices(m, l, r, q_sym, dq_sym)
    J = compute_jacobian(q_sym, l)
    J_fun = cs.Function('J', [q_sym], [J])

    # Cartesian acceleration expression
    xdd_sym = compute_symbolic_cartesian_acceleration(M, C, J, q_sym, dq_sym, tau_sym)
    xdd_fun = cs.Function("xdd", [q_sym, dq_sym, tau_sym], [xdd_sym])

    # Create forward kinematics functions for each link
    T1 = forward_kinematics_homogeneous(q_sym, l, site=1)
    T2 = forward_kinematics_homogeneous(q_sym, l, site=2)
    T3 = forward_kinematics_homogeneous(q_sym, l, site=3)

    # Extract positions from transformation matrices
    p1 = T1[:2, 2]  # x,y position of link 1
    p2 = T2[:2, 2]  # x,y position of link 2
    p3 = T3[:2, 2]  # x,y position of link 3 (end-effector)

    # Create CasADi functions for each link's position
    fk1_fun = cs.Function('fk1', [q_sym], [p1])
    fk2_fun = cs.Function('fk2', [q_sym], [p2])
    fk3_fun = cs.Function('fk3', [q_sym], [p3])


    # Normalize v_f direction
    v_f = cs.DM(v_f).reshape((2, 1))
    v_f_dir = v_f / (cs.norm_2(v_f) + 1e-8)

    # Estimate dq_f from q_f and v_f
    J_f = J_fun(q_f)  # 2x3 Jacobian at q_f
    J_f_pinv = damping_pseudoinverse(J_f)
    dq_f = cs.DM(cs.mtimes(J_f_pinv, v_f)).full().flatten()

    # Create an Opti instance
    opti = cs.Opti()

    # Decision variables: state and control trajectories
    Q = opti.variable(3, N+1)   # joint angles trajectory
    dQ = opti.variable(3, N+1)  # joint velocities trajectory
    Tau = opti.variable(3, N)   # joint torques (control input)

    # # Set initial values
    # opti.set_initial(Q, np.tile(q_f.reshape(-1, 1), (1, N+1)))
    # opti.set_initial(dQ, np.tile(dq_f.reshape(-1, 1), (1, N+1)))
    # opti.set_initial(Tau, np.zeros((3, N)))

    # Set final conditions
    opti.subject_to(Q[:, -1] == q_f)
    opti.subject_to(dQ[:, -1] == dq_f)


    # Cost accumulator
    total_cost = 0

    # sense if peak velocity is reached
    is_v_peak = False

    # Reverse-time integration loop
    for k in reversed(range(N)):
        q_next = Q[:, k+1]
        dq_next = dQ[:, k+1]
        tau_k = Tau[:, k]

        # Dynamics
        M_k = cs.Function('M', [q_sym], [M])(q_next)
        C_k = cs.Function('C', [q_sym, dq_sym], [C])(q_next, dq_next)
        ddq_k = cs.solve(M_k, tau_k - C_k @ dq_next)

        # Reverse Euler integration
        opti.subject_to(Q[:, k] == q_next - dt * dq_next)
        opti.subject_to(dQ[:, k] == dq_next - dt * ddq_k)

        # Enforce joint, velocity, torque limits
        opti.subject_to(q_lower <= Q[:, k])
        opti.subject_to(Q[:, k] <= q_upper)
        opti.subject_to(dq_lower <= dQ[:, k])
        opti.subject_to(dQ[:, k] <= dq_upper)
        opti.subject_to(tau_lower <= tau_k)
        opti.subject_to(tau_k <= tau_upper)

        # adding constraints on keeping each joint cartiesian position >=0
        # Add position constraints for each link
        p1_k = fk1_fun(q_next)
        p2_k = fk2_fun(q_next)
        p3_k = fk3_fun(q_next)
        
        # Constrain y positions to be above 0 (or any desired height)
        opti.subject_to(p1_k[1] >= 0)  # Link 1 y-position
        opti.subject_to(p2_k[1] >= 0)  # Link 2 y-position
        opti.subject_to(p3_k[1] >= 0)  # Link 3 y-position


        # smoothness of the joint velocity 
        # This will certainly make the cirve smoother but away from bang-bang
        # dq_k_next = dQ[:, k+1]
        # dq_diff = dq_k_next - dQ[:, k]
        # cost_dq_smooth = cs.sumsqr(dq_diff)
        # total_cost += weight_v_smooth * cost_dq_smooth

        if k < N - 1:
            # Minimize end-effector velocity in v_f direction
            J_k = J_fun(q_next)  # 2x3 Jacobian
            v_ee = cs.mtimes(J_k, dq_next) # end-effector velocity (2x1)
            total_cost += -cs.sumsqr(v_ee) * weight_v

            # Minimize Cartesian acceleration in v_f direction 
            # Not good, removed (Update 04/11/25)
            # xdd_k = xdd_fun(q_next, dq_next, tau_k)
            # xdd_proj = cs.dot(v_f_dir, xdd_k)
            # total_cost += xdd_proj * weight_xdd
            
            # TODO can add omega as a desired impact feature!!!

            # the acceleration direction constraint is not helping if only at the end
            # Instead of maximizing decel towards the direction of v_f at all time,
            # encourage normalized acceleration ?
            # F_ee = cs.mtimes(J_k.T, tau_k)  # end-effector force (2x1)
            # TODO the idea is that at some pose, same torque would result in much larger acceleration
            # than at other pose, so we want to keep the acceleration normalized
            xdd_k = xdd_fun(q_next, dq_next, tau_k)
            ramp = (k / N)**2  # or (k / N)**4 for stronger emphasis
            xdd_proj = cs.dot(v_f_dir, xdd_k)
            total_cost += xdd_proj * weight_xdd * ramp

            # Minimize jerkiness (smoothness) of the torque
            tau_k_next = Tau[:, k+1]
            tau_diff = tau_k_next - tau_k
            cost_tau_smooth = cs.sumsqr(tau_diff)
            total_cost += weight_tau_smooth * cost_tau_smooth

        # if k < N - 2:
        #     xdd_k_next = xdd_fun(Q[:, k+2], dQ[:, k+2], Tau[:, k+1])
        #     jerk = xdd_k_next - xdd_k
        #     total_cost += weight_tau_smooth * cs.sumsqr(jerk)

        if k == N - 1:

             # Minimize Cartesian acceleration in v_f direction
            xdd_k = xdd_fun(q_next, dq_next, tau_k)
            xdd_proj = cs.dot(v_f_dir, xdd_k)
            total_cost += xdd_proj * weight_terminal * N


            

    # Initial velocity zero constraint (start from rest)
    opti.subject_to(dQ[:, 0] == 0)

    # Set objective
    opti.minimize(total_cost)

    # Solve
    p_opts = {"print_time": True}
    s_opts = {"print_level": 1}
    opti.solver("ipopt", p_opts, s_opts)
    try:
        sol = opti.solve()
    except RuntimeError as e:
        print("[Solver failed]")
        opti.debug.show_infeasibilities()
        raise e

    return {
        "q": sol.value(Q),
        "dq": sol.value(dQ),
        "tau": sol.value(Tau),
        "cost": sol.value(total_cost),
        "t_f": T,
        "m": m,
        "l": l,
        "r": r,
    }


def time_optimal_trajectory_cartesian_accel(q_f, v_f, m, l, r,
                        q_lower, q_upper, dq_lower, dq_upper,
                        tau_lower, tau_upper,
                        t_lower, t_upper, t_init,
                        N,
                        weight_v=1e-3, weight_xdd=1e-3, weight_tf=0,
                        weight_tau_smooth = 1e-3, weight_terminal = 1e-1):
    """
    Optimize a trajectory for the 3DOF planar arm.
    
    The trajectory ends q_f and dq_f, and over a time horizon T with N steps,
    the optimizer aims to drive the end-effector (p3) velocity toward v_f from zero
    
              
        
    Parameters:
      q_f    : (3x1 np.array) final joint angles.
      v_f      : (2d np.array) desired end-effector velocity.
      m, l, r    : lists/arrays of link masses, lengths, and center-of-mass distances.
      q_lower, q_upper: (3x1 arrays) joint limits.
      dq_lower, dq_upper: (3x1 arrays) joint velocity limits.
      tau_lower, tau_upper: (3x1 arrays) joint torque limits.
      T          : total time horizon.
      N          : number of discretization steps.

    Returns:
      sol : dictionary with optimal trajectories for q, dq, and tau.
    """

    # Define symbolic variables
    q_sym = cs.SX.sym('q', 3)
    dq_sym = cs.SX.sym('dq', 3)
    tau_sym = cs.SX.sym('tau', 3)

    # Symbolic dynamics
    M, C, _, _ = formulate_symbolic_dynamic_matrices(m, l, r, q_sym, dq_sym)
    J = compute_jacobian(q_sym, l)
    J_fun = cs.Function('J', [q_sym], [J])

    # Cartesian acceleration expression
    xdd_sym = compute_symbolic_cartesian_acceleration(M, C, J, q_sym, dq_sym, tau_sym)
    xdd_fun = cs.Function("xdd", [q_sym, dq_sym, tau_sym], [xdd_sym])

    # Normalize v_f direction
    v_f = cs.DM(v_f).reshape((2, 1))
    v_f_dir = v_f / (cs.norm_2(v_f) + 1e-8)

    # Estimate dq_f from q_f and v_f
    J_f = J_fun(q_f)  # 2x3 Jacobian at q_f
    J_f_pinv = damping_pseudoinverse(J_f)
    dq_f = cs.DM(cs.mtimes(J_f_pinv, v_f)).full().flatten()

    # Create an Opti instance
    opti = cs.Opti()

    # Decision variables: time
    T_var = opti.variable()  # Total time variable
    dt = T_var / N  # Time step

    # Decision variables: state and control trajectories
    Q = opti.variable(3, N+1)   # joint angles trajectory
    dQ = opti.variable(3, N+1)  # joint velocities trajectory
    Tau = opti.variable(3, N)   # joint torques (control input)

    # Set initial values
    opti.set_initial(Q, np.tile(q_f.reshape(-1, 1), (1, N+1)))
    opti.set_initial(dQ, np.tile(dq_f.reshape(-1, 1), (1, N+1)))
    opti.set_initial(Tau, np.zeros((3, N)))
    opti.set_initial(T_var, t_init)  # Initial guess for total time

    # Set final conditions
    opti.subject_to(Q[:, -1] == q_f)
    opti.subject_to(dQ[:, -1] == dq_f)
    opti.subject_to(T_var >= t_lower)  # Minimum time constraint
    opti.subject_to(T_var <= t_upper)  # Maximum time constraint

    p_opts = {"print_time": True}
    s_opts = {"print_level": 5}  # Increase to a higher level for more detailed output
    opti.solver('ipopt', p_opts, s_opts)

    # Cost accumulator
    total_cost = 0

    # Reverse-time integration loop
    for k in reversed(range(N)):
        q_next = Q[:, k+1]
        dq_next = dQ[:, k+1]
        tau_k = Tau[:, k]

        # Dynamics
        M_k = cs.Function('M', [q_sym], [M])(q_next)
        C_k = cs.Function('C', [q_sym, dq_sym], [C])(q_next, dq_next)
        ddq_k = cs.solve(M_k, tau_k - C_k @ dq_next)

        # Reverse Euler integration
        opti.subject_to(Q[:, k] == q_next - dt * dq_next)
        opti.subject_to(dQ[:, k] == dq_next - dt * ddq_k)

        # Enforce joint, velocity, torque limits
        opti.subject_to(q_lower <= Q[:, k])
        opti.subject_to(Q[:, k] <= q_upper)
        opti.subject_to(dq_lower <= dQ[:, k])
        opti.subject_to(dQ[:, k] <= dq_upper)
        opti.subject_to(tau_lower <= tau_k)
        opti.subject_to(tau_k <= tau_upper)


        
        if k < N - 1:
            # Minimize Cartesian acceleration in v_f direction
            xdd_k = xdd_fun(q_next, dq_next, tau_k)
            xdd_proj = cs.dot(v_f_dir, xdd_k)
            total_cost += xdd_proj * weight_xdd

            # Minimize end-effector velocity in v_f direction
            J_k = J_fun(q_next)  # 2x3 Jacobian
            v_ee = cs.mtimes(J_k, dq_next) # end-effector velocity (2x1)
            total_cost += -cs.sumsqr(v_ee) * weight_v
        if k == N - 1:

             # Minimize Cartesian acceleration in v_f direction
            xdd_k = xdd_fun(q_next, dq_next, tau_k)
            xdd_proj = cs.dot(v_f_dir, xdd_k)
            total_cost += xdd_proj * weight_terminal

            # smoothness of the joint velocity right before the end
            dq_k_next = dQ[:, k+1]
            dq_diff = dq_k_next - dQ[:, k]
            cost_dq_smooth = cs.sumsqr(dq_diff)
            total_cost += weight_terminal * cost_dq_smooth

        

        # Minimize jerkiness (smoothness) of the torque
        if k < N - 1:
            tau_k_next = Tau[:, k+1]
            tau_diff = tau_k_next - tau_k
            cost_tau_smooth = cs.sumsqr(tau_diff)
            total_cost += weight_tau_smooth * cost_tau_smooth

    # Initial velocity zero constraint (start from rest)
    opti.subject_to(dQ[:, 0] == 0)

    # Set objective
    opti.minimize(total_cost + T_var*weight_tf)

    # Solve
    p_opts = {"print_time": True}
    s_opts = {"print_level": 5}
    opti.solver("ipopt", p_opts, s_opts)
    try:
        sol = opti.solve()
    except RuntimeError as e:
        print("[Solver failed]")
        opti.debug.show_infeasibilities()
        raise e

    return {
        "q": sol.value(Q),
        "dq": sol.value(dQ),
        "tau": sol.value(Tau),
        "t_f": sol.value(T_var),
        "cost": sol.value(total_cost),
        "m": m,
        "l": l,
        "r": r,
    }


def optimize_trajectory_cartesian_accel_flex_pose(p_f, v_f, m, l, r,
                        q_lower, q_upper, dq_lower, dq_upper,
                        tau_lower, tau_upper,
                        T, N,
                        weight_v=1e-3, weight_xdd=1e-3,
                        weight_tau_smooth = 1e-3, weight_terminal = 1e-3,
                        boundary_epsilon=1e-6):
    """
    Optimize a trajectory for the 3DOF planar arm.
    
    The trajectory ends q_f and dq_f, and over a time horizon T with N steps,
    the optimizer aims to drive the end-effector (p3) velocity toward v_f from zero

    This take final ee p_f and v_f as input, and optimize the trajectory
    
              
        
    Parameters:
      q_f    : (3x1 np.array) final joint angles.
      v_f      : (2d np.array) desired end-effector velocity.
      m, l, r    : lists/arrays of link masses, lengths, and center-of-mass distances.
      q_lower, q_upper: (3x1 arrays) joint limits.
      dq_lower, dq_upper: (3x1 arrays) joint velocity limits.
      tau_lower, tau_upper: (3x1 arrays) joint torque limits.
      T          : total time horizon.
      N          : number of discretization steps.

    Returns:
      sol : dictionary with optimal trajectories for q, dq, and tau.
    """
    dt = T / N

    # Define symbolic variables
    q_sym = cs.SX.sym('q', 3)
    dq_sym = cs.SX.sym('dq', 3)
    tau_sym = cs.SX.sym('tau', 3)

    # Symbolic Forward kinematics
    FK_fun = cs.Function("FK", [q_sym], [end_effector_position(q_sym, l)])
    
    # Create forward kinematics functions for each link
    T1 = forward_kinematics_homogeneous(q_sym, l, site=1)
    T2 = forward_kinematics_homogeneous(q_sym, l, site=2)
    T3 = forward_kinematics_homogeneous(q_sym, l, site=3)

    # Extract positions from transformation matrices
    p1 = T1[:2, 2]  # x,y position of link 1
    p2 = T2[:2, 2]  # x,y position of link 2
    p3 = T3[:2, 2]  # x,y position of link 3 (end-effector)

    # Create CasADi functions for each link's position
    fk1_fun = cs.Function('fk1', [q_sym], [p1])
    fk2_fun = cs.Function('fk2', [q_sym], [p2])
    fk3_fun = cs.Function('fk3', [q_sym], [p3])

    # Symbolic dynamics
    M, C, _, _ = formulate_symbolic_dynamic_matrices(m, l, r, q_sym, dq_sym)
    J = compute_jacobian(q_sym, l)
    J_fun = cs.Function('J', [q_sym], [J])

    # Cartesian acceleration expression
    xdd_sym = compute_symbolic_cartesian_acceleration(M, C, J, q_sym, dq_sym, tau_sym)
    xdd_fun = cs.Function("xdd", [q_sym, dq_sym, tau_sym], [xdd_sym])

    # Normalize v_f direction
    v_f = cs.DM(v_f).reshape((2, 1))
    v_f_dir = v_f / (cs.norm_2(v_f) + 1e-8)


    # Create an Opti instance
    opti = cs.Opti()

    # Decision variables: state and control trajectories
    Q = opti.variable(3, N+1)   # joint angles trajectory
    dQ = opti.variable(3, N+1)  # joint velocities trajectory
    Tau = opti.variable(3, N)   # joint torques (control input)


    # # Set initial values
    # opti.set_initial(Q, np.tile(q_f.reshape(-1, 1), (1, N+1)))
    # opti.set_initial(dQ, np.tile(dq_f.reshape(-1, 1), (1, N+1)))
    # opti.set_initial(Tau, np.zeros((3, N)))

    # Set final conditions
    eef_final_pos = FK_fun(Q[:, -1])
    opti.subject_to(eef_final_pos == p_f)
    J_f = J_fun(Q[:, -1])  # 2x3 Jacobian at q_f
    opti.subject_to(J_f @ dQ[:, -1] == v_f)


    # Cost accumulator
    total_cost = 0

    # sense if peak velocity is reached
    is_v_peak = False

    # Reverse-time integration loop
    for k in reversed(range(N)):
        q_next = Q[:, k+1]
        dq_next = dQ[:, k+1]
        tau_k = Tau[:, k]

        # Dynamics
        M_k = cs.Function('M', [q_sym], [M])(q_next)
        C_k = cs.Function('C', [q_sym, dq_sym], [C])(q_next, dq_next)
        ddq_k = cs.solve(M_k, tau_k - C_k @ dq_next)

        # Reverse Euler integration
        opti.subject_to(Q[:, k] == q_next - dt * dq_next)
        opti.subject_to(dQ[:, k] == dq_next - dt * ddq_k)

        # Enforce joint, velocity, torque limits
        opti.subject_to(q_lower <= Q[:, k])
        opti.subject_to(Q[:, k] <= q_upper)
        opti.subject_to(dq_lower <= dQ[:, k])
        opti.subject_to(dQ[:, k] <= dq_upper)
        opti.subject_to(tau_lower <= tau_k)
        opti.subject_to(tau_k <= tau_upper)

        # adding constraints on keeping each joint cartiesian position >=0
        # Add position constraints for each link
        p1_k = fk1_fun(q_next)
        p2_k = fk2_fun(q_next)
        p3_k = fk3_fun(q_next)
        
        # Constrain y positions to be above 0 (or any desired height)
        opti.subject_to(p1_k[1] >= 0-boundary_epsilon)  # Link 1 y-position
        opti.subject_to(p2_k[1] >= 0-boundary_epsilon)  # Link 2 y-position
        opti.subject_to(p3_k[1] >= 0-boundary_epsilon)  # Link 3 y-position


        # smoothness of the joint velocity 
        # This will certainly make the cirve smoother but away from bang-bang
        # dq_k_next = dQ[:, k+1]
        # dq_diff = dq_k_next - dQ[:, k]
        # cost_dq_smooth = cs.sumsqr(dq_diff)
        # total_cost += weight_v_smooth * cost_dq_smooth

        if k < N - 1:
            # Minimize end-effector velocity in v_f direction
            J_k = J_fun(q_next)  # 2x3 Jacobian
            v_ee = cs.mtimes(J_k, dq_next) # end-effector velocity (2x1)
            total_cost += -cs.sumsqr(v_ee) * weight_v

            # Minimize Cartesian acceleration in v_f direction 
            # Not good, removed (Update 04/11/25)
            # xdd_k = xdd_fun(q_next, dq_next, tau_k)
            # xdd_proj = cs.dot(v_f_dir, xdd_k)
            # total_cost += xdd_proj * weight_xdd

            # the acceleration direction constraint is not helping if only at the end
            # Instead of maximizing decel towards the direction of v_f at all time,
            # encourage 
            xdd_k = xdd_fun(q_next, dq_next, tau_k)
            ramp = (k / N)**2  # or (k / N)**4 for stronger emphasis
            xdd_proj = cs.dot(v_f_dir, xdd_k)
            total_cost += xdd_proj * weight_xdd * ramp

            # # make ee angular velocity zero
            # J_rot = cs.DM([1,1,1]).T
            # ee_omega = cs.mtimes(J_rot, dq_next) # end-effector angular velocity (1x1)
            # ee_omega_sq = cs.sumsqr(ee_omega)
            # total_cost += ee_omega_sq * weight_xdd * ramp

            # Minimize jerkiness (smoothness) of the torque
            tau_k_next = Tau[:, k+1]
            tau_diff = tau_k_next - tau_k
            cost_tau_smooth = cs.sumsqr(tau_diff)
            total_cost += weight_tau_smooth * cost_tau_smooth

        # if k < N - 2:
        #     xdd_k_next = xdd_fun(Q[:, k+2], dQ[:, k+2], Tau[:, k+1])
        #     jerk = xdd_k_next - xdd_k
        #     total_cost += weight_tau_smooth * cs.sumsqr(jerk)

        if k == N - 1:

            # maximize Cartesian deceleration in v_f direction
            xdd_k = xdd_fun(q_next, dq_next, tau_k)
            xdd_proj = cs.dot(v_f_dir, xdd_k)
            total_cost += xdd_proj * weight_terminal * N

            # make ee angular velocity zero
            J_rot = cs.DM([1,1,1]).T
            ee_omega = cs.mtimes(J_rot, dq_next) # end-effector angular velocity (1x1)
            ee_omega_sq = cs.sumsqr(ee_omega)
            total_cost += weight_terminal * ee_omega_sq * N

            

    # Initial velocity zero constraint (start from rest)
    opti.subject_to(dQ[:, 0] == 0)

    # Set objective
    opti.minimize(total_cost)

    # Solve
    # p_opts = {"print_time": True}
    # s_opts = {"print_level": 5}
    p_opts = {"print_time": True}
    s_opts = {"print_level": 1}
    opti.solver("ipopt", p_opts, s_opts)
    try:
        sol = opti.solve()
    except RuntimeError as e:
        print("[Solver failed]")
        opti.debug.show_infeasibilities()
        raise e

    return {
        "q": sol.value(Q),
        "dq": sol.value(dQ),
        "tau": sol.value(Tau),
        "cost": sol.value(total_cost),
        "t_f": T,
        "m": m,
        "l": l,
        "r": r,
    }

# ================================ Main Execution ==============================
if __name__ == "__main__":
    # Arm parameters (example values)
    m = [0.1/3, 0.1/3, 0.1/3]
    l = [0.5, 0.5, 0.5]
    r = [l[0]/20, l[1]/20, l[2]/20]
    
    # Initial conditions
    q_start = np.array([0, np.pi*0.3, np.pi*0.5])
    dq_start = np.array([0.0, 0.0, 0.0])
    
    # Desired end-effector velocity (2D)
    # v_des = np.array([0.5, 0.0])
    # optimize for only magnitude of velocity
    v_ee_mag_des = 11.0
    
    # Bounds
    q_lower = -4/5*np.pi * np.ones(3)
    q_upper = 4/5*np.pi * np.ones(3)
    dq_lower = -2*np.pi * np.ones(3)
    dq_upper = 2*np.pi * np.ones(3)
    tau_lower = -10*np.ones(3)
    tau_upper = 10*np.ones(3)
    
    # Time horizon and discretization steps
    T = 0.5  # seconds
    N = 30
    
    # Call the optimizer
    solution = optimize_trajectory(q_start, dq_start, v_ee_mag_des, m, l, r,
                                   q_lower, q_upper, dq_lower, dq_upper,
                                   tau_lower, tau_upper,
                                   T, N)
    
    # Display and save the solution
    display_and_save_solution(solution, 'opt_trajectory.npy')

