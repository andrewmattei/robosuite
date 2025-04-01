import numpy as np
import casadi as cs
import matplotlib.pyplot as plt
import casadi as ca
import os

# ============================== Helper Functions ==============================
def compute_jacobian(q, l):
    """Compute Jacobian for 3DOF planar arm"""
    # Assuming q is a CasADi MX matrix
    theta1 = q[0]
    theta2 = q[1]
    theta3 = q[2]

    # Create a 2x3 zero matrix for the Jacobian using MX instead of SX
    J = ca.MX.zeros(2, 3)

    # Fill in the Jacobian matrix with symbolic expressions for 2D position
    J[0, 0] = -l[0] * ca.sin(theta1) - l[1] * ca.sin(theta1 + theta2) - l[2] * ca.sin(theta1 + theta2 + theta3)
    J[0, 1] = -l[1] * ca.sin(theta1 + theta2) - l[2] * ca.sin(theta1 + theta2 + theta3)
    J[0, 2] = -l[2] * ca.sin(theta1 + theta2 + theta3)
    J[1, 0] = l[0] * ca.cos(theta1) + l[1] * ca.cos(theta1 + theta2) + l[2] * ca.cos(theta1 + theta2 + theta3)
    J[1, 1] = l[1] * ca.cos(theta1 + theta2) + l[2] * ca.cos(theta1 + theta2 + theta3)
    J[1, 2] = l[2] * ca.cos(theta1 + theta2 + theta3)

    return J


def formulate_symbolic_dynamic_matrices(m, l, r, q_sym=None, dq_sym=None):
    """Symbolic representation of M(q) and C(q, dq) for a 3DOF planar arm"""
    if q_sym is None or dq_sym is None:
        # Create symbolic variables
        q_sym = cs.MX.sym('q', 3, 1)   # Column vector (3x1)
        dq_sym = cs.MX.sym('dq', 3, 1) # Column vector (3x1)
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
    C = cs.MX.zeros(3, 3)
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


# ========================== Trajectory Optimization ==========================


def optimize_trajectory(q_start, dq_start, v_des, m, l, r,
                        q_lower, q_upper, dq_lower, dq_upper,
                        tau_lower, tau_upper,
                        T, N,
                        weight_v=0.001, weight_M00=10, weight_tau=0.001):
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
    q_sym = cs.MX.sym('q', 3)
    dq_sym = cs.MX.sym('dq', 3)
    M, C, M_fun, C_fun = formulate_symbolic_dynamic_matrices(m, l, r, q_sym, dq_sym)
    dM00_dq = cs.jacobian(M[0, 0], q_sym)
    dM00_dq_fun = cs.Function('dM00_dq', [q_sym], [dM00_dq])
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

    # Add a terminal cost on the final inertia value
    M00_N = M_fun(Q[:, -1])[0, 0]
    cost_M00_N = - M00_N
    total_cost = total_cost + weight_M00 * cost_M00_N

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
    
    return {'q': q_opt, 'dq': dq_opt, 'tau': tau_opt, 'cost': sol.value(total_cost)}

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
    v_ee_mag_des = 15.0
    
    # Bounds
    q_lower = -4/5*np.pi * np.ones(3)
    q_upper = 4/5*np.pi * np.ones(3)
    dq_lower = -2*np.pi * np.ones(3)
    dq_upper = 2*np.pi * np.ones(3)
    tau_lower = -10*np.ones(3)
    tau_upper = 10*np.ones(3)
    
    # Time horizon and discretization steps
    T = 0.1  # seconds
    N = 30
    
    # Call the optimizer
    solution = optimize_trajectory(q_start, dq_start, v_ee_mag_des, m, l, r,
                                   q_lower, q_upper, dq_lower, dq_upper,
                                   tau_lower, tau_upper,
                                   T, N)
    
    # Display the results
    num_steps = solution['q'].shape[1]
    v_ee = np.zeros((2, num_steps))
    q_sym = cs.MX.sym('q', 3)
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
    U_opt = np.hstack((solution['tau'], np.zeros((3,1))))
    T_opt = np.linspace(0, T, num_steps)
    [Z_dense] = match_trajectories(t_dense, T_opt, Z_opt)

    # Save the trajectory data
    trajectory_data = {
        'T_opt': T_opt,
        'Z_opt': Z_opt,
        'U_opt': U_opt,
    }
    # get current file location
    current_path = os.path.dirname(os.path.realpath(__file__))
    save_path = os.path.join(current_path, 'opt_trajectory.npy')
    np.save(save_path, trajectory_data)

    # plot torque, v_ee, and v_ee_mag
    plt.figure(figsize=(6, 8))
    plt.subplot(4, 1, 1)
    plt.plot(time[:-1], solution['tau'].T)
    plt.ylabel('Torque (Nm)')
    plt.subplot(4, 1, 2)
    plt.plot(time, v_ee_mag)
    plt.ylabel('End-effector velocity (m/s)')
    plt.subplot(4, 1, 3)
    plt.plot(time, v_ee[0, :], label='x')
    plt.plot(time, v_ee[1, :], label='y')
    plt.xlabel('Time (s)')
    plt.ylabel('End-effector velocity components (m/s)')
    plt.subplot(4, 1, 4)
    plt.plot(t_dense, Z_dense[0:3, :].T)
    plt.plot(time, solution['q'].T,'o')
    plt.xlabel('Time (s)')
    plt.tight_layout()
    plt.show()
    
