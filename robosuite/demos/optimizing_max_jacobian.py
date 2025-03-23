import numpy as np
import casadi as cs
import matplotlib.pyplot as plt
import casadi as ca

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

# ========================== Configuration Optimization ========================
def find_optimal_configurations(l, m, q_init):
    # Optimization 1: Maximize velocity capability (σ_max)
    q_vel = cs.MX.sym('q_vel', 3)
    J_vel = compute_jacobian(q_vel, l)  # Different symbolic variable
    sigma_max, _ = compute_singular_values(J_vel)
    
    # Optimization 2: Maximize force capability (1/σ_min)
    q_force = cs.MX.sym('q_force', 3)
    J_force = compute_jacobian(q_force, l)  # Separate symbolic variable
    _, sigma_min = compute_singular_values(J_force)
    
    # Solve independently
    solver_vel = cs.nlpsol('solver_vel', 'ipopt', {'x': q_vel, 'f': -sigma_max})
    sol_vel = solver_vel(x0=q_init)
    q_opt = sol_vel['x'].full().flatten()

    solver_force = cs.nlpsol('solver_force', 'ipopt', {'x': q_force, 'f': sigma_min})
    sol_force = solver_force(x0=q_init)
    q_start = sol_force['x'].full().flatten()

    return q_start, q_opt

# ========================== Trajectory Optimization ===========================
def time_optimal_trajectory(q_start, q_opt, m, l, tau_max):
    """Solve time-optimal trajectory problem"""
    opti = cs.Opti()
    T = opti.variable()                # Total time
    N = 50                             # Number of control intervals
    X = opti.variable(6, N+1)          # States [q; dq]
    U = opti.variable(3, N)            # Controls
    
    # Parameters
    dt = T/N
    m_sym = cs.MX.sym('m', 3)
    l_sym = cs.MX.sym('l', 3)
    
    # Dynamics constraints
    for k in range(N):
        q_k = X[0:3, k]
        dq_k = X[3:6, k]
        tau_k = U[:, k]
        
        # Compute dynamics
        M, C = compute_dynamics_matrices(q_k, dq_k, m_sym, l_sym)
        ddq_k = cs.solve(M, tau_k - C @ dq_k)
        
        # State propagation
        opti.subject_to(X[0:3, k+1] == X[0:3, k] + dt * dq_k)
        opti.subject_to(X[3:6, k+1] == X[3:6, k] + dt * ddq_k)
    
    # Boundary conditions
    opti.subject_to(X[:, 0] == cs.vertcat(q_start, 0, 0, 0))
    opti.subject_to(X[:, -1] == cs.vertcat(q_opt, 0, 0, 0))
    
    # Control constraints
    opti.subject_to(opti.bounded(-tau_max, U, tau_max))
    
    # Time constraints
    opti.subject_to(T >= 0.1)  # Minimum time
    
    # Objective: minimize time
    opti.minimize(T)
    
    # Solve
    opti.set_value(m_sym, m)
    opti.set_value(l_sym, l)
    opti.solver('ipopt')
    sol = opti.solve()
    
    return sol.value(X), sol.value(U), sol.value(T)

# ================================ Main Execution ==============================
if __name__ == "__main__":
    # Robot parameters
    m = [1.0, 1.0, 1.0]          # Link masses [kg]
    l = [1.0, 1.0, 1.0]           # Link lengths [m]
    tau_max = np.array([50.0, 50.0, 50.0])  # Torque limits [Nm]
    
    # Initial guess
    q_init = np.array([np.pi/4, np.pi/4, np.pi/4])
    
    # Find optimal configurations
    q_start, q_opt = find_optimal_configurations(l, m, q_init)
    print(f"q_start: {np.degrees(q_start)}, q_opt: {np.degrees(q_opt)}")
    
    # Compute optimal trajectory
    X_opt, U_opt, T_opt = time_optimal_trajectory(q_start, q_opt, m, l, tau_max)
    
    # Visualization
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(X_opt[0,:], label='theta1')
    plt.plot(X_opt[1,:], label='theta2')
    plt.plot(X_opt[2,:], label='theta3')
    plt.title('Optimal Joint Angles')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(U_opt[0,:], label='tau1')
    plt.plot(U_opt[1,:], label='tau2')
    plt.plot(U_opt[2,:], label='tau3')
    plt.title('Optimal Torques')
    plt.legend()
    plt.tight_layout()
    plt.show()