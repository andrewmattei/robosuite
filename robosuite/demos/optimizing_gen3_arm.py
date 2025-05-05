# kinova_casadi_utils.py

import pinocchio as pin
from pinocchio.utils import zero
import casadi as cs
import numpy as np
from scipy.interpolate import interp1d
import os

# ------------------------ Model loading & CasADi builders -----------------------

def load_kinova_model(urdf_path, package_dirs=None):
    """
    Load the Kinova Gen3 URDF into a Pinocchio Model/Data pair.
    """
    if package_dirs is None:
        package_dirs = []
    model = pin.buildModelFromUrdf(urdf_path, package_dirs)
    data = model.createData()
    return model, data

def build_casadi_kinematics_dynamics(model, data, frame_name):
    """
    Build CasADi functions for FK, Jacobian, dynamics (M, C, G) for a given end‐effector frame.
    Returns:
      fk_fun(q)       → 4×4 homogeneous transform
      pos_fun(q)      → 3×1 end‐effector position
      jac_fun(q)      → 6×nv spatial Jacobian (LOCAL_WORLD_ALIGNED)
      M_fun(q)        → nv×nv mass matrix
      C_fun(q, dq)    → nv×nv Coriolis/centrifugal matrix
      G_fun(q)        → nv×1 gravity torque vector
    """
    nq, nv = model.nq, model.nv

    # symbolic variables
    q  = cs.SX.sym('q',  nq)
    dq = cs.SX.sym('dq', nv)

    # Forward kinematics & Jacobian
    pin.forwardKinematics(model, data, q, dq)
    pin.updateFramePlacements(model, data)
    frame_id = model.getFrameId(frame_name)

    # Homogeneous transform
    oMf = data.oMf[frame_id].homogeneous
    fk_fun  = cs.Function('fk',  [q], [oMf])

    # Position only
    pos = oMf[0:3, 3]
    pos_fun = cs.Function('pos', [q], [pos])

    # Spatial Jacobian in world frame
    pin.computeJointJacobians(model, data, q)
    pin.updateFramePlacements(model, data)
    J6 = pin.getFrameJacobian(model, data, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    jac_fun = cs.Function('J6', [q], [J6])

    # Dynamics: M, C, G
    M_mat = pin.crba(model, data, q)
    C_mat = pin.computeCoriolisMatrix(model, data, q, dq)
    # Gravity torques via RNEA with zero velocities/accelerations
    zero_dq = cs.SX.zeros(nv)
    zero_ddq = cs.SX.zeros(nv)
    G_vec = pin.rnea(model, data, q, zero_dq, zero_ddq)

    M_fun = cs.Function('M', [q],    [M_mat])
    C_fun = cs.Function('C', [q, dq], [C_mat])
    G_fun = cs.Function('G', [q],    [G_vec])

    return fk_fun, pos_fun, jac_fun, M_fun, C_fun, G_fun

# ------------------------ Basic kinematic helpers ------------------------

def forward_kinematics_homogeneous(q, fk_fun):
    """
    Evaluate the 4×4 homogeneous end-effector transform at joint configuration q.
    """
    return fk_fun(q)

def end_effector_position(q, pos_fun):
    """
    Evaluate just the 3D position of the end-effector.
    """
    return pos_fun(q)

def inverse_kinematics_casadi(target_pose, fk_fun, q_init=None, lb=None, ub=None):
    """
    Solve IK via a CasADi NLP: minimize ||fk(q) - target_pose||₂.
    Optionally enforce q ∈ [lb, ub].
    """
    nq = fk_fun.size1_in(0)
    q = cs.SX.sym('q', nq)

    # objective: Frobenius norm of homogeneous error
    T_err = fk_fun(q) - target_pose
    obj = cs.norm_2(cs.reshape(T_err, -1, 1))

    nlp = {'x': q, 'f': obj}
    opts = {'ipopt.print_level':0, 'print_time':0}
    solver = cs.nlpsol('ik', 'ipopt', nlp, opts)

    x0 = q_init if q_init is not None else np.zeros(nq)
    bounds = {}
    if lb is not None and ub is not None:
        bounds = {'lbx': lb, 'ubx': ub}

    sol = solver(x0=x0, **bounds)
    return sol['x']

def compute_jacobian(q, jac_fun):
    """
    Evaluate the spatial Jacobian at q.
    """
    return jac_fun(q)

# ------------------------ Dynamics & acceleration ------------------------

def formulate_symbolic_dynamic_matrices(model, data):
    """
    Build & return CasADi functions M(q), C(q,dq), G(q).
    """
    _, _, _, _, M_fun, C_fun, G_fun = build_casadi_kinematics_dynamics(model, data, frame_name=model.frames[-1].name)
    return M_fun, C_fun, G_fun

def compute_dynamics_matrices(M_fun, C_fun, q, dq):
    """
    Numeric evaluation of M and C at (q, dq).
    """
    M = M_fun(q)
    C = C_fun(q, dq)
    return M, C

def compute_singular_values(J):
    """
    Return vector of singular values of J via CasADi's SVD.
    """
    _, S, _ = cs.svd(J)
    return S

def damping_pseudoinverse(J, damping=1e-4):
    """
    Damped least-squares pseudoinverse: Jᵗ (J Jᵗ + λ² I)⁻¹.
    """
    m, n = J.size1(), J.size2()
    JJt = J @ J.T
    inv = cs.solve(JJt + (damping**2)*cs.eye(m), cs.eye(m))
    return J.T @ inv

def compute_symbolic_cartesian_acceleration(q, dq, tau, M_fun, C_fun, jac_fun, G_fun=None, damping=1e-4):
    """
    ẍ = J M⁻¹ (τ - C dq - G + M J⁺ ḊJ dq)
    """
    # pull out sizes
    nv = dq.size1()

    M = M_fun(q)
    C = C_fun(q, dq)
    G = G_fun(q) if G_fun is not None else cs.SX.zeros(nv)

    J = jac_fun(q)           # 6×nv
    # compute J̇
    Jdot = cs.SX.zeros(*J.size())
    for i in range(q.size1()):
        dJ_dqi = cs.jacobian(J[:, i], q)
        Jdot += dJ_dqi @ dq[i]

    J_pinv = damping_pseudoinverse(J, damping)
    acc_dyn = tau - C @ dq - G + M @ (J_pinv @ (Jdot @ dq))
    ddq = cs.solve(M, acc_dyn)
    xdd = J @ ddq
    return xdd

# ------------------------ Trajectory & linearization ------------------------

def linearize_dynamics_along_trajectory(T_opt, U_opt, Z_opt, M_fun, C_fun, G_fun=None):
    """
    Discrete‐time linearization about a (time, torque, state) trajectory.
    Returns lists of A_d, B_d matrices.
    """
    A_list, B_list = [], []
    z_sym   = cs.SX.sym('z', Z_opt.shape[0])
    tau_sym = cs.SX.sym('tau', U_opt.shape[0])

    def f_cont(z, tau):
        n = z.size1() // 2
        q  = z[:n]
        dq = z[n:]
        M = M_fun(q)
        C = C_fun(q, dq)
        G = G_fun(q) if G_fun is not None else cs.SX.zeros(n)
        ddq = cs.solve(M, tau - C @ dq - G)
        return cs.vertcat(dq, ddq)

    f_jacobian = cs.Function('f_jac', [z_sym, tau_sym],
                             [cs.vertcat(dq, ddq := cs.vertcat(dq, cs.solve(M_fun(z_sym[:n]),
                                                                                tau_sym - C_fun(z_sym[:n], z_sym[n:]) @ z_sym[n:] - (G_fun(z_sym[:n]) if G_fun else zero(n)))))])
    # (Above line is illustrative; you can instead build A_k_fun and B_k_fun below.)

    for k in range(len(T_opt)-1):
        dt = T_opt[k+1] - T_opt[k]
        z_k = Z_opt[:, k]
        u_k = U_opt[:, k]
        # continuous linearization
        A_k = cs.jacobian(f_cont(z_sym, tau_sym), z_sym)
        B_k = cs.jacobian(f_cont(z_sym, tau_sym), tau_sym)
        A_fun = cs.Function(f"A_{k}", [z_sym, tau_sym], [A_k])
        B_fun = cs.Function(f"B_{k}", [z_sym, tau_sym], [B_k])
        A_d = np.eye(Z_opt.shape[0]) + dt * A_fun(z_k, u_k).full()
        B_d = dt *              B_fun(z_k, u_k).full()
        A_list.append(A_d)
        B_list.append(B_d)

    return A_list, B_list

# ------------------------ Interpolation & I/O ------------------------

def match_trajectories(T_des, *args):
    """
    Reinterpolate any number of (T_i, Z_i) pairs at common times T_des.
    Usage: match_trajectories(T_des, T1, Z1, T2, Z2, ..., interp_type='linear')
    Returns list of Z_i matched to T_des.
    """
    results = []
    interp_type = 'linear'
    if isinstance(args[-1], str):
        interp_type = args[-1]
        pairs = args[:-1]
    else:
        pairs = args

    for i in range(0, len(pairs), 2):
        T = np.asarray(pairs[i]).ravel()
        Z = np.asarray(pairs[i+1])
        if T.size == 1 and np.allclose(T_des, T):
            result = Z
        else:
            f = interp1d(T, Z.T, kind=interp_type, axis=0,
                         bounds_error=False, fill_value="extrapolate")
            result = f(T_des).T
        results.append(result)
    return results

def save_solution_to_npy(solution: dict, filename: str):
    """
    Write a dict of NumPy arrays to .npy (via savez).
    """
    np.savez(filename, **solution)

def display_and_save_solution(solution: dict, filename: str=None):
    """
    (Optional) visualize q, dq, τ, end-effector velocity over time.
    """
    # This will be highly application‐specific. Stubbed here.
    raise NotImplementedError("Adapt plotting code from your planar version to 7DOF format.")

# ------------------------ Planar‐specific / Optimization stubs ------------------------

def sample_qf_given_y_line_casadi(*args, **kwargs):
    raise NotImplementedError("Planar‐only sampling—no direct 7DOF equivalent.")

def sample_pf_vf_grid(*args, **kwargs):
    raise NotImplementedError("Planar‐only sampling—no direct 7DOF equivalent.")

def optimize_trajectory(*args, **kwargs):
    raise NotImplementedError("Requires full OCP formulation; please re-derive for 7DOF Kinova.")

def optimize_trajectory_cartesian_accel(*args, **kwargs):
    raise NotImplementedError("Requires full OCP formulation; please re-derive for 7DOF Kinova.")

def time_optimal_trajectory_cartesian_accel(*args, **kwargs):
    raise NotImplementedError("Requires full time‐optimal control problem formulation.")

def optimize_trajectory_cartesian_accel_flex_pose(*args, **kwargs):
    raise NotImplementedError("Requires full OCP formulation; please re-derive for 7DOF Kinova.")

if __name__ == "__main__":
    # Example usage
    urdf_path = os.path.join(os.path.dirname(__file__), os.pardir, 'models', 'assets', 'robots',
                                'dual_kinova3', 'leonardo.urdf')
    # model, data = load_kinova_model(urdf_path)
    # fk_fun, pos_fun, jac_fun, M_fun, C_fun, G_fun = build_casadi_kinematics_dynamics(model, data, 'tool_frame')

    # Example FK
    
    # # Example IK
    # target_pose = np.eye(4)  # Replace with actual target pose
    # q_init = np.zeros(model.nq)
    # q_sol = inverse_kinematics_casadi(target_pose, fk_fun, q_init)
    # print("IK solution:", q_sol)