# kinova_casadi_utils.py

import pinocchio as pin
import pinocchio.casadi as cpin
import casadi as cs
import numpy as np
np.set_printoptions(precision=3, suppress=True)
from scipy.interpolate import interp1d
import os

# ------------------------ Model loading ------------------------

def load_kinova_model(urdf_path):
    """
    Load the Kinova Gen3 URDF into a Pinocchio Model/Data pair.
    """
    model = pin.buildModelFromUrdf(urdf_path)
    data = model.createData()
    return model, data

# ------------------------ CasADi‐symbolic builders ------------------------

def standard_to_pinocchio_sx(model, q_std):
    """
    Convert standard joint angles (7×1 SX) to Pinocchio configuration (nq×1 SX).
    """
    q_pin = cs.SX.zeros(model.nq)
    for joint in model.joints[1:]:  # skip universe joint
        idx_v = joint.idx_v
        idx_q = joint.idx_q
        if joint.nq == 1:
            q_pin[idx_q] = q_std[idx_v]
        else:
            # multi-dof joints (cos, sin)
            angle = q_std[idx_v]
            q_pin[idx_q]   = cs.cos(angle)
            q_pin[idx_q+1] = cs.sin(angle)
    return q_pin


def pinocchio_to_standard_sx(model, q_pin):
    """
    Convert Pinocchio configuration (nq×1 SX) back to standard angles (7×1 SX).
    """
    q_std = cs.SX.zeros(model.nv)
    for joint in model.joints[1:]:
        idx_v = joint.idx_v
        idx_q = joint.idx_q
        if joint.nq == 1:
            q_std[idx_v] = q_pin[idx_q]
        else:
            # reconstruct angle from cos/sin
            c = q_pin[idx_q]; s = q_pin[idx_q+1]
            q_std[idx_v] = cs.atan2(s, c)
    return q_std


def build_casadi_kinematics_dynamics(model, data, frame_name):
    """
    Build CasADi functions (SX) for FK, Jacobian, dynamics (M, C, G) 
    accepting standard 7-DOF q_std and dq_std.
    """
    # Template model for SX
    cmodel = cpin.Model(model)
    cdata  = cmodel.createData()

    # Symbolic standard-state variables
    q_std   = cs.SX.sym('q',   model.nv)
    dq_std  = cs.SX.sym('dq',  model.nv)

    # Convert to Pinocchio internal config
    q_pin = standard_to_pinocchio_sx(model, q_std)

    # Forward kinematics
    cpin.forwardKinematics(cmodel, cdata, q_pin, dq_std, cs.SX.zeros(model.nv))
    cpin.updateFramePlacements(cmodel, cdata)
    frame_id = model.getFrameId(frame_name)
    oMf      = cdata.oMf[frame_id].homogeneous  # 4×4 SX
    fk_fun   = cs.Function('fk',   [q_std], [oMf])
    pos_fun  = cs.Function('pos',  [q_std], [oMf[0:3, 3]])

    # Spatial Jacobian
    cpin.computeJointJacobians(cmodel, cdata, q_pin)
    cpin.updateFramePlacements(cmodel, cdata)
    J6       = cpin.getFrameJacobian(
                  cmodel, cdata, frame_id,
                  pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
              )  # 6×nv SX
    jac_fun  = cs.Function('J6',  [q_std], [J6])

    # Dynamics: M, C, G
    M_mat = cpin.crba(cmodel, cdata, q_pin)
    C_mat = cpin.computeCoriolisMatrix(cmodel, cdata, q_pin, dq_std)
    G_vec = cpin.rnea(cmodel, cdata, q_pin,
                      cs.SX.zeros(model.nv),
                      cs.SX.zeros(model.nv))

    M_fun = cs.Function('M',   [q_std],       [M_mat])
    C_fun = cs.Function('C',   [q_std, dq_std], [C_mat])
    G_fun = cs.Function('G',   [q_std],       [G_vec])

    return fk_fun, pos_fun, jac_fun, M_fun, C_fun, G_fun


# ------------------------ Kinematic helpers ------------------------
def standard_to_pinocchio(model, q: np.ndarray) -> np.ndarray:
    """Convert standard joint angles (rad) to Pinocchio joint angles"""
    q_pin = np.zeros(model.nq)
    for i, j in enumerate(model.joints[1:]):
        if j.nq == 1:
            q_pin[j.idx_q] = q[j.idx_v]
        else:
            # cos(theta), sin(theta)
            q_pin[j.idx_q:j.idx_q+2] = np.array([np.cos(q[j.idx_v]), np.sin(q[j.idx_v])])
    return q_pin


def pinocchio_to_standard(model, q_pin: np.ndarray) -> np.ndarray:
    """Convert Pinocchio joint angles to standard joint angles (rad)"""
    q = np.zeros(model.nv)
    for i, j in enumerate(model.joints[1:]):
        if j.nq == 1:
            q[j.idx_v] = q_pin[j.idx_q]
        else:
            q_back = np.arctan2(q_pin[j.idx_q+1], q_pin[j.idx_q])
            q[j.idx_v] = q_back + 2*np.pi if q_back < 0 else q_back
    return q


def forward_kinematics_homogeneous(q, fk_fun):
    return fk_fun(q)

def end_effector_position(q, pos_fun):
    return pos_fun(q)

def inverse_kinematics_casadi(
    target_pose, fk_fun,
    q_init=None, lb=None, ub=None,
    tol=1e-9, max_iter=500
):
    """
    Solve IK for `fk_fun(q) ≈ target_pose` with high precision and no IPOPT banner.\n
    Args:
      target_pose : 4×4 SX or DM homogeneous matrix
      fk_fun      : CasADi Function mapping q→oMf
      q_init      : (nq,) initial guess
      lb, ub      : optional bounds on q
      tol         : desired NLP tolerance (default 1e-8)
      max_iter    : maximum IPOPT iterations (default 500)
    Returns:
      q_sol       : (nq,) SX or DM solution
    """
    nq = fk_fun.size1_in(0)
    q  = cs.SX.sym('q', nq)

    # objective: Frobenius‐norm of pose error
    T_err = fk_fun(q) - target_pose
    obj   = cs.norm_2(cs.reshape(T_err, -1, 1))

    nlp = {'x': q, 'f': obj}
    opts = {
      # suppress all IPOPT output
      'ipopt.print_level':       0,
      'print_time':        False,
      'ipopt.tol':            tol,
      'ipopt.max_iter':       max_iter,
    }
    solver = cs.nlpsol('ik', 'ipopt', nlp, opts)
    # set tol=1e-8 for IPOPT

    x0 = q_init if q_init is not None else np.zeros(nq)
    args = {'x0': x0}
    if lb is not None and ub is not None:
      args.update(lbx=lb, ubx=ub)

    sol = solver(**args)
    return sol['x']

def compute_jacobian(q, jac_fun):
    return jac_fun(q)

# ------------------------ Dynamics & acceleration ------------------------

def formulate_symbolic_dynamic_matrices(model, data):
    """
    Return CasADi functions (M, C, G) from the build routine.
    """
    _, _, _, _, M_fun, C_fun, G_fun = \
        build_casadi_kinematics_dynamics(
            model, data,
            frame_name=model.frames[-1].name
        )
    return M_fun, C_fun, G_fun


def compute_dynamics_matrices(M_fun, C_fun, q, dq):
    return M_fun(q), C_fun(q, dq)


def compute_singular_values(J):
    _, S, _ = cs.svd(J)
    return S


def damping_pseudoinverse(J, damping=1e-4):
    """
    Computes the damped pseudoinverse of a Jacobian matrix J using the formula:
    J⁺ = Jᵀ (JJᵀ + λ²I)⁻¹
    where λ is the damping factor and I is the identity matrix.
    """
    m, _ = J.size1(), J.size2()
    return J.T @ cs.solve(J@J.T + damping**2*cs.SX.eye(m), cs.SX.eye(m))


def compute_symbolic_cartesian_acceleration(
    q, dq, tau, M_fun, C_fun, jac_fun, G_fun=None, damping=1e-4
):
    nv = dq.size1()
    M  = M_fun(q)
    C  = C_fun(q, dq)
    G  = G_fun(q) if G_fun is not None else cs.SX.zeros(nv)
    J  = jac_fun(q)
    # Compute J̇ via total derivative
    Jdot = cs.SX.zeros(*J.size())
    for i in range(q.size1()):
        Jdot += cs.jacobian(J[:, i], q) @ dq[i]
    J_pinv = damping_pseudoinverse(J, damping)
    acc_dyn = tau - C@dq - G + M@(J_pinv@(Jdot@dq))
    ddq = cs.solve(M, acc_dyn)
    xdd = J @ ddq
    return xdd

# ------------------------ Linearization ------------------------

def linearize_dynamics_along_trajectory(
    T_opt, U_opt, Z_opt, M_fun, C_fun, G_fun=None
):
    A_list, B_list = [], []
    z_sym   = cs.SX.sym('z',   Z_opt.shape[0])
    tau_sym = cs.SX.sym('tau', U_opt.shape[0])

    def f_cont(z, tau):
        n  = z.size1()//2
        q  = z[:n]
        dq = z[n:]
        ddq = cs.solve(
            M_fun(q),
            tau - C_fun(q, dq)@dq - (G_fun(q) if G_fun else cs.SX.zeros(n))
        )
        return cs.vertcat(dq, ddq)

    A_k = cs.jacobian(f_cont(z_sym, tau_sym), z_sym)
    B_k = cs.jacobian(f_cont(z_sym, tau_sym), tau_sym)
    A_fun = cs.Function('A', [z_sym, tau_sym], [A_k])
    B_fun = cs.Function('B', [z_sym, tau_sym], [B_k])

    for k in range(len(T_opt)-1):
        dt  = T_opt[k+1] - T_opt[k]
        z_k = Z_opt[:, k]
        u_k = U_opt[:, k]
        A_d = np.eye(Z_opt.shape[0]) + dt*A_fun(z_k, u_k).full()
        B_d = dt*B_fun(z_k, u_k).full()
        A_list.append(A_d)
        B_list.append(B_d)
    return A_list, B_list

# ------------------------ Interpolation & I/O ------------------------

def match_trajectories(T_des, *args):
    results    = []
    interp_type= 'linear'
    pairs      = args[:-1] if isinstance(args[-1], str) else args
    if isinstance(args[-1], str): interp_type = args[-1]
    for i in range(0, len(pairs), 2):
        T = np.asarray(pairs[i]).ravel()
        Z = np.asarray(pairs[i+1])
        if T.size==1 and np.allclose(T_des, T):
            results.append(Z)
        else:
            f = interp1d(T, Z.T, kind=interp_type, axis=0,
                         bounds_error=False, fill_value="extrapolate")
            results.append(f(T_des).T)
    return results

def save_solution_to_npy(solution: dict, filename: str):
    np.savez(filename, **solution)


def display_and_save_solution(solution: dict, filename: str=None):
    raise NotImplementedError("Adapt plotting code for 7DOF.")

# ------------------------ Planar‐specific stubs ------------------------

def sample_qf_given_y_line_casadi(*args, **kwargs): raise NotImplementedError()

def sample_pf_vf_grid(*args, **kwargs):          raise NotImplementedError()

def optimize_trajectory(*args, **kwargs):         raise NotImplementedError()

def optimize_trajectory_cartesian_accel(*args, **kwargs): raise NotImplementedError()

def time_optimal_trajectory_cartesian_accel(*args, **kwargs): raise NotImplementedError()

def optimize_trajectory_cartesian_accel_flex_pose(
    model, data, frame_name,
    p_f, v_f,
    q_lower, q_upper,
    dq_lower, dq_upper,
    tau_lower, tau_upper,
    T, N,
    weight_v=1e-3,
    weight_xdd=1e-3,
    weight_tau_smooth=1e-3,
    weight_terminal=1e-3,
    boundary_epsilon=1e-6
):
    """
    Optimize a trajectory for a 7-DOF Kinova Gen3 arm using modular CasADi components.

    Args:
      model, data         : Pinocchio model and data instances
      frame_name          : name of the end-effector frame in the model
      p_f (3,)            : desired final end-effector position
      v_f (3,)            : desired final end-effector linear velocity
      q_lower, q_upper    : (7,) joint position bounds
      dq_lower, dq_upper  : (7,) joint velocity bounds
      tau_lower, tau_upper: (7,) joint torque bounds
      T                   : total time horizon
      N                   : number of discretization steps
      weights             : cost weights
      boundary_epsilon    : small tolerance for position constraints

    Returns:
      dict with keys:
        q     : (7×(N+1)) optimal joint positions
        dq    : (7×(N+1)) optimal joint velocities
        tau   : (7×N)   optimal joint torques
        cost  : scalar total cost
        Z     : (14×(N+1)) stacked [q; dq]
        t_f   : final time T
    """
    dt = T/float(N)
    nv = model.nv

    # Build modular CasADi functions
    fk_fun, pos_fun, J6_fun, M_fun, C_fun, G_fun = \
        build_casadi_kinematics_dynamics(model, data, frame_name)

    # Define CasADi symbols
    q_sym  = cs.SX.sym('q',  nv)
    dq_sym = cs.SX.sym('dq', nv)
    tau    = cs.SX.sym('tau', nv)

    # Symbolic Cartesian acceleration
    xdd_sym = compute_symbolic_cartesian_acceleration(
        q_sym, dq_sym, tau,
        M_fun, C_fun, J6_fun, G_fun
    )
    xdd_fun = cs.Function('xdd', [q_sym, dq_sym, tau], [xdd_sym])

    # Create optimization problem
    opti = cs.Opti()
    Q   = opti.variable(nv, N+1)
    dQ  = opti.variable(nv, N+1)
    Tau = opti.variable(nv, N)

    # Final pose and velocity constraints
    opti.subject_to(pos_fun(Q[:, -1]) == p_f)
    J6_f = J6_fun(Q[:, -1])
    opti.subject_to(J6_f[0:3, :] @ dQ[:, -1] == v_f)

    total_cost = 0
    for k in reversed(range(N)):
        qn, dqn, ta_k = Q[:, k+1], dQ[:, k+1], Tau[:, k]

        # Dynamics
        M_k = M_fun(qn)
        C_k = C_fun(qn, dqn)
        G_k = G_fun(qn)
        ddq_k = cs.solve(M_k, ta_k - C_k @ dqn - G_k)

        # Reverse-Euler integration
        opti.subject_to(Q[:, k]  == qn  - dt * dqn)
        opti.subject_to(dQ[:, k] == dqn - dt * ddq_k)

        # Joint, velocity, and torque limits
        opti.subject_to(q_lower   <= Q[:, k]); opti.subject_to(Q[:, k]   <= q_upper)
        opti.subject_to(dq_lower  <= dQ[:, k]); opti.subject_to(dQ[:, k]  <= dq_upper)
        opti.subject_to(tau_lower <= ta_k);   opti.subject_to(ta_k         <= tau_upper)

        # End-effector linear velocity cost
        v_ee = J6_fun(qn)[0:3, :] @ dqn
        total_cost += -cs.sumsqr(v_ee) * weight_v

        # Cartesian acceleration cost (ramped)
        xdd_k = xdd_fun(qn, dqn, ta_k)
        total_cost += cs.sumsqr(xdd_k[0:3]) * weight_xdd * ((k/float(N))**2)

        # Torque smoothness
        if k < N-1:
            total_cost += weight_tau_smooth * cs.sumsqr(Tau[:, k+1] - ta_k)

        # Terminal deceleration boost
        if k == N-1:
            total_cost += weight_terminal * cs.sumsqr(xdd_k[0:3]) * N

    # Start from rest
    opti.subject_to(dQ[:, 0] == 0)

    # Objective and solver
    opti.minimize(total_cost)
    opti.solver('ipopt', { 'print_time': True }, { 'print_level': 5 })
    sol = opti.solve()

    return {
        'q':   sol.value(Q),
        'dq':  sol.value(dQ),
        'tau': sol.value(Tau),
        'cost': sol.value(total_cost),
        'Z':   np.vstack([sol.value(Q), sol.value(dQ)]),
        't_f': T
    }
    


if __name__ == "__main__":
    # Example usage
    urdf_path = os.path.join(os.path.dirname(__file__), os.pardir, 'models', 'assets', 'robots',
                                'dual_kinova3', 'leonardo.urdf')
    model, data = load_kinova_model(urdf_path)
    fk_fun, pos_fun, jac_fun, M_fun, C_fun, G_fun = build_casadi_kinematics_dynamics(model, data, 'tool_frame')

    # # Example FK
    # q_test = np.ones(model.nv)*0.3
    # # q_test_pin = standard_to_pinocchio(model, q_test)
    # fk_result = forward_kinematics_homogeneous(q_test, fk_fun)
    # # display result
    # print("FK result:\n", fk_result.full())


    # example optimization
    p_f = np.array([0.4, 0.0, 0.2])    # meters
    v_f = np.array([0, 0, -0.5])   # m/s

    # # Example IK
    desired_target_pose = np.array([
        [0, 0, 1, p_f[0]],
        [0, 1, 0, p_f[1]],
        [-1, 0, 0, p_f[2]],
        [0, 0, 0, 1]
    ])
    target_pose = desired_target_pose  # Replace with actual target pose
    q_init = np.ones(model.nv)*0.4
    q_sol = inverse_kinematics_casadi(target_pose, fk_fun, q_init)
    print("IK solution:", q_sol)
    resulted_pose = fk_fun(q_sol)
    print("Resulted pose from IK:\n", resulted_pose.full())

    # 3) Extract bounds from model (example values here; replace with real ones)
    q_lower   =  pinocchio_to_standard(model, model.lowerPositionLimit)
    q_upper   =  pinocchio_to_standard(model, model.upperPositionLimit)
    dq_lower  =  -model.velocityLimit
    dq_upper  =  model.velocityLimit
    tau_lower =  -model.effortLimit
    tau_upper =  model.effortLimit

    frame_name = model.frames[-1].name  # end-effector frame name

    # 4) Planning horizon
    T = 2.0    # seconds
    N = 40     # 50 ms per step

    weight_v=1e-1, 
    weight_xdd=1e-2,
    weight_tau_smooth = 0, 
    weight_terminal = 1e-3,
    boundary_epsilon=1e-3

    # 5) Solve
    solution = optimize_trajectory_cartesian_accel_flex_pose(
        model, data, frame_name,
        p_f, v_f,
        q_lower, q_upper,
        dq_lower, dq_upper,
        tau_lower, tau_upper,
        T, N,
        weight_v=weight_v,
        weight_xdd=weight_xdd,
        weight_tau_smooth=weight_tau_smooth,
        weight_terminal=weight_terminal,
        boundary_epsilon=boundary_epsilon
    )

    # 6) Unpack and replay
    q_opt   = solution['q']     # shape (7, N+1)
    dq_opt  = solution['dq']    # shape (7, N+1)
    tau_opt = solution['tau']   # shape (7, N)

    # e.g. send (q_opt[:,k], dq_opt[:,k]) at each control tick,
    # or forward‐kinematics to plot end-effector path
    ee_path = [forward_kinematics_homogeneous(q_opt[:,k]).full()[0:3,3].flatten() for k in range(q_opt.shape[1]) ]