# kinova_casadi_utils.py

import pinocchio as pin
import pinocchio.casadi as cpin
import casadi as cs
import numpy as np
np.set_printoptions(precision=3, suppress=True)
from scipy.interpolate import interp1d
import os
from matplotlib import pyplot as plt

urdf_path = os.path.join(os.path.dirname(__file__), os.pardir, 'models', 'assets', 'robots',
                                'dual_kinova3', 'leonardo.urdf')

# ------------------------ Model loading ------------------------

def load_kinova_model(urdf_path=urdf_path):
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


def build_casadi_kinematics_dynamics(model, frame_name=None):
    """
    Build CasADi functions (SX) for FK, Jacobian, dynamics (M, C, G) 
    accepting standard 7-DOF q_std and dq_std.
    """
    # Template model for SX
    cmodel = cpin.Model(model)

    cdata  = cmodel.createData()

    # use the last frame (usually that means 'tool_frame')
    if not frame_name:
        frame_name = model.frames[-1].name

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
    tol=1e-9, max_iter=5000,
    w_pose = 10.0, w_config=0.01
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

    # objective: decomposed pose and orientation error
    T_curr = fk_fun(q)
    p_curr, R_curr = T_curr[:3,3], T_curr[:3,:3]
    p_des, R_des = target_pose[:3,3], target_pose[:3,:3]
    pos_error   = cs.sumsqr(p_curr-p_des)
    ER = cs.mtimes(R_des, R_curr.T)
    # trace_ER = ER[0,0] + ER[1,1] + ER[2,2]
    # cos_phi = (trace_ER - 1.0) / 2.0
    # cos_phi_clamped = cs.fmin(cs.fmax(cos_phi, -1.0), 1.0)
    # phi = cs.acos(cos_phi_clamped)
    # pose_error = pos_error + phi**2
    R_err = ER - cs.SX_eye(3)
    r_err_flat = cs.sumsqr(cs.reshape(R_err, -1, 1))
    pose_error = pos_error + r_err_flat*100
    
    obj = w_pose * pose_error
    if q_init is not None:
        # reduce the difference on the last link to make it very similar to q_init
        config_error = cs.sumsqr(q-q_init)
        obj += w_config * config_error

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


def inverse_kinematics_casadi_elbow_above(
    target_pose, fk_fun,
    q_init=None, lb=None, ub=None,
    tol=1e-9, max_iter=500,
    elbow_pos_fun = None,  # Name of elbow link in URDF
    min_height=0.0  # Minimum height above table
):
    """
    Solve IK with elbow height constraint.
    Args:
        ...existing args...
        elbow_link_name: Name of the elbow link to constrain
        min_height: Minimum allowed height above table (z=0)
    """
    nq = fk_fun.size1_in(0)
    q = cs.SX.sym('q', nq)

    # Get position of elbow link using forward kinematics
    # This requires creating a new FK function for the elbow link

    # Objective: Frobenius-norm of pose error
    T_err = fk_fun(q) - target_pose
    obj = cs.norm_2(cs.reshape(T_err, -1, 1))

    # Create NLP with elbow height constraint
    nlp = {
        'x': q,
        'f': obj,
        'g': elbow_pos_fun(q)[2]  # z-coordinate of elbow
    }
    
    # Add elbow height constraint
    opts = {
        'ipopt.print_level': 0,
        'print_time': False,
        'ipopt.tol': tol,
        'ipopt.max_iter': max_iter,
    }
    
    solver = cs.nlpsol('ik', 'ipopt', nlp, opts)

    # Set up bounds including elbow height constraint
    x0 = q_init if q_init is not None else np.zeros(nq)
    args = {
        'x0': x0,
        'lbg': min_height,  # Minimum z-height
        'ubg': cs.inf       # No upper bound
    }
    
    if lb is not None and ub is not None:
        args.update(lbx=lb, ubx=ub)

    sol = solver(**args)
    return sol['x']


def inverse_kinematics_pin(
    model,              # pinocchio.Model  (NumPy version)
    data,               # pinocchio.Data
    frame_id,           # integer frame ID of the end-effector
    q_init,             # (7,) NumPy array in standard rep (angles)
    target_pose,        # pin.SE3  (desired EE pose)  OR   4×4 NumPy array
    fk_fun,             # CasADi Function: q_std (7×1) → 4×4 EE homogeneous
    jac_fun,            # CasADi Function: q_std (7×1) → 6×7 spatial Jacobian (world‐frame)
    tol=1e-9,
    max_iter=100,
):
    """
    A fast iterative IK that uses:
      • `fk_fun(q_std)`    → 4×4 SX/DM “current EE pose”
      • `jac_fun(q_std)`   → 6×7 SX/DM spatial Jacobian (world‐aligned)
      • Pinocchio’s `integrate` and `log` (NumPy) for Δq on the manifold.

    Args:
      model, data   :  Pinocchio model/data (NumPy version)
      frame_id      :  index of ee frame
      q_init        :  (7,) NumPy “standard” joint angles
      target_pose   :  pin.SE3  or 4×4 NumPy (desired EE pose)
      fk_fun        :  CasADi Function: q_std → 4×4 homogeneous (SX/DM)
      jac_fun       :  CasADi Function: q_std → 6×7 Jacobian (SX/DM)
      tol           :  convergence tol on ‖err6‖
      max_iter      :  max Newton steps
      damp          :  damping weight for pseudo‐inverse

    Returns:
      (q_sol, success_flag)
        • q_sol is a (7,) NumPy array (standard rep)
        • success_flag is True if ‖err6‖ < tol within max_iter
    """

    R_des = target_pose[0:3, 0:3]
    p_des = target_pose[0:3, 3]
    oMf_des = pin.SE3(R_des, p_des)

    # 2) Initialize q (standard 7×1) and success flag
    q = q_init.copy()
    success = False

    for i in range(max_iter):
        # 3) Evaluate FK via CasADi: returns a 4×4 DM or SX; convert to NumPy
        Tcur_cas = fk_fun(q)   # returns a 4×4 DM if q is numpy; or SX if q is SX
        # We want numeric, so force .full() if it’s DM:
        if isinstance(Tcur_cas, cs.DM):
            Tcur_num = np.array(Tcur_cas.full())
        else:
            # If somehow we get SX (rare, since q is numeric), explicitly evaluate:
            Tcur_num = np.array(cs.Function('tmp', [], [Tcur_cas])().full())

        Rcur = Tcur_num[0:3, 0:3]
        tcur = Tcur_num[0:3, 3]
        oMf_cur = pin.SE3(Rcur, tcur)

        # 4) Compute SE(3) error: dM = oMf_des * oMf_cur⁻¹
        dM = oMf_des.actInv(oMf_cur)         # pin.SE3
        err6 = pin.log(dM).vector            # 6×1 NumPy

        # 5) Check convergence
        err_norm = np.linalg.norm(err6)
        if err_norm < tol:
            success = True
            break

        # 6) Evaluate spatial Jacobian (6×7) via CasADi, convert to NumPy
        Jcas = jac_fun(q)
        if isinstance(Jcas, cs.DM):
            J6 = np.array(Jcas.full())
        else:
            J6 = np.array(cs.Function('tmpJ', [], [Jcas])().full())

        # 7) Damped least‐squares:  solve  (J6 J6ᵀ + damp·I) x = err6
        JJt = J6.dot(J6.T) + damp * np.eye(6)
        sol = np.linalg.solve(JJt, err6)    # (6,)
        dq  = -J6.T.dot(sol)                # (7,)

        # 8) Integrate dq on the manifold with Pinocchio
        #    a) Convert current q (standard) → Pinocchio rep
        q_pin = standard_to_pinocchio(model, q)  # (nq,) NumPy = (13,) if there’s free-flyer; but on Kinova it’s (7 joint dims but pinocchio uses 7+? 
                                                # Actually, for Kinova “buildModelFromUrdf” yields an 7‐joint chain with no root free‐flyer, so nq=nq_pin=7.)
        #    b) Integrate
        q_pin_next = pin.integrate(model, q_pin, dq)
        #    c) Convert back to “standard” angles
        q = pinocchio_to_standard(model, q_pin_next)

        # 9) (Optional) Print every 10 iters
        if i % 10 == 0:
            print(f"[fast IK] iter {i:3d}  |  ‖err6‖ = {err_norm:.3e}")

    return q, success


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

def build_trace_grad_fun(M_fun):
    """
    Given M_fun: q → (n×n) inertia matrix,
    returns grad_trace(q): q → (1×n) gradient of trace(M(q)).
    """
    # assume M_fun takes a single SX vector of length nv
    # create a fresh symbol q of the right length:
    nv = M_fun.size1_in(0)  # number of rows in input
    q = cs.SX.sym('q', nv)

    Mq = M_fun(q)                 # (nv×nv) SX
    f = cs.trace(Mq)              # scalar SX
    df = cs.jacobian(f, q)        # (1×nv) SX
    grad_fun = cs.Function('grad_trace', [q], [df])
    return grad_fun

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


def save_solution_to_npy(solution: dict, filename: str):
    np.savez(filename, **solution)


def display_and_save_solution(solution: dict, J6_fun, filename: str=None):
    """
    Display kinematic and dynamic results for a 7-DOF Gen3 solution and optionally save to disk.

    Args:
        solution : dict returned by optimize_trajectory_cartesian_accel_flex_pose
        filename : if given, saves {T_opt, Z_opt, U_opt} as a .npz under this name
    """

    # Color definitions
    colors = {
        'joint1': '#1f77b4',    # bright blue
        'joint2': '#ff7f0e',    # bright orange
        'joint3': '#2ca02c',    # bright green
        'joint4': '#d62728',    # bright red
        'joint5': '#9467bd',    # purple
        'joint6': '#8c564b',    # brown
        'joint7': '#e377c2',    # pink
        'vel_x': '#7f7f7f',     # gray
        'vel_y': '#bcbd22',     # olive
        'vel_z': '#17becf',     # cyan
        'vel_mag': '#ff9896'    # light red
    }

    # Unpack
    q   = solution['q']    # (7, N+1)
    dq  = solution['dq']   # (7, N+1)
    tau = solution['tau']  # (7, N)
    T   = solution['t_f']

    # Time vector
    Np1   = q.shape[1]
    T_opt = np.linspace(0, T, Np1)

    # Compute linear portion of EEF velocity
    v_ee = np.zeros((3, Np1))
    for k in range(Np1):
        qk  = q[:, k]
        dqk = dq[:, k]
        J6k = J6_fun(qk)              # 6×7
        v_ee[:, k] = np.array(J6k[0:3, :]) @ dqk
    v_ee_mag = np.linalg.norm(v_ee, axis=0)

    # Densify the state trajectory for smooth plotting
    Z_opt   = np.vstack((q, dq))                 # (14, N+1)
    U_opt   = np.hstack((tau, tau[:, -1:]))      # pad to (7, N+1)

    # Save the trajectory data
    trajectory_data = {
        'T_opt': T_opt,
        'Z_opt': Z_opt,
        'U_opt': U_opt,
    }

    # Save if requested
    if filename:
        curr = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(curr, filename)
        np.save(path, trajectory_data)
        print(f"Saved trajectory data to {path}")

    # Plotting with updated colors
    plt.figure(figsize=(8, 13))

    # 1) Joint torques
    plt.subplot(5, 1, 1)
    for j in range(7):
        plt.plot(T_opt[:-1], tau[j].T, 
                color=colors[f'joint{j+1}'], 
                label=f'τ{j+1}',
                linewidth=2)
    plt.ylabel('Torque (Nm)')
    plt.legend(ncol=4, loc='upper right')
    plt.grid(True)

    # 2) EEF speed magnitude
    plt.subplot(5, 1, 2)
    plt.plot(T_opt, v_ee_mag, 
            color=colors['vel_mag'], 
            label='|v|',
            linewidth=2)
    plt.ylabel('EEF speed (m/s)')
    plt.legend()
    plt.grid(True)

    # 3) EEF velocity components
    plt.subplot(5, 1, 3)
    vel_labels = ['x', 'y', 'z']
    vel_colors = ['vel_x', 'vel_y', 'vel_z']
    for i in range(3):
        plt.plot(T_opt, v_ee[i, :], 
                color=colors[vel_colors[i]], 
                label=f'v_{vel_labels[i]}',
                linewidth=2)
    plt.legend(ncol=3)
    plt.ylabel('EEF vel components')
    plt.grid(True)

    # 4) Joint angles
    plt.subplot(5, 1, 4)
    for j in range(7):
        plt.plot(T_opt, q[j, :], 
                color=colors[f'joint{j+1}'], 
                label=f'q{j+1}',
                linewidth=2)
    plt.legend(ncol=4)
    plt.ylabel('Joint angles (rad)')

    # 5) Joint velocities
    plt.subplot(5, 1, 5)
    for j in range(7):
        plt.plot(T_opt, dq[j, :], 
                color=colors[f'joint{j+1}'], 
                label=f'dq{j+1}',
                linewidth=2)
    plt.legend(ncol=4)
    plt.ylabel('Joint velocities (rad/s)')

    plt.xlabel('Time (s)')
    plt.grid(True)

    plt.tight_layout()

    # Save if requested
    if filename:
        curr = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(curr, filename)
        
        # Also save the figure
        plt_path = path.replace('.npy', '.png')
        plt.savefig(plt_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {plt_path}")

    plt.show()
    return trajectory_data

# ------------------------ Planar‐specific stubs ------------------------

def sample_qf_given_y_line_casadi(*args, **kwargs): raise NotImplementedError()

def sample_pf_vf_grid(*args, **kwargs):          raise NotImplementedError()

def optimize_trajectory(*args, **kwargs):         raise NotImplementedError()

def optimize_trajectory_cartesian_accel(*args, **kwargs): raise NotImplementedError()

def time_optimal_trajectory_cartesian_accel(*args, **kwargs): raise NotImplementedError()


def back_propagate_traj_using_manip_ellipsoid(
    v_f, q_f, fk_fun, jac_fun, N=100, dt=0.01, v_p_mag=None
):
    """
    Back-propagate from zero joint-velocity up to (q_f, dq_f) while
    transitioning into the 6D max-manipulability twist, preserving the
    constant angle between tool-axis and path-direction, and keeping
    the SVD sign consistent—without any damping.
    """
    # Initialize
    q_curr          = q_f.astype(float).copy()
    v_f             = -np.asarray(v_f).reshape(3,)
    v_f_mag       = np.linalg.norm(v_f)
    if v_f_mag < 1e-8:
        raise ValueError("||v_f|| must be nonzero")
    
    if v_p_mag is not None:
        # create a velocity profile that peaks at v_peak at some point such that
        # the acceleration and deceleration equals and are constant
        # let i_p be the index where the peak occurs
        # since v_0 = 0, the acceleration slope is v_p_mag / i_p
        # the time it takes to go from v_f to 0 is i_f0_mag = v_f_mag*i_p / v_p_mag
        # and N = 2*i_p - i_f0_mag
        # so we can express i_p as:
        # N = 2*i_p - v_f_mag*i_p / v_p_mag
        # rearranging gives:
        i_p = int(N / (2 - v_f_mag / v_p_mag))
        # i_p = int(N * v_p_mag / (2 * v_p_mag - v_f_mag))
        v_acc_profile = np.linspace(0, v_p_mag, i_p)
        v_dec_profile = np.linspace(v_p_mag, v_f_mag, N - i_p)
        v_profile = np.concatenate([v_acc_profile, v_dec_profile])
        # reverse the profile so that it starts at v_f_mag and ends at 0
        v_profile = v_profile[::-1]


    # 1) initial unit-twist and seed for sign‐check
    twist_init       = np.concatenate([v_f/v_f_mag, np.zeros(3)]) * v_f_mag
    twist_prev_unit  = twist_init / v_f_mag
    u1_prev          = twist_init[:3] / v_f_mag

    traj = []
    # STEP 0: exact final qd so J6·qd = [v_f; 0]
    J6       = jac_fun(q_curr).full()
    J6_damp = cs.DM(damping_pseudoinverse(jac_fun(q_curr))).full()
    # J6_pinv0 = np.linalg.pinv(J6)
    qd_final = J6_damp @ twist_init
    traj.append((q_curr.copy(), qd_final.copy()))

    # BACK-PROPAGATION LOOP
    for i in range(1, N+1):
        alpha = (i / float(N))**4

        # recompute Jacobian
        J6     = jac_fun(q_curr).full()
        J_vel  = J6[0:3, :]

        # 2) principal translation direction via SVD
        Uvel, _, _ = np.linalg.svd(J_vel)
        u1         = Uvel[:, 0]

        # 3) enforce sign consistency
        if np.dot(u1, u1_prev) < 0:
            u1 = -u1
        u1_prev = u1

        # 4) build manipulability-only twist
        twist_mm = np.concatenate([u1, np.zeros(3)])

        # 5) blend and renormalize
        twist_unit = (1 - alpha) * (twist_init/v_f_mag) + alpha * twist_mm
        twist_unit /= np.linalg.norm(twist_unit)
        # twist_unit = twist_init/v_f_mag

        # 6) compute translational velocity and its derivative
        if v_p_mag is not None:
            v_curr = twist_unit[:3] * v_profile[i-1]
        else:
            v_curr = twist_unit[:3] * v_f_mag
        # v_prev = twist_prev_unit[:3] * v_f_mag
        # dv_dt  = (v_curr - v_prev) / dt

        # # # 7) parallel-transport ω = (v × v̇)/‖v‖²
        # omega = np.cross(v_curr, dv_dt)
        # omega /= np.dot(v_curr, v_curr)

        # # 8) assemble 6D twist and solve for q̇
        # twist_des = np.concatenate([v_curr, np.zeros(3)])
        # J6_damp = cs.DM(damping_pseudoinverse(jac_fun(q_curr))).full()
        # # J6_pinv   = np.linalg.pinv(J6)
        # qd        = J6_damp @ twist_des

        # 8.1) use 3d velocity to compute joint velocities
        J_vel_damp = cs.DM(damping_pseudoinverse(jac_fun(q_curr)[0:3, :])).full()
        qd = J_vel_damp @ v_curr

        # 9) Euler-step backwards
        q_prev = q_curr + dt * qd
        # since we are back propagating, we need to flip the sign of qd
        traj.append((q_prev.copy(), -qd.copy()))

        # update for next iter
        q_curr          = q_prev
        # twist_prev_unit = twist_unit
        # if i > 70:
        #     print(f"[backprop] step {i:3d}  |  ")

    # reverse so traj[0] is “start” and traj[-1] is (q_f, qd_f)
    traj.reverse()

    # pack into solution dict
    q  = np.stack([t[0] for t in traj], axis=1)   # (7, N+1)
    dq = np.stack([t[1] for t in traj], axis=1)   # (7, N+1)

    return {
        'q':  q,
        'dq': dq,
        'U':  np.zeros((q.shape[0], q.shape[1])),
        'T':  dt * np.arange(q.shape[1]),
        'Z':  np.vstack((q, dq))
    }

def back_propagate_traj_using_max_inertial_direction(
        v_f, q_f, grad_trace_fun, jac_fun, N=100, dt=0.01
):
    """
    Back‐propagate from q_f so that:
      • each q̇ points along +∇_q tr M,
      • and ‖J_vel(q)·q̇‖ = ‖v_f‖ (constant linear speed).
    Args:
      q_f           (n,)         : final joint angles
      v_f           (3,)         : desired end-effector linear velocity
      jac_fun       (q→6×n)      : CasADi Function returning spatial J
      grad_trace_fun (q→1×n)     : from build_trace_grad_fun
      N, dt                      : steps & backward timestep
    Returns:
      traj: [(q0, qd0), …, (q_f, qd_f)]
    """
    q_curr    = q_f.astype(float).copy()
    v_f_mag = np.linalg.norm(v_f)
    if v_f_mag < 1e-8:
        raise ValueError("‖v_f‖ must be nonzero")

    traj = []

    J6_sym    = jac_fun(q_curr)
    J_vel_sym = J6_sym[0:3, :]  
    J_vel_pinv = cs.DM(damping_pseudoinverse(J_vel_sym)).full()
    qd_f       = J_vel_pinv @ v_f
    traj.append((q_curr.copy(), qd_f.copy()))

    for step in range(N):
        alpha = (N - step) / float(N)

        # 1) get ∇_q tr M and normalize in joint‐space
        grad = grad_trace_fun(q_curr).full().ravel()   # (n,)
        grad_norm = grad / (np.linalg.norm(grad) + 1e-12)

        # 2) map grad_dir → v_dir in ℝ³
        J6    = jac_fun(q_curr).full()
        J_vel = J6[0:3, :]                          # 3×n
        v_dir = J_vel @ grad_norm                   # (3,)
        mag   = np.linalg.norm(v_dir) + 1e-12

        # 3) desired task-space vel = unit v_dir * v_f_mag
        v_inertial = (v_dir / mag) * v_f_mag

        # 3) blend v_des with v_f
        v_des = (1-alpha) * v_inertial + alpha * (-v_f)

        # 4) pull back via damped pseudo-inverse
        J_vel_pinv = cs.DM(damping_pseudoinverse(jac_fun(q_curr)[0:3, :])).full()
        qd         = J_vel_pinv @ v_des

        # 5) step forward (will be reversed later)
        q_prev = q_curr + dt * qd
        traj.append((q_prev.copy(), qd.copy()))
        q_curr = q_prev

    # reverse so that traj[0] is “start” and traj[-1] = (q_f, qd_f)
    traj.reverse()

    # convert to solution dict
    q = np.array([t[0] for t in traj]).T  # (7, N+1)
    dq = np.array([t[1] for t in traj]).T  # (7, N+1)
    traj = {
        'q': q,  # (7, N+1)
        'dq': dq,  # (7, N+1)
        'U': np.zeros((7, len(traj))),  # (7, N)
        'T': dt * np.arange(len(traj)),  # time vector
        'Z': np.vstack((q, dq))  # (14, N+1)
    }
    return traj
    

# def get_U_from_Z_using_inverse_dynamics(
#     Z, M_fun, C_fun, G_fun=None
# ):
#     """
#     Compute joint torques from joint positions and velocities using inverse dynamics.
    
#     Args:
#         Z (14, N+1) : stacked [q; dq] trajectory
#         M_fun       : function q → M(q) inertia matrix
#         C_fun       : function q, qd → C(q, qd) Coriolis matrix
#         G_fun       : function q → G(q) gravity vector (optional)
#     Returns:
#         U (7, N+1)    : joint torques at each time step
#     """
#     N = Z.shape[1] - 1  # number of time steps
#     nv = M_fun.size1_in(0)  # number of joints

#     U = np.zeros((nv, N+1))  # (7, N)

#     for k in range(N):
#         qk  = Z[:nv, k]
#         qdk = Z[nv:, k]
#         qddk = Z[nv:, k+1] - Z[nv:, k]  # finite difference
#         M_k = M_fun(qk)  # (nv, nv)
#         C_k = C_fun(qk, qdk)  # (nv, nv)
#         G_k = G_fun(qk) if G_fun is not None else np.zeros(nv)  # (nv,)
#         # Compute joint torques using inverse dynamics
#         U[:, k] = (M_k @ qddk + C_k @ qdk + G_k).full().flatten()

#     # Last column repeats the last torque
#     U[:, -1] = U[:, -2]  # repeat last torque for the final time step

#     return U


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
    boundary_epsilon=1e-6,
    init_traj=None
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
      init_traj           : optional initial trajectory dict with keys:
                            - 'T' (1D array of time points)
                            - 'Z' (2D array of stacked [q; dq])
                            - 'U' (2D array of joint torques)

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
        build_casadi_kinematics_dynamics(model, frame_name)

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

    # normalize v_f direction
    v_f = cs.DM(v_f).reshape((3,1))
    v_f_dir = v_f / (cs.norm_2(v_f) + 1e-8) # avoid division by zero

    # Create optimization problem
    opti = cs.Opti()
    Q   = opti.variable(nv, N+1)
    dQ  = opti.variable(nv, N+1)
    Tau = opti.variable(nv, N)

    # Initialize trajectory if provided
    if init_traj is not None:
        # Extract initial guess from init_traj
        T_init = init_traj['T']
        Z_init = init_traj['Z']
        U_init = init_traj['U']

        # Interpolate to match our discretization
        T_des = np.linspace(0, T, N+1)
        [U_matched, Z_matched] = match_trajectories(T_des, 
                                                  T_init, U_init,
                                                  T_init, Z_init)
        U_matched = U_matched[:, :-1]  # remove last column to match N
        # Set initial guess
        opti.set_initial(Q, Z_matched[:nv, :])
        opti.set_initial(dQ, Z_matched[nv:, :])
        opti.set_initial(Tau, U_matched)


    # Final pose and velocity constraints
    if init_traj is not None:
        # Use the last position and velocity from the initial trajectory
        q_f_des = Z_matched[:nv, -1]
        dq_f_des = Z_matched[nv:, -1]
        opti.subject_to(Q[:, -1] == q_f_des)
        opti.subject_to(dQ[:, -1] == dq_f_des)
    else:
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


        
        if k < N-1:
             # End-effector linear velocity cost
            v_ee = J6_fun(qn)[0:3, :] @ dqn
            total_cost += -cs.sumsqr(v_ee) * weight_v

            # Torque smoothness
            total_cost += weight_tau_smooth * cs.sumsqr(Tau[:, k+1] - ta_k)

            # ramped Cartesian acceleration cost
            xdd_k = xdd_fun(qn, dqn, ta_k)
            ramp = (k/float(N))**2
            xdd_proj = cs.dot(xdd_k[0:3], v_f_dir)
            total_cost += weight_xdd * xdd_proj * ramp

            # pinalize near singularity
            epsm = 1e-6
            Jvel = J6_fun(qn)[0:3, :]
            M_manip = Jvel @ Jvel.T + epsm * cs.MX.eye(3)
            singularity_cost = cs.trace(cs.inv(M_manip))
            weight_singularity = 1e-2
            total_cost += weight_singularity * singularity_cost

        # Terminal deceleration boost
        if k == N-1:
            # maximize the cartesian deceleration in the direction of v_f
            xdd_k = xdd_fun(qn, dqn, ta_k)
            xdd_proj = cs.dot(xdd_k[0:3], v_f_dir)
            total_cost += xdd_proj * weight_terminal * N

            # make ee angular velocity zero
            omega_ee = J6_fun(qn)[3:6, :] @ dqn
            total_cost += weight_terminal * cs.sumsqr(omega_ee) * N

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
    model, data = load_kinova_model(urdf_path)
    fk_fun, pos_fun, jac_fun, M_fun, C_fun, G_fun = build_casadi_kinematics_dynamics(model, 'tool_frame')

    # Example FK
    q_test = np.ones(model.nv)*0.0
    # q_test_pin = standard_to_pinocchio(model, q_test)
    fk_result = forward_kinematics_homogeneous(q_test, fk_fun)
    # display result
    print("FK result:\n", fk_result.full())


    # example optimization
    p_f = np.array([0.4, 0.0, 0.2])    # meters
    v_f = np.array([0, 0, -0.5])   # m/s

    # # Example IK
    q_init = np.array([0.000, 0.650, 0.000, 1.890, 0.000, 0.600, -np.pi / 2])
    q_init_2 = np.radians([0, 15, 180, 230, 0, -35, 90])
    target_pose = fk_fun(q_init).full()
    target_pose[0:3, 3] = p_f
    q_sol = inverse_kinematics_casadi(target_pose, fk_fun, q_init).full().flatten()
    print("IK solution:", q_sol)
    resulted_pose = fk_fun(q_sol)
    print("Resulted pose from IK:\n", resulted_pose.full())

    # 3) Extract bounds from model (example values here; replace with real ones)
    rev_lim = np.pi
    q_lower   =  np.array([-rev_lim, -2.41, -rev_lim, -2.66, -rev_lim, -2.23, -rev_lim])
    q_upper   =  np.array([ rev_lim,  2.41,  rev_lim,  2.66,  rev_lim,  2.23,  rev_lim])
    dq_lower  =  -model.velocityLimit
    dq_upper  =  model.velocityLimit
    tau_lower =  -model.effortLimit
    tau_upper =  model.effortLimit

    frame_name = model.frames[-1].name  # end-effector frame name

    # 4) generating a state path for the arm ending at the desired pose
    pth = back_propagate_traj_using_manip_ellipsoid(
        v_f, q_sol, fk_fun, jac_fun,
        N=100, dt=0.01
    )
    print("Back-propagated path length:", len(pth))


    # 4.2 # or using the max inertial direction
    traj = back_propagate_traj_using_max_inertial_direction(
        v_f, q_sol, build_trace_grad_fun(M_fun), jac_fun,
        N=100, dt=0.01
    )
    print("Back-propagated path length:", len(traj))

    # # 4) Planning horizon
    # T = 2.0    # seconds
    # N = 40     # 50 ms per step

    # weight_v=1e-1, 
    # weight_xdd=1e-2,
    # weight_tau_smooth = 0, 
    # weight_terminal = 1e-3,
    # boundary_epsilon=1e-3

    # # 5) Solve
    # solution = optimize_trajectory_cartesian_accel_flex_pose(
    #     model, data, frame_name,
    #     p_f, v_f,
    #     q_lower, q_upper,
    #     dq_lower, dq_upper,
    #     tau_lower, tau_upper,
    #     T, N,
    #     weight_v=weight_v,
    #     weight_xdd=weight_xdd,
    #     weight_tau_smooth=weight_tau_smooth,
    #     weight_terminal=weight_terminal,
    #     boundary_epsilon=boundary_epsilon
    # )

    # # 6) Unpack and replay
    # display_and_save_solution(
    #     solution, jac_fun,
    #     filename='kinova_gen3_opt_trajectory_flex_pose'
    # )

    # e.g. send (q_opt[:,k], dq_opt[:,k]) at each control tick,
    # or forward‐kinematics to plot end-effector path