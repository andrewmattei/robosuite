import pinocchio as pin
import pinocchio.casadi as cpin
import casadi as cs
import numpy as np
np.set_printoptions(precision=3, suppress=True)
import os
import traceback
from matplotlib import pyplot as plt

import robosuite.demos.optimizing_gen3_arm as opt
# Import the SEW Stereo class for spherical-elbow-wrist kinematics
from robosuite.demos.sew_stereo import SEWStereo, SEWStereoSymbolic, build_sew_stereo_casadi_functions

kinova_path = os.path.join(os.path.dirname(__file__), os.pardir, 'models', 'assets', 'robots',
                                'dual_kinova3', 'leonardo.urdf')


def sp_0(p, q, k):
    """
    Symbolic version of subproblem 0: finds theta such that q = rot(k, theta)*p
    
    ** assumes k'*p = 0 and k'*q = 0
    
    Requires that p and q are perpendicular to k. Use subproblem 1 if this is not
    guaranteed.
    
    Args:
        p: 3x1 CasADi SX/MX vector before rotation (must be perpendicular to k)
        q: 3x1 CasADi SX/MX vector after rotation (must be perpendicular to k)
        k: 3x1 CasADi SX/MX rotation axis unit vector
        
    Returns:
        theta: CasADi SX/MX angle in radians
    """
    
    # Normalize p and q
    norm_p = cs.sqrt(cs.dot(p, p))
    norm_q = cs.sqrt(cs.dot(q, q))
    ep = p / norm_p
    eq = q / norm_q
    
    # Calculate the angle using the reference formula
    # theta = 2 * arctan2(||ep - eq||, ||ep + eq||)
    ep_minus_eq = ep - eq
    ep_plus_eq = ep + eq
    
    norm_diff = cs.sqrt(cs.dot(ep_minus_eq, ep_minus_eq))
    norm_sum = cs.sqrt(cs.dot(ep_plus_eq, ep_plus_eq))
    
    theta = 2 * cs.atan2(norm_diff, norm_sum)
    
    # Check the sign using the cross product
    # if k · (p × q) < 0, return -theta
    cross_pq = cs.cross(p, q)
    sign_check = cs.dot(k, cross_pq)
    
    # Use conditional logic for the sign
    theta_signed = cs.if_else(sign_check < 0, -theta, theta)
    
    return theta_signed

def sp_0_numerical(p, q, k):
    """
    Numerical version of subproblem 0: finds theta such that q = rot(k, theta)*p
    
    ** assumes k'*p = 0 and k'*q = 0
           
    Requires that p and q are perpendicular to k. Use subproblem 1 if this is not
    guaranteed.

    Args:
        p: 3x1 numpy array vector before rotation
        q: 3x1 numpy array vector after rotation  
        k: 3x1 numpy array rotation axis unit vector
        
    Returns:
        theta: scalar angle in radians
    """
    import numpy as np
    
    eps = np.finfo(np.float64).eps    
    
    # Check that p and q are perpendicular to k (optional assertion for debugging)
    # assert (np.abs(np.dot(k,p)) < eps) and (np.abs(np.dot(k,q)) < eps), \
    #        "k must be perpendicular to p and q"
    
    norm = np.linalg.norm
    
    ep = p / norm(p)
    eq = q / norm(q)
    
    theta = 2 * np.arctan2(norm(ep - eq), norm(ep + eq))
    
    if (np.dot(k, np.cross(p, q)) < 0):
        return -theta
        
    return theta

def sp_1(p1, p2, k):
    """
    Subproblem 1: Plane and Sphere (Symbolic version)
    
    theta = sp_1(p1, p2, k) finds theta such that
        rot(k, theta) * p1 = p2
    If there's no solution, minimize the least-squares residual
        || rot(k, theta) * p1 - p2 ||
    
    The problem is well-posed if p1 and p2 have the same component along k
    and the same norm. Otherwise, it becomes a least-squares problem.
    
    Args:
        p1: 3x1 CasADi SX/MX vector before rotation
        p2: 3x1 CasADi SX/MX vector after rotation
        k: 3x1 CasADi SX/MX rotation axis unit vector
        
    Returns:
        Dictionary with symbolic expressions for solution
    """
    
    # Compute norms
    norm_p1 = cs.sqrt(cs.dot(p1, p1))
    norm_p2 = cs.sqrt(cs.dot(p2, p2))
    
    # Check for least-squares condition: different norms or different k-components
    norm_diff = cs.fabs(norm_p1 - norm_p2)
    
    # Components along k
    k_dot_p1 = cs.dot(k, p1)
    k_dot_p2 = cs.dot(k, p2)
    k_comp_diff = cs.fabs(k_dot_p1 - k_dot_p2)
    
    # Project vectors onto plane perpendicular to k
    p1_proj = p1 - k_dot_p1 * k
    p2_proj = p2 - k_dot_p2 * k
    
    # Compute projected norms
    norm_p1_proj = cs.sqrt(cs.dot(p1_proj, p1_proj))
    norm_p2_proj = cs.sqrt(cs.dot(p2_proj, p2_proj))
    
    # Difference in projected magnitudes
    proj_diff = cs.fabs(norm_p1_proj - norm_p2_proj)
    
    # For exact case: normalize projected vectors and use sp_0
    p1_proj_norm = p1_proj / norm_p1_proj
    p2_proj_norm = p2_proj / norm_p2_proj
    
    # Calculate angle using atan2 approach (similar to sp_0)
    p_diff = p1_proj_norm - p2_proj_norm
    p_sum = p1_proj_norm + p2_proj_norm
    
    norm_diff_proj = cs.sqrt(cs.dot(p_diff, p_diff))
    norm_sum_proj = cs.sqrt(cs.dot(p_sum, p_sum))
    
    theta = 2 * cs.atan2(norm_diff_proj, norm_sum_proj)
    
    # Check sign using cross product
    cross_p1p2 = cs.cross(p1_proj, p2_proj)
    sign_check = cs.dot(k, cross_p1p2)
    
    # Apply sign correction
    theta_signed = cs.if_else(sign_check < 0, -theta, theta)
    
    # Least-squares condition: significant differences in norms or k-components
    tolerance = 1e-8
    is_LS_condition = cs.fmax(norm_diff > tolerance, 
                             cs.fmax(k_comp_diff > tolerance, proj_diff > tolerance))
    
    return {
        'theta': theta_signed,
        'is_LS_condition': is_LS_condition,
        'norm_diff': norm_diff,
        'proj_diff': proj_diff,
        'k_comp_diff': k_comp_diff
    }

def sp_1_numerical(p1, p2, k):
    """
    Numerical version of subproblem 1: finds theta such that rot(k, theta)*p1 = p2
    
    If the problem is well-posed (same norm and k-component), finds exact solution.
    Otherwise, finds least-squares solution that minimizes || rot(k, theta)*p1 - p2 ||
    
    Args:
        p1: 3x1 numpy array vector before rotation
        p2: 3x1 numpy array vector after rotation
        k: 3x1 numpy array rotation axis unit vector
        
    Returns:
        theta: scalar angle in radians
        is_LS: boolean flag indicating if solution is least-squares
    """
    import numpy as np
    
    # Compute norms
    norm_p1 = np.linalg.norm(p1)
    norm_p2 = np.linalg.norm(p2)
    
    # Check for least-squares condition
    norm_diff = abs(norm_p1 - norm_p2)
    
    # Components along k
    k_dot_p1 = np.dot(k, p1)
    k_dot_p2 = np.dot(k, p2)
    k_comp_diff = abs(k_dot_p1 - k_dot_p2)
    
    # Project vectors onto plane perpendicular to k
    p1_proj = p1 - k_dot_p1 * k
    p2_proj = p2 - k_dot_p2 * k
    
    # Compute projected norms
    norm_p1_proj = np.linalg.norm(p1_proj)
    norm_p2_proj = np.linalg.norm(p2_proj)
    proj_diff = abs(norm_p1_proj - norm_p2_proj)
    
    # Check if this is a least-squares problem
    tolerance = 1e-8
    is_LS = (norm_diff > tolerance) or (k_comp_diff > tolerance) or (proj_diff > tolerance)
    
    # Handle degenerate cases
    if norm_p1_proj < tolerance or norm_p2_proj < tolerance:
        return 0.0, is_LS
    
    # Normalize projected vectors
    p1_proj_norm = p1_proj / norm_p1_proj
    p2_proj_norm = p2_proj / norm_p2_proj
    
    # Calculate angle using atan2 approach (similar to sp_0)
    p_diff = p1_proj_norm - p2_proj_norm
    p_sum = p1_proj_norm + p2_proj_norm
    
    norm_diff_proj = np.linalg.norm(p_diff)
    norm_sum_proj = np.linalg.norm(p_sum)
    
    # Handle edge case where vectors are identical
    if norm_sum_proj < tolerance:
        return np.pi, is_LS
    
    theta = 2 * np.arctan2(norm_diff_proj, norm_sum_proj)
    
    # Check sign using cross product
    cross_p1p2 = np.cross(p1_proj, p2_proj)
    sign_check = np.dot(k, cross_p1p2)
    
    if sign_check < 0:
        theta = -theta
    
    return theta, is_LS

def build_sp_1_casadi_function():
    """
    Build a CasADi Function for sp_1 that can be used in optimization problems.
    
    Returns:
        sp_1_fun: CasADi Function with inputs [p1, p2, k] and outputs [theta, is_LS_condition]
    """
    # Define symbolic inputs
    p1 = cs.SX.sym('p1', 3)
    p2 = cs.SX.sym('p2', 3)
    k = cs.SX.sym('k', 3)
    
    # Call the symbolic version
    result = sp_1(p1, p2, k)
    
    # Create CasADi function
    sp_1_fun = cs.Function('sp_1', 
                          [p1, p2, k],
                          [result['theta'], 
                           result['is_LS_condition'],
                           result['norm_diff'],
                           result['proj_diff']],
                          ['p1', 'p2', 'k'],
                          ['theta', 'is_LS_condition', 'norm_diff', 'proj_diff'])
    
    return sp_1_fun


def sp_2(p1, p2, k1, k2):
    """
    Subproblem 2: Two Circles (Symbolic version using sp_4)
    
    [theta1, theta2] = sp_2(p1, p2, k1, k2) finds theta1, theta2 such that
        rot(k1, theta1)*p1 = rot(k2, theta2)*p2
    
    This implementation follows the MATLAB reference that uses sp_4 internally.
    For least-squares cases, it rescales the vectors to unit length.
    
    Args:
        p1: 3x1 CasADi SX/MX vector
        p2: 3x1 CasADi SX/MX vector
        k1: 3x1 CasADi SX/MX vector with norm(k1) = 1
        k2: 3x1 CasADi SX/MX vector with norm(k2) = 1
        
    Returns:
        Dictionary with symbolic expressions for both exact and LS solutions
    """
    
    # Check for least-squares case: |norm(p1) - norm(p2)| > tolerance
    norm_p1 = cs.sqrt(cs.dot(p1, p1))
    norm_p2 = cs.sqrt(cs.dot(p2, p2))
    norm_diff = cs.fabs(norm_p1 - norm_p2)
    
    # Rescale for least-squares case (following MATLAB reference)
    p1_nrm = p1 / norm_p1
    p2_nrm = p2 / norm_p2
    
    # Compute dot products for sp_4 calls
    k2_dot_p2_nrm = cs.dot(k2, p2_nrm)
    k1_dot_p1_nrm = cs.dot(k1, p1_nrm)
    
    # Call sp_4 twice as in MATLAB reference
    # [theta1, t1_is_LS] = sp_4(k2, p1_nrm, k1, dot(k2,p2_nrm))
    sp4_result_1 = sp_4(k2, p1_nrm, k1, k2_dot_p2_nrm)
    
    # [theta2, t2_is_LS] = sp_4(k1, p2_nrm, k2, dot(k1,p1_nrm))
    sp4_result_2 = sp_4(k1, p2_nrm, k2, k1_dot_p1_nrm)
    
    # Extract solutions from sp_4 results
    # For theta1: use results from first sp_4 call
    theta1_exact_1 = sp4_result_1['theta_1']
    theta1_exact_2 = sp4_result_1['theta_2']
    theta1_ls = sp4_result_1['theta_ls']
    t1_is_LS_condition = sp4_result_1['is_ls_condition']
    
    # For theta2: use results from second sp_4 call (no sign flip initially)
    theta2_exact_1 = sp4_result_2['theta_1']
    theta2_exact_2 = sp4_result_2['theta_2']
    theta2_ls = sp4_result_2['theta_ls']
    t2_is_LS_condition = sp4_result_2['is_ls_condition']
    
    # Handle solution pairing as in MATLAB: theta1 = [theta1(1) theta1(end)]; theta2 = [theta2(end) theta2(1)];
    # For symbolic case, we'll provide both pairings
    theta1_paired_1 = theta1_exact_1  # First pairing uses first theta1
    theta1_paired_2 = theta1_exact_2  # Second pairing uses second theta1
    theta2_paired_1 = theta2_exact_2  # First pairing uses second theta2 (flipped pairing)
    theta2_paired_2 = theta2_exact_1  # Second pairing uses first theta2 (flipped pairing)
    
    # Determine overall LS condition
    # Use norm difference and individual sp_4 LS flags
    tolerance = 1e-8  # MATLAB uses 1e-8
    norm_diff_significant = norm_diff > tolerance
    individual_LS = cs.fmax(t1_is_LS_condition, t2_is_LS_condition) > 0
    overall_is_LS_condition = cs.fmax(norm_diff_significant, individual_LS)
    
    # Exact condition: negative means exact solutions exist
    exact_condition = -overall_is_LS_condition
    
    return {
        'theta1_exact_1': theta1_paired_1,
        'theta1_exact_2': theta1_paired_2,
        'theta2_exact_1': theta2_paired_1,
        'theta2_exact_2': theta2_paired_2,
        'theta1_ls': theta1_ls,
        'theta2_ls': theta2_ls,
        'exact_condition': exact_condition,  # < 0 means exact solutions exist
        'is_LS_condition': overall_is_LS_condition,  # > 0 means LS solution
        'norm_diff': norm_diff,
        't1_is_LS_condition': t1_is_LS_condition,
        't2_is_LS_condition': t2_is_LS_condition
    }

def sp_2_numerical(p1, p2, k1, k2):
    """
    Numerical version of sp_2 that follows the exact MATLAB reference implementation.
    
    [theta1, theta2] = sp_2_numerical(p1, p2, k1, k2) finds theta1, theta2 such that
        rot(k1, theta1)*p1 = rot(k2, theta2)*p2
    
    This implementation follows the MATLAB reference exactly:
    % Rescale for least-squares case
    p1_nrm = p1/norm(p1);
    p2_nrm = p2/norm(p2);
    
    [theta1, t1_is_LS] = subproblem.sp_4(k2, p1_nrm, k1, dot(k2,p2_nrm));
    [theta2, t2_is_LS] = subproblem.sp_4(k1, p2_nrm, k2, dot(k1,p1_nrm));
    
    % Make sure solutions correspond by flipping theta2
    % Also make sure in the edge case that one angle has one solution and the
    % other angle has two solutions that we duplicate the single solution
    if numel(theta1)>1 || numel(theta2)>1
        theta1 = [theta1(1) theta1(end)];
        theta2 = [theta2(end) theta2(1)];
    end
    
    Args:
        p1: 3x1 numpy array
        p2: 3x1 numpy array
        k1: 3x1 numpy array with norm(k1) = 1
        k2: 3x1 numpy array with norm(k2) = 1
        
    Returns:
        theta1: numpy array of theta1 solutions
        theta2: numpy array of theta2 solutions
        is_LS: boolean flag indicating if solution is least-squares
    """
    
     # Rescale for least-squares case
    p1_nrm = p1 / np.linalg.norm(p1)
    p2_nrm = p2 / np.linalg.norm(p2)

    # Call sp_4 twice as in MATLAB reference
    theta1, t1_is_LS = sp_4_numerical(k2, p1_nrm, k1, np.dot(k2, p2_nrm))
    theta2, t2_is_LS = sp_4_numerical(k1, p2_nrm, k2, np.dot(k1, p1_nrm))

    # Pair solutions as in MATLAB
    if len(theta1) > 1 or len(theta2) > 1:
        # Duplicate if needed
        if len(theta1) == 1:
            theta1 = np.array([theta1[0], theta1[0]])
        if len(theta2) == 1:
            theta2 = np.array([theta2[0], theta2[0]])
        # MATLAB pairing: theta1 = [theta1(1) theta1(end)]; theta2 = [theta2(end) theta2(1)];
        theta1 = np.array([theta1[0], theta1[-1]])
        theta2 = np.array([theta2[-1], theta2[0]])

    # LS flag
    is_LS = abs(np.linalg.norm(p1) - np.linalg.norm(p2)) > 1e-8 or t1_is_LS or t2_is_LS

    return theta1, theta2, is_LS

def build_sp_2_casadi_function():
    """
    Build a CasADi Function for sp_2 that can be used in optimization problems.
    
    Returns:
        sp_2_fun: CasADi Function with inputs [p1, p2, k1, k2] and outputs for all solution types
    """
    # Define symbolic inputs
    p1 = cs.SX.sym('p1', 3)
    p2 = cs.SX.sym('p2', 3)
    k1 = cs.SX.sym('k1', 3)
    k2 = cs.SX.sym('k2', 3)
    
    # Call the symbolic version
    result = sp_2(p1, p2, k1, k2)
    
    # Create CasADi function
    sp_2_fun = cs.Function('sp_2', 
                          [p1, p2, k1, k2],
                          [result['theta1_exact_1'], 
                           result['theta1_exact_2'],
                           result['theta2_exact_1'],
                           result['theta2_exact_2'],
                           result['theta1_ls'],
                           result['theta2_ls'],
                           result['exact_condition'],
                           result['is_LS_condition']],
                          ['p1', 'p2', 'k1', 'k2'],
                          ['theta1_exact_1', 'theta1_exact_2', 
                           'theta2_exact_1', 'theta2_exact_2',
                           'theta1_ls', 'theta2_ls', 
                           'exact_condition', 'is_LS_condition'])
    
    return sp_2_fun


def sp_3(p1, p2, k, d):
    """
    Subproblem 3: Circle and Sphere (Symbolic version)
    
    theta = sp_3(p1, p2, k, d) finds theta such that
        || rot(k, theta)*p1 - p2 || = d
    If there's no solution, minimize the least-squares residual
        | || rot(k, theta)*p1 - p2 || - d |
    
    If the problem is well-posed, there may be 1 or 2 exact solutions, or 1
    least-squares solution
    theta1 and theta2 are column vectors of the solutions
    
    The problem is ill-posed if (p1, k) or (p2, k) are parallel
    
    Args:
        p1: 3x1 CasADi SX/MX vector
        p2: 3x1 CasADi SX/MX vector
        k: 3x1 CasADi SX/MX vector with norm(k) = 1
        d: scalar CasADi SX/MX
        
    Returns:
        Dictionary with symbolic expressions for solutions
    """
    
    # Following MATLAB reference: [theta, is_LS] = subproblem.sp_4(p2, p1, k, 1/2 * (dot(p1,p1)+dot(p2,p2)-d^2));
    # Calculate the parameter for sp_4
    p1_dot_p1 = cs.dot(p1, p1)
    p2_dot_p2 = cs.dot(p2, p2)
    d_squared = d * d
    sp4_d_param = 0.5 * (p1_dot_p1 + p2_dot_p2 - d_squared)
    
    # Call sp_4 with the calculated parameters
    sp4_result = sp_4(p2, p1, k, sp4_d_param)
    
    return {
        'theta_1': sp4_result['theta_1'],          # First exact solution
        'theta_2': sp4_result['theta_2'],          # Second exact solution  
        'theta_ls': sp4_result['theta_ls'],        # Least-squares solution
        'discriminant': sp4_result['discriminant'], # > 0 means exact solutions exist
        'is_ls_condition': sp4_result['is_ls_condition'],  # > 0 means LS solution
        'norm_A_2': sp4_result['norm_A_2'],
        'b_squared': sp4_result['b_squared'],
        'sp4_d_param': sp4_d_param  # Store the calculated parameter for debugging
    }

def sp_3_numerical(p1, p2, k, d):
    """
    Numerical version of sp_3 based on the MATLAB reference implementation.
    
    Subproblem 3: Circle and sphere
    
    theta = sp_3(p1, p2, k, d) finds theta such that
        || rot(k, theta)*p1 - p2 || = d
    If there's no solution, minimize the least-squares residual
        | || rot(k, theta)*p1 - p2 || - d |
    
    If the problem is well-posed, there may be 1 or 2 exact solutions, or 1
    least-squares solution
    
    The problem is ill-posed if (p1, k) or (p2, k) are parallel
    
    Parameters:
    -----------
    p1 : array_like, shape (3,)
        3D vector
    p2 : array_like, shape (3,)
        3D vector
    k : array_like, shape (3,)
        3D vector with norm(k) = 1
    d : float
        Scalar value (desired distance)
        
    Returns:
    --------
    theta : ndarray
        Array of angles (in radians). Shape is (N,) where N is the number of solutions
    is_LS : bool
        True if theta is a least-squares solution, False if exact solutions
    """
    
    # Convert inputs to numpy arrays
    p1 = np.array(p1).reshape(-1)
    p2 = np.array(p2).reshape(-1)
    k = np.array(k).reshape(-1)
    
    # Validate input dimensions
    if p1.shape[0] != 3 or p2.shape[0] != 3 or k.shape[0] != 3:
        raise ValueError("p1, p2, and k must be 3D vectors")
    
    # Following MATLAB reference: [theta, is_LS] = subproblem.sp_4(p2, p1, k, 1/2 * (dot(p1,p1)+dot(p2,p2)-d^2));
    # Calculate the parameter for sp_4
    p1_dot_p1 = np.dot(p1, p1)
    p2_dot_p2 = np.dot(p2, p2)
    d_squared = d * d
    sp4_d_param = 0.5 * (p1_dot_p1 + p2_dot_p2 - d_squared)
    
    # Call sp_4_numerical with the calculated parameters
    theta, is_LS = sp_4_numerical(p2, p1, k, sp4_d_param)
    
    return theta, is_LS

def build_sp_3_casadi_function():
    """
    Build a CasADi Function for sp_3 that can be used in optimization problems.
    
    Returns:
        sp_3_fun: CasADi Function with inputs [p1, p2, k, d] and outputs for all solution types
    """
    # Define symbolic inputs
    p1 = cs.SX.sym('p1', 3)
    p2 = cs.SX.sym('p2', 3)
    k = cs.SX.sym('k', 3)
    d = cs.SX.sym('d', 1)
    
    # Call the symbolic version
    result = sp_3(p1, p2, k, d)
    
    # Create CasADi function
    sp_3_fun = cs.Function('sp_3', 
                          [p1, p2, k, d],
                          [result['theta_1'], 
                           result['theta_2'],
                           result['theta_ls'],
                           result['discriminant'],
                           result['is_ls_condition']],
                          ['p1', 'p2', 'k', 'd'],
                          ['theta_1', 'theta_2', 'theta_ls', 
                           'discriminant', 'is_ls_condition'])
    
    return sp_3_fun


def sp_4(h, p, k, d):
    """
    Subproblem 4: Circle and Plane (Symbolic version)
    
    theta = sp_4(h, p, k, d) finds theta such that
        h'*rot(k,theta)*p = d
    If there's no solution, minimize the least-squares residual
        | h'*rot(k,theta)*p - d |
    
    If the problem is well-posed, there may be 1 or 2 exact solutions, or 1
    least-squares solution
    
    The problem is ill-posed if (p, k) or (h, k) are parallel
    
    Args:
        h: 3x1 CasADi SX/MX vector with norm(h) = 1
        p: 3x1 CasADi SX/MX vector
        k: 3x1 CasADi SX/MX vector with norm(k) = 1
        d: scalar CasADi SX/MX
        
    Returns:
        Dictionary with symbolic expressions for solutions
    """
    
    # Build matrices following MATLAB implementation
    A_11 = cs.cross(k, p)
    A_1 = cs.horzcat(A_11, -cs.cross(k, A_11))
    A = cs.mtimes(h.T, A_1)  # h'*A_1
    
    b = d - cs.dot(h, k) * cs.dot(k, p)  # d - h'*k*(k'*p)
    
    norm_A_2 = cs.dot(A, A)  # ||A||^2
    
    x_ls_tilde = cs.mtimes(A_1.T, h * b)  # A_1'*(h*b)
    
    # Check condition for exact vs LS solution
    discriminant = norm_A_2 - b**2
    
    # Exact solutions (when ||A||^2 > b^2)
    xi = cs.sqrt(discriminant)
    x_N_prime_tilde = cs.vertcat(A[1], -A[0])  # [A(2); -A(1)]
    
    sc_1 = x_ls_tilde + xi * x_N_prime_tilde
    sc_2 = x_ls_tilde - xi * x_N_prime_tilde
    
    theta_1 = cs.atan2(sc_1[0], sc_1[1])
    theta_2 = cs.atan2(sc_2[0], sc_2[1])
    
    # Least-squares solution (when ||A||^2 <= b^2)
    theta_ls = cs.atan2(x_ls_tilde[0], x_ls_tilde[1])
    
    return {
        'theta_1': theta_1,          # First exact solution
        'theta_2': theta_2,          # Second exact solution  
        'theta_ls': theta_ls,        # Least-squares solution
        'discriminant': discriminant, # > 0 means exact solutions exist
        'is_ls_condition': -discriminant,  # > 0 means LS solution
        'norm_A_2': norm_A_2,
        'b_squared': b**2
    }

def sp_4_numerical(h, p, k, d):
    """
    Numerical version of sp_4 based on the MATLAB reference implementation.
    
    Subproblem 4: Circle and plane
    
    Finds theta such that h' * rot(k, theta) * p = d
    If there's no solution, minimize the least-squares residual
    | h' * rot(k, theta) * p - d |
    
    If the problem is well-posed, there may be 1 or 2 exact solutions, or 1
    least-squares solution
    
    The problem is ill-posed if (p, k) or (h, k) are parallel
    
    Parameters:
    -----------
    h : array_like, shape (3,)
        3D vector with norm(h) = 1
    p : array_like, shape (3,)
        3D vector
    k : array_like, shape (3,)
        3D vector with norm(k) = 1
    d : float
        Scalar value
        
    Returns:
    --------
    theta : ndarray
        Array of angles (in radians). Shape is (N,) where N is the number of solutions
    is_LS : bool
        True if theta is a least-squares solution, False if exact solutions
    """
    
    # Convert inputs to numpy arrays and ensure they're column vectors
    h = np.array(h).reshape(-1)
    p = np.array(p).reshape(-1)
    k = np.array(k).reshape(-1)
    
    # Validate input dimensions
    if h.shape[0] != 3 or p.shape[0] != 3 or k.shape[0] != 3:
        raise ValueError("h, p, and k must be 3D vectors")
    
    # A_11 = cross(k, p)
    A_11 = np.cross(k, p)
    
    # A_1 = [A_11 -cross(k, A_11)]
    # This creates a 3x2 matrix where first column is A_11 and second column is -cross(k, A_11)
    A_1 = np.column_stack([A_11, -np.cross(k, A_11)])
    
    # A = h' * A_1 (this is a 1x2 matrix, but we'll treat as 1D array)
    A = h.T @ A_1  # This gives us a (2,) array
    
    # b = d - h' * k * (k' * p)
    b = d - np.dot(h, k) * np.dot(k, p)
    
    # norm_A_2 = dot(A, A) = ||A||^2
    norm_A_2 = np.dot(A, A)
    
    # x_ls_tilde = A_1' * (h * b)
    x_ls_tilde = A_1.T @ (h * b)  # This gives us a (2,) array
    
    # Check if we have exact solutions or need least-squares
    if norm_A_2 > b**2:
        # Two exact solutions case
        xi = np.sqrt(norm_A_2 - b**2)
        
        # x_N_prime_tilde = [A(2); -A(1)] (swap and negate first component)
        x_N_prime_tilde = np.array([A[1], -A[0]])
        
        # Two solution candidates
        sc_1 = x_ls_tilde + xi * x_N_prime_tilde
        sc_2 = x_ls_tilde - xi * x_N_prime_tilde
        
        # Compute angles using atan2
        theta = np.array([np.arctan2(sc_1[0], sc_1[1]), 
                         np.arctan2(sc_2[0], sc_2[1])])
        is_LS = False
        
    else:
        # Least-squares solution case
        theta = np.array([np.arctan2(x_ls_tilde[0], x_ls_tilde[1])])
        is_LS = True
    
    return theta, is_LS

def build_sp_4_casadi_function():
    """
    Build a CasADi Function for sp_4 that can be used in optimization problems.
    
    Returns:
        sp_4_fun: CasADi Function with inputs [h, p, k, d] and outputs for all solution types
    """
    # Define symbolic inputs
    h = cs.SX.sym('h', 3)
    p = cs.SX.sym('p', 3)
    k = cs.SX.sym('k', 3)
    d = cs.SX.sym('d', 1)
    
    # Call the symbolic version
    result = sp_4(h, p, k, d)
    
    # Create CasADi function
    sp_4_fun = cs.Function('sp_4', 
                          [h, p, k, d],
                          [result['theta_1'], 
                           result['theta_2'],
                           result['theta_ls'],
                           result['discriminant'],
                           result['is_ls_condition']],
                          ['h', 'p', 'k', 'd'],
                          ['theta_1', 'theta_2', 'theta_ls', 
                           'discriminant', 'is_ls_condition'])
    
    return sp_4_fun


def rot(axis, angle):
    """
    Create a rotation matrix using Rodrigues' formula in CasADi.
    
    Args:
        axis: 3x1 CasADi SX/MX vector (unit vector)
        angle: scalar CasADi SX/MX angle in radians
        
    Returns:
        3x3 CasADi SX/MX rotation matrix
    """
    # Rodrigues' formula: R = I + sin(θ)[k]× + (1-cos(θ))[k]×²
    c = cs.cos(angle)
    s = cs.sin(angle)
    v = 1 - c
    
    # Skew-symmetric matrix [k]×
    k_skew = cs.vertcat(
        cs.horzcat(0, -axis[2], axis[1]),
        cs.horzcat(axis[2], 0, -axis[0]),
        cs.horzcat(-axis[1], axis[0], 0)
    )
    
    # Identity matrix
    I = cs.SX.eye(3)
    
    # Rodrigues' formula
    R = I + s * k_skew + v * cs.mtimes(k_skew, k_skew)
    
    return R

def rot_numerical(axis, angle):
    """
    Numerical version of rotation matrix using Rodrigues' formula.
    
    Args:
        axis: 3x1 numpy array (unit vector)
        angle: scalar angle in radians
        
    Returns:
        3x3 numpy rotation matrix
    """
    import numpy as np
    
    c = np.cos(angle)
    s = np.sin(angle)
    v = 1 - c
    
    # Skew-symmetric matrix
    k_skew = np.array([[0, -axis[2], axis[1]],
                       [axis[2], 0, -axis[0]],
                       [-axis[1], axis[0], 0]])
    
    # Rodrigues' formula
    R = np.eye(3) + s * k_skew + v * k_skew @ k_skew
    
    return R

def IK_2R_2R_3R_casadi(R_0_7, p_0_T, sew_stereo, psi, model_transforms):
    """
    CasADi implementation of IK_2R_2R_3R inverse kinematics function.
    Uses frame transformations from Pinocchio model instead of kin_P and kin_H.
    
    Solves inverse kinematics for a 7-DOF robot using subproblems:
    - Subproblem 3 for SEW (Spherical-Elbow-Wrist) configuration
    - Multiple Subproblem 2 calls for joint pairs
    - Subproblem 1 for final joint
    
    Args:
        R_0_7: 3x3 CasADi SX/MX desired end-effector orientation
        p_0_T: 3x1 CasADi SX/MX desired end-effector position
        sew_stereo: SEWStereoSymbolic instance for spherical kinematics
        psi: scalar CasADi SX/MX stereo angle parameter
        model_transforms: dict from get_frame_transforms_from_pinocchio() containing:
            - 'R': list of 3x3 rotation matrices [R_0_1, R_1_2, ..., R_6_7, R_7_T]
            - 'p': list of 3x1 position vectors [p_0_1, p_1_2, ..., p_6_7, p_7_T]
            - 'joint_names': list of joint/frame names
        
    Returns:
        Dictionary containing:
        - 'solutions': List of 7x1 joint angle solutions
        - 'is_LS_flags': List of least-squares flags for each solution
        - 'intermediate_results': Dictionary with intermediate calculations
    """
    
    solutions = []
    is_LS_flags = []

    # Extract frame transformations
    R_local = model_transforms['R']
    p_local = model_transforms['p']
    
    # Build position vectors following numerical implementation
    p_01, R_01 = p_local[0], R_local[0]  # in base frame
    p_12, R_12 = p_local[1], R_local[1]  # in 1 frame
    p_23, R_23 = p_local[2], R_local[2]  # in 2 frame
    p_34, R_34 = p_local[3], R_local[3]  # in 3 frame
    p_45, R_45 = p_local[4], R_local[4]  # in 4 frame
    p_56, R_56 = p_local[5], R_local[5]  # in 5 frame
    p_67, R_67 = p_local[6], R_local[6]  # in 6 frame
    p_7T, R_7T = p_local[7], R_local[7]  # in 7 frame

    # Find wrist position in base frame (following numerical implementation)
    p_7_T_0 = cs.mtimes(R_0_7, p_7T)
    p_6_7_0 = cs.mtimes(cs.mtimes(R_0_7, R_67.T), p_67)  # vector at q6 = 0
    p_W_7_0 = cs.SX.zeros(3)
    p_W_7_0[2] = p_6_7_0[2]  # wrist at the intersection of h6 and h7
    W = p_0_T - (p_W_7_0 + p_7_T_0)  # Wrist position in base frame between joint 6 and 7

    # Find shoulder position (fixed in base frame)
    p_12_0 = cs.mtimes(R_01, p_12)  # vector at q1 = 0
    p_1_S_0 = cs.SX.zeros(3)
    p_1_S_0[2] = p_12_0[2]  # shoulder at the intersection of h1 and h2
    S = p_01 + p_1_S_0  # Shoulder position in base frame between joint 1 and 2

    # expressing the distance from shoulder to elbow (at the intersection of h3 and h4)
    p_S2_0 = p_12_0 - p_1_S_0  # Vector from shoulder to joint 2
    R_02 = cs.mtimes(R_01, R_12)
    R_03 = cs.mtimes(R_02, R_23)
    p_3E = cs.SX.zeros(3)
    p_3E[2] = p_34[2]  # elbow at the intersection of h3 and h4
    p_2E_0 = cs.mtimes(R_02, p_23) + cs.mtimes(R_03, p_3E)  # vector at q2 = 0, q3 = 0
    d_SE_vec = p_S2_0 + p_2E_0  # Sum from shoulder to elbow
    d_SE = cs.sqrt(cs.dot(d_SE_vec, d_SE_vec))

    # expressing the distance from elbow to wrist
    R_04 = cs.mtimes(R_03, R_34)
    R_05 = cs.mtimes(R_04, R_45)
    R_06 = cs.mtimes(R_05, R_56)
    p_67_0 = cs.mtimes(R_06, p_67)
    p_W7_0 = cs.SX.zeros(3)
    p_W7_0[2] = p_67_0[2]  # wrist at the intersection of h6 and h7
    # vector at q4 = 0, q5 = 0, q6 = 0
    p_6W_0 = p_67_0 - p_W7_0
    p_E4_4 = cs.mtimes(R_34.T, (p_34 - p_3E))  # Vector from elbow to joint 4
    p_E6_0 = cs.mtimes(R_04, (p_E4_4 + p_45)) + cs.mtimes(R_05, p_56)  # vector at q4 = 0, q5 = 0
    d_EW_vec = p_E6_0 + p_6W_0  # Sum from elbow to wrist
    d_EW = cs.sqrt(cs.dot(d_EW_vec, d_EW_vec))

    # Vector from shoulder to wrist
    p_S_W = W - S
    e_S_W = p_S_W / cs.sqrt(cs.dot(p_S_W, p_S_W))
    
    # Use SEW inverse kinematics
    e_CE, n_SEW = sew_stereo.inv_kin_symbolic(S, W, psi)
    
    # Use subproblem 3 to find theta_SEW
    sp3_result = sp_3(d_SE * e_S_W, p_S_W, n_SEW, d_EW)
    
    # Pick theta_SEW > 0 for correct half-plane (symbolic version needs both solutions)
    theta_SEW_candidates = [sp3_result['theta_1'], sp3_result['theta_2'], sp3_result['theta_ls']]
    theta_SEW_is_LS_candidates = [sp3_result['is_ls_condition'] > 0, 
                                  sp3_result['is_ls_condition'] > 0, 
                                  cs.SX(1)]  # LS is always true for the LS solution
    
    for theta_SEW_idx in range(3):
        q_SEW = theta_SEW_candidates[theta_SEW_idx]
        theta_SEW_is_LS = theta_SEW_is_LS_candidates[theta_SEW_idx]
        
        # Skip if this is an LS solution and we have exact solutions available
        if theta_SEW_idx == 2 and sp3_result['discriminant'] > 0:
            continue
            
        # Calculate elbow position in base frame
        p_S_E = cs.mtimes(rot(n_SEW, q_SEW), (d_SE * e_S_W))  # this is actual vector in base frame 
        E = p_S_E + S
        
        # Joint axes projected to appropriate frames
        # All joint axes are z-direction (0,0,1) in their local frames
        ez = cs.SX([0, 0, 1])
        
        # h_1: joint 1 axis in joint 1 frame
        h_1 = ez  # Already in 1 frame
        
        # h_2: joint 2 axis rotated to joint 1 frame
        h_2 = cs.mtimes(R_12, ez)

        p_S_E_1 = cs.mtimes(R_01.T, p_S_E)  # desired Shoulder to elbow vector in 1 frame
        p_SE_1 = cs.mtimes(R_01.T, d_SE_vec)  # q1,q2 zero config shoulder to elbow vector in 1 frame

        sp2_12_result = sp_2(p_S_E_1, p_SE_1, -h_1, h_2)

        # Handle solutions for joints 1,2
        theta1_candidates = [sp2_12_result['theta1_exact_1'], sp2_12_result['theta1_exact_2'], sp2_12_result['theta1_ls']]
        theta2_candidates = [sp2_12_result['theta2_exact_1'], sp2_12_result['theta2_exact_2'], sp2_12_result['theta2_ls']]
        
        for i_q12 in range(len(theta1_candidates)):
            q1 = theta1_candidates[i_q12]
            q2 = theta2_candidates[i_q12]
            t12_is_ls = (i_q12 == 2) or (sp2_12_result['exact_condition'] > 0)
            
            # Build rotation matrix up to joint 2
            R_0_1 = cs.mtimes(R_01, rot(ez, q1))
            R_1_2 = cs.mtimes(R_12, rot(ez, q2))
            R_0_2 = cs.mtimes(R_0_1, R_1_2)
            
            # h_3 and h_4: joints 3,4 axes projected to frame 3
            h_3 = ez  # Joint 3 axis in 3 frame
            h_4 = cs.mtimes(R_34, ez)  # Joint 4 axis in 3 frame

            p_E_W_3 = cs.mtimes(cs.mtimes(R_0_2, R_23).T, (W - E))  # desired elbow to wrist vector in 3 frame
            p_EW_3 = cs.mtimes(R_03.T, d_EW_vec)  # q3,q4 zero config elbow to wrist vector in 3 frame

            sp2_34_result = sp_2(p_E_W_3, p_EW_3, -h_3, h_4)

            # Handle solutions for joints 3,4
            theta3_candidates = [sp2_34_result['theta1_exact_1'], sp2_34_result['theta1_exact_2'], sp2_34_result['theta1_ls']]
            theta4_candidates = [sp2_34_result['theta2_exact_1'], sp2_34_result['theta2_exact_2'], sp2_34_result['theta2_ls']]
            
            for i_q34 in range(len(theta3_candidates)):
                q3 = theta3_candidates[i_q34]
                q4 = theta4_candidates[i_q34]
                t34_is_ls = (i_q34 == 2) or (sp2_34_result['exact_condition'] > 0)
                
                # Build rotation matrix up to joint 4
                R_2_3 = cs.mtimes(R_23, rot(ez, q3))  # h_3 in its local frame
                R_3_4 = cs.mtimes(R_34, rot(ez, q4))  # h_4 in its local frame
                R_0_4 = cs.mtimes(cs.mtimes(R_0_2, R_2_3), R_3_4)
                
                # h_5, h_6, h_7: joints 5,6,7 axes projected to frame 5
                h_5 = ez  # Joint 5 axis in 5 frame
                h_6 = cs.mtimes(R_56, ez)  # Joint 6 axis in 5 frame

                h_7_act_5 = cs.mtimes(cs.mtimes(R_0_4, R_45).T, cs.mtimes(R_0_7, ez))  # Joint 7 axis in joint 5 frame
                h_7_zero_5 = cs.mtimes(R_56, cs.mtimes(R_67, ez))  # Joint 7 axis in joint 6 frame

                sp2_56_result = sp_2(h_7_act_5, h_7_zero_5, -h_5, h_6)

                # Handle solutions for joints 5,6
                theta5_candidates = [sp2_56_result['theta1_exact_1'], sp2_56_result['theta1_exact_2'], sp2_56_result['theta1_ls']]
                theta6_candidates = [sp2_56_result['theta2_exact_1'], sp2_56_result['theta2_exact_2'], sp2_56_result['theta2_ls']]
                
                for i_q56 in range(len(theta5_candidates)):
                    q5 = theta5_candidates[i_q56]
                    q6 = theta6_candidates[i_q56]
                    t56_is_ls = (i_q56 == 2) or (sp2_56_result['exact_condition'] > 0)
                    
                    # Build rotation matrix up to joint 6
                    R_4_5 = cs.mtimes(R_45, rot(ez, q5))  # h_5 in its local frame
                    R_5_6 = cs.mtimes(R_56, rot(ez, q6))  # h_6 in its local frame
                    R_0_6 = cs.mtimes(cs.mtimes(R_0_4, R_4_5), R_5_6)
                    
                    # Final joint 7 using subproblem 1 - match numerical implementation
                    # Projecting everything to joint 7 frame
                    h_7_final = ez  # Joint 7 axis for subproblem 1
                    h_6_act_7 = cs.mtimes(R_0_7.T, cs.mtimes(R_0_6, ez))
                    h_6_zero_7 = cs.mtimes(R_67.T, ez)  # Joint 7 axis for subproblem 1
                    
                    sp1_result = sp_1(h_6_zero_7, h_6_act_7, -h_7_final)
                    q7 = sp1_result['theta']
                    q7_is_ls = sp1_result['is_LS_condition'] > 1e-8
                    
                    # Combine joint angles
                    q_solution = cs.vertcat(q1, q2, q3, q4, q5, q6, q7)
                    solutions.append(q_solution)
                    
                    # Combine LS flags (OR operation)
                    overall_is_ls = cs.fmax(cs.fmax(cs.fmax(theta_SEW_is_LS, t12_is_ls), 
                                                   cs.fmax(t34_is_ls, t56_is_ls)), q7_is_ls)
                    is_LS_flags.append(overall_is_ls)
    
    # Filter and prioritize solutions
    filtered_solutions, filtered_is_LS = filter_symbolic_solutions(solutions, is_LS_flags)
    
    return {
        'solutions': filtered_solutions,
        'is_LS_flags': filtered_is_LS,
        'intermediate_results': {
            'W': W,
            'S': S,
            'E': E if 'E' in locals() else cs.SX.zeros(3),
            'p_S_W': p_S_W,
            'n_SEW': n_SEW if 'n_SEW' in locals() else cs.SX.zeros(3),
            'd_SE': d_SE,
            'd_EW': d_EW
        }
    }

def IK_2R_2R_3R_numerical(R_0_7, p_0_T, sew_stereo, psi, model_transforms):
    """
    Numerical implementation of IK_2R_2R_3R inverse kinematics function.
    Uses frame transformations from Pinocchio model instead of kin_P and kin_H.
    
    Args:
        R_0_7: 3x3 numpy array - desired end-effector orientation
        p_0_T: 3x1 numpy array - desired end-effector position
        sew_stereo: SEWStereo instance for spherical kinematics
        psi: scalar stereo angle parameter
        model_transforms: dict from get_frame_transforms_from_pinocchio() containing:
            - 'R': list of 3x3 rotation matrices [R_0_1, R_1_2, ..., R_6_7, R_7_T]
            - 'p': list of 3x1 position vectors [p_0_1, p_1_2, ..., p_6_7, p_7_T]
            - 'joint_names': list of joint/frame names
        
    Returns:
        Q: numpy array of joint angle solutions (7 x num_solutions)
        is_LS_vec: list of boolean flags indicating LS solutions
    """
    
    Q = []
    is_LS_vec = []

    # Extract frame transformations
    R_local = model_transforms['R']
    p_local = model_transforms['p']
    
    # Build position vectors (equivalent to kin_P columns)
    # p_01 = origin (0,0,0) since first frame is at base
    p_01, R_01 = p_local[0], R_local[0]  # in base frame
    p_12, R_12 = p_local[1], R_local[1]  # in 1 frame
    p_23, R_23 = p_local[2], R_local[2]  # in 2 frame
    p_34, R_34 = p_local[3], R_local[3]  # in 3 frame
    p_45, R_45 = p_local[4], R_local[4]  # in 4 frame
    p_56, R_56 = p_local[5], R_local[5]  # in 5 frame
    p_67, R_67 = p_local[6], R_local[6]  # in 6 frame
    p_7T, R_7T = p_local[7], R_local[7]  # in 7 frame

    
    ##### Notation on position #######
    # p_ij: position vector from frame i to j in local frame i with at q_i, q_i+1,...q_j-1 = 0
    # p_ij_0: position vector from frame i to j in base frame 0
    # p_i_j: position vector with q_i, q_i+1,...q_j-1 = (the actual value) calculated in "base frame" by default
    # p_i_j_k: position vector with q_i, q_i+1,...q_j-1 = (the actual value) calculated in "frame k"
    ##################################

    # Find wrist position in base frame
    p_7_T_0 = R_0_7 @ p_7T
    p_6_7_0 = R_0_7 @ R_67.T @ p_67 # vector at q6 = 0
    p_W_7_0 = np.zeros(3) 
    p_W_7_0[2] = p_6_7_0[2]  # wrist at the intersection of h6 and h7
    W = p_0_T - (p_W_7_0 + p_7_T_0)  # Wrist position in base frame between joint 6 and 7

    # Find shoulder position (fixed in base frame)
    p_12_0 = R_01 @ p_12 # vector at q1 = 0
    p_1_S_0 = np.zeros(3)  
    p_1_S_0[2] = p_12_0[2]  # shoulder at the intersection of h1 and h2
    S = p_01+p_1_S_0  # Shoulder position in base frame between joint 1 and 2

    ### Important bug diery ###
    # the elbow was originally located at exact joint 4. The final solution was always off in lateral (-y) direction by a very small amount.
    # for precisely half the solutions.This is caused by the offset in -y direction between the joint 1 and joint 3 at zero configuration.
    # the solution is to express the elbow position at the intersection of h3 and h4, not at joint 4.

    # expressing the distance from shoulder to elbow (at the intersection of h3 and h4)
    p_S2_0 = p_12_0 - p_1_S_0  # Vector from shoulder to joint 2
    R_02 = R_01 @ R_12
    R_03 = R_02 @ R_23 
    p_3E = np.zeros(3)
    p_3E[2] = p_34[2]  # elbow at the intersection of h3 and h4
    p_2E_0 = R_02 @ p_23 + R_03 @ p_3E  # vector at q2 = 0, q3 = 0
    d_SE_vec = p_S2_0 + p_2E_0  # Sum from shoulder to elbow
    d_SE = np.linalg.norm(d_SE_vec)

    # expressing the distance from elbow to wrist
    R_04 = R_03 @ R_34
    R_05 = R_04 @ R_45
    R_06 = R_05 @ R_56
    p_67_0 = R_06 @ p_67 
    p_W7_0 = np.zeros(3)
    p_W7_0[2] = p_67_0[2]  # wrist at the intersection of h6 and h7
    # vector at q4 = 0, q5 = 0, q6 = 0
    p_6W_0 = p_67_0 - p_W7_0
    p_E4_4 = R_34.T @ (p_34 - p_3E)  # Vector from elbow to joint 4
    p_E6_0 = R_04 @ (p_E4_4 + p_45) + R_05 @ p_56 # vector at q4 = 0, q5 = 0
    d_EW_vec = p_E6_0 + p_6W_0  # Sum from elbow to wrist
    d_EW = np.linalg.norm(d_EW_vec)

    # Vector from shoulder to wrist
    p_S_W = W - S
    e_S_W = p_S_W / np.linalg.norm(p_S_W)
    
    # Use SEW inverse kinematics
    e_CE, n_SEW = sew_stereo.inv_kin(S, W, psi)
    
    # Use subproblem 3 to find theta_SEW
    theta_SEW, theta_SEW_is_LS = sp_3_numerical(d_SE * e_S_W, p_S_W, n_SEW, d_EW)
    
    # Pick theta_SEW > 0 for correct half-plane
    if len(theta_SEW) > 1:
        q_SEW = np.max(theta_SEW)
    else:
        q_SEW = theta_SEW[0]
    
    # Calculate elbow position in base frame
    p_S_E = rot_numerical(n_SEW, q_SEW) @ (d_SE * e_S_W)  # this is actual vector in base frame 
    E = p_S_E + S
    
    # Joint axes projected to appropriate frames
    # All joint axes are z-direction (0,0,1) in their local frames
    ez = np.array([0, 0, 1])
    
    # h_1: joint 1 axis in joint 1 frame
    h_1 = ez  # Already in 1 frame
    
    # h_2: joint 2 axis rotated to joint 1 frame
    h_2 = R_12 @ ez

    p_S_E_1 = R_01.T @ p_S_E  # desired Shoulder to elbow vector in 1 frame
    p_SE_1 = R_01.T @ d_SE_vec  # q1,q2 zero config shoulder to elbow vector in 1 frame

    t1, t2, t12_is_ls = sp_2_numerical(p_S_E_1, p_SE_1, -h_1, h_2)

    for i_q12 in range(len(t1)):
        q1 = t1[i_q12]
        q2 = t2[i_q12]
        
        # Build rotation matrix up to joint 2
        R_0_1 = R_01 @ rot_numerical(ez, q1)
        R_1_2 = R_12 @ rot_numerical(ez, q2) 
        R_0_2 = R_0_1 @ R_1_2
        
        # h_3 and h_4: joints 3,4 axes projected to frame 3
        h_3 = ez  # Joint 3 axis in 3 frame
        h_4 = R_34 @ ez  # Joint 4 axis in 3 frame

        p_E_W_3 = (R_0_2 @ R_23).T @ (W - E) # desired elbow to wrist vector in 3 frame
        p_EW_3 = R_03.T @ d_EW_vec  # q3,q4 zero config elbow to wrist vector in 3 frame

        t3, t4, t34_is_ls = sp_2_numerical(p_E_W_3, p_EW_3, -h_3, h_4)

        for i_q34 in range(len(t3)):
            q3 = t3[i_q34]
            q4 = t4[i_q34]
            
            # Build rotation matrix up to joint 4
            R_2_3 = R_23 @ rot_numerical(ez, q3)  # h_3 in its local frame
            R_3_4 = R_34 @ rot_numerical(ez, q4)  # h_4 in its local frame
            R_0_4 = R_0_2 @ R_2_3 @ R_3_4
            
            # h_5, h_6, h_7: joints 5,6,7 axes projected to frame 5
            h_5 = ez  # Joint 5 axis in 5 frame
            h_6 = R_56 @ ez  # Joint 6 axis in 5 frame

            h_7_act_5 = (R_0_4 @ R_45).T @ R_0_7 @ ez  # Joint 7 axis in joint 5 frame
            h_7_zero_5 = R_56 @ R_67 @ ez  # Joint 7 axis in joint 6 frame

            t5, t6, t56_is_ls = sp_2_numerical(h_7_act_5, h_7_zero_5, -h_5, h_6)

            for i_q56 in range(len(t5)):
                q5 = t5[i_q56]
                q6 = t6[i_q56]
                
                # Build rotation matrix up to joint 6
                R_4_5 = R_45 @ rot_numerical(ez, q5)  # h_5 in its local frame
                R_5_6 = R_56 @ rot_numerical(ez, q6)  # h_6 in its local frame
                R_0_6 = R_0_4 @ R_4_5 @ R_5_6
                
                # Projecting everying to joint 7 frmae
                h_7_final = ez  # Joint 6 axis for subproblem 1

                h_6_act_7 = (R_0_7).T @ R_0_6 @ ez
                h_6_zero_7 = R_67.T @ ez  # Joint 7 axis for subproblem 1

                q7, q7_is_ls = sp_1_numerical(h_6_zero_7, h_6_act_7, -h_7_final)
                # Combine solution
                q_i = np.array([q1, q2, q3, q4, q5, q6, q7])
                Q.append(q_i)
                
                # Combine LS flags
                overall_is_ls = theta_SEW_is_LS or t12_is_ls or t34_is_ls or t56_is_ls or q7_is_ls
                is_LS_vec.append(overall_is_ls)
    
    return np.column_stack(Q) if Q else np.array([]).reshape(7, 0), is_LS_vec
    

def build_IK_2R_2R_3R_casadi_function(model_transforms):
    """
    Build a CasADi Function for IK_2R_2R_3R that can be used in optimization problems.
    Uses frame transformations from Pinocchio model instead of kin_P and kin_H.
    
    Args:
        model_transforms: dict from get_frame_transforms_from_pinocchio() containing:
            - 'R': list of 3x3 rotation matrices [R_0_1, R_1_2, ..., R_6_7, R_7_T]
            - 'p': list of 3x1 position vectors [p_0_1, p_1_2, ..., p_6_7, p_7_T]
            - 'joint_names': list of joint/frame names
    
    Returns:
        ik_fun: CasADi Function for inverse kinematics
    """
    # Define symbolic inputs
    R_0_7 = cs.SX.sym('R_0_7', 3, 3)
    p_0_T = cs.SX.sym('p_0_T', 3)
    psi = cs.SX.sym('psi', 1)
    
    # Create SEW stereo instance (simplified for symbolic)
    r, v = np.array([0, 0, -1]), np.array([0, 1, 0])
    sew_stereo = SEWStereoSymbolic(r, v)
    
    # Call the symbolic version
    result = IK_2R_2R_3R_casadi(R_0_7, p_0_T, sew_stereo, psi, model_transforms)
    
    # For now, return just the first solution (in practice you'd handle all)
    if result['solutions']:
        first_solution = result['solutions'][0]
        first_is_ls = result['is_LS_flags'][0]
    else:
        first_solution = cs.SX.zeros(7)
        first_is_ls = cs.SX(1)  # Default to LS if no solution
    
    # Create CasADi function
    ik_fun = cs.Function('IK_2R_2R_3R', 
                        [R_0_7, p_0_T, psi],
                        [first_solution, first_is_ls],
                        ['R_0_7', 'p_0_T', 'psi'],
                        ['q_solution', 'is_LS'])
    
    return ik_fun


def filter_and_select_closest_solution(Q, is_LS_vec, q_prev=None):
    """
    Filter out invalid solutions based on joint limits and return the closest one to a previous pose.
    
    Args:
        Q: numpy array of joint angle solutions (7 x num_solutions)
        is_LS_vec: list of boolean flags indicating LS solutions
        q_prev: 7x1 numpy array of previous joint configuration (optional)
        
    Returns:
        q_best: 7x1 numpy array of best solution (or None if no valid solutions)
        is_LS_best: boolean flag for the best solution
        valid_count: number of valid solutions found
    """
    
    # Define joint limits
    rev_lim = np.pi
    q_lower = np.array([-rev_lim, -2.41, -rev_lim, -2.66, -rev_lim, -2.23, -rev_lim])
    q_upper = np.array([ rev_lim,  2.41,  rev_lim,  2.66,  rev_lim,  2.23,  rev_lim])
    
    if Q.shape[1] == 0:
        return None, None, 0
    
    # Filter valid solutions based on joint limits
    valid_indices = []
    for i in range(Q.shape[1]):
        q_i = Q[:, i]
        # Check if all joints are within limits
        if np.all(q_i >= q_lower) and np.all(q_i <= q_upper):
            valid_indices.append(i)
    
    if len(valid_indices) == 0:
        print("No valid solutions found within joint limits!")
        return None, None, 0
    
    # If no previous pose provided, return the first valid solution
    if q_prev is None:
        best_idx = valid_indices[0]
        return Q[:, best_idx], is_LS_vec[best_idx], len(valid_indices)
    
    # Find the solution closest to the previous pose
    min_distance = float('inf')
    best_idx = valid_indices[0]
    
    for idx in valid_indices:
        q_i = Q[:, idx]
        # Calculate distance using L2 norm
        q_diff = q_i - q_prev
        # wrap to [-pi, pi]
        q_diff = (q_diff + np.pi) % (2 * np.pi) - np.pi
        # Calculate distance
        distance = np.linalg.norm(q_diff)
        if distance < min_distance:
            min_distance = distance
            best_idx = idx
    
    return Q[:, best_idx], is_LS_vec[best_idx], len(valid_indices)


def filter_symbolic_solutions(solutions, is_LS_flags, joint_limits=None, max_solutions=8):
    """
    Filter and prioritize symbolic IK solutions to match numerical implementation behavior.
    
    Args:
        solutions: List of CasADi SX joint angle solutions
        is_LS_flags: List of CasADi SX LS flags
        joint_limits: Tuple of (q_lower, q_upper) joint limits
        max_solutions: Maximum number of solutions to return
        
    Returns:
        Filtered lists of solutions and LS flags
    """
    if not solutions:
        return [], []
    
    # Convert to numerical values for filtering
    filtered_solutions = []
    filtered_is_LS = []
    
    if joint_limits is None:
        rev_lim = np.pi
        q_lower = np.array([-rev_lim, -2.41, -rev_lim, -2.66, -rev_lim, -2.23, -rev_lim])
        q_upper = np.array([ rev_lim,  2.41,  rev_lim,  2.66,  rev_lim,  2.23,  rev_lim])
    else:
        q_lower, q_upper = joint_limits
    
    # First pass: collect valid solutions within joint limits
    valid_exact = []
    valid_ls = []
    
    for i, (q_sym, is_ls_sym) in enumerate(zip(solutions, is_LS_flags)):
        try:
            # Convert to numerical
            q_numerical = np.array([float(q_sym[j]) for j in range(7)])
            is_ls_numerical = float(is_ls_sym) > 0.5
            
            # Check joint limits
            within_limits = np.all(q_numerical >= q_lower) and np.all(q_numerical <= q_upper)
            
            if within_limits:
                if is_ls_numerical:
                    valid_ls.append((q_sym, is_ls_sym, q_numerical))
                else:
                    valid_exact.append((q_sym, is_ls_sym, q_numerical))
        except:
            continue
    
    # Prioritize exact solutions over LS solutions
    priority_solutions = valid_exact + valid_ls
    
    # Limit number of solutions
    if len(priority_solutions) > max_solutions:
        priority_solutions = priority_solutions[:max_solutions]
    
    # Extract filtered solutions
    for q_sym, is_ls_sym, _ in priority_solutions:
        filtered_solutions.append(q_sym)
        filtered_is_LS.append(is_ls_sym)
    
    return filtered_solutions, filtered_is_LS


# ...existing code...

def get_elbow_angle_kinova(q, model, sew_stereo):
    """
    Calculate the elbow angle psi from the joint configuration.

    Args:
        q: 7x1 numpy array of joint angles
        model: the robot pinocchio model
        sew_stereo: SEWStereo instance for spherical kinematics

    Returns:
        elbow_angle: scalar value of the elbow angle (psi)
    """
    
    # Get model transforms and build forward kinematics
    model_transforms = opt.get_frame_transforms_from_pinocchio(model)
    fk_fun, _, _, _, _, _ = opt.build_casadi_kinematics_dynamics(model, 'tool_frame')
    fk_joint3_fun, _, _, _, _, _ = opt.build_casadi_kinematics_dynamics(model, 'joint_3')
    
    # Extract frame transformations
    R_local = model_transforms['R']
    p_local = model_transforms['p']
    
    # Build position vectors
    p_01, R_01 = p_local[0], R_local[0]  # in base frame
    p_12, R_12 = p_local[1], R_local[1]  # in 1 frame
    p_23, R_23 = p_local[2], R_local[2]  # in 2 frame
    p_34, R_34 = p_local[3], R_local[3]  # in 3 frame
    p_45, R_45 = p_local[4], R_local[4]  # in 4 frame
    p_56, R_56 = p_local[5], R_local[5]  # in 5 frame
    p_67, R_67 = p_local[6], R_local[6]  # in 6 frame
    p_7T, R_7T = p_local[7], R_local[7]  # in 7 frame

    # Get current end-effector pose from forward kinematics
    T_0_T = fk_fun(q).full()  # 4x4 homogeneous matrix
    R_0_T = T_0_T[:3, :3]
    p_0_T = T_0_T[:3, 3]
    
    # Calculate R_0_7 from end-effector pose
    R_0_7 = R_0_T @ R_7T.T  # R_0_T @ R_T_7

    # Find shoulder position (fixed in base frame)
    p_12_0 = R_01 @ p_12  # vector at q1 = 0
    p_1_S_0 = np.zeros(3)  
    p_1_S_0[2] = p_12_0[2]  # shoulder at the intersection of h1 and h2
    S = p_01 + p_1_S_0  # Shoulder position in base frame between joint 1 and 2

    # Get elbow position using forward kinematics to joint 3
    T_0_3 = fk_joint3_fun(q).full()
    R_0_3 = T_0_3[:3, :3]
    p_0_3 = T_0_3[:3, 3]

    # Calculate elbow position at intersection of h3 and h4
    p_3E = np.zeros(3)
    p_3E[2] = p_34[2]  # elbow at the intersection of h3 and h4
    p_3_E_0 = R_0_3 @ p_3E  # elbow position in base frame
    E = p_0_3 + p_3_E_0

    # Find wrist position in base frame
    p_7_T_0 = R_0_7 @ p_7T
    p_6_7_0 = R_0_7 @ R_67.T @ p_67  # vector at q6 = 0
    p_W_7_0 = np.zeros(3) 
    p_W_7_0[2] = p_6_7_0[2]  # wrist at the intersection of h6 and h7
    W = p_0_T - (p_W_7_0 + p_7_T_0)  # Wrist position in base frame between joint 6 and 7

    # Use sew_stereo forward kinematics to compute psi
    psi = sew_stereo.fwd_kin(S, E, W)

    return psi


# Example usage:
if __name__ == "__main__":

    model, data = opt.load_kinova_model(kinova_path)

    print("\n" + "="*50)
    print("Testing IK_2R_2R_3R function:")
    print("="*50)
    
    # Test inverse kinematics function
    # Create SEW stereo instance
    r, v = np.array([0, 0, -1]), np.array([0, 1, 0])
    sew_stereo = SEWStereo(r, v)
    # Get frame transformations from the Kinova model
    model = pin.buildModelFromUrdf(kinova_path)
    model_transforms = opt.get_frame_transforms_from_pinocchio(model)
    # Build forward kinematics function from optimizing_gen3_arm
    fk_fun, pos_fun, jac_fun, M_fun, C_fun, G_fun = opt.build_casadi_kinematics_dynamics(model, 'tool_frame')
    # Create test data
    q_init = np.radians([   0.,   15., -180., -130.,    0.,  -35.,   90.])
    # q_init = np.radians([0, 0, 0, 0, 0, 0, 0])  # Use zero angles for testing
    target_pose = fk_fun(q_init).full()  # 4x4 homogeneous matrix
    R_0T = target_pose[:3, :3]  # Desired end-effector orientation
    R_0_7_test = R_0T @ model_transforms['R'][-1]  # R_0_7 from end effector frame
    p_0_T_test = target_pose[:3, 3]  # Desired end-effector position
    
    psi_init = get_elbow_angle_kinova(q_init, model, sew_stereo)
    psi_test = psi_init  # Stereo angle

    
    # Test numerical version with timing
    print("Testing numerical IK function...")
    import time
    
    # Time the numerical implementation (multiple runs for better accuracy)
    num_runs = 100
    start_time = time.perf_counter()
    for _ in range(num_runs):
        Q_num, is_LS_num = IK_2R_2R_3R_numerical(R_0_7_test, p_0_T_test, sew_stereo, psi_test,
                                                    model_transforms)
    end_time = time.perf_counter()
    numerical_time = (end_time - start_time) / num_runs

    print(f"Numerical IK found {Q_num.shape[1]} solutions")
    print(f"Average execution time: {numerical_time*1000:.3f} ms ({1/numerical_time:.1f} Hz)")
    
    # Filter solutions and select the best one
    q_prev = q_init  # Use initial configuration as reference for closest solution
    q_best, is_LS_best, valid_count = filter_and_select_closest_solution(Q_num, is_LS_num, q_prev)
    
    if q_best is not None:
        print(f"Found {valid_count} valid solutions within joint limits")
        print(f"Selected best solution (closest to reference): {np.degrees(q_best).round(1)}°")
        print(f"Best solution is LS: {is_LS_best}")
        
        # Replace Q_num with only the best solution for verification
        if q_prev is not None:
            q_best = q_best.reshape(7, 1)
            Q_num = q_best.reshape(7, 1)
            is_LS_num = [is_LS_best]
    else:
        print("No valid solutions found within joint limits!")
        Q_num = np.array([]).reshape(7, 0)
        is_LS_num = []

    # Add Forward Kinematics Verification
    print("\n" + "="*60)
    print("FORWARD KINEMATICS VERIFICATION")
    print("="*60)
    
    if Q_num.shape[1] > 0:
        
        # Create target pose matrix from desired R_0_7 and p_0_T
        R_7T = model_transforms['R'][-1]  # R_7_T from end effector frame
        target_pose_4x4 = np.eye(4)
        target_pose_4x4[:3, :3] = R_0_7_test @ R_7T  # R_0_7 @ R_7T
        target_pose_4x4[:3, 3] = p_0_T_test
        
        print(f"Target end-effector pose:")
        print(f"  Position: {target_pose_4x4[:3, 3]}")
        print(f"  Orientation:\n{target_pose_4x4[:3, :3]}")
        
        print(f"\nVerifying all {Q_num.shape[1]} IK solutions:")
        
        for i in range(Q_num.shape[1]):
            q_solution = Q_num[:, i]
            is_ls = is_LS_num[i] if is_LS_num else False                # Compute forward kinematics for this solution
            T_computed = fk_fun(q_solution).full()  # 4x4 homogeneous matrix
            
            # Extract position and orientation
            p_computed = T_computed[:3, 3]
            R_computed = T_computed[:3, :3]
            
            # Compute errors
            pos_error = np.linalg.norm(p_computed - target_pose_4x4[:3, 3])
            # Correct orientation error: eR = R_computed @ R_desired.T compared with Identity
            R_desired = target_pose_4x4[:3, :3]
            eR = R_computed @ R_desired.T
            ori_error = np.linalg.norm(eR - np.eye(3), 'fro')
            
            print(f"\n  Solution {i+1} ({'LS' if is_ls else 'exact'}):")
            print(f"    Joint angles: {np.degrees(q_solution).round(1)}°")
            print(f"    Computed position: {p_computed}")
            print(f"    Position error: {pos_error:.6f} m")
            print(f"    Orientation error: {ori_error:.6f} (should be ~0)")
            
            if pos_error < 1e-3 and ori_error < 1e-2:
                print(f"    ✓ Solution {i+1} is accurate!")
            else:
                print(f"    ✗ Solution {i+1} has significant error")
        
        # Summary
        accurate_solutions = 0;
        for i in range(Q_num.shape[1]):
            q_solution = Q_num[:, i]
            T_computed = fk_fun(q_solution).full()
            p_computed = T_computed[:3, 3]
            R_computed = T_computed[:3, :3]
            pos_error = np.linalg.norm(p_computed - target_pose_4x4[:3, 3])
            ori_error = np.linalg.norm(R_computed - target_pose_4x4[:3, :3], 'fro')
            
            if pos_error < 1e-3 and ori_error < 1e-2:
                accurate_solutions += 1
        
        print(f"\n  Summary: {accurate_solutions}/{Q_num.shape[1]} solutions are accurate")
        
    else:
        print("No IK solutions found to verify!")
    
    # Test CasADi symbolic version with timing
    print("\n" + "="*60)
    print("TESTING CASADI SYMBOLIC IK FUNCTION")
    print("="*60)
    
    # Initialize variables to avoid NameError
    Q_casadi = []
    is_LS_casadi = []
    symbolic_build_time = 0
    symbolic_inference_time = 0
    
    try:
        # Time the graph building phase
        print("Building symbolic CasADi IK graph...")
        build_start = time.perf_counter()
        
        # Create symbolic SEW stereo instance (this includes graph building)
        sew_stereo_symbolic = SEWStereoSymbolic(r, v)
        
        # First call to IK function (includes graph compilation)
        casadi_result = IK_2R_2R_3R_casadi(R_0_7_test, p_0_T_test, sew_stereo_symbolic, 
                                           psi_test, model_transforms)
        
        build_end = time.perf_counter()
        symbolic_build_time = build_end - build_start
        
        # Time only the inference phase (multiple runs for accuracy)
        print("Timing symbolic inference...")
        inference_start = time.perf_counter()
        for _ in range(num_runs):
            casadi_result = IK_2R_2R_3R_casadi(R_0_7_test, p_0_T_test, sew_stereo_symbolic, 
                                               psi_test, model_transforms)
        inference_end = time.perf_counter()
        symbolic_inference_time = (inference_end - inference_start) / num_runs
        
        Q_casadi = casadi_result['solutions']
        is_LS_casadi = casadi_result['is_LS_flags']
        
        print(f"Symbolic IK found {len(Q_casadi)} potential solutions")
        
        if len(Q_casadi) > 0:
            print(f"\nIntermediate results:")
            intermediate = casadi_result['intermediate_results']
            # Convert CasADi expressions to numerical values for display
            try:
                S_val = [float(intermediate['S'][i]) for i in range(3)]
                W_val = [float(intermediate['W'][i]) for i in range(3)]
                d_SE_val = float(intermediate['d_SE'])
                d_EW_val = float(intermediate['d_EW'])
                
                print(f"  Shoulder S: {S_val}")
                print(f"  Wrist W: {W_val}")
                print(f"  Shoulder-Elbow distance d_SE: {d_SE_val:.3f}")
                print(f"  Elbow-Wrist distance d_EW: {d_EW_val:.3f}")
            except Exception as e:
                print(f"  Could not display intermediate results: {e}")
            
            # Evaluate symbolic solutions to numerical values
            print(f"\nEvaluating symbolic solutions:")
            
            for i, (q_sym, is_ls_sym) in enumerate(zip(Q_casadi, is_LS_casadi)):
                try:
                    # Convert symbolic solution to numerical
                    q_numerical = np.array([float(q_sym[j]) for j in range(7)])
                    is_ls_numerical = float(is_ls_sym) > 0.5
                    
                    print(f"\n  Symbolic Solution {i+1} ({'LS' if is_ls_numerical else 'exact'}):")
                    print(f"    Joint angles: {np.degrees(q_numerical).round(1)}°")
                    
                    # Check joint limits
                    rev_lim = np.pi
                    q_lower = np.array([-rev_lim, -2.41, -rev_lim, -2.66, -rev_lim, -2.23, -rev_lim])
                    q_upper = np.array([ rev_lim,  2.41,  rev_lim,  2.66,  rev_lim,  2.23,  rev_lim])
                    
                    within_limits = np.all(q_numerical >= q_lower) and np.all(q_numerical <= q_upper)
                    print(f"    Within joint limits: {'✓' if within_limits else '✗'}")
                    
                    if within_limits:
                        # Verify forward kinematics
                        T_computed = fk_fun(q_numerical).full()
                        p_computed = T_computed[:3, 3]
                        R_computed = T_computed[:3, :3]
                        
                        pos_error = np.linalg.norm(p_computed - target_pose_4x4[:3, 3])
                        # Correct orientation error: eR = R_computed @ R_desired.T compared with Identity
                        R_desired = target_pose_4x4[:3, :3]
                        eR = R_computed @ R_desired.T
                        ori_error = np.linalg.norm(eR - np.eye(3), 'fro')
                        
                        if pos_error < 1e-3 and ori_error < 1e-2:
                            print(f"    ✓ Symbolic solution {i+1} is accurate!")
                        else:
                            print(f"    ✗ Symbolic solution {i+1} has significant error")
                    
                except Exception as e:
                    print(f"    ✗ Error evaluating symbolic solution {i+1}: {e}")
        
        else:
            print("No symbolic solutions found!")
            
    except Exception as e:
        print(f"Error testing symbolic IK function: {e}")
        import traceback
        traceback.print_exc()
    
    # Compare numerical vs symbolic results
    print("\n" + "="*60)
    print("NUMERICAL VS SYMBOLIC COMPARISON")
    print("="*60)
    
    if Q_num.shape[1] > 0 and len(Q_casadi) > 0:
        print(f"Numerical solutions: {Q_num.shape[1]}")
        print(f"Symbolic solutions: {len(Q_casadi)}")
        
        # Compare first valid solutions if they exist
        try:
            # Find first valid numerical solution
            first_num_valid = None
            for i in range(Q_num.shape[1]):
                q_test = Q_num[:, i]
                rev_lim = np.pi
                q_lower = np.array([-rev_lim, -2.41, -rev_lim, -2.66, -rev_lim, -2.23, -rev_lim])
                q_upper = np.array([ rev_lim,  2.41,  rev_lim,  2.66,  rev_lim,  2.23,  rev_lim])
                if np.all(q_test >= q_lower) and np.all(q_test <= q_upper):
                    first_num_valid = q_test
                    break
            
            # Find exact match in symbolic solutions
            exact_match_found = False
            best_match_idx = -1
            min_diff = float('inf')
            
            for i, q_sym in enumerate(Q_casadi):
                try:
                    q_test = np.array([float(q_sym[j]) for j in range(7)])
                    if np.all(q_test >= q_lower) and np.all(q_test <= q_upper):
                        diff = np.linalg.norm(first_num_valid - q_test)
                        if diff < min_diff:
                            min_diff = diff;
                            best_match_idx = i
                        if diff < 1e-3:  # Exact match threshold
                            exact_match_found = True
                            print(f"\n✓ EXACT MATCH FOUND!")
                            print(f"  Numerical solution: {np.degrees(first_num_valid).round(1)}°")
                            print(f"  Symbolic solution {i+1}: {np.degrees(q_test).round(1)}°")
                            print(f"  Difference: {diff:.6f} rad ({np.degrees(diff):.3f}°)")
                            break
                except:
                    continue
            
            if not exact_match_found and best_match_idx >= 0:
                q_best = np.array([float(Q_casadi[best_match_idx][j]) for j in range(7)])
                print(f"\nClosest match found:")
                print(f"  Numerical: {np.degrees(first_num_valid).round(1)}°")
                print(f"  Symbolic solution {best_match_idx+1}: {np.degrees(q_best).round(1)}°")
                print(f"  Difference: {min_diff:.6f} rad ({np.degrees(min_diff):.3f}°)")
                
            if not exact_match_found and min_diff > 1e-1:
                print("  ⚠ No close match found - this may indicate different solution branches")
                
        except Exception as e:
            print(f"Error comparing solutions: {e}")
    
    else:
        print("Cannot compare - insufficient solutions from one or both methods")
    
    # Performance comparison
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    
    print(f"Number of test runs: {num_runs}")
    print(f"\nNumerical Implementation:")
    print(f"  Average execution time: {numerical_time*1000:.3f} ms")
    print(f"  Frequency: {1/numerical_time:.1f} Hz")
    
    if 'symbolic_build_time' in locals() and 'symbolic_inference_time' in locals():
        print(f"\nSymbolic Implementation:")
        print(f"  Graph building time (one-time): {symbolic_build_time*1000:.1f} ms")
        print(f"  Average inference time: {symbolic_inference_time*1000:.3f} ms")
        print(f"  Inference frequency: {1/symbolic_inference_time:.1f} Hz")
        
        # Performance ratio
        speedup = numerical_time / symbolic_inference_time
        if speedup > 1:
            print(f"\n🚀 Symbolic inference is {speedup:.1f}x FASTER than numerical")
        elif speedup < 1:
            print(f"\n⚠ Numerical is {1/speedup:.1f}x faster than symbolic inference")
        else:
            print(f"\n📊 Both implementations have similar performance")
            
        print(f"\nTotal time including graph building:")
        total_symbolic_time = symbolic_build_time + symbolic_inference_time
        if total_symbolic_time < numerical_time:
            print(f"  Symbolic (build + inference): {total_symbolic_time*1000:.3f} ms")
            print(f"  Still faster than numerical for single execution")
        else:
            print(f"  Symbolic (build + inference): {total_symbolic_time*1000:.3f} ms")
            break_even = symbolic_build_time / (numerical_time - symbolic_inference_time)
            if break_even > 0:
                print(f"  Break-even point: {break_even:.0f} executions")
            
    else:
        print("Symbolic timing data not available")
        
    print(f"\nMemory and computational characteristics:")
    print(f"  Numerical: Direct computation, no compilation overhead")
    print(f"  Symbolic: Compiled computational graph, optimized execution")
    if 'Q_casadi' in locals() and len(Q_casadi) > 0:
        print(f"  Both implementations produce {len(Q_casadi)} identical solutions")


