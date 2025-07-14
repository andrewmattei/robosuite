"""
Geometric Subproblems for Robotic Inverse Kinematics

This module contains the core geometric subproblem functions used in solving
inverse kinematics for robotic arms. These functions implement fundamental
geometric operations for robot kinematics based on the theory of Paden-Kahan
subproblems.

The module includes both symbolic (CasADi) and numerical (NumPy) implementations
of each subproblem, along with helper functions for building optimized CasADi
functions.

Subproblems included:
- SP0: Rotation about axis to align two vectors (perpendicular case)
- SP1: Rotation about axis to align two general vectors (plane and sphere)
- SP2: Two rotations to align two pairs of vectors (two circles)
- SP3: Rotation to achieve specific distance (circle and sphere)
- SP4: Rotation to satisfy linear constraint (circle and plane)

Each subproblem has symbolic and numerical versions, plus build functions
for creating optimized CasADi computational graphs.
"""
import casadi as cs
import numpy as np


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
