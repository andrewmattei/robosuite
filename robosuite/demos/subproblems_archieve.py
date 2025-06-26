import casadi as cs
import numpy as np

def sp_1(p1, p2, k):
    """
    Subproblem 1: Cone and Point
    
    theta = sp_1(p1, p2, k) finds theta such that
        rot(k, theta)*p1 = p2
    If there's no solution, theta minimizes the least-squares residual
        || rot(k, theta)*p1 - p2 ||
    
    The problem is ill-posed if p1 or p2 are parallel to k
    
    Args:
        p1: 3x1 CasADi SX/MX vector
        p2: 3x1 CasADi SX/MX vector
        k:  3x1 CasADi SX/MX vector with norm(k) = 1
        
    Returns:
        Dictionary with symbolic expressions for theta and LS condition
    """
    
    # Check if p1 and p2 are essentially the same (early return)
    diff_norm = cs.sqrt(cs.dot(p1 - p2, p1 - p2))
    eps = 1e-12
    
    # Normalize k to ensure it's a unit vector
    norm_k = cs.sqrt(cs.dot(k, k))
    k_normalized = k / norm_k
    
    # Project p1 and p2 onto the plane perpendicular to k
    # pp = p1 - (p1 · k) * k
    # qp = p2 - (p2 · k) * k
    kTp1 = cs.dot(k_normalized, p1)
    kTp2 = cs.dot(k_normalized, p2)
    
    pp = p1 - kTp1 * k_normalized
    qp = p2 - kTp2 * k_normalized
    
    # Normalize the projected vectors
    norm_pp = cs.sqrt(cs.dot(pp, pp))
    norm_qp = cs.sqrt(cs.dot(qp, qp))
    
    epp = pp / norm_pp
    eqp = qp / norm_qp
    
    # Use subproblem 0 to find the angle between projected vectors
    theta = sp_0(epp, eqp, k_normalized)
    
    # Least squares condition: ||p1|| != ||p2||
    norm_p1 = cs.sqrt(cs.dot(p1, p1))
    norm_p2 = cs.sqrt(cs.dot(p2, p2))
    norm_diff = cs.fabs(norm_p1 - norm_p2)
    proj_diff = cs.fabs(kTp1 - kTp2)
    is_LS_condition = cs.fmax(norm_diff, proj_diff)  # > 1e-8 means LS
    
    # Handle the case where vectors are too close
    theta_result = cs.if_else(diff_norm < eps, 0.0, theta)
    
    return {
        'theta': theta_result,
        'is_LS_condition': is_LS_condition,
        'norm_diff': norm_diff,
        'proj_diff': proj_diff
    }

def sp_1_numerical(p1, p2, k):
    """
    Numerical version of sp_1 that returns the rotation angle and LS flag.
    Based on the reference subproblem1 implementation.
    
    Args:
        p1: 3x1 numpy array
        p2: 3x1 numpy array
        k:  3x1 numpy array with norm(k) = 1
        
    Returns:
        theta: scalar rotation angle in radians
        is_LS: boolean flag indicating if solution is least-squares
    """
    import numpy as np
    import warnings
    
    eps = np.finfo(np.float64).eps
    norm = np.linalg.norm
    
    # Early return if p1 and p2 are essentially the same
    if norm(np.subtract(p1, p2)) < np.sqrt(eps):
        return 0.0, False
    
    # Normalize k to ensure it's a unit vector
    k = np.divide(k, norm(k))
    
    # Project p1 and p2 onto the plane perpendicular to k
    # pp = p1 - (p1 · k) * k
    # qp = p2 - (p2 · k) * k
    pp = np.subtract(p1, np.dot(p1, k) * k)
    qp = np.subtract(p2, np.dot(p2, k) * k)
    
    # Normalize the projected vectors
    epp = np.divide(pp, norm(pp))
    eqp = np.divide(qp, norm(qp))
    
    # Use subproblem 0 to find the angle between projected vectors
    theta = sp_0_numerical(epp, eqp, k)
    
    # Check least squares condition
    norm_diff = abs(norm(p1) - norm(p2))
    proj_diff = abs(np.dot(k, p1) - np.dot(k, p2))
    is_LS = norm_diff > 1e-8 or proj_diff > 1e-8
    
    # Warn if norms are significantly different (following reference implementation)
    if norm_diff > norm(p1) * 1e-2:
        warnings.warn("||p|| and ||q|| must be the same!!!")
    
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
    Subproblem 2: Two Cones
    
    [theta1, theta2] = sp_2(p1, p2, k1, k2) finds theta1, theta2 such that
        rot(k1, theta1)*p1 = rot(k2, theta2)*p2
    If there's no solution, minimize the least-squares residual
        || rot(k1, theta1)*p1 - rot(k2, theta2)*p2 ||
    
    If the problem is well-posed, there may be 1 or 2 solutions
    (These may be exact or least-squares solutions)
    theta1 and theta2 are column vectors of the solutions
    
    The third return value is_LS is a flag which is true if (theta1, theta2) 
    is a least-squares solution
    
    The problem is ill-posed if (p1, k1), (p2, k2), or (k1, k2) are parallel
    
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
    is_LS_condition = cs.fabs(norm_p1 - norm_p2)
    
    # Rescale for least-squares case
    p1_normalized = p1 / norm_p1
    p2_normalized = p2 / norm_p2
    
    # Cross products
    KxP1 = cs.cross(k1, p1_normalized)
    KxP2 = cs.cross(k2, p2_normalized)
    
    # Build matrices A_1 and A_2
    A_1 = cs.horzcat(KxP1, -cs.cross(k1, KxP1))
    A_2 = cs.horzcat(KxP2, -cs.cross(k2, KxP2))
    
    # Compute radius squares
    radius_1_sq = cs.dot(KxP1, KxP1)
    radius_2_sq = cs.dot(KxP2, KxP2)
    
    # Dot products
    k1_d_p1 = cs.dot(k1, p1_normalized)
    k2_d_p2 = cs.dot(k2, p2_normalized)
    k1_d_k2 = cs.dot(k1, k2)
    
    # Least-squares fraction
    ls_frac = 1 / (1 - k1_d_k2**2)
    
    # Alpha values
    alpha_1 = ls_frac * (k1_d_p1 - k1_d_k2 * k2_d_p2)
    alpha_2 = ls_frac * (k2_d_p2 - k1_d_k2 * k1_d_p1)
    
    # Least-squares solution components
    x_ls_1 = alpha_2 * cs.mtimes(A_1.T, k2) / radius_1_sq
    x_ls_2 = alpha_1 * cs.mtimes(A_2.T, k1) / radius_2_sq
    x_ls = cs.vertcat(x_ls_1, x_ls_2)
    
    # Normal vector for perturbation
    n_sym = cs.cross(k1, k2)
    
    # Pseudo-inverses
    pinv_A1 = A_1.T / radius_1_sq
    pinv_A2 = A_2.T / radius_2_sq
    A_perp_tilde = cs.vertcat(cs.mtimes(pinv_A1, n_sym), cs.mtimes(pinv_A2, n_sym))
    
    # Check condition for exact vs LS solution
    x_ls_12_norm_sq = cs.dot(x_ls[0:2], x_ls[0:2])
    
    # Exact solutions (when ||x_ls(1:2)|| < 1)
    xi = cs.sqrt(1 - x_ls_12_norm_sq) / cs.sqrt(cs.dot(A_perp_tilde[0:2], A_perp_tilde[0:2]))
    sc_1 = x_ls + xi * A_perp_tilde
    sc_2 = x_ls - xi * A_perp_tilde
    
    # Extract angles for both solutions
    theta1_exact_1 = cs.atan2(sc_1[0], sc_1[1])
    theta1_exact_2 = cs.atan2(sc_2[0], sc_2[1])
    theta2_exact_1 = cs.atan2(sc_1[2], sc_1[3])
    theta2_exact_2 = cs.atan2(sc_2[2], sc_2[3])
    
    # Least-squares solution (when ||x_ls(1:2)|| >= 1)
    theta1_ls = cs.atan2(x_ls[0], x_ls[1])
    theta2_ls = cs.atan2(x_ls[2], x_ls[3])
    
    return {
        'theta1_exact_1': theta1_exact_1,
        'theta1_exact_2': theta1_exact_2,
        'theta2_exact_1': theta2_exact_1,
        'theta2_exact_2': theta2_exact_2,
        'theta1_ls': theta1_ls,
        'theta2_ls': theta2_ls,
        'x_ls_12_norm_sq': x_ls_12_norm_sq,
        'is_LS_condition': is_LS_condition,
        'exact_condition': x_ls_12_norm_sq - 1  # < 0 means exact solutions exist
    }

def sp_2_numerical(p, q, k1, k2):
    """
    Solves canonical geometric subproblem 2, solve for two coincident, nonparallel
    axes rotation a link according to
    
        q = rot(k1, theta1) * rot(k2, theta2) * p
    
    solves by looking for the intersection between cones of
    
        rot(k1,-theta1)q = rot(k2, theta2) * p
        
    may have 0, 1, or 2 solutions
       
    
    :type    p: numpy.array
    :param   p: 3 x 1 vector before rotations
    :type    q: numpy.array
    :param   q: 3 x 1 vector after rotations
    :type    k1: numpy.array
    :param   k1: 3 x 1 rotation axis 1 unit vector
    :type    k2: numpy.array
    :param   k2: 3 x 1 rotation axis 2 unit vector
    :rtype:  tuple
    :return: (theta1_array, theta2_array, is_LS) where theta angles are arrays of solutions in radians
    """
    import numpy as np
    import warnings
    
    eps = np.finfo(np.float64).eps
    norm = np.linalg.norm
    
    k12 = np.dot(k1, k2)
    pk = np.dot(p, k2)
    qk = np.dot(q, k1)
    
    # check if solution exists
    if (np.abs(1 - k12**2) < eps):
        warnings.warn("No solution - k1 != k2")
        return np.array([]), np.array([]), True
    
    a = np.matmul([[k12, -1], [-1, k12]], [pk, qk]) / (k12**2 - 1)
    
    bb = (np.dot(p, p) - np.dot(a, a) - 2*a[0]*a[1]*k12)
    if (np.abs(bb) < eps): 
        bb = 0
    
    if (bb < 0):
        warnings.warn("No solution - no intersection found between cones")
        return np.array([]), np.array([]), True
    
    gamma = np.sqrt(bb) / norm(np.cross(k1, k2))
    if (np.abs(gamma) < eps):
        cm = np.array([k1, k2, np.cross(k1, k2)]).T
        c1 = np.dot(cm, np.hstack((a, gamma)))
        theta2, _ = sp_1_numerical(p, c1, k2)
        theta1, _ = sp_1_numerical(q, c1, k1)
        theta1 = -theta1  # Apply negative sign as in reference
        return np.array([theta1]), np.array([theta2]), False
    
    cm = np.array([k1, k2, np.cross(k1, k2)]).T
    c1 = np.dot(cm, np.hstack((a, gamma)))
    c2 = np.dot(cm, np.hstack((a, -gamma)))
    theta1_1, _ = sp_1_numerical(q, c1, k1)
    theta1_2, _ = sp_1_numerical(q, c2, k1)
    theta1_1 = -theta1_1  # Apply negative sign as in reference
    theta1_2 = -theta1_2  # Apply negative sign as in reference
    theta2_1, _ = sp_1_numerical(p, c1, k2)
    theta2_2, _ = sp_1_numerical(p, c2, k2)
    
    return np.array([theta1_1, theta1_2]), np.array([theta2_1, theta2_2]), False

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


def sp_3(p, q, k, d):
    """
    Subproblem 3: Cone and Sphere (Symbolic version)
    
    theta = sp_3(p, q, k, d) finds theta such that
        || q + rot(k, theta)*p || = d
    If there's no solution, minimize the least-squares residual
        | || rot(k, theta)*p - q || - d |
    
    If the problem is well-posed, there may be 1 or 2 exact solutions
    
    The problem is ill-posed if (p, k) or (q, k) are parallel
    
    Args:
        p: 3x1 CasADi SX/MX vector - position vector of point p
        q: 3x1 CasADi SX/MX vector - position vector of point q
        k: 3x1 CasADi SX/MX vector - rotation axis for point p  
        d: scalar CasADi SX/MX - desired distance
        
    Returns:
        Dictionary with symbolic expressions for solutions
    """
    
    # Project p and q onto the plane perpendicular to k
    # pp = p - (p · k) * k
    # qp = q - (q · k) * k
    kTp = cs.dot(k, p)
    kTq = cs.dot(k, q)
    
    pp = p - kTp * k
    qp = q - kTq * k
    
    # Compute the discriminant for the distance constraint
    k_dot_pq = cs.dot(k, p + q)
    dpsq = d**2 - k_dot_pq**2
    
    # Compute the cosine of the half-angle between projected vectors
    norm_pp = cs.sqrt(cs.dot(pp, pp))
    norm_qp = cs.sqrt(cs.dot(qp, qp))
    
    bb = -(cs.dot(pp, pp) + cs.dot(qp, qp) - dpsq) / (2 * norm_pp * norm_qp)
    
    # Find the base angle between projected vectors using subproblem 1
    pp_normalized = pp / norm_pp
    qp_normalized = qp / norm_qp
    sp1_result = sp_1(pp_normalized, qp_normalized, k)
    theta_base = sp1_result['theta']
    
    # Compute the cone angle
    phi = cs.acos(cs.fabs(bb))  # Use fabs to ensure valid acos argument
    
    # Two potential solutions
    theta_1 = theta_base + phi
    theta_2 = theta_base - phi
    
    # Condition checks
    dpsq_valid = dpsq >= 0
    bb_valid = cs.fabs(bb) <= 1
    solution_exists = dpsq_valid * bb_valid
    phi_nonzero = cs.fabs(phi) > 1e-10
    
    return {
        'theta_1': theta_1,          # First solution
        'theta_2': theta_2,          # Second solution  
        'theta_single': theta_base,  # Single solution when phi ≈ 0
        'phi': phi,                  # Cone half-angle
        'solution_exists': solution_exists,  # Whether valid solutions exist
        'two_solutions': phi_nonzero,        # Whether there are two solutions
        'is_ls_condition': 1 - solution_exists  # > 0 means no exact solution
    }

def sp_3_numerical(p, q, k, d):
    """
    Numerical version of sp_3 based on the reference subproblem3 implementation.
    
    Solves canonical geometric subproblem 3, solve for theta in
    an elbow joint according to
    
        || q + rot(k, theta)*p || = d
        
    may have 0, 1, or 2 solutions
    
    Args:
        p: 3x1 numpy array - position vector of point p
        q: 3x1 numpy array - position vector of point q  
        k: 3x1 numpy array - rotation axis for point p
        d: scalar - desired distance between p and q after rotation
        
    Returns:
        theta: numpy array of solution(s)
        is_LS: boolean flag (always False for this geometric approach)
    """
    import warnings
    
    norm = np.linalg.norm
    
    # Project p and q onto the plane perpendicular to k
    pp = np.subtract(p, np.dot(np.dot(p, k), k))
    qp = np.subtract(q, np.dot(np.dot(q, k), k))
    
    # Compute the discriminant for the distance constraint
    dpsq = d**2 - ((np.dot(k, np.add(p, q)))**2)
    
    # Check if solution exists
    if dpsq < 0:
        warnings.warn("No solution - no rotation can achieve specified distance (dpsq < 0)")
        return np.array([]), True
    
    # Compute the cosine of the half-angle between projected vectors
    bb = -(np.dot(pp, pp) + np.dot(qp, qp) - dpsq) / (2 * norm(pp) * norm(qp))
    
    if np.abs(bb) > 1:
        warnings.warn("No solution - no rotation can achieve specified distance (|bb| > 1)")
        return np.array([]), True
    
    # Find the base angle between projected vectors using subproblem 1
    theta, _ = sp_1_numerical(pp/norm(pp), qp/norm(qp), k)
    
    # Compute the cone angle
    phi = np.arccos(np.abs(bb))  # Use abs to ensure valid arccos argument
    
    if np.abs(phi) > 1e-10:  # Two solutions
        return np.array([theta + phi, theta - phi]), False
    else:  # Single solution
        return np.array([theta]), False

def build_sp_3_casadi_function():
    """
    Build a CasADi Function for sp_3 that can be used in optimization problems.
    
    Returns:
        sp_3_fun: CasADi Function with inputs [p, q, k, d] and outputs [theta_1, theta_2, theta_single, is_ls_condition]
    """
    # Define symbolic inputs
    p = cs.SX.sym('p', 3)
    q = cs.SX.sym('q', 3)
    k = cs.SX.sym('k', 3)
    d = cs.SX.sym('d', 1)
    
    # Call the symbolic version
    result = sp_3(p, q, k, d)
    
    # Create CasADi function
    sp_3_fun = cs.Function('sp_3', 
                          [p, q, k, d],
                          [result['theta_1'], 
                           result['theta_2'], 
                           result['theta_single'],
                           result['is_ls_condition']],
                          ['p', 'q', 'k', 'd'],
                          ['theta_1', 'theta_2', 'theta_single', 'is_ls_condition'])
    
    return sp_3_fun