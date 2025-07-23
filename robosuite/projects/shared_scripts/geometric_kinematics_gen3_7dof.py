import pinocchio as pin
import pinocchio.casadi as cpin

from robosuite.projects.shared_scripts.geometric_subproblems import *

np.set_printoptions(precision=3, suppress=True)
import os
import traceback
from matplotlib import pyplot as plt

import robosuite.projects.shared_scripts.optimizing_gen3_arm as opt
# Import the SEW Stereo class for spherical-elbow-wrist kinematics
from robosuite.projects.shared_scripts.sew_stereo import SEWStereo, SEWStereoSymbolic, build_sew_stereo_casadi_functions

kinova_path = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'models', 'assets', 'robots',
                                'dual_kinova3', 'leonardo.urdf')


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
    p_W7 = np.zeros(3) 
    p_W7[1] = p_67[1]      # wrist at the intersection of h6 and h7
    p_W_7_0 = R_0_7 @ R_67.T @ p_W7 # vector at q6 = 0
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


def IK_2R_2R_3R_auto_elbow(R_0_7, p_0_T, sew_stereo, model_transforms, q_prev=None):
    """
    Numerical implementation of IK_2R_2R_3R inverse kinematics function.
    Uses frame transformations from Pinocchio model instead of kin_P and kin_H.
    Automatically determines elbow position based on the end-effector position.
    Automatically stabilizes singularity solutions. 
    TODO: haven't solved subproblem 4 for elbow angle yet, might need to redefine
    an SEW system.
    
    Args:
        R_0_7: 3x3 numpy array - desired end-effector orientation
        p_0_T: 3x1 numpy array - desired end-effector position
        sew_stereo: SEWStereo instance for spherical kinematics,
            Assume the following setup for humanoid bimanual:
            r, v = np.array([1, 0, 0]), np.array([0, 1, 0])
            sew_stereo = SEWStereo(r, v)
        model_transforms: dict from get_frame_transforms_from_pinocchio() containing:
            - 'R': list of 3x3 rotation matrices [R_0_1, R_1_2, ..., R_6_7, R_7_T]
            - 'p': list of 3x1 position vectors [p_0_1, p_1_2, ..., p_6_7, p_7_T]
            - 'joint_names': list of joint/frame names
        q_prev: if None, then assume Right arm configuration, return all solutions.

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
    p_W7 = np.zeros(3) 
    p_W7[1] = p_67[1]      # wrist at the intersection of h6 and h7
    p_W_7_0 = R_0_7 @ R_67.T @ p_W7 # vector at q6 = 0
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
    
    # create a fake elbow in -y -z direction of the T frame
    E_fake = np.array([0, -0.5, -0.3])  # Fake elbow position in base frame
    R_0_T = R_0_7 @ R_7T  # Rotation matrix from base to T frame
    E_fake_0 = p_0_T + R_0_T @ E_fake  # Fake elbow position in 0 frame
    psi_auto = sew_stereo.fwd_kin(S, E_fake_0, W)  # Calculate stereo angle for fake elbow

    # Use SEW inverse kinematics
    e_CE, n_SEW = sew_stereo.inv_kin(S, W, psi_auto)
    
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


def IK_2R_2R_3R_SEW(S_human, E_human, W_human, model_transforms, sol_ids=None, R_0_T_kinova=None):
    """
    Numerical inverse kinematics function that matches robot arm SEW (Shoulder-Elbow-Wrist) 
    angles with human poses in a forward fashion.
    
    The approach:
    1. Joints 1,2: Use subproblem 2 to orient shoulder segment toward human shoulder-elbow direction
    2. Joints 3,4: Use subproblem 2 to orient elbow segment toward human elbow-wrist direction  
    3. Joints 5,6,7: Set to zeros by default, or if R_0_T_kinova is provided, use subproblem 2 for q5,q6 
       and subproblem 1 for q7 to match end-effector orientation
    
    Args:
        S_human: 3x1 numpy array - human shoulder position in robot base frame
        E_human: 3x1 numpy array - human elbow position in robot base frame
        W_human: 3x1 numpy array - human wrist position in robot base frame
        model_transforms: dict from get_frame_transforms_from_pinocchio() containing:
            - 'R': list of 3x3 rotation matrices [R_0_1, R_1_2, ..., R_6_7, R_7_T]
            - 'p': list of 3x1 position vectors [p_0_1, p_1_2, ..., p_6_7, p_7_T]
            - 'joint_names': list of joint/frame names
        sol_ids: dict with solution indices for consistent solution selection:
            - 'q12_idx': index for joints 1,2 solution (0 or 1)
            - 'q34_idx': index for joints 3,4 solution (0 or 1)
            - 'q56_idx': index for joints 5,6 solution (when R_0_T_kinova is provided)
            If None, returns all solutions
        R_0_T_kinova: 3x3 numpy array - desired end-effector orientation in robot base frame
               If provided, will solve for wrist joints (q5,q6,q7) to match this orientation
    
    Returns:
        Q: numpy array of joint angle solutions (7 x num_solutions)
        is_LS_vec: list of boolean flags indicating LS solutions
        human_vectors: dict with human pose information
        sol_ids_used: dict with the solution indices actually used (for initialization)
    """
    
    Q = []
    is_LS_vec = []
    sol_ids_used = {'q12_idx': [], 'q34_idx': [], 'q56_idx': []}
    
    # Extract frame transformations
    R_local = model_transforms['R']
    p_local = model_transforms['p']
    
    # Build position vectors following the standard implementation
    p_01, R_01 = p_local[0], R_local[0]  # base to joint 1
    p_12, R_12 = p_local[1], R_local[1]  # joint 1 to joint 2
    p_23, R_23 = p_local[2], R_local[2]  # joint 2 to joint 3
    p_34, R_34 = p_local[3], R_local[3]  # joint 3 to joint 4
    p_45, R_45 = p_local[4], R_local[4]  # joint 4 to joint 5
    p_56, R_56 = p_local[5], R_local[5]  # joint 5 to joint 6
    p_67, R_67 = p_local[6], R_local[6]  # joint 6 to joint 7
    p_7T, R_7T = p_local[7], R_local[7]  # joint 7 to end-effector
    
    # Calculate human pose vectors
    SE_human = E_human - S_human  # Human shoulder to elbow vector
    EW_human = W_human - E_human  # Human elbow to wrist vector
    SW_human = W_human - S_human  # Human shoulder to wrist vector
    
    # Normalize human vectors
    SE_human_norm = SE_human / (np.linalg.norm(SE_human) + 1e-8)
    EW_human_norm = EW_human / (np.linalg.norm(EW_human) + 1e-8)
    SW_human_norm = SW_human / (np.linalg.norm(SW_human) + 1e-8)

    # Robot shoulder position (intersection of joint 1 and 2 axes)
    p_12_0 = R_01 @ p_12  # vector at q1 = 0
    p_1_S_0 = np.zeros(3)
    p_1_S_0[2] = p_12_0[2]  # shoulder at the intersection of h1 and h2
    S_robot = p_01 + p_1_S_0  # Robot shoulder position in base frame
    
    # Robot shoulder-to-elbow vector at zero configuration
    p_S2_0 = p_12_0 - p_1_S_0  # Vector from shoulder to joint 2
    R_02 = R_01 @ R_12
    R_03 = R_02 @ R_23
    p_3E = np.zeros(3)
    p_3E[2] = p_34[2]  # elbow at the intersection of h3 and h4
    p_2E_0 = R_02 @ p_23 + R_03 @ p_3E  # vector at q2 = 0, q3 = 0
    SE_robot_zero = p_S2_0 + p_2E_0  # Robot shoulder-to-elbow vector at zero config
    E_robot = S_robot + SE_robot_zero # Robot elbow position at zero configuration in base frame
    
    p_E4_0 = R_03 @ (p_34 - p_3E)  # Vector from elbow to joint 4 at zero config
    
    # Robot elbow-to-wrist vector at zero configuration
    R_04 = R_03 @ R_34
    R_05 = R_04 @ R_45
    R_06 = R_05 @ R_56
    p_45_0 = R_04 @ p_45
    p_56_0 = R_05 @ p_56
    p_67_0 = R_06 @ p_67
    p_W7_0 = np.zeros(3)
    p_W7_0[2] = p_67_0[2]  # wrist at the intersection of h6 and h7
    p_6W_0 = p_67_0 - p_W7_0
    EW_robot_zero = p_E4_0 + p_45_0 + p_56_0 + p_6W_0  # Robot elbow-to-wrist vector at zero config


    
    # Joint axes (all joints rotate about z-axis in their local frames)
    ez = np.array([0, 0, 1])
    
    # === STAGE 1: Solve joints 1,2 to match shoulder-elbow direction ===
    # Project human SE vector to joint 1 frame for subproblem 2
    SE_human_1 = R_01.T @ SE_human_norm  # Human SE vector in joint 1 frame
    SE_robot_1 = R_01.T @ SE_robot_zero  # Robot SE vector in joint 1 frame at zero config
    SE_robot_1_norm = SE_robot_1 / np.linalg.norm(SE_robot_1)
    
    # Joint axes in joint 1 frame
    h_1 = ez  # Joint 1 axis in joint 1 frame
    h_2 = R_12 @ ez  # Joint 2 axis in joint 1 frame
    
    # Use subproblem 2 to find q1, q2 that align robot SE with human SE
    theta1_12, theta2_12, is_LS_12 = sp_2_numerical(SE_human_1, SE_robot_1_norm, -h_1, h_2)
    
    # Ensure we have arrays (sp_2_numerical might return scalars)
    if np.isscalar(theta1_12):
        theta1_12 = np.array([theta1_12])
    if np.isscalar(theta2_12):
        theta2_12 = np.array([theta2_12])
    
    # If sol_ids provided, use specific solution indices
    if sol_ids is not None and 'q12_idx' in sol_ids:
        q12_indices = [sol_ids['q12_idx']]
    else:
        q12_indices = range(len(theta1_12))
    
    # Iterate through joint 1,2 solutions
    for i in q12_indices:
        if i >= len(theta1_12):
            continue  # Skip if index out of range
            
        q1 = theta1_12[i]
        q2 = theta2_12[i]
        
        # Build rotation matrices up to joint 2
        R_0_1 = R_01 @ rot_numerical(ez, q1)  # Joint 1 rotation
        R_1_2 = R_12 @ rot_numerical(ez, q2)  # Joint 2 rotation
        R_0_2 = R_0_1 @ R_1_2
        
        # === STAGE 2: Solve joints 3,4 to match elbow-wrist direction ===
        # Calculate actual robot elbow position after applying q1, q2
        p_S2_actual = R_0_1 @ (p_12 - p_1_S_0)  # Actual vector from shoulder to joint 2
        p_2E_actual = R_0_2 @ p_23 + (R_0_2 @ R_23) @ p_3E
        E_robot_actual = S_robot + p_S2_actual + p_2E_actual  # Actual robot elbow position
        
        # Project human EW vector to joint 3 frame for subproblem 2
        R_0_3 = R_0_2 @ R_23
        EW_human_3 = R_0_3.T @ EW_human_norm  # Human EW vector in joint 3 frame
        EW_robot_3 = R_03.T @ EW_robot_zero  # Robot EW vector in joint 3 frame at zero config
        EW_robot_3_norm = EW_robot_3 / np.linalg.norm(EW_robot_3)
        
        # Joint axes in joint 3 frame
        h_3 = ez  # Joint 3 axis in 3 frame
        h_4 = R_34 @ ez  # Joint 4 axis in 3 frame
        
        # Use subproblem 2 to find q3, q4 that align robot EW with human EW
        theta1_34, theta2_34, is_LS_34 = sp_2_numerical(EW_human_3, EW_robot_3_norm, -h_3, h_4)
        
        # Ensure we have arrays
        if np.isscalar(theta1_34):
            theta1_34 = np.array([theta1_34])
        if np.isscalar(theta2_34):
            theta2_34 = np.array([theta2_34])
        
        # If sol_ids provided, use specific solution indices
        if sol_ids is not None and 'q34_idx' in sol_ids:
            q34_indices = [sol_ids['q34_idx']]
        else:
            q34_indices = range(len(theta1_34))
        
        # Iterate through joint 3,4 solutions
        for j in q34_indices:
            if j >= len(theta1_34):
                continue  # Skip if index out of range
                
            q3 = theta1_34[j]
            q4 = theta2_34[j]
            
            # === STAGE 3: Set joints 5,6,7 ===
            # Build rotation matrices up to joint 4
            R_0_3 = R_0_2 @ R_23
            R_2_3 = R_23 @ rot_numerical(ez, q3)
            R_3_4 = R_34 @ rot_numerical(ez, q4)
            R_0_4 = R_0_2 @ R_2_3 @ R_3_4
            
            if R_0_T_kinova is not None:
                # Use the provided end-effector orientation to solve for wrist joints
                # Following the same approach as IK_2R_2R_3R_numerical()
                R_0_7 = R_0_T_kinova @ R_7T.T
                
                # Joint axes in their local frames
                h_5 = ez  # Joint 5 axis in 5 frame
                h_6 = R_56 @ ez  # Joint 6 axis in 5 frame
                
                # Desired joint 7 axis in frame 5 (based on desired end-effector orientation)
                h_7_act_5 = (R_0_4 @ R_45).T @ R_0_7 @ ez
                
                # Joint 7 axis at zero configuration in frame 5
                h_7_zero_5 = R_56 @ R_67 @ ez
                
                # Use subproblem 2 to find q5, q6
                theta5_56, theta6_56, is_LS_56 = sp_2_numerical(h_7_act_5, h_7_zero_5, -h_5, h_6)
                
                # Ensure we have arrays
                if np.isscalar(theta5_56):
                    theta5_56 = np.array([theta5_56])
                if np.isscalar(theta6_56):
                    theta6_56 = np.array([theta6_56])
                
                # If sol_ids provided, use specific solution indices for q5, q6
                if sol_ids is not None and 'q56_idx' in sol_ids:
                    q56_indices = [sol_ids['q56_idx']]
                else:
                    q56_indices = range(len(theta5_56))
                
                # Iterate through q5, q6 solutions
                for k in q56_indices:
                    if k >= len(theta5_56):
                        continue  # Skip if index out of range
                        
                    q5 = theta5_56[k]
                    q6 = theta6_56[k]
                    
                    # Build rotation matrix up to joint 6
                    R_4_5 = R_45 @ rot_numerical(ez, q5)
                    R_5_6 = R_56 @ rot_numerical(ez, q6)
                    R_0_6 = R_0_4 @ R_4_5 @ R_5_6
                    
                    # Use subproblem 1 to find q7
                    h_7_final = ez  # Joint 7 axis in frame 7
                    h_6_act_7 = R_0_7.T @ R_0_6 @ ez  # Actual orientation of joint 6 axis in frame 7
                    h_6_zero_7 = R_67.T @ ez  # Joint 6 axis at zero config in frame 7
                    
                    q7, q7_is_ls = sp_1_numerical(h_6_zero_7, h_6_act_7, -h_7_final)
                    
                    # Combine joint angles
                    q_solution = np.array([q1, q2, q3, q4, q5, q6, q7])
                    Q.append(q_solution)
                    
                    # Combine LS flags (OR operation)
                    overall_is_ls = is_LS_12 or is_LS_34 or is_LS_56 or q7_is_ls
                    is_LS_vec.append(overall_is_ls)
                    
                    # Store solution indices used
                    sol_ids_used['q12_idx'].append(i)
                    sol_ids_used['q34_idx'].append(j)
                    sol_ids_used['q56_idx'].append(k)
            else:
                # Default behavior: set wrist joints to zero
                q5 = 0.0
                q6 = 0.0
                q7 = 0.0
                
                # Combine joint angles
                q_solution = np.array([q1, q2, q3, q4, q5, q6, q7])
                Q.append(q_solution)
                
                # Combine LS flags (OR operation)
                overall_is_ls = is_LS_12 or is_LS_34
                is_LS_vec.append(overall_is_ls)
                
                # Store solution indices used (wrist joints set to 0 index for consistency)
                sol_ids_used['q12_idx'].append(i)
                sol_ids_used['q34_idx'].append(j)
                sol_ids_used['q56_idx'].append(0)  # Single solution for zero configuration
    
    # Convert to numpy array if we have solutions
    if Q:
        Q = np.column_stack(Q)  # 7 x num_solutions
    else:
        Q = np.zeros((7, 0))  # Empty array with correct shape
    
    # Store human vectors for analysis
    human_vectors = {
        'S_human': S_human,
        'E_human': E_human,
        'W_human': W_human,
        'SE_human': SE_human,
        'EW_human': EW_human,
        'SW_human': SW_human,
        'SE_human_norm': SE_human_norm,
        'EW_human_norm': EW_human_norm,
        'SW_human_norm': SW_human_norm,
        'S_robot': S_robot,
        'SE_robot_zero': SE_robot_zero,
        'EW_robot_zero': EW_robot_zero
    }
    
    return Q, is_LS_vec, human_vectors, sol_ids_used


def get_robot_SEW_from_q(q, model):
    """
    Extract robot SEW (Shoulder, Elbow, Wrist) positions from joint angles.
    
    Args:
        q: 7x1 numpy array of joint angles
        model_transforms: dict from get_frame_transforms_from_pinocchio() containing:
            - 'R': list of 3x3 rotation matrices [R_0_1, R_1_2, ..., R_6_7, R_7_T]
            - 'p': list of 3x1 position vectors [p_0_1, p_1_2, ..., p_6_7, p_7_T]
            - 'joint_names': list of joint/frame names
    
    Returns:
        dict with 'S', 'E', 'W' positions in robot base frame
    """
    
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
    
    return {
        'S': S,
        'E': E,
        'W': W
    }


def filter_and_select_closest_solution(Q, is_LS_vec, q_prev=None):
    """
    Filter out invalid solutions based on joint limits and return the closest one to a previous pose.
    If the closest solution is outside joint limits, return q_prev.
    
    Args:
        Q: numpy array of joint angle solutions (7 x num_solutions)
        is_LS_vec: list of boolean flags indicating LS solutions
        q_prev: 7x1 numpy array of previous joint configuration (optional)
        
    Returns:
        q_best: 7x1 numpy array of best solution (or q_prev if closest is invalid, or None if q_prev is None)
        is_LS_best: boolean flag for the best solution (or None if q_prev is used/None)
        joint_limit_violated: boolean flag indicating joint limit violations caused fallback behavior
    """
    
    # Define joint limits
    rev_lim = np.pi
    q_lower = np.array([-rev_lim, -2.41, -rev_lim, -2.66, -rev_lim, -2.23, -rev_lim])
    q_upper = np.array([ rev_lim,  2.41,  rev_lim,  2.66,  rev_lim,  2.23,  rev_lim])
    
    if Q.shape[1] == 0:
        return None, None, False
    
    # If no previous pose provided, return None
    if q_prev is None:
        return None, None, True  # joint_limit_violated = True (no q_prev, no valid solution)
    
    # Find the solution closest to the previous pose (regardless of joint limits)
    min_distance = float('inf')
    best_idx = 0
    
    for idx in range(Q.shape[1]):
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
    
    # Get the closest solution
    q_closest = Q[:, best_idx]
    
    # Check if the closest solution is within joint limits
    within_limits = np.all(q_closest >= q_lower) and np.all(q_closest <= q_upper)
    
    if within_limits:
        # Return the closest valid solution
        return q_closest, is_LS_vec[best_idx], False  # joint_limit_violated = False
    else:
        # Return q_prev as fallback
        return q_prev, None, True  # joint_limit_violated = True (used q_prev due to limits)


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
    r, v = np.array([1, 0, 0]), np.array([0, 1, 0])
    sew_stereo = SEWStereo(r, v)
    # Get frame transformations from the Kinova model
    model = pin.buildModelFromUrdf(kinova_path)
    model_transforms = opt.get_frame_transforms_from_pinocchio(model)
    # Build forward kinematics function from optimizing_gen3_arm
    fk_fun, pos_fun, jac_fun, M_fun, C_fun, G_fun = opt.build_casadi_kinematics_dynamics(model, 'tool_frame')
    # Create test data
    # q_init = np.radians([   0.,   15., -180., -130.,    0.,  -35.,   90.])
    # q_init = np.radians([   90.,   90., -90., 90.,    0.,  0.,   90.])
    q_init = np.array([1.571, 1.571, -1.571, 1.571, 0.000, 0.524, 1.571])
    # q_init = np.radians([0, 45, 0, 0, 0, 0, 0])  # Use zero angles for testing
    target_pose = fk_fun(q_init).full()  # 4x4 homogeneous matrix
    R_0T = target_pose[:3, :3]  # Desired end-effector orientation
    R_0_7_test = R_0T @ model_transforms['R'][-1]  # R_0_7 from end effector frame
    p_0_T_test = target_pose[:3, 3]  # Desired end-effector position

    # autoelbow IK:
    # q_autoEB = IK_2R_2R_3R_auto_elbow(R_0_7_test, p_0_T_test, sew_stereo, model_transforms, None)

    psi_init = get_elbow_angle_kinova(q_init, model, sew_stereo)
    # psi_test = psi_init  # Stereo angle
    psi_test = 0
    print(f"Elbow angle (psi) for initial configuration: {psi_test:.3f} rad ({np.degrees(psi_test):.1f}°)")

    
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
        print(f"  Desired Elbow angle (psi): {psi_test:.3f} rad ({np.degrees(psi_test):.1f}°)")
        
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
    
    # # Test CasADi symbolic version with timing
    # print("\n" + "="*60)
    # print("TESTING CASADI SYMBOLIC IK FUNCTION")
    # print("="*60)
    
    # # Initialize variables to avoid NameError
    # Q_casadi = []
    # is_LS_casadi = []
    # symbolic_build_time = 0
    # symbolic_inference_time = 0
    
    # try:
    #     # Time the graph building phase
    #     print("Building symbolic CasADi IK graph...")
    #     build_start = time.perf_counter()
        
    #     # Create symbolic SEW stereo instance (this includes graph building)
    #     sew_stereo_symbolic = SEWStereoSymbolic(r, v)
        
    #     # First call to IK function (includes graph compilation)
    #     casadi_result = IK_2R_2R_3R_casadi(R_0_7_test, p_0_T_test, sew_stereo_symbolic, 
    #                                        psi_test, model_transforms)
        
    #     build_end = time.perf_counter()
    #     symbolic_build_time = build_end - build_start
        
    #     # Time only the inference phase (multiple runs for accuracy)
    #     print("Timing symbolic inference...")
    #     inference_start = time.perf_counter()
    #     for _ in range(num_runs):
    #         casadi_result = IK_2R_2R_3R_casadi(R_0_7_test, p_0_T_test, sew_stereo_symbolic, 
    #                                            psi_test, model_transforms)
    #     inference_end = time.perf_counter()
    #     symbolic_inference_time = (inference_end - inference_start) / num_runs
        
    #     Q_casadi = casadi_result['solutions']
    #     is_LS_casadi = casadi_result['is_LS_flags']
        
    #     print(f"Symbolic IK found {len(Q_casadi)} potential solutions")
        
    #     if len(Q_casadi) > 0:
    #         print(f"\nIntermediate results:")
    #         intermediate = casadi_result['intermediate_results']
    #         # Convert CasADi expressions to numerical values for display
    #         try:
    #             S_val = [float(intermediate['S'][i]) for i in range(3)]
    #             W_val = [float(intermediate['W'][i]) for i in range(3)]
    #             d_SE_val = float(intermediate['d_SE'])
    #             d_EW_val = float(intermediate['d_EW'])
                
    #             print(f"  Shoulder S: {S_val}")
    #             print(f"  Wrist W: {W_val}")
    #             print(f"  Shoulder-Elbow distance d_SE: {d_SE_val:.3f}")
    #             print(f"  Elbow-Wrist distance d_EW: {d_EW_val:.3f}")
    #         except Exception as e:
    #             print(f"  Could not display intermediate results: {e}")
            
    #         # Evaluate symbolic solutions to numerical values
    #         print(f"\nEvaluating symbolic solutions:")
            
    #         for i, (q_sym, is_ls_sym) in enumerate(zip(Q_casadi, is_LS_casadi)):
    #             try:
    #                 # Convert symbolic solution to numerical
    #                 q_numerical = np.array([float(q_sym[j]) for j in range(7)])
    #                 is_ls_numerical = float(is_ls_sym) > 0.5
                    
    #                 print(f"\n  Symbolic Solution {i+1} ({'LS' if is_ls_numerical else 'exact'}):")
    #                 print(f"    Joint angles: {np.degrees(q_numerical).round(1)}°")
                    
    #                 # Check joint limits
    #                 rev_lim = np.pi
    #                 q_lower = np.array([-rev_lim, -2.41, -rev_lim, -2.66, -rev_lim, -2.23, -rev_lim])
    #                 q_upper = np.array([ rev_lim,  2.41,  rev_lim,  2.66,  rev_lim,  2.23,  rev_lim])
                    
    #                 within_limits = np.all(q_numerical >= q_lower) and np.all(q_numerical <= q_upper)
    #                 print(f"    Within joint limits: {'✓' if within_limits else '✗'}")
                    
    #                 if within_limits:
    #                     # Verify forward kinematics
    #                     T_computed = fk_fun(q_numerical).full()
    #                     p_computed = T_computed[:3, 3]
    #                     R_computed = T_computed[:3, :3]
                        
    #                     pos_error = np.linalg.norm(p_computed - target_pose_4x4[:3, 3])
    #                     # Correct orientation error: eR = R_computed @ R_desired.T compared with Identity
    #                     R_desired = target_pose_4x4[:3, :3]
    #                     eR = R_computed @ R_desired.T
    #                     ori_error = np.linalg.norm(eR - np.eye(3), 'fro')
                        
    #                     if pos_error < 1e-3 and ori_error < 1e-2:
    #                         print(f"    ✓ Symbolic solution {i+1} is accurate!")
    #                     else:
    #                         print(f"    ✗ Symbolic solution {i+1} has significant error")
                    
    #             except Exception as e:
    #                 print(f"    ✗ Error evaluating symbolic solution {i+1}: {e}")
        
    #     else:
    #         print("No symbolic solutions found!")
            
    # except Exception as e:
    #     print(f"Error testing symbolic IK function: {e}")
    #     import traceback
    #     traceback.print_exc()
    
    # # Compare numerical vs symbolic results
    # print("\n" + "="*60)
    # print("NUMERICAL VS SYMBOLIC COMPARISON")
    # print("="*60)
    
    # if Q_num.shape[1] > 0 and len(Q_casadi) > 0:
    #     print(f"Numerical solutions: {Q_num.shape[1]}")
    #     print(f"Symbolic solutions: {len(Q_casadi)}")
        
    #     # Compare first valid solutions if they exist
    #     try:
    #         # Find first valid numerical solution
    #         first_num_valid = None
    #         for i in range(Q_num.shape[1]):
    #             q_test = Q_num[:, i]
    #             rev_lim = np.pi
    #             q_lower = np.array([-rev_lim, -2.41, -rev_lim, -2.66, -rev_lim, -2.23, -rev_lim])
    #             q_upper = np.array([ rev_lim,  2.41,  rev_lim,  2.66,  rev_lim,  2.23,  rev_lim])
    #             if np.all(q_test >= q_lower) and np.all(q_test <= q_upper):
    #                 first_num_valid = q_test
    #                 break
            
    #         # Find exact match in symbolic solutions
    #         exact_match_found = False
    #         best_match_idx = -1
    #         min_diff = float('inf')
            
    #         for i, q_sym in enumerate(Q_casadi):
    #             try:
    #                 q_test = np.array([float(q_sym[j]) for j in range(7)])
    #                 if np.all(q_test >= q_lower) and np.all(q_test <= q_upper):
    #                     diff = np.linalg.norm(first_num_valid - q_test)
    #                     if diff < min_diff:
    #                         min_diff = diff;
    #                         best_match_idx = i
    #                     if diff < 1e-3:  # Exact match threshold
    #                         exact_match_found = True
    #                         print(f"\n✓ EXACT MATCH FOUND!")
    #                         print(f"  Numerical solution: {np.degrees(first_num_valid).round(1)}°")
    #                         print(f"  Symbolic solution {i+1}: {np.degrees(q_test).round(1)}°")
    #                         print(f"  Difference: {diff:.6f} rad ({np.degrees(diff):.3f}°)")
    #                         break
    #             except:
    #                 continue
            
    #         if not exact_match_found and best_match_idx >= 0:
    #             q_best = np.array([float(Q_casadi[best_match_idx][j]) for j in range(7)])
    #             print(f"\nClosest match found:")
    #             print(f"  Numerical: {np.degrees(first_num_valid).round(1)}°")
    #             print(f"  Symbolic solution {best_match_idx+1}: {np.degrees(q_best).round(1)}°")
    #             print(f"  Difference: {min_diff:.6f} rad ({np.degrees(min_diff):.3f}°)")
                
    #         if not exact_match_found and min_diff > 1e-1:
    #             print("  ⚠ No close match found - this may indicate different solution branches")
                
    #     except Exception as e:
    #         print(f"Error comparing solutions: {e}")
    
    # else:
    #     print("Cannot compare - insufficient solutions from one or both methods")
    
    # # Performance comparison
    # print("\n" + "="*60)
    # print("PERFORMANCE COMPARISON")
    # print("="*60)
    
    # print(f"Number of test runs: {num_runs}")
    # print(f"\nNumerical Implementation:")
    # print(f"  Average execution time: {numerical_time*1000:.3f} ms")
    # print(f"  Frequency: {1/numerical_time:.1f} Hz")
    
    # if 'symbolic_build_time' in locals() and 'symbolic_inference_time' in locals():
    #     print(f"\nSymbolic Implementation:")
    #     print(f"  Graph building time (one-time): {symbolic_build_time*1000:.1f} ms")
    #     print(f"  Average inference time: {symbolic_inference_time*1000:.3f} ms")
    #     print(f"  Inference frequency: {1/symbolic_inference_time:.1f} Hz")
        
    #     # Performance ratio
    #     speedup = numerical_time / symbolic_inference_time
    #     if speedup > 1:
    #         print(f"\n🚀 Symbolic inference is {speedup:.1f}x FASTER than numerical")
    #     elif speedup < 1:
    #         print(f"\n⚠ Numerical is {1/speedup:.1f}x faster than symbolic inference")
    #     else:
    #         print(f"\n📊 Both implementations have similar performance")
            
    #     print(f"\nTotal time including graph building:")
    #     total_symbolic_time = symbolic_build_time + symbolic_inference_time
    #     if total_symbolic_time < numerical_time:
    #         print(f"  Symbolic (build + inference): {total_symbolic_time*1000:.3f} ms")
    #         print(f"  Still faster than numerical for single execution")
    #     else:
    #         print(f"  Symbolic (build + inference): {total_symbolic_time*1000:.3f} ms")
    #         break_even = symbolic_build_time / (numerical_time - symbolic_inference_time)
    #         if break_even > 0:
    #             print(f"  Break-even point: {break_even:.0f} executions")
            
    # else:
    #     print("Symbolic timing data not available")
        
    # print(f"\nMemory and computational characteristics:")
    # print(f"  Numerical: Direct computation, no compilation overhead")
    # print(f"  Symbolic: Compiled computational graph, optimized execution")
    # if 'Q_casadi' in locals() and len(Q_casadi) > 0:
    #     print(f"  Both implementations produce {len(Q_casadi)} identical solutions")


