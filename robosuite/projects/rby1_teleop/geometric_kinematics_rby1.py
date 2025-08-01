"""
Geometric kinematics functions for the RBY1 robot.
This module also includes functions to load the RBY1 model and extract frame transformations.
Author: Chuizheng Kong
Date created: 2025-07-27
"""

import pinocchio as pin
import numpy as np
np.set_printoptions(precision=3, suppress=True)
import os
import robosuite.projects.shared_scripts.geometric_subproblems as gsp
import robosuite.utils.tool_box_no_ros as tb

# urdf_path = os.path.join(os.path.dirname(__file__), 'rby1a', 'urdf', 'model.urdf')  # has some problem with "capsule"
urdf_path = os.path.join(os.path.dirname(__file__), 'rbyxhand_v2', 'model_modified.urdf')

def load_rby1_model(urdf_path=urdf_path):
    """
    Load the RBY1 URDF into a Pinocchio Model/Data pair.
    """
    model = pin.buildModelFromUrdf(urdf_path)
    data = model.createData()
    return model, data

def get_frame_transforms_from_pinocchio(model, part_name):
    """
    Extract frame transformations R_i_i+1, p_i_i+1, and joint axes h_i from Pinocchio model
    for a specific part of the RBY1 robot.
    
    Args:
        model: Pinocchio model
        part_name: str, one of:
            - "right_arm": right arm joints (right_arm_0 to right_arm_6)
            - "left_arm": left arm joints (left_arm_0 to left_arm_6) 
            - "torso": torso joints (torso_0 to torso_4)
            - "base": mobile base (left_wheel, right_wheel)
            - "head": head joints (head_0, head_1)
            - "right_hand": right hand attachment (right_xhand_attach)
            - "left_hand": left hand attachment (left_xhand_attach)
            - "right_hand_thumb": right thumb joints
            - "left_hand_thumb": left thumb joints
            - "right_hand_index": right index finger joints
            - "left_hand_index": left index finger joints
            - "right_hand_mid": right middle finger joints
            - "left_hand_mid": left middle finger joints
            - "right_hand_ring": right ring finger joints
            - "left_hand_ring": left ring finger joints
            - "right_hand_pinky": right pinky finger joints
            - "left_hand_pinky": left pinky finger joints
        
    Returns:
        dict with:
        - 'R': list of 3x3 rotation matrices [R_0_1, R_1_2, ..., R_n_T]
        - 'p': list of 3x1 position vectors [p_0_1, p_1_2, ..., p_n_T] 
        - 'h': list of 3x1 joint axis vectors [h_0, h_1, ..., h_n]
        - 'joint_names': list of joint names
    """
    
    # Define joint sequences for each part
    joint_sequences = {
        "right_arm": ["right_arm_0", "right_arm_1", "right_arm_2", "right_arm_3", 
                     "right_arm_4", "right_arm_5", "right_arm_6"],
        "left_arm": ["left_arm_0", "left_arm_1", "left_arm_2", "left_arm_3",
                    "left_arm_4", "left_arm_5", "left_arm_6"], 
        "torso": ["torso_0", "torso_1", "torso_2", "torso_3", "torso_4"],  # Removed torso_base (fixed joint)
        "base": ["right_wheel", "left_wheel"],
        "head": ["head_0", "head_1"],  # Removed head_base (fixed joint)
        "right_hand": ["right_xhand_attach"],  # Attachment joint to hand
        "left_hand": ["left_xhand_attach"],   # Attachment joint to hand
        # Full finger sequences for detailed hand analysis
        "right_hand_thumb": ["right_hand_thumb_bend_joint", "right_hand_thumb_rota_joint1", "right_hand_thumb_rota_joint2"],
        "left_hand_thumb": ["left_hand_thumb_bend_joint", "left_hand_thumb_rota_joint1", "left_hand_thumb_rota_joint2"],
        "right_hand_index": ["right_hand_index_bend_joint", "right_hand_index_joint1", "right_hand_index_joint2"],
        "left_hand_index": ["left_hand_index_bend_joint", "left_hand_index_joint1", "left_hand_index_joint2"],
        "right_hand_mid": ["right_hand_mid_joint1", "right_hand_mid_joint2"],
        "left_hand_mid": ["left_hand_mid_joint1", "left_hand_mid_joint2"],
        "right_hand_ring": ["right_hand_ring_joint1", "right_hand_ring_joint2"],
        "left_hand_ring": ["left_hand_ring_joint1", "left_hand_ring_joint2"],
        "right_hand_pinky": ["right_hand_pinky_joint1", "right_hand_pinky_joint2"],
        "left_hand_pinky": ["left_hand_pinky_joint1", "left_hand_pinky_joint2"]
    }
    
    # Define end effector frames for each part
    end_effector_frames = {
        "right_arm": "right_hand_link",
        "left_arm": "left_hand_link", 
        "torso": "link_torso_4",
        "base": "base_link",
        "head": "link_head_2",
        "right_hand": "right_hand_ee_link",
        "left_hand": "left_hand_ee_link",
        "right_hand_thumb": "right_hand_thumb_rota_tip",
        "left_hand_thumb": "left_hand_thumb_rota_tip",
        "right_hand_index": "right_hand_index_rota_tip",
        "left_hand_index": "left_hand_index_rota_tip",
        "right_hand_mid": "right_hand_mid_tip",
        "left_hand_mid": "left_hand_mid_tip",
        "right_hand_ring": "right_hand_ring_tip",
        "left_hand_ring": "left_hand_ring_tip",
        "right_hand_pinky": "right_hand_pinky_tip",
        "left_hand_pinky": "left_hand_pinky_tip"
    }
    
    if part_name not in joint_sequences:
        raise ValueError(f"Unknown part_name: {part_name}. Available: {list(joint_sequences.keys())}")
    
    joint_names = joint_sequences[part_name]
    R_transforms = []
    p_transforms = []
    h_transforms = []
    
    print(f"Processing {part_name} with {len(joint_names)} joints")
    
    # Loop through joints in the specified part
    for joint_name in joint_names:
        try:
            # Get joint ID by name
            joint_id = model.getJointId(joint_name)
            
            # Get the joint from the model
            joint = model.joints[joint_id]
            
            # Get joint placement (local transformation)
            M_local = model.jointPlacements[joint_id]
            
            # Extract rotation and translation
            R_i_iplus1 = M_local.rotation.copy()     # 3x3 numpy array
            p_i_iplus1 = M_local.translation.copy()  # 3x1 numpy array
            
            # Get joint axis in local frame
            # For revolute joints, we need to extract the axis differently in Pinocchio
            if joint.shortname() == "JointModelRZ":
                h_i = np.array([0.0, 0.0, 1.0])  # Z-axis rotation
            elif joint.shortname() == "JointModelRY":  
                h_i = np.array([0.0, 1.0, 0.0])  # Y-axis rotation
            elif joint.shortname() == "JointModelRX":
                h_i = np.array([1.0, 0.0, 0.0])  # X-axis rotation
            elif hasattr(joint, 'axis'):
                h_i = joint.axis.copy()  # 3x1 joint axis
            else:
                # For fixed joints or unknown types
                print(f"Warning: Unknown joint type {joint.shortname()} for joint {joint_name}, using zero axis")
                h_i = np.array([0.0, 0.0, 0.0])
            
            R_transforms.append(R_i_iplus1)
            p_transforms.append(p_i_iplus1)
            h_transforms.append(h_i)
            
            print(f"Joint {joint_name}: R shape={R_i_iplus1.shape}, p={p_i_iplus1}, h={h_i}")
            
        except Exception as e:
            print(f"Error processing joint {joint_name}: {e}")
            continue
    
    # Add transformation from last joint to end effector frame if it exists
    end_effector_frame = end_effector_frames.get(part_name)
    if end_effector_frame:
        try:
            # Check if frame exists in model
            if model.existFrame(end_effector_frame):
                frame_id = model.getFrameId(end_effector_frame)
                frame = model.frames[frame_id]
                
                # Get transformation from parent joint to end effector
                M_joint_to_ee = frame.placement
                R_joint_to_ee = M_joint_to_ee.rotation.copy()
                p_joint_to_ee = M_joint_to_ee.translation.copy()
                
                R_transforms.append(R_joint_to_ee)
                p_transforms.append(p_joint_to_ee)
                h_transforms.append(np.array([0.0, 0.0, 0.0]))  # No axis for fixed frame
                
                print(f"Added end effector frame {end_effector_frame}")
            else:
                print(f"Warning: End effector frame {end_effector_frame} not found in model")
                
        except Exception as e:
            print(f"Error adding end effector frame {end_effector_frame}: {e}")
    
    return {
        'R': R_transforms,
        'p': p_transforms, 
        'h': h_transforms,
        'joint_names': joint_names + ([end_effector_frame] if end_effector_frame and model.existFrame(end_effector_frame) else [])
    }


def IK_3R_R_3R_SEW(S_human, E_human, W_human, model_transforms, sol_ids=None, R_0_7=None):
    """
    Geometric inverse kinematics function that matches robot arm SEW (Shoulder-Elbow-Wrist) 
    angles with human poses in a forward fashion.

    The approach:
    1. Define an elbow frame that locks the orientation of the elbow joint (joint 4) to face certain direction based on SEW_human,
        if the SEW_human is colinear, the elbow frame will just be at q3=0 config since it's defined in joint 3 frame.
    2. Joint 1,2,3: Use subproblem 2 + subproblem 1 to solve for the spherical shoulder.
    3. Joint 4: Use subproblem 1 to solve using EW_human and the elbow frame.
    4. Joint 5,6,7: If R_0_7 is provided, use subproblem 2 + subproblem 1 to solve for the wrist.

    Note that this function is designed to work with the RBY1 robot model and will only accept the joint 7 frame (R_0_7)
    instead of the end-effector frame (R_0_T) since different gripper have different end-effector frames.

    Args:
        S_human: 3x1 numpy array, human shoulder position
        E_human: 3x1 numpy array, human elbow position
        W_human: 3x1 numpy array, human wrist position
        model_transforms: dict from get_frame_transforms_from_pinocchio() containing:
            - 'R': list of 3x3 rotation matrices [R_0_1, R_1_2, ..., R_6_7, R_7_T]
            - 'p': list of 3x1 position vectors [p_0_1, p_1_2, ..., p_6_7, p_7_T]
            - 'h': list of 3x1 joint axis vectors [h_0, h_1, ..., h_6, h_7]
            - 'joint_names': list of joint/frame names
        sol_ids: list, solution IDs to consider
        R_0_7: 3x3 numpy array, optional end-effector rotation, if provided, it will be used to solve for the wrist joints (5,6,7).

    Returns:
        Q: numpy array of joint angle solutions (7 x num_solutions)
        is_LS_vec: list of boolean flags indicating LS solutions
        human_vectors: dict with human pose information
        sol_ids_used: dict with the solution indices actually used (for initialization)
    """

    ##### Notation on position #######
    # p_ij: position vector from frame i to j in local frame i with at q_i, q_i+1,...q_j-1 = 0
    # p_ij_0: position vector from frame i to j in base frame 0
    # p_i_j: position vector with q_i, q_i+1,...q_j-1 = (the actual value) calculated in "base frame" by default
    # p_i_j_k: position vector with q_i, q_i+1,...q_j-1 = (the actual value) calculated in "frame k"
    ##################################

    Q = []
    is_LS_vec = {'q12': None, 'q3': None, 'q4': None, 'q56': None, 'q7': None}
    sol_ids_used = {'q12_idx': [], 'q56_idx': []}
    
    # Extract frame transformations
    R_local = model_transforms['R']
    p_local = model_transforms['p']
    h_local = model_transforms['h']

    # Build position vectors following the standard implementation
    p_01, R_01, h1 = p_local[0], R_local[0], h_local[0]  # Base to joint 1
    p_12, R_12, h2 = p_local[1], R_local[1], h_local[1]  # Joint 1 to joint 2
    p_23, R_23, h3 = p_local[2], R_local[2], h_local[2]  # Joint 2 to joint 3
    p_34, R_34, h4 = p_local[3], R_local[3], h_local[3]  # Joint 3 to joint 4
    p_45, R_45, h5 = p_local[4], R_local[4], h_local[4]  # Joint 4 to joint 5
    p_56, R_56, h6 = p_local[5], R_local[5], h_local[5]  # Joint 5 to joint 6
    p_67, R_67, h7 = p_local[6], R_local[6], h_local[6]  # Joint 6 to joint 7

    # Robot shoulder position in base frame
    S_robot_actual = p_01

    # Robot shoulder-to-elbow vector at zero configuration
    R_02 = R_01 @ R_12 
    R_03 = R_02 @ R_23
    p_12_0 = R_01 @ p_12
    p_23_0 = R_02 @ p_23
    p_34_0 = R_03 @ p_34
    p_04 = p_01 + p_12_0 + p_23_0 + p_34_0  # Position of joint 4 in base frame
    E_robot_zero = p_04  # Robot elbow position in base frame
    SE_robot_zero = E_robot_zero - S_robot_actual  # Robot shoulder to elbow vector at zero configuration

    # Robot elbow-to-wrist vector at zero configuration
    R_04 = R_03 @ R_34
    R_05 = R_04 @ R_45
    p_45_0 = R_04 @ p_45
    p_56_0 = R_05 @ p_56
    p_06 = p_04 + p_45_0 + p_56_0  # Position of joint 6 in base frame
    W_robot_zero = p_06  # Robot wrist position in base frame
    EW_robot_zero = W_robot_zero - E_robot_zero  # Robot elbow to wrist vector at zero configuration


    # === STAGE 1: Define elbow frame to lock joint 3 orientation ===
    # Calculate human pose vectors
    SE_human = E_human - S_human  # Human shoulder to elbow vector
    EW_human = W_human - E_human  # Human elbow to wrist vector
    SW_human = W_human - S_human  # Human shoulder to wrist vector
    
    # Normalize human vectors
    SE_human_norm = SE_human / (np.linalg.norm(SE_human) + 1e-12)
    EW_human_norm = EW_human / (np.linalg.norm(EW_human) + 1e-12)
    SW_human_norm = SW_human / (np.linalg.norm(SW_human) + 1e-12)

    # TODO: experiment with singularity management with entry point memory, 
    #       since our spherical range of motion is limited, we can almost do a "wrap-to-pi" style index switching
    #       to reduce over rotations. Will default to zero facing for now.
    # Calculate the elbow frame position in rby1 base frame
    rE_ez = -SE_human_norm  # Elbow frame z-axis (pointing from elbow to shoulder)
    rE_ey = np.cross(rE_ez, EW_human_norm)  # Elbow frame y-axis (perpendicular to SE and EW)
    rE_ey_norm = np.linalg.norm(rE_ey)
    if rE_ey_norm < 1e-8: # Handle colinear case
        # opt1 crossing with h1 in base frame
        h1_0 = R_01 @ h1
        rE_ex = np.cross(SE_human_norm, h1_0)  # Elbow frame x-axis (perpendicular to SE and h1)
        rE_ex_norm = np.linalg.norm(rE_ex)
        if rE_ex_norm < 1e-8:  # If still colinear, then it's the rare case when SEW_human colinear with h1, use x-axis in base
            rE_ex = np.array([1.0, 0.0, 0.0])  # Default x-axis
        else:
            rE_ex = rE_ex / rE_ex_norm  # Normalize x-axis
        rE_ey = np.cross(rE_ez, rE_ex)
    else:
        rE_ey = rE_ey / rE_ey_norm  # Normalize y-axis
        rE_ex = np.cross(rE_ey, rE_ez)

    R_0_E = np.column_stack((rE_ex, rE_ey, rE_ez))  # Elbow frame rotation in base frame

    # calculate the angle between h3 and rE_ez in joint 3 frame (need R_3E)
    # this can be done in zero config first
    SE_robot_zero_3 = R_03.T @ -SE_robot_zero  # Shoulder to elbow vector in joint 3 frame
    theta_3E = np.arctan2(SE_robot_zero_3[0], SE_robot_zero_3[2])

    # create R_3E using rotation about the y-axis
    ey = np.array([0.0, 1.0, 0.0])
    R_3E = gsp.rot_numerical(ey, -theta_3E)
    R_0_3 = R_0_E @ R_3E.T

    # === STAGE 2: Solve for joint 1, 2, 3 angles using subproblem 2 + subproblem 1 ===
    h3_actual_1 = R_01.T @ R_0_3 @ h3  # Joint 3 axis in base frame
    h3_zero_1 = R_01.T @ R_03 @ h3
    h2_1 = R_12 @ h2
    theta1_12, theta2_12, is_LS_12 = gsp.sp_2_numerical(h3_actual_1, h3_zero_1, -h1, h2_1)
    is_LS_vec['q12'] = is_LS_12  # Store LS flag for joint 1,2
    

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
        R_0_1 = R_01 @ gsp.rot_numerical(h1, q1)  # Joint 1 rotation
        R_1_2 = R_12 @ gsp.rot_numerical(h2, q2)  # Joint 2 rotation
        R_0_2 = R_0_1 @ R_1_2  # Combined rotation from base to joint 2

        # use subproblem 1 to solve for joint 3 angle
        # now that the z-axis (h3) of joint 3 frame is aligned, need to align x-axis (h2) in joint 3 frame
        h2_actual_3 = R_0_3.T @ R_0_2 @ h2  # Joint 2 axis in joint 3 frame
        h2_zero_3 = R_23.T @ h2  # Joint 2 axis in zero configuration
        q3, q3_is_LS = gsp.sp_1_numerical(h2_zero_3, h2_actual_3, -h3)
        is_LS_vec['q3'] = q3_is_LS  # Store LS flag for joint 3

        # === STAGE 3: Solve for joint 4 angle using subproblem 1 ===
        # calculate human elbow-wrist vector in joint 4 frame
        EW_human_norm_4 = R_34.T @ R_0_3.T @ EW_human_norm  # Human elbow to wrist vector in joint 4 frame
        EW_robot_zero_4 = R_34.T @ R_03.T @ EW_robot_zero  # Robot elbow to wrist vector at zero configuration
        EW_robot_zero_4_norm = EW_robot_zero_4 / (np.linalg.norm(EW_robot_zero_4))
        q4, q4_is_LS = gsp.sp_1_numerical(EW_robot_zero_4_norm, EW_human_norm_4, h4)
        is_LS_vec['q4'] = q4_is_LS  # Store LS flag for joint 4

        # === STAGE 4: Solve for joint 5,6,7 angles using subproblem 2 + subproblem 1 ===
        if R_0_7 is not None:
            # normalize R_0_7 using nearest orthogonalization
            U_0_7,_, V_0_7_T = np.linalg.svd(R_0_7)
            R_0_7 = U_0_7 @ V_0_7_T

            # Calculate wrist rotation in joint 5 frame
            R_3_4 = R_34 @ gsp.rot_numerical(h4, q4)
            R_0_4 = R_0_3 @ R_3_4  # Rotation from base to joint 4
            h6_5 = R_56 @ h6  # Joint 6 axis in joint 5 frame

            # desired joint 7 axis in frame 5 based on desired R_0_7
            h7_actual_5 = (R_0_4 @ R_45).T @ R_0_7 @ h7  # Joint 7 axis in joint 5 frame

            # joint 7 axis in zero configuration in frame 5
            h7_zero_5 = R_56 @ R_67 @ h7

            # Use subproblem 2 to find q5, q6
            theta5_56, theta6_56, is_LS_56 = gsp.sp_2_numerical(h7_actual_5, h7_zero_5, -h5, h6_5)
            is_LS_vec['q56'] = is_LS_56  # Store LS flag for joint 5,6

            # If sol_ids provided, use specific solution indices for q5, q6
            if sol_ids is not None and 'q56_idx' in sol_ids:
                q56_indices = [sol_ids['q56_idx']]
            else:
                q56_indices = range(len(theta5_56))

            # Iterate through joint 5,6 solutions
            for j in q56_indices:
                if j >= len(theta5_56):
                    continue

                q5 = theta5_56[j]
                q6 = theta6_56[j]

                # Build rotation matrices up to joint 6
                R_4_5 = R_45 @ gsp.rot_numerical(h5, q5)
                R_5_6 = R_56 @ gsp.rot_numerical(h6, q6)
                R_0_6 = R_0_4 @ R_4_5 @ R_5_6  # Combined rotation from base to joint 6

                # Use subproblem 1 to solve for joint 7 angle
                h6_actual_7 = R_0_7.T @ R_0_6 @ h6  # Joint 6 axis in joint 7 frame
                h6_zero_7 = R_67.T @ h6  # Joint 6 axis in zero configuration
                q7, q7_is_LS = gsp.sp_1_numerical(h6_zero_7, h6_actual_7, -h7)
                is_LS_vec['q7'] = q7_is_LS  # Store LS flag for joint 7

                # Combine joint angles
                q_solution = np.array([q1, q2, q3, q4, q5, q6, q7])
                Q.append(q_solution)
                
                # Store solution indices used
                sol_ids_used['q12_idx'].append(i)
                sol_ids_used['q56_idx'].append(j)
        else:
            q5 = 0.0
            q6 = 0.0
            q7 = 0.0

            # Combine joint angles
            q_solution = np.array([q1, q2, q3, q4, q5, q6, q7])
            Q.append(q_solution)
            
            # Combine LS flags (OR operation)

            sol_ids_used['q12_idx'].append(i)
            sol_ids_used['q56_idx'].append(0)

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
        'SE_robot_zero': SE_robot_zero,
        'EW_robot_zero': EW_robot_zero
    }
    
    return Q, is_LS_vec, human_vectors, sol_ids_used


def IK_3R_R_3R_SEW_wrist_lock(S_human, E_human, W_human, model_transforms, sol_ids=None, R_0_7=None):
    """
    Geometric inverse kinematics function that matches robot arm SEW (Shoulder-Elbow-Wrist) 
    angles with human poses in a forward fashion.

    The approach:
    1. Define an elbow frame that locks the orientation of the elbow joint (joint 4) to face certain direction based on SEW_human,
        if the SEW_human is colinear, the elbow frame will just be at q3=0 config since it's defined in joint 3 frame.
    2. Joint 1,2,3: Use subproblem 2 + subproblem 1 to solve for the spherical shoulder.
    3. Joint 4: Use subproblem 1 to solve using EW_human and the elbow frame.
    4. Joint 5,6,7: If R_0_7 is provided, use subproblem 2 + subproblem 1 to solve for the wrist.

    Note that this function is designed to work with the RBY1 robot model and will only accept the joint 7 frame (R_0_7)
    instead of the end-effector frame (R_0_T) since different gripper have different end-effector frames.

    Args:
        S_human: 3x1 numpy array, human shoulder position
        E_human: 3x1 numpy array, human elbow position
        W_human: 3x1 numpy array, human wrist position
        model_transforms: dict from get_frame_transforms_from_pinocchio() containing:
            - 'R': list of 3x3 rotation matrices [R_0_1, R_1_2, ..., R_6_7, R_7_T]
            - 'p': list of 3x1 position vectors [p_0_1, p_1_2, ..., p_6_7, p_7_T]
            - 'h': list of 3x1 joint axis vectors [h_0, h_1, ..., h_6, h_7]
            - 'joint_names': list of joint/frame names
        sol_ids: list, solution IDs to consider
        R_0_7: 3x3 numpy array, optional end-effector rotation, if provided, it will be used to solve for the wrist joints (5,6,7).

    Returns:
        Q: numpy array of joint angle solutions (7 x num_solutions)
        is_LS_vec: list of boolean flags indicating LS solutions
        human_vectors: dict with human pose information
        sol_ids_used: dict with the solution indices actually used (for initialization)
    """

    ##### Notation on position #######
    # p_ij: position vector from frame i to j in local frame i with at q_i, q_i+1,...q_j-1 = 0
    # p_ij_0: position vector from frame i to j in base frame 0
    # p_i_j: position vector with q_i, q_i+1,...q_j-1 = (the actual value) calculated in "base frame" by default
    # p_i_j_k: position vector with q_i, q_i+1,...q_j-1 = (the actual value) calculated in "frame k"
    ##################################

    Q = []
    is_LS_vec = []
    sol_ids_used = {'q12_idx': [], 'q56_idx': []}
    
    # Extract frame transformations
    R_local = model_transforms['R']
    p_local = model_transforms['p']
    h_local = model_transforms['h']

    # Build position vectors following the standard implementation
    p_01, R_01, h1 = p_local[0], R_local[0], h_local[0]  # Base to joint 1
    p_12, R_12, h2 = p_local[1], R_local[1], h_local[1]  # Joint 1 to joint 2
    p_23, R_23, h3 = p_local[2], R_local[2], h_local[2]  # Joint 2 to joint 3
    p_34, R_34, h4 = p_local[3], R_local[3], h_local[3]  # Joint 3 to joint 4
    p_45, R_45, h5 = p_local[4], R_local[4], h_local[4]  # Joint 4 to joint 5
    p_56, R_56, h6 = p_local[5], R_local[5], h_local[5]  # Joint 5 to joint 6
    p_67, R_67, h7 = p_local[6], R_local[6], h_local[6]  # Joint 6 to joint 7

    # Robot shoulder position in base frame
    S_robot_actual = p_01

    # Robot shoulder-to-elbow vector at zero configuration
    R_02 = R_01 @ R_12 
    R_03 = R_02 @ R_23
    p_12_0 = R_01 @ p_12
    p_23_0 = R_02 @ p_23
    p_34_0 = R_03 @ p_34
    p_04 = p_01 + p_12_0 + p_23_0 + p_34_0  # Position of joint 4 in base frame
    E_robot_zero = p_04  # Robot elbow position in base frame
    SE_robot_zero = E_robot_zero - S_robot_actual  # Robot shoulder to elbow vector at zero configuration

    # Robot elbow-to-wrist vector at zero configuration
    R_04 = R_03 @ R_34
    R_05 = R_04 @ R_45
    p_45_0 = R_04 @ p_45
    p_56_0 = R_05 @ p_56
    p_06 = p_04 + p_45_0 + p_56_0  # Position of joint 6 in base frame
    W_robot_zero = p_06  # Robot wrist position in base frame
    EW_robot_zero = W_robot_zero - E_robot_zero  # Robot elbow to wrist vector at zero configuration


    # === STAGE 1: Define elbow frame to lock joint 3 orientation ===
    # Calculate human pose vectors
    SE_human = E_human - S_human  # Human shoulder to elbow vector
    EW_human = W_human - E_human  # Human elbow to wrist vector
    SW_human = W_human - S_human  # Human shoulder to wrist vector
    
    # Normalize human vectors
    SE_human_norm = SE_human / (np.linalg.norm(SE_human) + 1e-12)
    EW_human_norm = EW_human / (np.linalg.norm(EW_human) + 1e-12)
    SW_human_norm = SW_human / (np.linalg.norm(SW_human) + 1e-12)

    # Calculate the elbow frame position in rby1 base frame
    rE_ez = -SE_human_norm  # Elbow frame z-axis (pointing from elbow to shoulder)
    rE_ey = np.cross(rE_ez, EW_human_norm)  # Elbow frame y-axis (perpendicular to SE and EW)
    rE_ey_norm = np.linalg.norm(rE_ey)
    if rE_ey_norm < 1e-8: # Handle colinear case
        # opt1 crossing with h1 in base frame
        h1_0 = R_01 @ h1
        rE_ex = np.cross(SE_human_norm, h1_0)  # Elbow frame x-axis (perpendicular to SE and h1)
        rE_ex_norm = np.linalg.norm(rE_ex)
        if rE_ex_norm < 1e-8:  # If still colinear, then it's the rare case when SEW_human colinear with h1, use x-axis in base
            rE_ex = np.array([1.0, 0.0, 0.0])  # Default x-axis
        else:
            rE_ex = rE_ex / rE_ex_norm  # Normalize x-axis
        rE_ey = np.cross(rE_ez, rE_ex)
    else:
        rE_ey = rE_ey / rE_ey_norm  # Normalize y-axis
        rE_ex = np.cross(rE_ey, rE_ez)

    R_0_E = np.column_stack((rE_ex, rE_ey, rE_ez))  # Elbow frame rotation in base frame

    # calculate the angle between h3 and rE_ez in joint 3 frame (need R_3E)
    # this can be done in zero config first
    SE_robot_zero_3 = R_03.T @ -SE_robot_zero  # Shoulder to elbow vector in joint 3 frame
    theta_3E = np.arctan2(SE_robot_zero_3[0], SE_robot_zero_3[2])

    # create R_3E using rotation about the y-axis
    ey = np.array([0.0, 1.0, 0.0])
    R_3E = gsp.rot_numerical(ey, -theta_3E)
    R_0_3 = R_0_E @ R_3E.T

    # === STAGE 2: Solve for joint 1, 2, 3 angles using subproblem 2 + subproblem 1 ===
    h3_actual_1 = R_01.T @ R_0_3 @ h3  # Joint 3 axis in base frame
    h3_zero_1 = R_01.T @ R_03 @ h3
    h2_1 = R_12 @ h2
    theta1_12, theta2_12, is_LS_12 = gsp.sp_2_numerical(h3_actual_1, h3_zero_1, -h1, h2_1)
    

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
        R_0_1 = R_01 @ gsp.rot_numerical(h1, q1)  # Joint 1 rotation
        R_1_2 = R_12 @ gsp.rot_numerical(h2, q2)  # Joint 2 rotation
        R_0_2 = R_0_1 @ R_1_2  # Combined rotation from base to joint 2

        # use subproblem 1 to solve for joint 3 angle
        # now that the z-axis (h3) of joint 3 frame is aligned, need to align x-axis (h2) in joint 3 frame
        h2_actual_3 = R_0_3.T @ R_0_2 @ h2  # Joint 2 axis in joint 3 frame
        h2_zero_3 = R_23.T @ h2  # Joint 2 axis in zero configuration
        q3, q3_is_LS = gsp.sp_1_numerical(h2_zero_3, h2_actual_3, -h3)

        # === STAGE 3: Solve for joint 4 angle using subproblem 1 ===
        # calculate human elbow-wrist vector in joint 4 frame
        EW_human_norm_4 = R_34.T @ R_0_3.T @ EW_human_norm  # Human elbow to wrist vector in joint 4 frame
        EW_robot_zero_4 = R_34.T @ R_03.T @ EW_robot_zero  # Robot elbow to wrist vector at zero configuration
        EW_robot_zero_4_norm = EW_robot_zero_4 / (np.linalg.norm(EW_robot_zero_4))
        q4, q4_is_LS = gsp.sp_1_numerical(EW_robot_zero_4_norm, EW_human_norm_4, h4)

        # === STAGE 4: Solve for joint 5,6,7 angles using subproblem 2 + subproblem 1 ===
        if R_0_7 is not None:
            # normalize R_0_7 using nearest orthogonalization
            U_0_7,_, V_0_7_T = np.linalg.svd(R_0_7)
            R_0_7 = U_0_7 @ V_0_7_T

            # Calculate wrist rotation in joint 5 frame
            R_3_4 = R_34 @ gsp.rot_numerical(h4, q4)
            R_0_4 = R_0_3 @ R_3_4  # Rotation from base to joint 4

            # Decompose R_0_7 into rpy in R_0_5 frame with zero q5 config
            R_0_5_zero = R_0_4 @ R_45
            R_567 = R_0_5_zero.T @ R_0_7  # Rotation from joint 5 to joint 7 in zero configuration
            r_x, r_y, r_z = tb.rotation_matrix_to_euler(R_567)
            # if S_human[1] > 0: # right side
            #     print("r_x{:.2f}, r_y{:.2f}, r_z{:.2f}".format(r_x, r_y, r_z))
            q5 = r_z
            q6 = r_y
            q7 = 0

            # Combine joint angles
            q_solution = np.array([q1, q2, q3, q4, q5, q6, q7])
            Q.append(q_solution)
                
            # Combine LS flags (OR operation)
            overall_is_ls = is_LS_12 or q3_is_LS or q4_is_LS
            is_LS_vec.append(overall_is_ls)
            
            # Store solution indices used
            sol_ids_used['q12_idx'].append(i)
            sol_ids_used['q56_idx'].append(0)

        else:
            q5 = 0.0
            q6 = 0.0
            q7 = 0.0

            # Combine joint angles
            q_solution = np.array([q1, q2, q3, q4, q5, q6, q7])
            Q.append(q_solution)
            
            # Combine LS flags (OR operation)
            overall_is_ls = is_LS_12 or q3_is_LS or q4_is_LS
            is_LS_vec.append(overall_is_ls)

            sol_ids_used['q12_idx'].append(i)
            sol_ids_used['q56_idx'].append(0)

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
        'SE_robot_zero': SE_robot_zero,
        'EW_robot_zero': EW_robot_zero
    }
    
    return Q, is_LS_vec, human_vectors, sol_ids_used


if __name__ == "__main__":
    # Example usage
    model, data = load_rby1_model()
    model_transforms = get_frame_transforms_from_pinocchio(model, "right_arm")
    
    # Define human pose (example values)
    S_human = np.array([0.0, -0.2, 0.0])
    E_human = np.array([0.0, -0.2, -0.25])
    W_human = np.array([0.0, -0.2, -0.5])

    R_0_7 = gsp.rot_numerical(-np.array([0.0, 1.0, 0.0]), np.pi/4)  # Example wrist rotation
    
    # Call the IK function
    Q, is_LS_vec, human_vectors, sol_ids_used = IK_3R_R_3R_SEW(S_human, E_human, W_human, model_transforms, R_0_7=R_0_7)

    print("Joint angles (Q):", Q)
    print("Is LS solution:", is_LS_vec) 
