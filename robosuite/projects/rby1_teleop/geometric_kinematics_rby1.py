import pinocchio as pin
import numpy as np
np.set_printoptions(precision=3, suppress=True)
import os

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
    is_LS_vec = []
    sol_ids_used = {'q12_idx': [], 'q34_idx': [], 'q56_idx': []}
    
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
    
    # Calculate human pose vectors
    SE_human = E_human - S_human  # Human shoulder to elbow vector
    EW_human = W_human - E_human  # Human elbow to wrist vector
    SW_human = W_human - S_human  # Human shoulder to wrist vector
    
    # Normalize human vectors
    SE_human_norm = SE_human / (np.linalg.norm(SE_human) + 1e-8)
    EW_human_norm = EW_human / (np.linalg.norm(EW_human) + 1e-8)
    SW_human_norm = SW_human / (np.linalg.norm(SW_human) + 1e-8)

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
    R_0_3 = R_0_E # locking the robot elbow axis facing to the human elbow frame

    # Robot shoulder position in base frame
    S_robot = p_01

    # Robot shoulder-to-elbow vector at zero configuration
    R_02 = R_01 @ R_12 
    R_03 = R_02 @ R_23
    p_12_0 = R_01 @ p_12
    p_23_0 = R_02 @ p_23
    p_34_0 = R_03 @ p_34
    p_04 = p_01 + p_12_0 + p_23_0 + p_34_0  # Position of joint 4 in base frame
    E_robot_zero = p_04  # Robot elbow position in base frame
    SE_robot_zero = E_robot_zero - S_robot  # Robot shoulder to elbow vector at zero configuration

    # Robot elbow-to-wrist vector at zero configuration
    R_04 = R_03 @ R_34
    R_05 = R_04 @ R_45
    p_45_0 = R_04 @ p_45
    p_56_0 = R_05 @ p_56
    p_06 = p_04 + p_45_0 + p_56_0  # Position of joint 6 in base frame
    W_robot_zero = p_06  # Robot wrist position in base frame
    EW_robot_zero = W_robot_zero - E_robot_zero  # Robot elbow to wrist vector at zero configuration
