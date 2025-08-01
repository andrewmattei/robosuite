import numpy as np
from collections.abc import Iterable
from scipy.spatial.transform import Rotation

from robosuite.controllers.parts.controller import Controller
from robosuite.projects.shared_scripts.geometric_kinematics_gen3_7dof import IK_2R_2R_3R_SEW, filter_and_select_closest_solution, get_robot_SEW_from_q, rot_numerical
from robosuite.projects.shared_scripts.geometric_kinematics_gen3_7dof import IK_2R_2R_3R_SEW_wrist_lock
import robosuite.projects.shared_scripts.optimizing_gen3_arm as opt
from robosuite.projects.shared_scripts.geometric_kinematics_gen3_7dof import kinova_path


class SEWMimicController(Controller):
    """
    Controller for controlling robot arm by mimicking human SEW (Shoulder, Elbow, Wrist) poses.
    Uses the IK_2R_2R_3R_SEW function to compute joint angles that match human arm configurations,
    then applies joint-space PD control to reach those configurations.

    The controller takes human SEW poses in body-centric coordinates as input and outputs joint torques. 
    Body-centric coordinates have the shoulder center as origin with:
    - X-axis: forward
    - Y-axis: right to left  
    - Z-axis: down to up (hip to shoulder)
    
    The controller transforms these to robot base frame before computing inverse kinematics.
    If any of the SEW poses are empty/None, the controller holds the current pose.

    Args:
        sim (MjSim): Simulator instance this controller will pull robot state updates from

        ref_name (str): Name of controlled robot arm's end effector (from robot XML)

        joint_indexes (dict): Each key contains sim reference indexes to relevant robot joint information, namely:

            :`'joints'`: list of indexes to relevant robot joints
            :`'qpos'`: list of indexes to relevant robot joint positions
            :`'qvel'`: list of indexes to relevant robot joint velocities

        actuator_range (2-tuple of array of float): 2-Tuple (low, high) representing the robot joint actuator range

        input_max (float or Iterable of float): Maximum safe workspace bounds for clipping. Can be either be
            a scalar (same value for all action dimensions), or a list (specific values for each dimension). 
            Default is 1.5 meters. Actions will be clipped to stay within these bounds.

        input_min (float or Iterable of float): Minimum safe workspace bounds for clipping. Can be either be
            a scalar (same value for all action dimensions), or a list (specific values for each dimension). 
            Default is -1.5 meters. Actions will be clipped to stay within these bounds.

        kp (float or Iterable of float): Proportional gain for joint space PD control. Can be either a scalar
            (same value for all joints), or a list (specific values for each joint)

        kd (float or Iterable of float): Derivative gain for joint space PD control. Can be either a scalar
            (same value for all joints), or a list (specific values for each joint)

        policy_freq (int): Frequency at which actions from the robot policy are fed into this controller

        lite_physics (bool): Whether to optimize for mujoco forward and step calls to reduce total simulation overhead.
            Set to False to preserve backward compatibility with datasets collected in robosuite <= 1.4.1.

        **kwargs: Does nothing; placeholder to "sink" any additional arguments so that instantiating this controller
            via an argument dict that has additional extraneous arguments won't raise an error
    """

    def __init__(
        self,
        sim,
        ref_name,
        joint_indexes,
        actuator_range,
        input_max=1.5,
        input_min=-1.5,
        kp=150,
        kd=None,
        policy_freq=20,
        lite_physics=True,
        part_name=None,
        naming_prefix=None,
        **kwargs,  # does nothing; used so no error raised when dict is passed with extra terms
    ):

        super().__init__(
            sim,
            ref_name=ref_name,
            joint_indexes=joint_indexes,
            actuator_range=actuator_range,
            part_name=part_name,
            naming_prefix=naming_prefix,
            lite_physics=lite_physics,
        )

        # Control dimension can be 9 (SEW positions only) or 18 (SEW positions + wrist rotation matrix)
        self.control_dim = 18  # Maximum dimension, but will accept 9 or 18

        # input max and min for safety constraints (clipping bounds)
        # For positions: reasonable workspace bounds
        # For rotation matrix elements: [-1, 1] since they're normalized
        # can define better rotation limit later on
        pos_max = self.nums2array(input_max, 9) if isinstance(input_max, (int, float)) else np.array(input_max)
        rot_max = np.ones(9)  # Rotation matrix elements are in [-1, 1]
        self.input_max = np.concatenate([pos_max, rot_max])
        
        pos_min = self.nums2array(input_min, 9) if isinstance(input_min, (int, float)) else np.array(input_min)
        rot_min = -np.ones(9)  # Rotation matrix elements are in [-1, 1]
        self.input_min = np.concatenate([pos_min, rot_min])

        # kp and kd gains
        self.kp = self.nums2array(kp, self.joint_dim)
        self.kd = self.nums2array(2 * np.sqrt(self.kp) if kd is None else kd, self.joint_dim)

        # control frequency
        self.control_freq = policy_freq

        # Load kinova model and initialize transforms
        self.pin_model, _ = opt.load_kinova_model(kinova_path)
        self.fk_fun, pos_fun, jac_fun, M_fun, C_fun, G_fun = opt.build_casadi_kinematics_dynamics(self.pin_model, 'tool_frame')
        self.model_transforms = opt.get_frame_transforms_from_pinocchio(self.pin_model)

        # Initialize goal SEW poses and wrist rotation
        self.goal_S = None
        self.goal_E = None
        self.goal_W = None
        self.goal_R_base_wrist = None  # Wrist rotation matrix in body frame
        
        # Previous joint configuration for solution selection
        self.q_prev = None

        # Initialize torques
        self.torques = np.zeros(self.joint_dim)

        # Initialize origin pos and ori (similar to osc_geo)
        self.origin_pos = None
        self.origin_ori = None
        
        # Solution indices for consistent IK solution selection
        self.sol_ids = None
        self._initialize_solution_indices()

    def set_goal(self, action):
        """
        Sets goal SEW poses and wrist rotation based on input action.

        Args:
            action (np.array): Action to execute. Should be of size (18,) representing:
                [S_x, S_y, S_z, E_x, E_y, E_z, W_x, W_y, W_z, 
                 R11, R12, R13, R21, R22, R23, R31, R32, R33]
                
                SEW positions + 3x3 wrist rotation matrix (flattened row-wise) in body frame.
                The controller will transform these to robot base frame using its
                world_to_origin_frame transformation.
                
                If any SEW triplet contains NaN or inf values, that pose will be ignored
                and the controller will hold the current configuration.
                
                If rotation matrix elements contain NaN or inf values, wrist rotation
                will be ignored and only SEW positions will be used.
        """

        # Ensure action is a numpy array
        action = np.array(action)
        
        # Validate that the action has the correct size (now always 18)
        if action.size != 18:
            raise ValueError(f"SEW action must have 18 elements (SEW + rotation matrix), got {action.size}")

        # Clip action to safe bounds
        action = np.clip(action, self.input_min, self.input_max)

        # Extract SEW positions from action (body-centric coordinates, already clipped to safe bounds)
        S_body = action[0:3]
        E_body = action[3:6]
        W_body = action[6:9]

        # Check if any of the SEW poses contain invalid values (NaN or inf)
        # If so, set them to None to indicate they should be ignored
        if np.any(~np.isfinite(S_body)):
            self.goal_S = None
        else:
            # Transform from body-centric to robot base frame
            self.goal_S = self._transform_body_to_robot_frame(S_body)

        if np.any(~np.isfinite(E_body)):
            self.goal_E = None
        else:
            self.goal_E = self._transform_body_to_robot_frame(E_body)

        if np.any(~np.isfinite(W_body)):
            self.goal_W = None
        else:
            self.goal_W = self._transform_body_to_robot_frame(W_body)

        # Handle wrist rotation matrix (elements 9-17)
        R_body_wrist_flat = action[9:18]
        
        # Check if rotation matrix elements are valid
        if np.any(~np.isfinite(R_body_wrist_flat)):
            self.goal_R_base_wrist = None
        elif np.all(R_body_wrist_flat == 0):
            # If all elements are zero, treat as no wrist rotation
            self.goal_R_base_wrist = None
        else:
            # Reshape to 3x3 matrix (row-wise flattening)
            R_body_wrist = R_body_wrist_flat.reshape(3, 3)
            
            # Validate that it's close to a proper rotation matrix
            # Check if determinant is close to 1 and matrix is orthogonal
            det = np.linalg.det(R_body_wrist)
            if abs(det - 1.0) > 0.2:  # Allow some tolerance for numerical errors
                # Not a valid rotation matrix - use identity instead
                if hasattr(self, 'debug') and self.debug:
                    print(f"Warning: Invalid rotation matrix (det={det:.3f}), using identity")
                R_body_wrist = np.eye(3)

            # Transform rotation matrix from body frame to robot base frame
            self.goal_R_base_wrist = self._transform_rotation_body_to_robot_frame(R_body_wrist)

    def _initialize_solution_indices(self):
        """
        Initialize solution indices using the robot's actual initial joint configuration.
        This determines which solution branch to consistently select for IK.
        Uses the robot's current SEW positions at initialization to set up indexing.
        """
        try:
            # Get the robot's actual initial joint configuration from the simulation
            q_init = np.array(self.sim.data.qpos[self.qpos_index])
            fk_result = opt.forward_kinematics_homogeneous(q_init, self.fk_fun).full()
            R_0_T_init = fk_result[:3, :3]  # Extract rotation part from FK result

            # Extract robot SEW positions at the initial configuration
            robot_sew = get_robot_SEW_from_q(q_init, self.pin_model)
            S_init = robot_sew['S']
            E_init = robot_sew['E'] 
            W_init = robot_sew['W']
            
            # Get all possible solutions for the initial SEW configuration
            Q_solutions, is_LS_vec, human_vectors, sol_ids_used = IK_2R_2R_3R_SEW(
                S_init, E_init, W_init, self.model_transforms, sol_ids=None, R_0_T_kinova=R_0_T_init
            )

            # currently IK_SEW doesn't have wrist so last three q_init should be ignored
            q_init_no_wrist = q_init
            q_init_no_wrist[4:] = np.zeros(3)  # Set wrist joints to zero for IK
            if Q_solutions.shape[1] > 0:
                # Select the solution closest to the initial joint configuration
                q_selected, is_least_square, joint_limit_violated = filter_and_select_closest_solution(
                    Q_solutions, is_LS_vec, q_init_no_wrist
                )
                
                if q_selected is not None and not joint_limit_violated:
                    # Find which solution was selected and store the corresponding sol_ids
                    best_distance = float('inf')
                    best_sol_idx = 0
                    for i in range(Q_solutions.shape[1]):
                        distance = np.linalg.norm(Q_solutions[:, i] - q_selected)
                        if distance < best_distance:
                            best_distance = distance
                            best_sol_idx = i
                    
                    # Extract the solution indices for the selected solution
                    if best_sol_idx < len(sol_ids_used['q12_idx']):
                        self.sol_ids = {
                            'q12_idx': sol_ids_used['q12_idx'][best_sol_idx],
                            'q34_idx': sol_ids_used['q34_idx'][best_sol_idx],
                            'q56_idx': sol_ids_used['q56_idx'][best_sol_idx] if 'q56_idx' in sol_ids_used and best_sol_idx < len(sol_ids_used['q56_idx']) else 0
                        }
                        print(f"SEW Controller: Initialized solution indices from q_init={q_init.round(3)}: {self.sol_ids}")
                    else:
                        print(f"SEW Controller: Warning - could not extract solution indices from q_init={q_init.round(3)}")
                else:
                    print(f"SEW Controller: Warning - initial solution violates joint limits or is invalid for q_init={q_init.round(3)}")
            else:
                print(f"SEW Controller: Warning - no IK solutions found for initial configuration q_init={q_init.round(3)}")
                
        except Exception as e:
            print(f"SEW Controller: Solution indices initialization failed: {e}")
            # Fall back to no specific indexing
            self.sol_ids = None

    def _transform_body_to_robot_frame(self, body_pos):
        """
        Transform body-centric coordinates to robot base frame using world_to_origin_frame.
        
        Args:
            body_pos (np.array): Position in body-centric frame (meters)
            
        Returns:
            np.array: Position in robot base frame
        """
        # Body-centric frame: X=forward, Y=right-to-left, Z=down-to-up (hip to shoulder)
        # Robot base frame: depends on robot setup, typically X=forward, Y=left, Z=up
        
        # Basic transformation from body-centric to robot base coordinates
        # This can be adjusted based on the specific robot setup
        
        # Transform from world to origin frame if origin is set
        if self.origin_pos is not None and self.origin_ori is not None:
            robot_pos = self.world_to_origin_frame(body_pos)
        
        return robot_pos

    def _transform_rotation_body_to_robot_frame(self, R_body_wrist):
        """
        Transform wrist rotation matrix from body frame to robot base frame.
        
        Args:
            R_body_wrist (np.array): 3x3 rotation matrix representing wrist orientation 
                                   in body-centric frame
            
        Returns:
            np.array: 3x3 rotation matrix representing wrist orientation in robot base frame
        """
        # Get the rotation matrix from body frame to robot base frame
        if self.origin_ori is not None:
            # R_base_body is the rotation from body frame to robot base frame
            R_base_body = self.origin_ori.T
            
            # Transform: R_base_wrist = R_base_body @ R_body_wrist
            R_base_wrist = R_base_body @ R_body_wrist 
            
            return R_base_wrist
        else:
            # If no transformation is set, assume body frame = robot base frame
            return R_body_wrist

    def world_to_origin_frame(self, vec):
        """
        Transform vector from world to reference coordinate frame.
        Adapted from osc_geo controller.
        
        Args:
            vec (np.array): Vector in world frame
            
        Returns:
            np.array: Vector in origin frame
        """
        import robosuite.utils.transform_utils as T
        
        # world rotation matrix is just identity
        world_frame = np.eye(4)
        world_frame[:3, 3] = vec

        # mediapipe origin is different from mujoco origin
        origin_pos_mp = np.zeros(3)
        origin_frame = T.make_pose(origin_pos_mp, self.origin_ori)
        origin_frame_inv = T.pose_inv(origin_frame)
        vec_origin_pose = T.pose_in_A_to_pose_in_B(world_frame, origin_frame_inv)
        vec_origin_pos, _ = T.mat2pose(vec_origin_pose)
        return vec_origin_pos

    def run_controller(self):
        """
        Calculates the torques required to reach the desired SEW configuration.

        Uses IK_2R_2R_3R_SEW to compute joint angles that match the human SEW poses,
        then applies joint-space PD control to reach those joint angles.

        Returns:
             np.array: Command torques
        """
        # Update state
        self.update()

        # Determine target joint configuration
        q_desired = self.joint_pos.copy()  # Default: hold current pose

        # Only compute IK if all SEW poses are provided
        if self.goal_S is not None and self.goal_E is not None and self.goal_W is not None:
            try:
                if self.goal_R_base_wrist is not None:

                    # wrist rotation conversion 
                    ez = np.array([0, 0, 1])  # Assuming wrist rotation is around Z-axis
                    ey = np.array([0, 1, 0])  # Assuming wrist rotation is around Y-axis
                    if self.part_name == "left":
                        ez = -ez  # Adjust for left arm if needed
                        ey = -ey  # Adjust for left arm if needed

                    R_T_W = Rotation.from_euler('zyx', [np.pi/2, 0, np.pi/2]).as_matrix()  # Identity rotation for wrist

                    R_0_T_kinova = self.goal_R_base_wrist @ R_T_W  # Transform wrist rotation to robot base frame

                    # Future: Pass wrist rotation matrix to enhanced IK function
                else:
                    R_0_T_kinova = None

                q7_bias = self.initial_joint[6]  # Use initial q7 value as bias for wrist rotation
                    

                # Q_solutions, is_LS_vec, human_vectors, _ = IK_2R_2R_3R_SEW(
                #         self.goal_S, self.goal_E, self.goal_W, self.model_transforms,
                #         sol_ids=self.sol_ids, R_0_T_kinova= R_0_T_kinova
                #     )

                Q_solutions, is_LS_vec, human_vectors, _ = IK_2R_2R_3R_SEW_wrist_lock(
                        self.goal_S, self.goal_E, self.goal_W, self.model_transforms,
                        sol_ids=self.sol_ids, R_0_T_kinova= R_0_T_kinova, q7_bias=q7_bias
                    )

                # With sol_ids, we should get the specific solution we want
                if Q_solutions.shape[1] > 0:  # Check if we have solutions
                    # Use the first (and should be only) solution returned
                    q_desired = Q_solutions[:, 0]
                    
                    # If wrist rotation matrix is available, it can be used to compute q5,q6,q7
                    # This will be implemented when IK_2R_2R_3R_SEW is updated
                    # For now, the IK function will output the previous q5,q6,q7 values as mentioned
                
            except Exception as e:
                # If IK fails, hold current pose
                print(f"SEW IK failed: {e}")
                # q_desired remains as current joint positions

        # Compute joint position and velocity errors
        joint_pos_error = q_desired - self.joint_pos
        # Wrap joint position error to [-pi, pi]
        joint_pos_error = (joint_pos_error + np.pi) % (2 * np.pi) - np.pi
        joint_vel_error = -self.joint_vel  # Desired velocity is zero

        # Compute desired joint accelerations using PD control
        qdd_desired = self.kp * joint_pos_error + self.kd * joint_vel_error

        # Compute torques using inverse dynamics: tau = M * qdd + gravity_compensation
        self.torques = np.dot(self.mass_matrix, qdd_desired) + self.torque_compensation

        # Always run superclass call for any cleanups at the end
        super().run_controller()

        return self.torques

    def reset_goal(self):
        """
        Resets the goal SEW poses and wrist rotation to None, causing the controller to hold current pose.
        """
        self.goal_S = None
        self.goal_E = None
        self.goal_W = None
        self.goal_R_base_wrist = None

    @property
    def control_limits(self):
        """
        Returns the limits over this controller's action space.
        For SEW controller, these represent safe workspace clipping bounds for positions
        and normalized bounds for rotation matrix elements.

        Returns:
            2-tuple:
                - (np.array) minimum action values: SEW position bounds (meters) + rotation matrix bounds [-1,1]
                - (np.array) maximum action values: SEW position bounds (meters) + rotation matrix bounds [-1,1]
        """
        return self.input_min, self.input_max

    @property
    def name(self):
        return "SEW_MIMIC"

    @staticmethod
    def nums2array(nums, dim):
        """
        Convert input @nums into numpy array of length @dim. If @nums is a single number, broadcasts it to the
        corresponding dimension size @dim before converting into a numpy array

        Args:
            nums (numeric or Iterable): Either single value or array of numbers
            dim (int): Size of array to broadcast input to

        Returns:
            np.array: Array filled with values specified in @nums
        """
        # First run sanity check to make sure no strings are being passed
        if isinstance(nums, str):
            raise TypeError("Error: Only numeric types are supported for this operation!")

        # Check if input is an Iterable, if so, we simply convert the input to np.array and return
        # Else, we assume it's a single value and broadcast it to the desired size.
        return np.array(nums) if isinstance(nums, Iterable) else np.ones(dim) * nums

    def update_origin(self, origin_pos, origin_ori):
        """
        Optional function to implement in subclass controllers that will take in @origin_pos and @origin_ori and update
        internal configuration to account for changes in the respective states. Useful for controllers in which the origin
        is a frame of reference that is dynamically changing, e.g., adapting the arm to move along with a moving base.
        Adapted from osc_geo controller.

        Args:
            origin_pos (3-tuple): x,y,z position of controller reference in mujoco world coordinates
            origin_ori (np.array): 3x3 rotation matrix orientation of controller reference in mujoco world coordinates
        """
        self.origin_pos = origin_pos
        self.origin_ori = origin_ori

    
