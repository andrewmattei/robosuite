import numpy as np
from collections.abc import Iterable

from robosuite.controllers.parts.controller import Controller
from robosuite.demos.geometric_kinematics_gen3_7dof import IK_2R_2R_3R_SEW, filter_and_select_closest_solution, get_robot_SEW_from_q
import robosuite.demos.optimizing_gen3_arm as opt
from robosuite.demos.geometric_kinematics_gen3_7dof import kinova_path


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

        # Control dimension is 9 (3 for each of S, E, W positions)
        self.control_dim = 9

        # input max and min for safety constraints (clipping bounds)
        self.input_max = self.nums2array(input_max, self.control_dim)
        self.input_min = self.nums2array(input_min, self.control_dim)

        # kp and kd gains
        self.kp = self.nums2array(kp, self.joint_dim)
        self.kd = self.nums2array(2 * np.sqrt(self.kp) if kd is None else kd, self.joint_dim)

        # control frequency
        self.control_freq = policy_freq

        # Load kinova model and initialize transforms
        self.pin_model, _ = opt.load_kinova_model(kinova_path)
        self.model_transforms = opt.get_frame_transforms_from_pinocchio(self.pin_model)

        # Initialize goal SEW poses
        self.goal_S = None
        self.goal_E = None
        self.goal_W = None
        
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
        Sets goal SEW poses based on input action.

        Args:
            action (np.array): Action to execute. Should be of size (9,) representing
                [S_x, S_y, S_z, E_x, E_y, E_z, W_x, W_y, W_z] where S, E, W are
                shoulder, elbow, and wrist positions respectively in body-centric frame.
                The controller will transform these to robot base frame using its
                world_to_origin_frame transformation.
                If any triplet contains NaN or inf values, that pose will be ignored
                and the controller will hold the current configuration.
        """

        # Ensure action is a numpy array
        action = np.array(action)
        
        # Validate that the action has the correct size
        if action.size != 9:
            raise ValueError(f"SEW action must have 9 elements (3 for S, E, W each), got {action.size}")

        # Clip action to be within safe bounds (no scaling, just clipping)
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

    def _initialize_solution_indices(self):
        """
        Initialize solution indices using the robot's actual initial joint configuration.
        This determines which solution branch to consistently select for IK.
        Uses the robot's current SEW positions at initialization to set up indexing.
        """
        try:
            # Get the robot's actual initial joint configuration from the simulation
            q_init = np.array(self.sim.data.qpos[self.qpos_index])
            
            # Extract robot SEW positions at the initial configuration
            robot_sew = get_robot_SEW_from_q(q_init, self.pin_model)
            S_init = robot_sew['S']
            E_init = robot_sew['E'] 
            W_init = robot_sew['W']
            
            # Get all possible solutions for the initial SEW configuration
            Q_solutions, is_LS_vec, human_vectors, sol_ids_used = IK_2R_2R_3R_SEW(
                S_init, E_init, W_init, self.model_transforms, sol_ids=None
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
                            'q34_idx': sol_ids_used['q34_idx'][best_sol_idx]
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
                # Call IK_2R_2R_3R_SEW with stored solution indices for consistent solution selection
                Q_solutions, is_LS_vec, human_vectors, _ = IK_2R_2R_3R_SEW(
                    self.goal_S, self.goal_E, self.goal_W, self.model_transforms, sol_ids=self.sol_ids
                )

                # With sol_ids, we should get the specific solution we want
                if Q_solutions.shape[1] > 0:  # Check if we have solutions
                    # Use the first (and should be only) solution returned
                    q_desired = Q_solutions[:, 0]
                
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
        Resets the goal SEW poses to None, causing the controller to hold current pose.
        """
        self.goal_S = None
        self.goal_E = None
        self.goal_W = None

    @property
    def control_limits(self):
        """
        Returns the limits over this controller's action space.
        For SEW controller, these represent safe workspace clipping bounds.

        Returns:
            2-tuple:
                - (np.array) minimum action values for SEW positions (clipping bounds in meters)
                - (np.array) maximum action values for SEW positions (clipping bounds in meters)
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

    
