import math

import numpy as np
from scipy.spatial.transform import Rotation

import robosuite.utils.transform_utils as T
from robosuite.controllers.parts.controller import Controller
from robosuite.utils.control_utils import *

from robosuite.projects.shared_scripts.geometric_kinematics_gen3_7dof import IK_2R_2R_3R_numerical, kinova_path, get_elbow_angle_kinova, filter_and_select_closest_solution
from robosuite.projects.shared_scripts.geometric_kinematics_gen3_7dof import IK_2R_2R_3R_auto_elbow
from robosuite.projects.shared_scripts.sew_stereo import SEWStereo
import robosuite.projects.shared_scripts.optimizing_gen3_arm as opt

# Supported impedance modes
IMPEDANCE_MODES = {"fixed", "variable", "variable_kp"}

# TODO: Maybe better naming scheme to differentiate between input / output min / max and pos/ori limits, etc.


class OperationalSpaceControllerGeo(Controller):
    """
    Controller for controlling robot arm via operational space control. Allows position and / or orientation control
    of the robot's end effector. For detailed information as to the mathematical foundation for this controller, please
    reference http://khatib.stanford.edu/publications/pdfs/Khatib_1987_RA.pdf

    NOTE: Control input actions can either be taken to be relative to the current position / orientation of the
    end effector or absolute values. In either case, a given action to this controller is assumed to be of the form:
    (x, y, z, ax, ay, az) if controlling pos and ori or simply (x, y, z) if only controlling pos

    Args:
        sim (MjSim): Simulator instance this controller will pull robot state updates from

        joint_indexes (dict): Each key contains sim reference indexes to relevant robot joint information, namely:

            :`'joints'`: list of indexes to relevant robot joints
            :`'qpos'`: list of indexes to relevant robot joint positions
            :`'qvel'`: list of indexes to relevant robot joint velocities

        actuator_range (2-tuple of array of float): 2-Tuple (low, high) representing the robot joint actuator range

        input_max (float or Iterable of float): Maximum above which an inputted action will be clipped. Can be either be
            a scalar (same value for all action dimensions), or a list (specific values for each dimension). If the
            latter, dimension should be the same as the control dimension for this controller

        input_min (float or Iterable of float): Minimum below which an inputted action will be clipped. Can be either be
            a scalar (same value for all action dimensions), or a list (specific values for each dimension). If the
            latter, dimension should be the same as the control dimension for this controller

        output_max (float or Iterable of float): Maximum which defines upper end of scaling range when scaling an input
            action. Can be either be a scalar (same value for all action dimensions), or a list (specific values for
            each dimension). If the latter, dimension should be the same as the control dimension for this controller

        output_min (float or Iterable of float): Minimum which defines upper end of scaling range when scaling an input
            action. Can be either be a scalar (same value for all action dimensions), or a list (specific values for
            each dimension). If the latter, dimension should be the same as the control dimension for this controller

        kp (float or Iterable of float): positional gain for determining desired torques based upon the pos / ori error.
            Can be either be a scalar (same value for all action dims), or a list (specific values for each dim)

        damping_ratio (float or Iterable of float): used in conjunction with kp to determine the velocity gain for
            determining desired torques based upon the joint pos errors. Can be either be a scalar (same value for all
            action dims), or a list (specific values for each dim)

        impedance_mode (str): Impedance mode with which to run this controller. Options are {"fixed", "variable",
            "variable_kp"}. If "fixed", the controller will have fixed kp and damping_ratio values as specified by the
            @kp and @damping_ratio arguments. If "variable", both kp and damping_ratio will now be part of the
            controller action space, resulting in a total action space of (6 or 3) + 6 * 2. If "variable_kp", only kp
            will become variable, with damping_ratio fixed at 1 (critically damped). The resulting action space will
            then be (6 or 3) + 6.

        kp_limits (2-list of float or 2-list of Iterable of floats): Only applicable if @impedance_mode is set to either
            "variable" or "variable_kp". This sets the corresponding min / max ranges of the controller action space
            for the varying kp values. Can be either be a 2-list (same min / max for all kp action dims), or a 2-list
            of list (specific min / max for each kp dim)

        damping_ratio_limits (2-list of float or 2-list of Iterable of floats): Only applicable if @impedance_mode is
            set to "variable". This sets the corresponding min / max ranges of the controller action space for the
            varying damping_ratio values. Can be either be a 2-list (same min / max for all damping_ratio action dims),
            or a 2-list of list (specific min / max for each damping_ratio dim)

        policy_freq (int): Frequency at which actions from the robot policy are fed into this controller

        position_limits (2-list of float or 2-list of Iterable of floats): Limits (m) below and above which the
            magnitude of a calculated goal eef position will be clipped. Can be either be a 2-list (same min/max value
            for all cartesian dims), or a 2-list of list (specific min/max values for each dim)

        orientation_limits (2-list of float or 2-list of Iterable of floats): Limits (rad) below and above which the
            magnitude of a calculated goal eef orientation will be clipped. Can be either be a 2-list
            (same min/max value for all joint dims), or a 2-list of list (specific min/mx values for each dim)

        interpolator_pos (Interpolator): Interpolator object to be used for interpolating from the current position to
            the goal position during each timestep between inputted actions

        interpolator_ori (Interpolator): Interpolator object to be used for interpolating from the current orientation
            to the goal orientation during each timestep between inputted actions

        control_ori (bool): Whether inputted actions will control both pos and ori or exclusively pos

        input_type (str): Whether to control the robot using delta ("delta") or absolute commands ("absolute").
            This is wrt the contorller reference frame (see input_ref_frame field)

        input_ref_frame (str): Reference frame for controller. Current supported options are:
            "base": actions are wrt to the robot body (i.e., the base)
            "world": actions are wrt the world coordinate frame

        uncouple_pos_ori (bool): Whether to decouple torques meant to control pos and torques meant to control ori

        lite_physics (bool): Whether to optimize for mujoco forward and step calls to reduce total simulation overhead.
            Set to False to preserve backward compatibility with datasets collected in robosuite <= 1.4.1.

        **kwargs: Does nothing; placeholder to "sink" any additional arguments so that instantiating this controller
            via an argument dict that has additional extraneous arguments won't raise an error

    Raises:
        AssertionError: [Invalid impedance mode]
    """

    def __init__(
        self,
        sim,
        ref_name,
        joint_indexes,
        actuator_range,
        input_max=1,
        input_min=-1,
        output_max=(0.05, 0.05, 0.05, 0.5, 0.5, 0.5),
        output_min=(-0.05, -0.05, -0.05, -0.5, -0.5, -0.5),
        kp=150,
        damping_ratio=1,
        impedance_mode="fixed",
        kp_limits=(0, 300),
        damping_ratio_limits=(0, 100),
        policy_freq=20,
        position_limits=None,
        orientation_limits=None,
        interpolator_pos=None,
        interpolator_ori=None,
        control_ori=True,
        input_type="delta",
        input_ref_frame="base",
        uncouple_pos_ori=True,
        lite_physics=True,
        **kwargs,  # does nothing; used so no error raised when dict is passed with extra terms used previously
    ):

        super().__init__(
            sim,
            ref_name=ref_name,
            joint_indexes=joint_indexes,
            actuator_range=actuator_range,
            lite_physics=lite_physics,
            part_name=kwargs.get("part_name", None),
            naming_prefix=kwargs.get("naming_prefix", None),
        )
        # Determine whether this is pos ori or just pos
        self.use_ori = control_ori
        # Determine whether we want to use delta or absolute values as inputs
        self.input_type = input_type
        assert self.input_type in ["delta", "absolute"], f"Input type must be delta or absolute, got: {self.input_type}"

        # determine reference frame wrt actions are set
        self.input_ref_frame = input_ref_frame
        assert self.input_ref_frame in [
            "world",
            "base",
        ], f"Input reference frame must be world or base, got: {self.input_ref_frame}"

        # Control dimension
        self.control_dim = 6 if self.use_ori else 3
        self.name_suffix = "POSE" if self.use_ori else "POSITION"

        # input and output max and min (allow for either explicit lists or single numbers)
        self.input_max = self.nums2array(input_max, self.control_dim)
        self.input_min = self.nums2array(input_min, self.control_dim)
        self.output_max = self.nums2array(output_max, self.control_dim)
        self.output_min = self.nums2array(output_min, self.control_dim)

        # kp kd
        self.kp = self.nums2array(kp, 6)
        self.kd = 2 * np.sqrt(self.kp) * damping_ratio

        # kp and kd limits
        self.kp_min = self.nums2array(kp_limits[0], 6)
        self.kp_max = self.nums2array(kp_limits[1], 6)
        self.damping_ratio_min = self.nums2array(damping_ratio_limits[0], 6)
        self.damping_ratio_max = self.nums2array(damping_ratio_limits[1], 6)

        # Verify the proposed impedance mode is supported
        assert impedance_mode in IMPEDANCE_MODES, (
            "Error: Tried to instantiate OSC controller for unsupported "
            "impedance mode! Inputted impedance mode: {}, Supported modes: {}".format(impedance_mode, IMPEDANCE_MODES)
        )

        # Impedance mode
        self.impedance_mode = impedance_mode

        # Add to control dim based on impedance_mode
        if self.impedance_mode == "variable":
            self.control_dim += 12
        elif self.impedance_mode == "variable_kp":
            self.control_dim += 6

        # limits
        self.position_limits = np.array(position_limits) if position_limits is not None else position_limits
        self.orientation_limits = np.array(orientation_limits) if orientation_limits is not None else orientation_limits

        # control frequency
        self.control_freq = policy_freq

        # interpolator
        self.interpolator_pos = interpolator_pos
        self.interpolator_ori = interpolator_ori

        # whether or not pos and ori want to be uncoupled
        self.uncoupling = uncouple_pos_ori

        # initialize goals
        self.goal_pos = None
        self.goal_ori = None

        # initialize orientation references
        self.relative_ori = np.zeros(3)
        self.ori_ref = None

        # initialize origin pos and ori
        self.origin_pos = None
        self.origin_ori = None

        # initialize ik-geo modules
        self.EE_id = self.sim.model.body_name2id(self.naming_prefix+self.part_name+'_end_effector')
        R_wd_ee = self.sim.data.body_xmat[self.EE_id].reshape(3, 3)
        p_wd_ee = self.sim.data.body_xpos[self.EE_id]
        R_wd_ref = self.ref_ori_mat
        p_wd_ref = self.ref_pos
        self.R_ref_ee = R_wd_ref.T @ R_wd_ee  # Rotation from reference to end effector
        self.p_ref_ee = R_wd_ref.T @ (p_wd_ee - p_wd_ref)  # Position from reference to end effector

        pin_model, _ = opt.load_kinova_model(kinova_path)
        r, v = np.array([-1, 0, 0]), np.array([0, 1, 0]) # r is e_t, v is e_r
        self.sew_stereo = SEWStereo(r, v)
        self.model_transforms = opt.get_frame_transforms_from_pinocchio(pin_model)
        self.elbow_angle = get_elbow_angle_kinova(self.initial_joint, pin_model, self.sew_stereo)
        self.fk_fun, _,_,_,_,_ = opt.build_casadi_kinematics_dynamics(pin_model, 'tool_frame')
        print(f"Elbow angle: {self.elbow_angle}")


    def set_goal(self, action):
        """
        Sets goal based on input @action. If self.impedance_mode is not "fixed", then the input will be parsed into the
        delta values to update the goal position / pose and the kp and/or damping_ratio values to be immediately updated
        internally before executing the proceeding control loop.

        Note that @action expected to be in the following format, based on impedance mode!

            :Mode `'fixed'`: [joint pos command]
            :Mode `'variable'`: [damping_ratio values, kp values, joint pos command]
            :Mode `'variable_kp'`: [kp values, joint pos command]

        Args:
            action (Iterable): Desired relative joint position goal state
        """
        # Update state
        self.update()

        # Parse action based on the impedance mode, and update kp / kd as necessary
        if self.impedance_mode == "variable":
            damping_ratio, kp, delta = action[:6], action[6:12], action[12:]
            self.kp = np.clip(kp, self.kp_min, self.kp_max)
            self.kd = 2 * np.sqrt(self.kp) * np.clip(damping_ratio, self.damping_ratio_min, self.damping_ratio_max)
        elif self.impedance_mode == "variable_kp":
            kp, delta = action[:6], action[6:]
            self.kp = np.clip(kp, self.kp_min, self.kp_max)
            self.kd = 2 * np.sqrt(self.kp)  # critically damped
        else:  # This is case "fixed"
            delta = action

        # If we're using deltas, interpret actions as such
        if self.input_type == "delta":
            scaled_delta = self.scale_action(delta)
            self.goal_pos = self.compute_goal_pos(scaled_delta[0:3])
            if self.use_ori is True:
                self.goal_ori = self.compute_goal_ori(scaled_delta[3:6])
            else:
                self.goal_ori = self.compute_goal_ori(np.zeros(3))
        # Else, interpret actions as absolute values
        elif self.input_type == "absolute":
            self.goal_pos = action[0:3]
            if self.use_ori is True:
                self.goal_ori = Rotation.from_rotvec(action[3:6]).as_matrix()
            else:
                self.goal_ori = self.compute_goal_ori(np.zeros(3))
        else:
            raise ValueError(f"Unsupport input_type {self.input_type}")

        if self.interpolator_pos is not None:
            self.interpolator_pos.set_goal(self.goal_pos)

        if self.interpolator_ori is not None:
            self.ori_ref = np.array(self.ref_ori_mat)  # reference is the current orientation at start
            self.interpolator_ori.set_goal(
                orientation_error(self.goal_ori, self.ori_ref)
            )  # goal is the total orientation error
            self.relative_ori = np.zeros(3)  # relative orientation always starts at 0

    def world_to_origin_frame(self, vec):
        """
        transform vector from world to reference coordinate frame
        """

        # world rotation matrix is just identity
        world_frame = np.eye(4)
        world_frame[:3, 3] = vec

        origin_frame = T.make_pose(self.origin_pos, self.origin_ori)
        origin_frame_inv = T.pose_inv(origin_frame)
        vec_origin_pose = T.pose_in_A_to_pose_in_B(world_frame, origin_frame_inv)
        vec_origin_pos, _ = T.mat2pose(vec_origin_pose)
        return vec_origin_pos

    def goal_origin_to_eef_pose(self):
        origin_pose = T.make_pose(self.origin_pos, self.origin_ori)
        ee_pose = T.make_pose(self.ref_pos, self.ref_ori_mat)
        origin_pose_inv = T.pose_inv(origin_pose)
        return T.pose_in_A_to_pose_in_B(ee_pose, origin_pose_inv)

    def compute_goal_pos(self, delta, goal_update_mode=None):
        """
        Compute new goal position, given a delta to update. Can either update the new goal based on
        current achieved position or current deisred goal. Updating based on current deisred goal can be useful
        if we want the robot to adhere with a sequence of target poses as closely as possible,
        without lagging or overshooting.

        Args:
            delta (np.array): Desired relative change in position [x, y, z]
            goal_update_mode (str): either "achieved" (achieved position) or "desired" (desired goal)

        Returns:
            np.array: updated goal position in the controller frame
        """
        if goal_update_mode is None:
            goal_update_mode = self._goal_update_mode
        assert goal_update_mode in ["achieved", "desired"]

        if self.goal_pos is None:
            # if goal is not already set, set it to current position (in controller ref frame)
            if self.input_ref_frame == "base":
                self.goal_pos = self.world_to_origin_frame(self.ref_pos)
            elif self.input_ref_frame == "world":
                self.goal_pos = self.ref_pos
            else:
                raise ValueError

        if goal_update_mode == "desired":
            # update new goal wrt current desired goal
            goal_pos = self.goal_pos + delta
        elif goal_update_mode == "achieved":
            # update new goal wrt current achieved position
            if self.input_ref_frame == "base":
                goal_pos = self.world_to_origin_frame(self.ref_pos) + delta
            elif self.input_ref_frame == "world":
                goal_pos = self.ref_pos + delta
            else:
                raise ValueError

        if self.position_limits is not None:
            # to be implemented later
            raise NotImplementedError

        return goal_pos

    def compute_goal_ori(self, delta, goal_update_mode=None):
        """
        Compute new goal orientation, given a delta to update. Can either update the new goal based on
        current achieved position or current deisred goal. Updating based on current deisred goal can be useful
        if we want the robot to adhere with a sequence of target poses as closely as possible,
        without lagging or overshooting.

        Args:
            delta (np.array): Desired relative change in orientation, in axis-angle form [ax, ay, az]
            goal_update_mode (str): either "achieved" (achieved position) or "desired" (desired goal)

        Returns:
            np.array: updated goal orientation in the controller frame
        """
        if goal_update_mode is None:
            goal_update_mode = self._goal_update_mode
        assert goal_update_mode in ["achieved", "desired"]

        if self.goal_ori is None:
            # if goal is not already set, set it to current orientation (in controller ref frame)
            if self.input_ref_frame == "base":
                self.goal_ori = self.goal_origin_to_eef_pose()[:3, :3]
            elif self.input_ref_frame == "world":
                self.goal_ori = self.ref_ori_mat
            else:
                raise ValueError

        # convert axis-angle value to rotation matrix
        quat_error = T.axisangle2quat(delta)
        rotation_mat_error = T.quat2mat(quat_error)

        if self._goal_update_mode == "desired":
            # update new goal wrt current desired goal
            goal_ori = np.dot(rotation_mat_error, self.goal_ori)
        elif self._goal_update_mode == "achieved":
            # update new goal wrt current achieved orientation
            if self.input_ref_frame == "base":
                curr_goal_ori = self.goal_origin_to_eef_pose()[:3, :3]
            elif self.input_ref_frame == "world":
                curr_goal_ori = self.ref_ori_mat
            else:
                raise ValueError
            goal_ori = np.dot(rotation_mat_error, curr_goal_ori)
        else:
            raise ValueError

        # check for orientation limits
        if np.array(self.orientation_limits).any():
            # to be implemented later
            raise NotImplementedError
        return goal_ori

    def run_controller(self):
        """
        Calculates the torques required to reach the desired setpoint.

        Executes Operational Space Control (OSC) -- either position only or position and orientation.

        A detailed overview of derivation of OSC equations can be seen at:
        http://khatib.stanford.edu/publications/pdfs/Khatib_1987_RA.pdf

        Returns:
             np.array: Command torques
        """
        # Update state
        self.update()


        # desired POSE in robot base frame
        if self.input_ref_frame == "world":
            p_des_0_ref = self.world_to_origin_frame(self.goal_pos)
            R_des_0_ref = self.origin_ori.T @ self.goal_ori
            p_des_0_ee = p_des_0_ref + R_des_0_ref @ self.p_ref_ee
            R_des_0_ee = R_des_0_ref @ self.R_ref_ee
        elif self.input_ref_frame == "base":
            p_des_0_ref = self.goal_pos
            R_des_0_ee = self.goal_ori
            p_des_0_ee = p_des_0_ref + R_des_0_ref @ self.p_ref_ee
            R_des_0_ee = R_des_0_ref @ self.R_ref_ee
        else:
            raise ValueError(f"Unsupported input reference frame: {self.input_ref_frame}")

        # Use IK_2R_2R_3R_numerical to compute desired joint angles
        # Convert desired pose to homogeneous transformation matrix
        R_0_7_desired = R_des_0_ee @ self.model_transforms['R'][-1]
        p_0_T_desired = p_des_0_ee


        # # debug
        # pin_T_0_T = self.fk_fun(self.joint_pos).full()

        # R_0_T_curr = self.origin_ori.T @ self.sim.data.body_xmat[self.EE_id].reshape(3, 3)
        # p_0_T_curr = self.world_to_origin_frame(self.sim.data.body_xpos[self.EE_id])
        
        # p_0_ref_curr_alt = self.world_to_origin_frame(self.ref_pos)
        # R_0_ref_curr_alt = self.origin_ori.T @ self.ref_ori_mat
        # R_0_T_curr_alt = R_0_ref_curr_alt @ self.R_ref_ee
        # p_0_T_curr_alt= p_0_ref_curr_alt + R_0_ref_curr_alt @ self.p_ref_ee 
        # print(pin_T_0_T)
        # print(f"Desired position: {p_0_T_desired}, Current EE position: {p_0_T_curr_alt}")
        # print(f"Desired orientation: \n{R_des_0_ee}, \nCurrent EE orientation: \n{R_0_T_curr_alt}")

        # Call IK solver to get desired joint angles
        try:
            angle = np.pi/4
            # if left robot, use minus sign
            if self.part_name == "left":
                test_elbow_angle = self.elbow_angle + angle
            else:
                test_elbow_angle = self.elbow_angle - angle
            Q_solutions, is_LS_vec = IK_2R_2R_3R_numerical(
                R_0_7_desired, p_0_T_desired, self.sew_stereo, test_elbow_angle, self.model_transforms
            )

            # Q_solutions, is_LS_vec = IK_2R_2R_3R_auto_elbow(
            #     R_0_7_desired, p_0_T_desired, self.sew_stereo, self.model_transforms
            # )
            
            # Select the closest solution to current joint configuration
            if len(Q_solutions) > 0:
                q_desired, is_least_square, joint_limit_violated = filter_and_select_closest_solution(Q_solutions, is_LS_vec, self.joint_pos)
            else:
                # If no solution found, use current joint positions as fallback
                q_desired = self.joint_pos.copy()
            if joint_limit_violated:
                print("Joint limit violated in IK solution, using current joint positions as fallback.")
                q_desired = self.joint_pos.copy()
                
        except Exception as e:
            # If IK fails, use current joint positions as fallback
            print(f"IK failed: {e}")
            q_desired = self.joint_pos.copy()
        
        # debug
        # pin_T_0_T_des = self.fk_fun(q_desired).full()
        # print(pin_T_0_T_des)

        # Compute joint position and velocity errors
        joint_pos_error = q_desired - self.joint_pos
        # wrap joint position error to [-pi, pi]
        joint_pos_error = (joint_pos_error + np.pi) % (2 * np.pi) - np.pi 
        joint_vel_error = -self.joint_vel  # Desired velocity is zero
        
        # Compute desired joint accelerations using PD control
        # qdd_des = kp * joint_pos_error + kd * joint_vel_error
        kp_joint = self.kp[0]  # Use first element of kp as joint space gain
        kd_joint = self.kd[0]  # Use first element of kd as joint space damping
        
        qdd_desired = kp_joint * joint_pos_error + kd_joint * joint_vel_error
        
        # Compute torques using inverse dynamics: tau = M * qdd + gravity_compensation
        self.torques = np.dot(self.mass_matrix, qdd_desired) + self.torque_compensation

        # Always run superclass call for any cleanups at the end
        super().run_controller()

        return self.torques

    def update_origin(self, origin_pos, origin_ori):
        """
        Optional function to implement in subclass controllers that will take in @origin_pos and @origin_ori and update
        internal configuration to account for changes in the respective states. Useful for controllers in which the origin
        is a frame of reference that is dynamically changing, e.g., adapting the arm to move along with a moving base.

        Args:
            origin_pos (3-tuple): x,y,z position of controller reference in mujoco world coordinates
            origin_ori (np.array): 3x3 rotation matrix orientation of controller reference in mujoco world coordinates
        """
        self.origin_pos = origin_pos
        self.origin_ori = origin_ori

    def update_initial_joints(self, initial_joints):
        # First, update from the superclass method
        super().update_initial_joints(initial_joints)

        # We also need to reset the goal in case the old goals were set to the initial confguration
        self.reset_goal()

    def set_goal_update_mode(self, goal_update_mode):
        self._goal_update_mode = goal_update_mode

    def reset_goal(self, goal_update_mode="achieved"):
        """
        Resets the goal to the current state of the robot.

        Args:
            goal_update_mode (str): set mode for updating controller goals,
                either "achieved" (achieved position) or "desired" (desired goal).
        """
        self.goal_ori = np.array(self.ref_ori_mat)
        self.goal_pos = np.array(self.ref_pos)

        assert goal_update_mode in ["achieved", "desired"]
        self._goal_update_mode = goal_update_mode

        # Also reset interpolators if required

        if self.interpolator_pos is not None:
            self.interpolator_pos.set_goal(self.goal_pos)

        if self.interpolator_ori is not None:
            self.ori_ref = np.array(self.ref_ori_mat)  # reference is the current orientation at start
            self.interpolator_ori.set_goal(
                orientation_error(self.goal_ori, self.ori_ref)
            )  # goal is the total orientation error
            self.relative_ori = np.zeros(3)  # relative orientation always starts at 0

    @property
    def control_limits(self):
        """
        Returns the limits over this controller's action space, overrides the superclass property
        Returns the following (generalized for both high and low limits), based on the impedance mode:

            :Mode `'fixed'`: [joint pos command]
            :Mode `'variable'`: [damping_ratio values, kp values, joint pos command]
            :Mode `'variable_kp'`: [kp values, joint pos command]

        Returns:
            2-tuple:

                - (np.array) minimum action values
                - (np.array) maximum action values
        """
        if self.impedance_mode == "variable":
            low = np.concatenate([self.damping_ratio_min, self.kp_min, self.input_min])
            high = np.concatenate([self.damping_ratio_max, self.kp_max, self.input_max])
        elif self.impedance_mode == "variable_kp":
            low = np.concatenate([self.kp_min, self.input_min])
            high = np.concatenate([self.kp_max, self.input_max])
        else:  # This is case "fixed"
            low, high = self.input_min, self.input_max
        return low, high

    def delta_to_abs_action(self, delta_ac, goal_update_mode):
        """
        helper function that converts delta action into absolute action
        """
        abs_pos = self.compute_goal_pos(delta_ac[0:3], goal_update_mode=goal_update_mode)
        abs_ori = self.compute_goal_ori(delta_ac[3:6], goal_update_mode=goal_update_mode)
        abs_rot = T.quat2axisangle(T.mat2quat(abs_ori))
        abs_action = np.concatenate([abs_pos, abs_rot])
        return abs_action

    @property
    def name(self):
        return "OSC_GEO_" + self.name_suffix
