from kortex_api.autogen.client_stubs.ActuatorConfigClientRpc import ActuatorConfigClient
from kortex_api.autogen.client_stubs.ActuatorCyclicClientRpc import ActuatorCyclicClient
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.client_stubs.DeviceConfigClientRpc import DeviceConfigClient
from kortex_api.autogen.client_stubs.DeviceManagerClientRpc import DeviceManagerClient
from kortex_api.autogen.messages import Session_pb2, ActuatorConfig_pb2, Base_pb2, BaseCyclic_pb2, Common_pb2
from kortex_api.RouterClient import RouterClientSendOptions
from kortex_api.autogen.client_stubs.ControlConfigClientRpc import ControlConfigClient

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
sys.path.append(os.path.dirname(__file__))

import robosuite.utils.tool_box_no_ros as tb
import robosuite.utils.kortex_utilities as kortex_utils
import optimizing_gen3_arm as opt

import numpy as np
np.set_printoptions(precision=4, suppress=True)
import time
import threading
import traceback
from copy import deepcopy
from collections import OrderedDict
import matplotlib.pyplot as plt
import casadi as cs


class TCPArguments:
    def __init__(self):
        self.ip = "192.168.0.10"
        self.username = "admin"
        self.password = "admin"


class Kinova3HardwareController:
    def __init__(self, router, router_real_time, home_pose="Home", use_friction_compensation=False):
        # Maximum allowed waiting time during actions (in seconds)
        self.ACTION_TIMEOUT_DURATION = 20
        self.home_pose = home_pose
        self.use_friction_compensation = use_friction_compensation

        # Create required services
        device_manager = DeviceManagerClient(router)
        self.actuator_config = ActuatorConfigClient(router)
        self.base = BaseClient(router)
        self.base_cyclic = BaseCyclicClient(router_real_time)
        self.control_config = ControlConfigClient(router)

        self.base_command = BaseCyclic_pb2.Command()
        self.base_feedback = BaseCyclic_pb2.Feedback()
        self.base_custom_data = BaseCyclic_pb2.CustomData()

        # Detect all devices
        device_handles = device_manager.ReadAllDevices()
        self.actuator_count = self.base.GetActuatorCount().count

        # Setup for all actuators
        for handle in device_handles.device_handle:
            if handle.device_type == Common_pb2.BIG_ACTUATOR or handle.device_type == Common_pb2.SMALL_ACTUATOR:
                self.base_command.actuators.add()
                self.base_feedback.actuators.add()

        # Change send option to reduce max timeout at 3ms
        self.sendOption = RouterClientSendOptions()
        self.sendOption.andForget = False
        self.sendOption.delay_ms = 0
        self.sendOption.timeout_ms = 3

        self.cyclic_t_end = 30
        self.cyclic_thread = None
        self.kill_the_thread = False
        self.already_stopped = False
        self.cyclic_running = False
        self.action_aborted = False

        # Load pinocchio model and build dynamics functions
        self.model, self.data = opt.load_kinova_model()
        self.fk_fun, self.pos_fun, self.jac_fun, self.M_fun, self.C_fun, self.G_fun = \
            opt.build_casadi_kinematics_dynamics(self.model)
        
        # Build Jacobian derivative function for Lyapunov controller
        self.jac_pos_fun = opt.build_position_jacobian_derivative_function(self.jac_fun)
        self.jdot_fun = opt.build_jacobian_derivative_function_efficient(self.jac_fun)

        # target info
        self.p_f = None
        self.v_f = None
        self.v_p = None
        self.R_f = None

        # Joint limits
        rev_lim = np.pi*2
        self.q_lower = np.array([-rev_lim, -2.41, -rev_lim, -2.66, -rev_lim, -2.23, -rev_lim])
        self.q_upper = np.array([rev_lim, 2.41, rev_lim, 2.66, rev_lim, 2.23, rev_lim])
        self.dq_lower  =  -self.model.velocityLimit
        self.dq_upper  =  self.model.velocityLimit
        self.tau_lower = -self.model.effortLimit
        self.tau_upper = self.model.effortLimit

        # Control parameters
        self.init_jogging = True
        # for LQR controller
        self.reversed_traj = None
        self.linearization_cache = None
        self.reversed_linearization_cache = None

        # Data storage
        self.times = []
        self.q_pos = []
        self.q_vel = []
        self.ee_pos = []
        self.ee_vel = []
        self.tau_log = []
        self.tau_friction = []
        self.tau_measured = []

        self.impact_time = None
        self.impact_pos = None
        self.impact_vel = None

        self.retreat_distance = None
        self.retreat_q = None

        # Filters
        self.torque_lowpass = None
        self.velocity_lowpass = None

        # Friction compensation parameters (if enabled)
        if self.use_friction_compensation:
            self.motor_joint_stiffness = np.diag([4000, 4000, 4000, 4000, 3500, 3500, 3500])
            self.motor_joint_Kp = 2.0 * np.diag([20, 20, 20, 20, 10, 10, 10])
            self.motor_joint_Kd = np.diag([2, 2, 2, 2, 2, 2, 2])
            self.motor_inertia = np.diag([0.08, 0.08, 0.08, 0.08, 0.10, 0.10, 0.10])
            self.motor_friction_l = 3.0 * np.diag([25, 25, 25, 25, 20, 20, 20])
            self.motor_friction_lp = 0.1 * np.diag([2.5, 2.5, 2.5, 2.5, 2, 2, 2])
            
            # Previous states for friction observer
            self.previous_nominal_theta = None
            self.previous_nominal_theta_dot = None

        # Initialize Lyapunov retreat controller parameters
        self.configure_retreat_controller()

    def check_for_end_or_abort(self, e):
        """Return a closure checking for END or ABORT notifications"""
        def check(notification, e=e):
            print("EVENT : " + Base_pb2.ActionEvent.Name(notification.action_event))
            if notification.action_event == Base_pb2.ACTION_END:
                e.set()
            elif notification.action_event == Base_pb2.ACTION_ABORT:
                self.action_aborted = True
                e.set()
        return check

    def move_to_home_position(self):
        """Move arm to home position"""
        base_servo_mode = Base_pb2.ServoingModeInformation()
        base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        self.base.SetServoingMode(base_servo_mode)
        
        print("Moving the arm to home position")
        action_type = Base_pb2.RequestedActionType()
        action_type.action_type = Base_pb2.REACH_JOINT_ANGLES
        action_list = self.base.ReadAllActions(action_type)
        action_handle = None
        for action in action_list.action_list:
            if action.name == self.home_pose:
                action_handle = action.handle

        if action_handle is None:
            print("Can't reach home position. Exiting")
            return False
        self.action_aborted = False
        e = threading.Event()
        notification_handle = self.base.OnNotificationActionTopic(
            self.check_for_end_or_abort(e),
            Base_pb2.NotificationOptions()
        )

        self.base.ExecuteActionFromReference(action_handle)
        finished = e.wait(self.ACTION_TIMEOUT_DURATION)
        self.base.Unsubscribe(notification_handle)

        if finished:
            print("Home position reached")
        else:
            print("Timeout on action notification wait")
        return finished

    def init_low_level_control(self, sampling_time=0.001, t_end=30, target_func=None):
        """Initialize low-level torque control mode"""
        if self.cyclic_running:
            return True

        # # Move to home position first
        # if not self.move_to_home_position():
        #     return False

        print("Initializing low-level control")
        
        base_feedback = self.send_call_with_retry(self.base_cyclic.RefreshFeedback, 3)
        if base_feedback:
            self.base_feedback = base_feedback

            # Initialize command frame for all actuators
            for x in range(self.actuator_count):
                self.base_command.actuators[x].flags = 1  # servoing
                self.base_command.actuators[x].position = self.base_feedback.actuators[x].position

            # Initialize all actuators with current torque to ensure continuity
            for x in range(self.actuator_count):
                self.base_command.actuators[x].torque_joint = self.base_feedback.actuators[x].torque

            # Set arm in LOW_LEVEL_SERVOING
            base_servo_mode = Base_pb2.ServoingModeInformation()
            base_servo_mode.servoing_mode = Base_pb2.LOW_LEVEL_SERVOING
            self.base.SetServoingMode(base_servo_mode)

            # Send first frame
            self.base_feedback = self.base_cyclic.Refresh(self.base_command, 0, self.sendOption)

            # Set all actuators in torque mode
            control_mode_message = ActuatorConfig_pb2.ControlModeInformation()
            control_mode_message.control_mode = ActuatorConfig_pb2.ControlMode.Value('TORQUE')
            
            for actuator_id in range(1, self.actuator_count + 1):
                self.send_call_with_retry(self.actuator_config.SetControlMode, 3, 
                                        control_mode_message, actuator_id)

            # Initialize filters
            init_q, init_qdot = tb.get_realtime_q_qdot(self.base_feedback)
            init_torque = tb.get_realtime_torque(self.base_feedback)
            self.torque_lowpass = tb.LowPassFilter(init_torque, cutoff_freq=10)
            self.velocity_lowpass = tb.LowPassFilter(init_qdot, cutoff_freq=10)

            # Initialize friction compensation if enabled
            if self.use_friction_compensation:
                self.previous_nominal_theta = init_q.copy()
                self.previous_nominal_theta_dot = init_qdot.copy()
                self.norminal_theta_log = []
                self.norminal_theta_dot_log = []

            # Initialize data storage
            self.times = []
            self.q_pos = []
            self.q_vel = []
            self.ee_pos = []
            self.ee_vel = []
            self.tau_log = []
            self.tau_friction = []
            self.tau_measured = []

            self.impact_time = None
            self.impact_pos = None
            self.impact_vel = None


            # Start cyclic thread
            self.cyclic_t_end = t_end
            self.cyclic_thread = threading.Thread(target=target_func, args=(sampling_time,))
            self.cyclic_thread.daemon = True
            self.cyclic_thread.start()
            return True
        else:
            print("Failed to initialize low-level control")
            return False

    def stop_low_level_control(self):
        """Stop low-level control and return to position mode"""
        print("Stopping low-level control and returning to position mode...")
        if self.already_stopped:
            return

        # Kill the thread first
        if self.cyclic_running:
            print("Stopping cyclic thread...")
            self.kill_the_thread = True
            self.cyclic_thread.join()
        
        # Set all actuators back to position mode
        control_mode_message = ActuatorConfig_pb2.ControlModeInformation()
        control_mode_message.control_mode = ActuatorConfig_pb2.ControlMode.Value('POSITION')
        
        for actuator_id in range(1, self.actuator_count + 1):
            self.send_call_with_retry(self.actuator_config.SetControlMode, 3, 
                                    control_mode_message, actuator_id)
        
        base_servo_mode = Base_pb2.ServoingModeInformation()
        base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        self.base.SetServoingMode(base_servo_mode)
        
        self.already_stopped = True
        print('Clean exit from low-level control')

    @staticmethod
    def send_call_with_retry(call, retry, *args):
        """Send API call with retry mechanism"""
        for i in range(retry):
            try:
                return call(*args)
            except:
                continue
        print("Failed to communicate after retries")
        return None

    def compute_gravity_compensation(self, q):
        """Compute gravity compensation torques"""
        return self.G_fun(q).full().flatten()

    def inverse_dynamics(self, q, dq, ddq_des):
        """Compute inverse dynamics"""
        G_curr = self.G_fun(q).full().flatten()
        C_curr = self.C_fun(q, dq).full()
        M_curr = self.M_fun(q).full()
        tau = M_curr @ ddq_des + C_curr @ dq + G_curr
        return tau

    def pid_joint_jog(self, q_desired, q_current, dq_current, Kp=100, Kd=5, tol=0.1):
        """PID control for joint jogging"""
        """ doesn't work so good in real life because there's no region of attraction"""
        error = q_desired - q_current
        error = (error + np.pi) % (2 * np.pi) - np.pi
        ddq_des = Kp * error - Kd * dq_current
        print(f"error: {np.linalg.norm(error)}")

        if np.linalg.norm(error) < tol:
            self.init_jogging = False

        if self.init_jogging:
            torque = self.inverse_dynamics(q_current, dq_current, ddq_des)
        else:
            torque = self.inverse_dynamics(q_current, dq_current, np.zeros_like(ddq_des))
        return torque

    def lqr_tracking_controller(self, current_time, T_opt, U_opt, Z_opt, 
                              q_current, dq_current, horizon=5,
                              cache=None):
        """LQR tracking controller"""
        if cache is None:
            cache = self.linearization_cache
    

        A_list, B_list = cache

        # Interpolate feed-forward torque and reference state
        u_ff, z_ref = opt.match_trajectories(current_time, T_opt, U_opt, T_opt, Z_opt)

        # Find index for selecting local linearization
        idx = np.searchsorted(T_opt, current_time)
        idx = min(max(idx, 0), len(A_list) - 1)

        # Build cost weights
        state_dim = Z_opt.shape[0]
        control_dim = U_opt.shape[0]
        Q = np.eye(state_dim)
        Q[:control_dim, :control_dim] *= 5e6
        Q[control_dim:, control_dim:] *= 1e3
        R = np.eye(control_dim) * 1e-5

        # Backward Riccati recursion
        P = Q.copy()
        K_seq = []
        for k in reversed(range(horizon)):
            if k == horizon - 1:
                P = 10 * Q
            j = min(idx + k, len(A_list) - 1)
            A = A_list[j]
            B = B_list[j]
            K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
            P = Q + A.T @ P @ (A - B @ K)
            K_seq.insert(0, K)

        # First feedback gain
        K0 = K_seq[0]

        # Current state
        # account for q wrapping
        q_error_raw = q_current - z_ref[:control_dim].flatten()
        q_error = (q_error_raw + np.pi) % (2 * np.pi) - np.pi
        dq_error = dq_current - z_ref[control_dim:].flatten()
        z_error = np.concatenate((q_error, dq_error)).flatten()

        # Compute LQR command
        tau = u_ff.flatten() - K0 @ z_error

        # Zero out after trajectory end
        if current_time > T_opt[-1]:
            tau = self.inverse_dynamics(q_current, dq_current, np.zeros_like(dq_current))
            # print("Trajectory ended, zeroing torques")
        return tau

    def post_contact_joint_ctrl(self, q_end, q_current, dq_current, Kp=80, Kd=20):
        """Post-contact joint position control"""
        q_err = q_end - q_current
        # Handle angle differences at boundaries
        q_err = (q_err + np.pi) % (2 * np.pi) - np.pi
        ddq_des = Kp * q_err - Kd * dq_current
        return self.inverse_dynamics(q_current, dq_current, ddq_des)

    def get_ee_pose_in_base(self, q):
        """Get end-effector pose in base frame"""
        return self.fk_fun(q).full()

    def get_ee_velocity_in_base(self, q, dq):
        """Get end-effector velocity in base frame"""
        J6 = self.jac_fun(q).full()
        J_vel = J6[0:3, :]
        ee_vel = J_vel @ dq
        return ee_vel
    
    def get_ee_twist_in_base(self, q, dq):
        """Get end-effector twist in base frame
        return the twist as a 6D vector [linear_velocity, angular_velocity]
        """
        J6 = self.jac_fun(q).full()
        ee_twist = J6 @ dq
        return ee_twist

    def configure_retreat_controller(self, enable=True,Kp=200.0, Kd=60.0, alpha=1.0, max_time=5.0):
        """
        Configure the Lyapunov retreat controller parameters.
        
        Args:
            enable: Whether to use the Lyapunov retreat controller
            retreat_distance: Distance to retreat in meters (default: 10cm)
            Kp: Position gain for the controller
            Kd: Damping gain for the controller
            alpha: Convergence rate parameter
            max_time: Maximum time for retreat motion
        """
        self.use_lyapunov_retreat = enable
        self.retreat_Kp = Kp
        self.retreat_Kd = Kd
        self.retreat_alpha = alpha
        self.retreat_max_time = max_time
        print(f"Retreat controller configured: enabled={enable}, distance={self.retreat_distance}m")

    def lyapunov_ee_retreat_controller(self, v_f, d_up, R_des=None, current_time=0.0, 
                                     Kp=100.0, Kd=5.0, alpha=1.0,
                                     start_pos=None, max_time=5.0):
        """
        Lyapunov-based controller to move end-effector in opposite direction of v_f
        for a distance d_up with guaranteed convergence.
        
        Args:
            v_f: Final velocity vector (3,) - direction to move away from
            d_up: Distance to move in opposite direction (scalar)
            current_time: Current time for initialization
            Kp: Position gain (scalar or 3x3 matrix)
            Kd: Damping gain (scalar or 3x3 matrix)  
            alpha: Convergence rate parameter
            start_pos: Starting end-effector position (if None, uses current)
            max_time: Maximum time for the motion
            
        Returns:
            tau: Control torques (7,)
            finished: Boolean indicating if target reached
        """
        # Get current robot state - use hardware-specific methods
        q_curr, dq_curr_raw = tb.get_realtime_q_qdot(self.base_feedback)
        dq_curr = self.velocity_lowpass.filter(dq_curr_raw)
        
        # Get current end-effector position and velocity using hardware methods
        H_base_ee = self.get_ee_pose_in_base(q_curr)
        ee_pos_curr = H_base_ee[:3, 3]
        R_ee_curr = H_base_ee[:3, :3]

        ee_twist = self.get_ee_twist_in_base(q_curr, dq_curr)
        ee_vel_curr = ee_twist[:3]  # Linear velocity part
        ee_omega_curr = ee_twist[3:]  # Angular velocity part
        
        # Initialize starting position if not provided
        if not hasattr(self, '_retreat_start_pos') or start_pos is not None:
            self._retreat_start_pos = start_pos if start_pos is not None else ee_pos_curr.copy()
            self._retreat_start_time = current_time

            # set target orientation
            if R_des is not None:
                self._R_des = R_des.copy()
            else:
                # Use current orientation if no desired orientation provided
                self._R_des = R_ee_curr.copy()

        
        # Calculate retreat direction (opposite to v_f)
        v_f_norm = np.linalg.norm(v_f)
        if v_f_norm < 1e-8:
            print("Warning: v_f magnitude too small, using default z-direction")
            retreat_direction = np.array([0, 0, 1])
        else:
            # tilt towards the origin for 20 degrees
            normal_dir = -v_f / v_f_norm  # Opposite direction
            
            # Direction towards robot origin from contact point
            robot_origin = np.array([0.0, 0.0, 0.0])  # Robot base origin
            to_origin_vec = robot_origin - self.p_f  # Use contact point as reference
            to_origin_norm = np.linalg.norm(to_origin_vec)
            
            if to_origin_norm > 1e-8:
                to_origin_unit = to_origin_vec / to_origin_norm
                
                # Tilt by 20 degrees: blend normal_dir with to_origin_unit
                tilt_angle = np.radians(20)
                tilt_factor = np.sin(tilt_angle)  # How much to blend towards origin
                
                # Blend the directions
                tilted_direction = (1 - tilt_factor) * normal_dir + tilt_factor * to_origin_unit
                retreat_direction = tilted_direction / np.linalg.norm(tilted_direction)
            else:
                # Origin is at contact point, use normal direction
                retreat_direction = normal_dir
            

        # Target position: start_pos + d_up * retreat_direction
        p_target = self._retreat_start_pos + d_up * retreat_direction
        
        # Position error in task space
        p_error = p_target - ee_pos_curr
        p_error_norm = np.linalg.norm(p_error)

        # Orientation error in task space
        ER = R_ee_curr @ self._R_des.T # Desired orientation relative to current
        k, theta = tb.R2rot(ER)  # Axis-angle representation
        k = np.array(k)
        rot_error = -np.sin(theta/2) * k  # Error vector for orientation
        # print(ER[0, 0], ER[1, 1], ER[2, 2], ER[0, 1], ER[0, 2], ER[1, 2])
    
        # Check if target reached
        position_tolerance = 0.05  # 5mm
        velocity_tolerance = 0.01   # 1cm/s
        ee_vel_norm = np.linalg.norm(ee_vel_curr)

        # print(f"Retreat: p_error={p_error_norm:.4f}m, ee_vel={ee_vel_norm:.4f}m/s, target={p_target}, current={ee_pos_curr}")
        
        if p_error_norm < position_tolerance and ee_vel_norm < velocity_tolerance:
            # Target reached - apply zero torque
            print(f"Retreat target reached: error={p_error_norm:.4f}m, vel={ee_vel_norm:.4f}m/s")
            return self.inverse_dynamics(q_curr, dq_curr, np.zeros_like(dq_curr)), True
        
        # Check timeout
        if current_time - self._retreat_start_time > max_time:
            print(f"Retreat timeout reached: error={p_error_norm:.4f}m")
            return self.inverse_dynamics(q_curr, dq_curr, np.zeros_like(dq_curr)), True
        
        # Convert gains to matrices if scalars
        # rad_mm_error_scale = 5
        Kp_pos = Kp * np.eye(3)
        Kp_orient = 10 * np.eye(3)  # Lower gain for orientation

        Kd_pos = Kd * np.eye(3)
        Kd_orient = 10 * np.eye(3)  # Lower damping for orientation
        
        # Lyapunov-based control law in task space
        # V = 0.5 * p_error^T * Kp * p_error + 0.5 * ee_vel^T * ee_vel
        # For V_dot < 0, we choose: ee_accel_des = Kp * p_error - Kd * ee_vel - alpha * p_error
        
        # Desired end-effector acceleration
        ee_accel_pos_des = Kp_pos @ p_error - Kd_pos @ ee_vel_curr - alpha * p_error
        ee_accel_ori_des = Kp_orient @ rot_error - Kd_orient @ ee_omega_curr

        # clip the orientation acceleration to avoid large values
        max_omega_accel = 40 # rad/s^2
        ee_accel_ori_des = np.clip(ee_accel_ori_des, -max_omega_accel, max_omega_accel)

        ee_accel_des = np.concatenate((ee_accel_pos_des, ee_accel_ori_des))

        # print(f"Retreat: p_error={p_error_norm:.4f}m, rot_error={np.linalg.norm(rot_error):.4f}, "
        #       f"ee_accel_des={ee_accel_des}, "
        #       f"ee_omega_curr={ee_omega_curr}")
        # print(p_error_norm, np.linalg.norm(rot_error), ee_accel_des, ee_omega_curr)

        # Get current Jacobian
        J6 = self.jac_fun(q_curr)
        
        # Compute Jacobian time derivative for acceleration mapping
        J_dot = self.jdot_fun(q_curr, dq_curr).full()

        # Map desired task-space acceleration to joint space
        # ee_accel = J * ddq + J_dot * dq
        # Solve for ddq: ddq = J_pinv @ (ee_accel_des - J_dot @ dq)
        
        # Use damped pseudo-inverse for numerical stability
        J_damped = cs.DM(opt.damping_pseudoinverse(J6)).full()
        
        ddq_des = J_damped @ (ee_accel_des - J_dot @ dq_curr)
        
        # Add null-space control to maintain joint limits and comfort
        # Null-space projection: N = I - J_pinv @ J
        N = np.eye(self.actuator_count) - J_damped @ J6.full()

        # Null-space objective: move towards comfortable joint configuration
        q_diff = self.retreat_q - q_curr
        q_diff = (q_diff + np.pi) % (2 * np.pi) - np.pi  # Normalize to [-pi, pi]
        ddq_null = 10.0 * q_diff - 5.0 * dq_curr
        
        
        # Combine task-space and null-space control
        ddq_total = ddq_des + N @ ddq_null
        
        # Compute required torques using inverse dynamics
        tau = self.inverse_dynamics(q_curr, dq_curr, ddq_total)

        return tau, False

    def apply_friction_compensation(self, q, dq, q_des, dq_des, tau_task, tau_measured, dt):
        """Apply friction compensation if enabled"""
        if not self.use_friction_compensation:
            return tau_task

        # Friction observer implementation
        K_s_inv = np.linalg.inv(self.motor_joint_stiffness)
        K_r_inv = np.linalg.inv(self.motor_inertia)
        
        G = self.G_fun(q).full().flatten()
        theta_des = q_des - K_s_inv @ G
        theta_dot_des = dq_des

        # Handle angle differences at boundaries
        theta_diff = (self.previous_nominal_theta - theta_des + np.pi) % (2 * np.pi) - np.pi
        
        # tau_control = -self.motor_joint_Kp @ theta_diff - self.motor_joint_Kd @ (self.previous_nominal_theta_dot - theta_dot_des)
        nominal_theta_ddot = K_r_inv @ (tau_task - G - tau_measured)
        nominal_theta_dot = self.previous_nominal_theta_dot + dt * nominal_theta_ddot
        nominal_theta = self.previous_nominal_theta + dt * nominal_theta_dot

        # Update previous values
        self.previous_nominal_theta = nominal_theta
        self.previous_nominal_theta_dot = nominal_theta_dot

        # Compute friction compensation
        observer_theta_diff = (nominal_theta - q + np.pi) % (2 * np.pi) - np.pi
        tau_friction = self.motor_inertia @ self.motor_friction_l @ ((nominal_theta_dot - dq) + self.motor_friction_lp @ observer_theta_diff)
        self.norminal_theta_log.append(nominal_theta.copy())
        self.norminal_theta_dot_log.append(nominal_theta_dot.copy())

        return tau_friction

    def run_trajectory_execution(self, T_opt, U_opt, Z_opt, sampling_time=0.001):
        """Execute trajectory with optimized real-time control for 1000Hz"""
        self.cyclic_running = True
        print("Starting trajectory execution")
        
        cyclic_count = 0
        failed_cyclic_count = 0
        
        # High-resolution timing initialization - avoid time.time() in loop
        import time
        t_init = time.perf_counter()
        control_dt = sampling_time  # 0.001s for 1000Hz
        target_cycle_time = control_dt
        
        # Pre-calculate key values to minimize computation in loop
        n_actuators = self.actuator_count
        tau_lower, tau_upper = self.tau_lower, self.tau_upper
        
        # Cache frequently used arrays
        tau = np.zeros(n_actuators)
        q_curr = np.zeros(n_actuators)
        dq_curr = np.zeros(n_actuators)
        ee_pos = np.zeros(3)
        ee_vel = np.zeros(3)
        
        # Timing management with counter-based approach
        cycle_counter = 0

        # Initialize trajectory parameters
        phase = "tracking"
        impact_detected = False
        impact_time = None
        impact_pos = None
        impact_vel = None
        ref_time = 0.0
        # q_end = Z_opt[:self.actuator_count, int(len(T_opt) * 0.70)]

        # # post-contact control parameters
        # T_rev = self.reversed_traj['T']
        # Z_rev = self.reversed_traj['Z']
        # U_rev = self.reversed_traj['U']
        
        # Lyapunov retreat controller parameters
        retreat_finished = False
        
        # Pre-allocate for minimal memory allocation in loop
        H_base_ee = np.eye(4)
        
        # Main control loop - optimized for 1000Hz
        next_cycle_time = t_init + target_cycle_time

        # Initialize retreat parameters if not set
        if self.retreat_q is None:
            self.retreat_q = np.radians([0, 15, 180, -130, 0, -35, 90])

        while not self.kill_the_thread:
            cycle_counter += 1
            control_time = cycle_counter * control_dt  # Counter-based timing
            
            # High-resolution timing check
            current_time = time.perf_counter()
            
            # Sleep until next cycle if we're ahead (precision timing control)
            if current_time < next_cycle_time:
                sleep_time = next_cycle_time - current_time
                if sleep_time > 0.0001:  # Only sleep if significant time remains
                    time.sleep(sleep_time * 0.9)  # Sleep 90% to avoid oversleep
                # Busy wait for final precision
                while time.perf_counter() < next_cycle_time:
                    pass
            
            # Update next cycle target time
            next_cycle_time += target_cycle_time

            # Get current state - minimize function calls
            q_curr[:], dq_curr_raw = tb.get_realtime_q_qdot(self.base_feedback)
            dq_curr[:] = self.velocity_lowpass.filter(dq_curr_raw)
            
            # End-effector state computation - reuse H_base_ee memory
            H_base_ee = self.get_ee_pose_in_base(q_curr)
            ee_pos[:] = H_base_ee[:3, 3]
            ee_vel[:] = self.get_ee_velocity_in_base(q_curr, dq_curr)
            
            tau_meas = self.torque_lowpass.filter(tb.get_realtime_torque(self.base_feedback))

            # Check for contact
            # total_force = np.linalg.norm(tau_measured)
            # contact_detected = total_force > 15.0  # Threshold for contact detection
            # currently use z_position of the end-effector vs desired z_position
            contact_detected = ee_pos[2] <= self.p_f[2]
            traveling_away = np.dot(ee_vel, self.v_f) < 0.0  # Check if moving away from target
        

            # Control logic based on phase
            if phase == "tracking":
                if self.linearization_cache is not None:
                    tau = self.lqr_tracking_controller(control_time, T_opt, U_opt, Z_opt, q_curr, dq_curr)
                # else:
                #     # Simple feedforward + PD tracking
                #     tau_ref, z_ref = opt.match_trajectories(control_time, T_opt, U_opt, T_opt, Z_opt)
                #     n = self.actuator_count
                #     q_ref = z_ref[:n].flatten()
                #     dq_ref = z_ref[n:].flatten()
                    
                #     Kp, Kd = 80.0, 5.0
                #     q_err = q_ref - q_curr
                #     dq_err = dq_ref - dq_curr
                #     ddq_des = Kp * q_err + Kd * dq_err
                #     tau = self.inverse_dynamics(q_curr, dq_curr, ddq_des)

                # Check for impact
                if contact_detected and not impact_detected:
                    impact_detected = True
                    impact_time = control_time
                    impact_pos = ee_pos.copy()
                    impact_vel = ee_vel.copy()
                    print(f"Impact detected at time {impact_time:.3f}s")
                    phase = "post_impact"

            elif phase == "post_impact":
                # Use Lyapunov-based retreat controller
                tau, retreat_finished = self.lyapunov_ee_retreat_controller(
                    v_f=self.v_f,
                    d_up=self.retreat_distance,
                    R_des=self.R_f,
                    current_time=control_time,
                    Kp=self.retreat_Kp,
                    Kd=self.retreat_Kd,
                    alpha=self.retreat_alpha,
                    start_pos=self.p_f,
                    max_time=self.retreat_max_time
                )
                
                if retreat_finished:
                    print(f"Lyapunov retreat completed at time {control_time:.3f}s")
                    # Optionally transition to LQR control after retreat
                    # phase = "post_retreat_lqr"
                    break
                
                # if not self.use_lyapunov_retreat or retreat_finished:
                #     # Use LQR controller for post-impact control (original implementation)
                #     q_err = Z_rev[:self.actuator_count, 0].flatten() - q_curr
                #     q_err = (q_err + np.pi) % (2 * np.pi) - np.pi
                #     q_err_init = np.linalg.norm(q_err)
                #     # the closer the number, the more delay at the bottom
                #     if traveling_away and q_err_init < 0.12 and impact_detected:
                #         print(f"Contact finished at time {control_time:.3f}s")
                #         impact_detected = False
                #         t_away = control_time

                #     if impact_detected:
                #         # keep shooting for the first lqr point until reaching it.
                #         ref_time = 0.00
                #     else:
                #         ref_time = control_time - t_away

                #     tau = self.lqr_tracking_controller(
                #         ref_time, T_rev, U_rev, Z_rev, q_curr, dq_curr, horizon=5,
                #         cache=self.reversed_linearization_cache)
                #     if not impact_detected and control_time > t_away + T_rev[-1]+0.4:
                #         break

            # Apply friction compensation if enabled
            if self.use_friction_compensation:
                tau = np.clip(tau, tau_lower, tau_upper)
                tau_fric = self.apply_friction_compensation(q_curr, dq_curr, q_curr, dq_curr, tau, tau_meas, control_dt)
            # Clip torques to limits
            tau = np.clip(tau, tau_lower, tau_upper)

            # Apply torques to all actuators - optimized loop
            for i in range(n_actuators):
                self.base_command.actuators[i].position = self.base_feedback.actuators[i].position
                self.base_command.actuators[i].torque_joint = tau[i]

            # Log data - only when needed
            if impact_time is None or control_time < impact_time + 0.2:  # Log for trajectory duration + buffer
                self.times.append(control_time)
                self.q_pos.append(q_curr.copy())
                self.q_vel.append(dq_curr.copy())
                
                # Get end-effector state
                self.ee_pos.append(ee_pos.copy())
                self.ee_vel.append(ee_vel.copy())
                
                self.tau_log.append(tau.copy())
                if self.use_friction_compensation:
                    self.tau_friction.append(tau_fric.copy())
                else:
                    self.tau_friction.append(np.zeros_like(tau))
                self.tau_measured.append(tau_meas.copy())

            # Send command - optimized frame handling
            self.base_command.frame_id = (self.base_command.frame_id + 1) % 65536
            for i in range(n_actuators):
                self.base_command.actuators[i].command_id = self.base_command.frame_id

            try:
                self.base_feedback = self.base_cyclic.Refresh(self.base_command, 0, self.sendOption)
            except:
                failed_cyclic_count += 1
            
            cyclic_count += 1

            # Exit conditions - check less frequently to avoid timing impact
            if cyclic_count % 100 == 0:  # Check every 100ms
                current_elapsed = time.perf_counter() - t_init
                if self.cyclic_t_end != 0 and current_elapsed > self.cyclic_t_end:
                    print("Execution time limit reached")
                    break

            if control_time > T_opt[-1] + 5:  # 5 seconds after trajectory end
                print("Trajectory execution completed")
                break

        # Store impact information for plotting
        self.impact_time = impact_time
        self.impact_pos = impact_pos
        self.impact_vel = impact_vel

        self.cyclic_running = False
        return True

    def run_pid_jogging(self, q_desired, sampling_time=0.001, max_time=10.0):
        """Run PID jogging to desired joint positions with optimized timing"""
        self.cyclic_running = True
        self.init_jogging = True
        print(f"Starting PID jogging to target position")
        
        # High-resolution timing setup
        import time
        t_init = time.perf_counter()
        control_dt = sampling_time
        target_cycle_time = control_dt
        
        # Pre-allocate for performance
        n_actuators = self.actuator_count
        tau_lower, tau_upper = self.tau_lower, self.tau_upper
        tau = np.zeros(n_actuators)
        q_curr = np.zeros(n_actuators)
        dq_curr = np.zeros(n_actuators)
        H_base_ee = np.eye(4)
        
        cycle_counter = 0
        next_cycle_time = t_init + target_cycle_time

        while not self.kill_the_thread:
            cycle_counter += 1
            control_time = cycle_counter * control_dt
            
            # Precision timing control
            current_time = time.perf_counter()
            if current_time < next_cycle_time:
                sleep_time = next_cycle_time - current_time
                if sleep_time > 0.0001:
                    time.sleep(sleep_time * 0.9)
                while time.perf_counter() < next_cycle_time:
                    pass
            next_cycle_time += target_cycle_time

            # Get current state - optimized
            q_curr[:], dq_curr_raw = tb.get_realtime_q_qdot(self.base_feedback)
            dq_curr[:] = self.velocity_lowpass.filter(dq_curr_raw)

            # Compute control
            tau[:] = self.pid_joint_jog(q_desired, q_curr, dq_curr)
            tau = np.clip(tau, tau_lower, tau_upper)

            # Apply torques - optimized loop
            for i in range(n_actuators):
                self.base_command.actuators[i].position = self.base_feedback.actuators[i].position
                self.base_command.actuators[i].torque_joint = tau[i]

            # Log data
            self.times.append(control_time)
            self.q_pos.append(q_curr.copy())
            self.q_vel.append(dq_curr.copy())
            
            H_base_ee = self.get_ee_pose_in_base(q_curr)
            self.ee_pos.append(H_base_ee[:3, 3].copy())
            
            ee_vel = self.get_ee_velocity_in_base(q_curr, dq_curr)
            self.ee_vel.append(ee_vel.copy())
            
            self.tau_log.append(tau.copy())

            # Send command - optimized
            self.base_command.frame_id = (self.base_command.frame_id + 1) % 65536
            for i in range(n_actuators):
                self.base_command.actuators[i].command_id = self.base_command.frame_id

            try:
                self.base_feedback = self.base_cyclic.Refresh(self.base_command, 0, self.sendOption)
            except:
                pass

            # Check exit conditions
            if not self.init_jogging:
                print("Target position reached")
                break
            if control_time > max_time:
                print("Jogging timeout reached")
                break

        self.cyclic_running = False
        return True

    def plot_results(self, T_opt=None, Z_opt=None, ee_pos_opt=None, ee_vel_opt=None):
        """Plot execution results"""
        if not self.times:
            print("No data to plot")
            return

        # Convert lists to numpy arrays
        times = np.array(self.times)
        q_pos = np.array(self.q_pos)
        q_vel = np.array(self.q_vel)
        ee_pos = np.array(self.ee_pos)
        ee_vel = np.array(self.ee_vel)
        tau = np.array(self.tau_log)
        tau_measured = np.array(self.tau_measured)

        # Color definitions similar to simulation
        colors = {
            'actual': '#7f7f7f',    # gray
            'desired': '#bcbd22',   # olive
            'impact': '#9467bd',    # purple
            'target': '#8c564b',    # brown
            'joint1': '#1f77b4',    # bright blue
            'joint2': '#ff7f0e',    # bright orange
            'joint3': '#2ca02c',    # bright green
            'joint4': '#d62728',    # bright red
            'joint5': '#9467bd',    # purple
            'joint6': '#8c564b',    # brown
            'joint7': '#e377c2',    # pink
        }

        # Create plots - only 3 subplots now (hiding contact forces)
        if self.use_friction_compensation:
            fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
        else:
            fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

        # End-effector position error
        if T_opt is not None and ee_pos_opt is not None:
            pos_interp = np.vstack([
                np.interp(times, T_opt, ee_pos_opt[:, i]) for i in range(3)
            ]).T
            err = np.linalg.norm(ee_pos - pos_interp, axis=1)
            axes[0].plot(times, err, color=colors['actual'], label='EE pos error', linewidth=2)
        else:
            axes[0].plot(times, np.linalg.norm(ee_pos, axis=1), color=colors['actual'], 
                        label='EE position magnitude', linewidth=2)
        
        # Add impact time vertical line to pose error plot
        if hasattr(self, 'impact_time') and self.impact_time is not None:
            axes[0].axvline(x=self.impact_time, color=colors['impact'], 
                        linestyle='--', label='Actual Impact')
        
        axes[0].set_ylabel('Error/Position (m)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # End-effector velocity
        ee_speed = np.linalg.norm(ee_vel, axis=1)
        axes[1].plot(times, ee_speed, color=colors['actual'], label='actual', linewidth=2)
        
        if T_opt is not None and ee_vel_opt is not None:
            axes[1].plot(T_opt, np.linalg.norm(ee_vel_opt, axis=1), '--', 
                        color=colors['desired'], label='desired', linewidth=2)
            # Add desired impact velocity line
            imp_vel_des = np.linalg.norm(ee_vel_opt[-1])
            axes[1].axhline(y=imp_vel_des, color=colors['target'], linestyle='--', 
                        label=f'imp_vel_des={imp_vel_des:.2f}')
            # Add desired impact time
            axes[1].axvline(x=T_opt[-1], color=colors['target'], 
                        linestyle='--', label='Desired Impact Time')
        
        # Add impact annotations if available
        if hasattr(self, 'impact_time') and self.impact_time is not None:
            axes[1].axvline(x=self.impact_time, color=colors['impact'], 
                        linestyle='--', label='Actual Impact')
            if hasattr(self, 'impact_vel') and self.impact_vel is not None:
                impact_speed = np.linalg.norm(self.impact_vel)
                axes[1].annotate(f"Impact vel: {impact_speed:.3f} m/s", 
                            xy=(self.impact_time, impact_speed),
                            xytext=(self.impact_time + 0.1, impact_speed + 0.01),
                            arrowprops=dict(facecolor=colors['impact'], shrink=0.05))
        
        axes[1].set_ylabel('EE Speed (m/s)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Joint torques
        for j in range(min(7, tau.shape[1])):
            axes[2].plot(times, tau[:, j], color=colors[f'joint{j+1}'], 
                        label=f'τ{j+1}', linewidth=2)
        axes[2].set_ylabel('Torque (Nm)')
        axes[2].set_xlabel('Time (s)')
        axes[2].legend(ncol=4, fontsize='small')
        axes[2].grid(True, alpha=0.3)

        # plot measured torques on axis 3
        for j in range(min(7, tau_measured.shape[1])):
            axes[3].plot(times, tau_measured[:, j], color=colors[f'joint{j+1}'], 
                        label=f'Measured τ{j+1}', linestyle='--', linewidth=2)
        axes[3].set_ylabel('Measured Torque (Nm)')
        axes[3].set_xlabel('Time (s)')
        axes[3].legend(ncol=4, fontsize='small')
        axes[3].grid(True, alpha=0.3)

        # If friction compensation is enabled, plot friction torques
        if self.use_friction_compensation:
            for j in range(min(7, len(self.tau_friction[0]))):
                axes[4].plot(times, np.array(self.tau_friction)[:, j], 
                            color=colors[f'joint{j+1}'], 
                            label=f'Friction τ{j+1}', linewidth=2)
            axes[4].set_ylabel('Friction Torque (Nm)')
            axes[4].set_xlabel('Time (s)')
            axes[4].legend(ncol=4, fontsize='small')
            axes[4].grid(True, alpha=0.3)

        

        # Contact forces plot is now hidden
        # # Contact forces
        # axes[3].plot(times, tau_measured, 'r-', label='Contact Force', linewidth=2)
        # 
        # # Add impact information
        # if hasattr(self, 'impact_time') and self.impact_time is not None:
        #     axes[3].axvline(x=self.impact_time, color=colors['impact'], 
        #                 linestyle='--', label='Impact')
        #     if len(tau_measured) > 0:
        #         peak_force = max(tau_measured)
        #         peak_idx = np.argmax(tau_measured)
        #         peak_time = times[peak_idx] if peak_idx < len(times) else self.impact_time
        #         axes[3].annotate(f"Peak: {peak_force:.2f}N", 
        #                     xy=(peak_time, peak_force), 
        #                     xytext=(peak_time + 0.1, peak_force + 5),
        #                     arrowprops=dict(facecolor='red', shrink=0.05))
        # 
        # axes[3].set_ylabel('Force (N)')
        # axes[3].set_xlabel('Time (s)')
        # axes[3].legend()
        # axes[3].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_result_joints(self, T_opt=None, Z_opt=None):
        """ 
        plot (7,3) subplot joint positions velocities and torques horizontally across each joint
        Comparing between reference, actual, and nominal if friction compensation is enabled (overlaid)
        """
        if not self.times:
            print("No data to plot")
            return

        # Convert lists to numpy arrays
        times = np.array(self.times)
        q_pos = np.array(self.q_pos)
        q_vel = np.array(self.q_vel)
        tau = np.array(self.tau_log)

        # Create subplots for each joint
        fig, axes = plt.subplots(7, 3, figsize=(25, 20), sharex=True)

        for j in range(7):
            # Joint position
            axes[j, 0].plot(times, q_pos[:, j], color='blue', label='Actual')
            if T_opt is not None and Z_opt is not None:
                axes[j, 0].plot(T_opt, Z_opt[j, :], '--', color='orange', label='Reference')
            if self.use_friction_compensation and hasattr(self, 'norminal_theta_log'):
                nominal_theta = np.array(self.norminal_theta_log)[j, :] # TODO fix this maybe
                axes[j, 0].plot(times, nominal_theta, '--', color='green', label='Nominal')
            axes[j, 0].set_ylabel(f'Joint {j+1} Position (rad)')
            axes[j, 0].legend()
            axes[j, 0].grid(True)

            # Joint velocity
            axes[j, 1].plot(times, np.degrees(q_vel[:, j]), color='blue', label='Actual')
            if T_opt is not None and Z_opt is not None:
                axes[j, 1].plot(T_opt, np.degrees(Z_opt[7 + j, :]), '--', color='orange', label='Reference')
            if self.use_friction_compensation and hasattr(self, 'norminal_theta_dot_log'):
                nominal_theta_dot = np.array(self.norminal_theta_dot_log)[:, j]
                axes[j, 1].plot(times, np.degrees(nominal_theta_dot), '--', color='green', label='Nominal')
            axes[j, 1].set_ylabel(f'Joint {j+1} Velocity (deg/s)')
            axes[j, 1].legend()
            axes[j, 1].grid(True)

            # Joint torque
            axes[j, 2].plot(times, tau[:, j], color='blue', label='Actual')
            # plot measured torques if available
            if hasattr(self, 'tau_measured') and len(self.tau_measured) > 0:
                tau_measured = np.array(self.tau_measured)[:, j]
                axes[j, 2].plot(times, tau_measured, '--', color='red', label='Measured')
            if self.use_friction_compensation and hasattr(self, 'tau_friction'):
                tau_friction = np.array(self.tau_friction)[:, j]
                axes[j, 2].plot(times, tau_friction, '--', color='green', label='Friction Compensated')
            axes[j, 2].set_ylabel(f'Joint {j+1} Torque (Nm)')
            axes[j, 2].legend()
            axes[j, 2].grid(True)

        plt.tight_layout()
        plt.show()




def main():
    """Main execution function"""
    # Setup arguments
    args = TCPArguments()


    # Connect to robot and execute
    with kortex_utils.DeviceConnection.createTcpConnection(args) as router, \
         kortex_utils.DeviceConnection.createUdpConnection(args) as router_real_time:

        # Create controller
        gen3 = Kinova3HardwareController(
            router, router_real_time, 
            home_pose="Mujoco_Home",  # Adjust as needed
            use_friction_compensation=False  # Set to True if desired
        )
        ###### Example trajectory generation
        # Target pose and velocity
        # p_ee_to_ball_buttom = 0.05 + 0.025 + 0.2
        p_ee_to_ball_buttom = 0.081
        p_f = np.array([0.6, -0.05, p_ee_to_ball_buttom])    # meters
        v_f = np.array([0, 0, -0.5])   # m/s
        v_p_mag = np.linalg.norm(v_f)+0.5

        gen3.p_f = p_f
        gen3.v_f = v_f
        gen3.v_p = v_p_mag

        # Initial configuration
        q_init = np.radians([0, 15, 180, -130, 0, -35, 90])
        target_pose = gen3.fk_fun(q_init).full()
        target_pose[0:3, 3] = p_f

        gen3.R_f = target_pose[:3, :3]  # Rotation part of the target pose
        
        # Solve inverse kinematics
        q_sol = opt.inverse_kinematics_casadi(
        target_pose,
        gen3.fk_fun,
        q_init, gen3.q_lower, gen3.q_upper
        ).full().flatten()

        #target error
        target_error = gen3.fk_fun(q_sol).full()[:3,3] - target_pose[:3,3]
        print(f"Target pose error: {np.linalg.norm(target_error):.4f}m")

        # Generate trajectory
        T = 1.0  # seconds
        N = 150  # number of points
        dt = T / (N)  # time step
        traj = opt.back_propagate_traj_using_manip_ellipsoid(
            v_f, q_sol, gen3.fk_fun, gen3.jac_fun, N=N, dt=dt, v_p_mag=v_p_mag
        )

        T_opt = traj['T']
        U_opt = traj['U']
        Z_opt = traj['Z']

        # obtain linearization cache for LQR controller
        gen3.linearization_cache = opt.linearize_dynamics_along_trajectory(
            T_opt, U_opt, Z_opt, gen3.M_fun, gen3.C_fun, gen3.G_fun
        )

        # traj_reversed = opt.back_trace_from_traj(traj, gen3.jac_fun, ratio = 0.5)
        # gen3.reversed_linearization_cache = opt.linearize_dynamics_along_trajectory(
        #     traj_reversed['T'], traj_reversed['U'], traj_reversed['Z'],
        #     gen3.M_fun, gen3.C_fun, gen3.G_fun
        # )
        # gen3.reversed_traj = traj_reversed

        ## lyapunov retreat parameters
        gen3.retreat_distance = 0.15  # meters
        retreat_target_pose = target_pose.copy()
        retreat_target_pose[0:3, 3] += gen3.retreat_distance * (-v_f) / np.linalg.norm(v_f)
        q_retreat = opt.inverse_kinematics_casadi(
            retreat_target_pose,
            gen3.fk_fun,
            q_sol, gen3.q_lower, gen3.q_upper
        ).full().flatten()
        gen3.retreat_q = q_retreat
        print(f"Generated trajectory with {len(T_opt)} points, duration: {T_opt[-1]:.2f}s")
        # Initialize low-level control
        ########################
        try:
            # # Option 1: Run PID jogging to target position
            # success = gen3.init_low_level_control(
            #     sampling_time=0.001, t_end=30, 
            #     target_func=lambda dt: gen3.run_pid_jogging(Z_opt[:7, 0], dt, max_time=30.0)
            # )

            # TODO: tomorrow, debug the LQR controller and trajectory execution
            
            # Option 2: Execute trajectory
            q_start_360 = tb.to_kinova_joints(Z_opt[:7, 0])

            joint_speed = None  # degrees per second
            gen3.action_aborted = False
            result = tb.move_joints(gen3.base, q_start_360, 
                                    joint_speed,
                                    gen3.check_for_end_or_abort)

            if gen3.action_aborted or not result:
                print("Failed to move to start position")
                return

            print("Starting trajectory execution...")

            success = gen3.init_low_level_control(
                sampling_time=0.001, t_end=60,
                target_func=lambda dt: gen3.run_trajectory_execution(T_opt, U_opt, Z_opt, dt)
            )

            if success:
                print("Control initialized successfully. Running...")
                while gen3.cyclic_running:
                    try:
                        time.sleep(0.1)
                    except KeyboardInterrupt:
                        print("Keyboard interrupt received")
                        break

                # Stop control and plot results
                gen3.stop_low_level_control()
                
                # Prepare reference data for plotting
                num = len(T_opt)
                ee_pos_opt = np.zeros((num, 3))
                ee_vel_opt = np.zeros((num, 3))
                for k in range(num):
                    qk = Z_opt[:7, k]
                    dqk = Z_opt[7:, k]
                    ee_pos_opt[k] = gen3.pos_fun(qk).full().flatten()
                    jac_pos = gen3.jac_fun(qk)[0:3, :]
                    ee_vel_opt[k] = (jac_pos @ dqk).full().flatten()

                gen3.plot_result_joints(T_opt, Z_opt)
                gen3.plot_results(T_opt, Z_opt, ee_pos_opt, ee_vel_opt)
                
                print("Execution completed successfully")
            else:
                print("Failed to initialize control")

        except Exception as e:
            print(f"Error during execution: {e}")
            traceback.print_exc()
            gen3.stop_low_level_control()


if __name__ == "__main__":
    main()
    # TODO figure out the finishing trajectory, then start collecting data
    # TODO still buggy, post LQR not right, maybe lyapunov
    # TODO project to plane.