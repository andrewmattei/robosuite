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
from copy import deepcopy
from collections import OrderedDict
import matplotlib.pyplot as plt


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

        # Load pinocchio model and build dynamics functions
        self.model, self.data = opt.load_kinova_model()
        self.fk_fun, self.pos_fun, self.jac_fun, self.M_fun, self.C_fun, self.G_fun = \
            opt.build_casadi_kinematics_dynamics(self.model)
        

        # target info
        self.p_f = None
        self.v_f = None
        self.v_p = None

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
        self.linearization_cache = None

        # Data storage
        self.times = []
        self.q_pos = []
        self.q_vel = []
        self.ee_pos = []
        self.ee_vel = []
        self.tau_log = []
        self.contact_forces = []

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

    def check_for_end_or_abort(self, e):
        """Return a closure checking for END or ABORT notifications"""
        def check(notification, e=e):
            print("EVENT : " + Base_pb2.ActionEvent.Name(notification.action_event))
            if notification.action_event == Base_pb2.ACTION_END \
            or notification.action_event == Base_pb2.ACTION_ABORT:
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

            # Initialize data storage
            self.times = []
            self.q_pos = []
            self.q_vel = []
            self.ee_pos = []
            self.ee_vel = []
            self.tau_log = []
            self.contact_forces = []

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
                              q_current, dq_current, horizon=100):
        """LQR tracking controller"""
        if self.linearization_cache is None:
            raise RuntimeError("Call linearize_dynamics_along_trajectory before using this gen3.")

        A_list, B_list = self.linearization_cache

        # Interpolate feed-forward torque and reference state
        u_ff, z_ref = opt.match_trajectories(current_time, T_opt, U_opt, T_opt, Z_opt)

        # Find index for selecting local linearization
        idx = np.searchsorted(T_opt, current_time)
        idx = min(max(idx, 0), len(A_list) - 1)

        # Build cost weights
        state_dim = Z_opt.shape[0]
        control_dim = U_opt.shape[0]
        Q = np.eye(state_dim)
        Q[:control_dim, :control_dim] *= 1e4
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

        return tau

    def post_contact_joint_ctrl(self, q_end, q_current, dq_current, Kp=100, Kd=10):
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

    def apply_friction_compensation(self, q, dq, q_des, dq_des, tau_task, dt):
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
        
        tau_control = -self.motor_joint_Kp @ theta_diff - self.motor_joint_Kd @ (self.previous_nominal_theta_dot - theta_dot_des)
        nominal_theta_ddot = K_r_inv @ (tau_control - G - tau_task)
        nominal_theta_dot = self.previous_nominal_theta_dot + dt * nominal_theta_ddot
        nominal_theta = self.previous_nominal_theta + dt * nominal_theta_dot

        # Update previous values
        self.previous_nominal_theta = nominal_theta
        self.previous_nominal_theta_dot = nominal_theta_dot

        # Compute friction compensation
        observer_theta_diff = (nominal_theta - q + np.pi) % (2 * np.pi) - np.pi
        tau_friction = self.motor_inertia @ self.motor_friction_l @ ((nominal_theta_dot - dq) + self.motor_friction_lp @ observer_theta_diff)
        
        return tau_task + tau_friction

    def run_trajectory_execution(self, T_opt, U_opt, Z_opt, sampling_time=0.001):
        """Execute trajectory with real-time control"""
        self.cyclic_running = True
        print("Starting trajectory execution")
        
        cyclic_count = 0
        failed_cyclic_count = 0
        t_now = time.time()
        t_init = t_now
        t_start_control = t_now

        # Initialize trajectory parameters
        phase = "tracking"
        impact_detected = False
        impact_time = None
        impact_pos = None
        impact_vel = None
        q_end = Z_opt[:self.actuator_count, int(len(T_opt) * 0.80)]

        while not self.kill_the_thread:
            t_now = time.time()
            control_time = t_now - t_start_control

            # Get current state
            q_curr, dq_curr_raw = tb.get_realtime_q_qdot(self.base_feedback)
            dq_curr = self.velocity_lowpass.filter(dq_curr_raw)
            # ee_pos, ee_vel = tb.get_realtime_ee_p_v(self.base_feedback)  #not working ?
            H_base_ee = self.get_ee_pose_in_base(q_curr)
            ee_pos = H_base_ee[:3, 3]
            ee_vel = self.get_ee_velocity_in_base(q_curr, dq_curr)
            tau_measured = self.torque_lowpass.filter(tb.get_realtime_torque(self.base_feedback))

            # Check for contact
            total_force = np.linalg.norm(tau_measured)
            # contact_detected = total_force > 15.0  # Threshold for contact detection
            # currently use z_position of the end-effector vs desired z_position
            contact_detected = ee_pos[2] < self.p_f[2]+0.005  # 5cm threshold for contact detection

            # Control logic based on phase
            if phase == "tracking":
                if self.linearization_cache is not None:
                    tau = self.lqr_tracking_controller(control_time, T_opt, U_opt, Z_opt, q_curr, dq_curr)
                else:
                    # Simple feedforward + PD tracking
                    tau_ref, z_ref = opt.match_trajectories(control_time, T_opt, U_opt, T_opt, Z_opt)
                    n = self.actuator_count
                    q_ref = z_ref[:n].flatten()
                    dq_ref = z_ref[n:].flatten()
                    
                    Kp, Kd = 80.0, 5.0
                    q_err = q_ref - q_curr
                    dq_err = dq_ref - dq_curr
                    ddq_des = Kp * q_err + Kd * dq_err
                    tau = self.inverse_dynamics(q_curr, dq_curr, ddq_des)

                # Check for impact
                if contact_detected and not impact_detected:
                    impact_detected = True
                    impact_time = control_time
                    impact_pos = ee_pos.copy()
                    impact_vel = ee_vel.copy()
                    print(f"Impact detected at time {impact_time:.3f}s")
                    phase = "post_impact"

            elif phase == "post_impact":
                tau = self.post_contact_joint_ctrl(q_end, q_curr, dq_curr)
                error = np.linalg.norm(q_curr - q_end)
                if error < 0.04:
                    print(f"Target position reached at time {control_time:.3f}s")
                    break

            # Apply friction compensation if enabled
            if self.use_friction_compensation:
                tau = self.apply_friction_compensation(q_curr, dq_curr, q_curr, dq_curr, tau, sampling_time)
            # Clip torques to limits
            tau = np.clip(tau, self.tau_lower, self.tau_upper)

            # Apply torques to all actuators
            for i in range(self.actuator_count):
                self.base_command.actuators[i].position = self.base_feedback.actuators[i].position
                self.base_command.actuators[i].torque_joint = tau[i]

            # Log data
            if control_time < T_opt[-1] + 0.2:  # Log for trajectory duration + 2 seconds
                self.times.append(control_time)
                self.q_pos.append(q_curr.copy())
                self.q_vel.append(dq_curr.copy())
                
                # Get end-effector state
                self.ee_pos.append(ee_pos.copy())
                self.ee_vel.append(ee_vel.copy())
                
                self.tau_log.append(tau.copy())
                self.contact_forces.append(total_force)

            # Send command
            self.base_command.frame_id += 1
            if self.base_command.frame_id > 65535:
                self.base_command.frame_id = 0
            for i in range(self.actuator_count):
                self.base_command.actuators[i].command_id = self.base_command.frame_id

            try:
                self.base_feedback = self.base_cyclic.Refresh(self.base_command, 0, self.sendOption)
            except:
                failed_cyclic_count += 1
            
            cyclic_count += 1

            # Exit conditions
            if self.cyclic_t_end != 0 and (t_now - t_init > self.cyclic_t_end):
                print("Execution time limit reached")
                break
            
            if control_time > T_opt[-1] + 5.0:  # 5 seconds after trajectory end
                print("Trajectory execution completed")
                break

        # Store impact information for plotting
        self.impact_time = impact_time
        self.impact_pos = impact_pos
        self.impact_vel = impact_vel

        self.cyclic_running = False
        return True

    def run_pid_jogging(self, q_desired, sampling_time=0.001, max_time=10.0):
        """Run PID jogging to desired joint positions"""
        self.cyclic_running = True
        self.init_jogging = True
        print(f"Starting PID jogging to target position")
        
        t_now = time.time()
        t_init = t_now

        while not self.kill_the_thread:
            t_now = time.time()
            control_time = t_now - t_init

            # Get current state
            q_curr, dq_curr_raw = tb.get_realtime_q_qdot(self.base_feedback)
            dq_curr = self.velocity_lowpass.filter(dq_curr_raw)

            # Compute control
            tau = self.pid_joint_jog(q_desired, q_curr, dq_curr)
            tau = np.clip(tau, self.tau_lower, self.tau_upper)

            # Apply torques
            for i in range(self.actuator_count):
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

            # Send command
            self.base_command.frame_id += 1
            if self.base_command.frame_id > 65535:
                self.base_command.frame_id = 0
            for i in range(self.actuator_count):
                self.base_command.actuators[i].command_id = self.base_command.frame_id

            try:
                self.base_feedback = self.base_cyclic.Refresh(self.base_command, 0, self.sendOption)
            except:
                pass

            # Check if target reached or timeout
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
        contact_forces = np.array(self.contact_forces) if self.contact_forces else np.zeros_like(times)

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
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

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

        # Contact forces plot is now hidden
        # # Contact forces
        # axes[3].plot(times, contact_forces, 'r-', label='Contact Force', linewidth=2)
        # 
        # # Add impact information
        # if hasattr(self, 'impact_time') and self.impact_time is not None:
        #     axes[3].axvline(x=self.impact_time, color=colors['impact'], 
        #                 linestyle='--', label='Impact')
        #     if len(contact_forces) > 0:
        #         peak_force = max(contact_forces)
        #         peak_idx = np.argmax(contact_forces)
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
        p_ee_to_ball_buttom = 0.09
        p_f = np.array([0.5, 0.0, p_ee_to_ball_buttom])    # meters
        v_f = np.array([0, 0, -0.05])   # m/s
        v_p_mag = 1.0 # m/s           # m/s

        gen3.p_f = p_f
        gen3.v_f = v_f
        gen3.v_p = v_p_mag

        # Initial configuration
        q_init = np.radians([0, 15, 180, -130, 0, -35, 90])
        target_pose = gen3.fk_fun(q_init).full()
        target_pose[0:3, 3] = p_f
        
        # Solve inverse kinematics
        q_sol = opt.inverse_kinematics_casadi(
        target_pose,
        gen3.fk_fun,
        q_init, gen3.q_lower, gen3.q_upper
        ).full().flatten()

        # Generate trajectory
        traj = opt.back_propagate_traj_using_manip_ellipsoid(
            v_f, q_sol, gen3.fk_fun, gen3.jac_fun, N=100, dt=0.01, v_p_mag=v_p_mag
        )

        T_opt = traj['T']
        U_opt = traj['U']
        Z_opt = traj['Z']

        # obtain linearization cache for LQR controller
        gen3.linearization_cache = opt.linearize_dynamics_along_trajectory(
            T_opt, U_opt, Z_opt, gen3.M_fun, gen3.C_fun, gen3.G_fun
        )

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

            finished = tb.move_joints(gen3.base, q_start_360)

            if not finished:
                print("Failed to move to start position. Exiting.")
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

                gen3.plot_results(T_opt, Z_opt, ee_pos_opt, ee_vel_opt)
                
                print("Execution completed successfully")
            else:
                print("Failed to initialize control")

        except Exception as e:
            print(f"Error during execution: {e}")
            gen3.stop_low_level_control()


if __name__ == "__main__":
    main()
