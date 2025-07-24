"""
Note: mink IK is using mj_data, where the qpos is 35-dim, while the robot state feedback is 24 [7:31]
mujoco qpos setup:
joint # 0  world_j              → qpos[0:7] (7-dim)
joint # 1  right_wheel          → qpos[7:8] (1-dim)
joint # 2  left_wheel           → qpos[8:9] (1-dim)
joint # 3  torso_0              → qpos[9:10] (1-dim)
joint # 4  torso_1              → qpos[10:11] (1-dim)
joint # 5  torso_2              → qpos[11:12] (1-dim)
joint # 6  torso_3              → qpos[12:13] (1-dim)
joint # 7  torso_4              → qpos[13:14] (1-dim)
joint # 8  torso_5              → qpos[14:15] (1-dim)
joint # 9  right_arm_0          → qpos[15:16] (1-dim)
joint #10  right_arm_1          → qpos[16:17] (1-dim)
joint #11  right_arm_2          → qpos[17:18] (1-dim)
joint #12  right_arm_3          → qpos[18:19] (1-dim)
joint #13  right_arm_4          → qpos[19:20] (1-dim)
joint #14  right_arm_5          → qpos[20:21] (1-dim)
joint #15  right_arm_6          → qpos[21:22] (1-dim)
joint #16  gripper_finger_r1    → qpos[22:23] (1-dim)
joint #17  gripper_finger_r2    → qpos[23:24] (1-dim)
joint #18  left_arm_0           → qpos[24:25] (1-dim)
joint #19  left_arm_1           → qpos[25:26] (1-dim)
joint #20  left_arm_2           → qpos[26:27] (1-dim)
joint #21  left_arm_3           → qpos[27:28] (1-dim)
joint #22  left_arm_4           → qpos[28:29] (1-dim)
joint #23  left_arm_5           → qpos[29:30] (1-dim)
joint #24  left_arm_6           → qpos[30:31] (1-dim)
joint #25  gripper_finger_l1    → qpos[31:32] (1-dim)
joint #26  gripper_finger_l2    → qpos[32:33] (1-dim)
joint #27  head_0               → qpos[33:34] (1-dim)
joint #28  head_1               → qpos[34:35] (1-dim)

Get State return:
[r_wheel, l_wheel, torse*6, r_arm*7, l_arm*7, head*2] = 24
JointPositionCommandBuilder()
[torso*6, r_arm*7, l_arm*7]

# TODO: initial state after dead man switch is not good (maybe cache issue). Only enable the upper body joints. Full body will keep drifting.
"""

import rby1_sdk
import numpy as np
import sys
import time
import argparse
import re
from rby1_sdk import *
from pathlib import Path

import time
from threading import Lock
from typing import Dict, Tuple
from enum import Enum, auto

import tf2_ros
import rclpy
from builtin_interfaces.msg import Duration
from controller_manager_msgs.srv import ListControllers, SwitchController
from geometry_msgs.msg import Pose, PoseStamped, Quaternion, Vector3
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from rclpy.time import Time
from robot_learner_msgs.msg import GripperCommand, RobotPoseStamped, ControllerStamped
from std_msgs.msg import Empty
from sensor_msgs.msg import JointState
from tf2_ros import (
    ConnectivityException,
    ExtrapolationException,
    InvalidArgumentException,
    LookupException,
)
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from robot_learner.utils.ros_utils import nanosec_to_rclpy_time, pose_to_matrix

import mujoco

# from dm_control.viewer import user_input
# from loop_rate_limiters import RateLimiter
import mink
from mink.tasks.damping_task import DampingTask

D2R = np.pi / 180  # Degree to Radian conversion factor
MINIMUM_TIME = 0.01
RESET_TIME = 3.0
LINEAR_VELOCITY_LIMIT = 1.5
ANGULAR_VELOCITY_LIMIT = np.pi * 1.5
ACCELERATION_LIMIT = 1.0
STOP_ORIENTATION_TRACKING_ERROR = 1e-5
STOP_POSITION_TRACKING_ERROR = 1e-5
WEIGHT = 0.0015
STOP_COST = 1e-2
VELOCITY_TRACKING_GAIN = 0.01
MIN_DELTA_COST = 1e-4
PATIENCE = 10

## Setup for Mink
_XML = Path(
    "/home/aloha/colcon_ws/src/robot_learner/robot_learner/robots/rby1a/mujoco/model_act.xml"
)  # require posix path
_XML = _XML.parent / "model_act.xml"
# act model include the controller (pos for joint, vel for wheel

# fmt: off
joint_names = [
    # Base joints.
    "left_wheel", "right_wheel",
    # Arm joints.
    "torso_0", "torso_1", "torso_2", "torso_3", "torso_4", "torso_5",
    "right_arm_0", "right_arm_1", "right_arm_2", "right_arm_3", "right_arm_4", "right_arm_5", "right_arm_6",
    "gripper_finger_r1", "gripper_finger_r2",
    "left_arm_0",  "left_arm_1",  "left_arm_2",  "left_arm_3",  "left_arm_4",  "left_arm_5",  "left_arm_6",
    "gripper_finger_l1", "gripper_finger_l2",
    "head_0", "head_1",
]
actuator_names = [
    "left_wheel_act", "right_wheel_act",
    "link1_act", "link2_act", "link3_act", "link4_act", "link5_act", "link6_act",
    "right_arm_1_act", "right_arm_2_act", "right_arm_3_act", "right_arm_4_act", "right_arm_5_act", "right_arm_6_act", "right_arm_7_act",
    "left_arm_1_act", "left_arm_2_act", "left_arm_3_act", "left_arm_4_act", "left_arm_5_act", "left_arm_6_act", "left_arm_7_act",
    "head_0_act", "head_1_act",
    "right_finger_act", "left_finger_act",
]
# site name for control
hands = ["left_palm", "right_palm"] # site linked to end effector

class TeleopType(Enum):
    """Enum for different teleoperation types."""
    WHOLE_BODY = auto()
    UPPER_BODY = auto()
    # add more as needed

# source frame name and type by teleop type
SOURCE_FRAME_NAME_BY_TELEOP = {
    TeleopType.WHOLE_BODY: {"name": "world", "type": "body"},
    TeleopType.UPPER_BODY: {"name": "link_torso_5", "type": "body"},
}

class RBY1Controller(Node):
    def __init__(self, node_name="rby1_controller", teleop_type=TeleopType.WHOLE_BODY, robot_ip="localhost:50051"):
        super().__init__(node_name)

        # robot setup
        self.robot_ip = robot_ip  # for simulation
        self.servo = ".*"
        self.device = ".*"

        self.base_frame_id = "base"
        self.ee_right_frame_id = "ee_right"
        self.ee_left_frame_id = "ee_left"
        self.ctrl_loop_rate = 100
        self.max_position_delta = 0.05  # TODO: adjust later
        self.max_position_delta = 0.05  # TODO: adjust later

        self.right_last_command = None
        self.right_last_command_time = None
        self.left_last_command = None
        self.left_last_command_time = None
        self.right_command_test = None
        self.left_command_test = None

        self.robot = self.init_robot(self.robot_ip, self.servo, self.device)
        
        # joint limit setup
        dyn = self.robot.get_dynamics()
        dyn_state = dyn.make_state(["base"], Model_A().robot_joint_names)
        self.q_upper_limit = dyn.get_limit_q_upper(dyn_state)[8:8 + 14]
        self.q_lower_limit = dyn.get_limit_q_lower(dyn_state)[8:8 + 14]
        self.go_to_home_pose()
        self.get_logger().info("Robot initialized and home pose set.")
        
        self._lock = Lock()
        # IK setup
        self.IK_setup, self.IK_data = self.init_mink(teleop_type=teleop_type)
        
        #### Subscribers and publishers ####
        self.left_command_subscriber = self.create_subscription(
            RobotPoseStamped,  # Message type
            "/rby1_left/goal_pose_left",  # Topic name
            self.left_command_callback,  # Callback function
            1,  # QoS profile
        )
        
        self.right_command_subscriber = self.create_subscription(
            RobotPoseStamped,  # Message type
            "/rby1_right/goal_pose_right",  # Topic name
            self.right_command_callback,  # Callback function
            1,  # QoS profile
        )
        
        self.joint_state_publisher = self.create_publisher(
            JointState,
            "joint_states",
            1,  # QoS profile
        )

        self.command_stream = self.robot.create_command_stream()
        time.sleep(3)

        self.command_sender_timer = self.create_timer(1.0 / self.ctrl_loop_rate, self.joint_command_sender_timer_callback)
        self.joint_state_timer = self.create_timer(1.0 / 100, self.publish_joint_states) # TODO: check API response time
        self.get_logger().info("Registered timer callback")
        

    def init_mink(self, teleop_type=TeleopType.UPPER_BODY):
        # setup mink tasks and solver
        
        try:
            source_frame = SOURCE_FRAME_NAME_BY_TELEOP[teleop_type]
            print(f"Performing IK for teleop type: {teleop_type.name}, source frame: {source_frame}")
        except KeyError:
            raise ValueError(f"No frame defined for teleop type {teleop_type!r}")

        model = mujoco.MjModel.from_xml_path(_XML.as_posix())
        configuration = mink.Configuration(model)
        
        # For debugging: from mujoco import mjtObj, mj_id2name
        # # 2. Iterate over every joint index
        # for jid in range(model.njnt):
        #     # a) lookup its name
        #     name = mj_id2name(model, mjtObj.mjOBJ_JOINT, jid)
        #     # b) lookup its starting slot in data.qpos
        #     start = model.jnt_qposadr[jid]
        #     # c) figure out how many dims it occupies
        #     jtype = model.jnt_type[jid]
        #     dim = 7 if jtype == mujoco.mjtJoint.mjJNT_FREE   \
        #         else 4 if jtype == mujoco.mjtJoint.mjJNT_BALL \
        #         else 1
        #     print(f"joint #{jid:2d}  {name:20s} → qpos[{start}:{start+dim}] ({dim}-dim)")

        # fmt: on
        # dof_ids = np.array([model.joint(name).id for name in joint_names])
        # actuator_ids = np.array([model.actuator(name).id for name in actuator_names])

        #### Define all the IK related tasks ####
        # task for limiting the movement of base and torso
        # we prioritize the joint with less torque and intertia (for torso)
        pelvis_orientation_task = mink.FrameTask(
            frame_name="base",
            frame_type="body",
            position_cost=0.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        )
        # Main torso task
        torso_orientation_task = mink.FrameTask(
            frame_name="link_torso_5",
            frame_type="body",
            position_cost=0.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        )
        posture_task = mink.PostureTask(
            model, cost=1e-1
        )  # TODO: add different cost for different joints
        com_task = mink.ComTask(cost=1.0)
        damping_task = DampingTask(model, cost=1.0)

        # tracking the poses of both hands
        left_hand_task = mink.RelativeFrameTask(
            frame_name="left_palm",
            frame_type="site",
            root_name=source_frame['name'],
            root_type=source_frame['type'],
            position_cost=1.0,
            orientation_cost=1.0,
        )
        right_hand_task = mink.RelativeFrameTask(
            frame_name="right_palm",
            frame_type="site",
            root_name=source_frame['name'],
            root_type= source_frame['type'],
            position_cost=1.0,
            orientation_cost=1.0,
        )
        ### Define the constraints and limits ###
        # TODO: add self collision avoidance between hands and body
        # When move the base, mainly focus on the motion on xy plane, minimize the rotation.
        # posture_cost = np.zeros((model.nv,))
        # posture_cost[2] = 1e-3
        # posture_task = mink.PostureTask(model, cost=posture_cost)

        # immobile_base_cost = np.zeros((model.nv,))
        # immobile_base_cost[:2] = 100
        # immobile_base_cost[2] = 1e-3
        # damping_task = mink.DampingTask(model, immobile_base_cost)

        # tasks = [
        #     end_effector_task,
        #     posture_task,
        # ]

        limits = [
            mink.ConfigurationLimit(model),
        ]

        # IK settings.
        solver = "quadprog"
        pos_threshold = 1e-4
        ori_threshold = 1e-4
        max_iters = 20

        model = configuration.model
        data = configuration.data  # this create a link to the mink model data
        
        # initialize IK robot state to current robot state
        qpos_swap = np.copy(data.qpos) # swap or whatever
        qpos_curr = self.robot.get_state().position
        qpos_swap[7:22]  = np.copy(qpos_curr[0:15])  # wheel + torso + right arm, in rad
        qpos_swap[24:31] = np.copy(qpos_curr[15:22])  # left arm
        configuration.update(qpos_swap) # update qpos!!
        
        # mocap ID
        left_mocap_id = model.body("left_target").mocapid[0]
        right_mocap_id = model.body("right_target").mocapid[0]
        com_mid = model.body("com_target").mocapid[0]
        # body ID
        # link_torso5_id = model.body_name2id("link_torso_5")
        # print(f"link_torso5_id: {link_torso5_id}, left_mocap_id: {left_mocap_id}, right_mocap_id: {right_mocap_id}, com_mid: {com_mid}")
        
        # Initialize to the home keyframe.
        # configuration.update_from_keyframe("teleop")
        posture_task.set_target_from_configuration(configuration)
        pelvis_orientation_task.set_target_from_configuration(configuration)
        torso_orientation_task.set_target_from_configuration(configuration)
        com_task.set_target(data.mocap_pos[com_mid]) # NOTE: make it changable
        
        # Initialize the mocap targets to its cite pose
        mink.move_mocap_to_frame(model, data, "left_target", "left_palm", "site")
        mink.move_mocap_to_frame(model, data, "right_target", "right_palm", "site")
        
        # return setup, data, tasks of IKconfiguration
        IK_setup = dict(
            model=model,
            configuration=configuration,
            limits=limits,
            solver=solver,
            left_mocap_id=left_mocap_id,
            right_mocap_id=right_mocap_id,
            com_mid=com_mid,
            # link_torso5_id=link_torso5_id,
            base_tasks=[
                pelvis_orientation_task, 
                torso_orientation_task, 
                posture_task
            ],
            reg_tasks = [
                com_task, 
                damping_task
            ],
            hands_tasks=[left_hand_task, right_hand_task],
            source_frame=source_frame,
            teleop_type=teleop_type,
        )
        
        IK_data = data
        
        return IK_setup, IK_data

    def publish_joint_states(self):
        robot_state = self.robot.get_state()
        joint_state_msg = JointState()
        joint_state_msg.header.stamp = self.get_clock().now().to_msg()
        joint_state_msg.name = self.joint_names
        joint_positions = []
        joint_velocities = []
        joint_efforts = []
        for joint_state in robot_state.joint_states:
            joint_positions.append(joint_state.position)
            joint_velocities.append(joint_state.velocity)
            joint_efforts.append(joint_state.torque)
        joint_state_msg.position = joint_positions
        joint_state_msg.velocity = joint_velocities
        joint_state_msg.effort = joint_efforts
        self.joint_state_publisher.publish(joint_state_msg)

    def init_robot(self, address: str, servo: str, power_device: str):
        """
        Initialize the robot by setting up the necessary configurations and parameters.
        This function should be called before sending any commands to the robot.
        """
        # Initialize the robot connection
        print("Attempting to connect to the robot...")

        robot = rby1_sdk.create_robot_a(address)

        if not robot.connect():
            print("Error: Unable to establish connection to the robot at")
            sys.exit(1)

        print("Successfully connected to the robot")
        print("Starting state update...")

        def cb(rs):
            print(f"Timestamp: {rs.timestamp - rs.ft_sensor_right.time_since_last_update}")
            position = rs.position * 180 / 3.141592
            print(f"torso [deg]: {position[2:2 + 6]}")
            print(f"right arm [deg]: {position[8:8 + 7]}")
            print(f"left arm [deg]: {position[15:15 + 7]}")

        robot.start_state_update(cb, 0.1)

        robot.set_parameter("default.acceleration_limit_scaling", "0.8")
        robot.set_parameter("joint_position_command.cutoff_frequency", "5")
        robot.set_parameter("cartesian_command.cutoff_frequency", "5")
        robot.set_parameter("default.linear_acceleration_limit", "5")
        # robot.set_time_scale(1.0)

        print("parameters setting is done")

        if not robot.is_connected():
            print("Robot is not connected")
            exit(1)

        if not robot.is_power_on(power_device):
            rv = robot.power_on(power_device)
            if not rv:
                print("Failed to power on")
                exit(1)

        print(servo)
        if not robot.is_servo_on(servo):
            rv = robot.servo_on(servo)
            if not rv:
                print("Fail to servo on")
                exit(1)

        control_manager_state = robot.get_control_manager_state()

        if (
            control_manager_state.state == rby1_sdk.ControlManagerState.State.MinorFault
            or control_manager_state.state == rby1_sdk.ControlManagerState.State.MajorFault
        ):

            if control_manager_state.state == rby1_sdk.ControlManagerState.State.MajorFault:
                print("Warning: Detected a Major Fault in the Control Manager!!!!!!!!!!!!!!!.")
            else:
                print("Warning: Detected a Minor Fault in the Control Manager@@@@@@@@@@@@@@@@.")

            print("Attempting to reset the fault...")
            if not robot.reset_fault_control_manager():
                print("Error: Unable to reset the fault in the Control Manager.")
                sys.exit(1)
            print("Fault reset successfully.")

        print("Control Manager state is normal. No faults detected.")

        print("Enabling the Control Manager...")
        if not robot.enable_control_manager():
            print("Error: Failed to enable the Control Manager.")
            sys.exit(1)
        print("************Control Manager enabled successfully.************")

        self.stream = robot.create_command_stream()

        # Do some init for ROS publishing
        robot_info = robot.get_robot_info()
        joint_names = []
        for joint_info in robot_info.joint_infos:
            joint_names.append(joint_info.name)

        self.joint_names = joint_names
        return robot

    def go_to_home_pose(self):
        print("Resetting robot to home pose...")

        # Define joint positions
        q_joint_waist = np.array([0, 30, -60, 30, 0, 0]) * D2R
        q_joint_right_arm = np.array([-45, -30, 0, -90, 0, 45, 0]) * D2R
        q_joint_left_arm = np.array([-45, 30, 0, -90, 0, 45, 0]) * D2R

        # Combine joint positions
        q = np.concatenate([q_joint_waist, q_joint_right_arm, q_joint_left_arm])

        # Build command
        rc = RobotCommandBuilder().set_command(
            ComponentBasedCommandBuilder().set_body_command(
                BodyCommandBuilder().set_command(
                    JointPositionCommandBuilder().set_position(q).set_minimum_time(RESET_TIME)
                )
            )
        )

        rv = self.robot.send_command(rc, 10).get()

        if rv.finish_code != RobotCommandFeedback.FinishCode.Ok:
            print("Error: Failed to conduct demo motion.")
            return 1

        return 0

    def joint_command_sender_timer_callback(self) -> None:
        """Timer callback for the control loop.
        self.IK_setup should only be accessed and updated within this function.

        Publishes the next pose in the trajectory to the robot, or the current
        pose if no trajectory is available.
        """
        if self.right_last_command is None or self.left_last_command is None:
            # print("No command received yet. Enable bother left and right Quest controllers to send commands.")
            return
        
        #### Assign msg to IK task goals ####        
        with self._lock:
            left_pose = self.left_last_command
            right_pose = self.right_last_command
            
        # # TODO: add function for getting last command, perform transformation from the robot to arbitrary frames
        # link_torso5_id = self.IK_setup['link_torso5_id']
        # base_2_torso5_pos = self.IK.data.xpos[link_torso5_id]
        # base_2_torso5_quat = self.IK_data.xquat[link_torso5_id]
        # T_base_2_torso5 = transform_from_pos_quat(
        #     base_2_torso5_pos, base_2_torso5_quat
        # )
        # T_torso5_2_ee_left = np.pinv(T_base_2_torso5) @ pose_to_matrix(left_pose)
                    
        T_base_2_ee_left = pose_to_matrix(left_pose)
        left_mocap_id = self.IK_setup['left_mocap_id']
        self.IK_data.mocap_pos[left_mocap_id][0: 3] = T_base_2_ee_left[0: 3, 3]
        self.IK_data.mocap_quat[left_mocap_id][0] = left_pose.orientation.w
        self.IK_data.mocap_quat[left_mocap_id][1] = left_pose.orientation.x
        self.IK_data.mocap_quat[left_mocap_id][2] = left_pose.orientation.y
        self.IK_data.mocap_quat[left_mocap_id][3] = left_pose.orientation.z
        
        T_base_2_ee_right = pose_to_matrix(right_pose)
        right_mocap_id = self.IK_setup['right_mocap_id']
        self.IK_data.mocap_pos[right_mocap_id][0: 3] = T_base_2_ee_right[0: 3, 3] # x, y, z
        self.IK_data.mocap_quat[right_mocap_id][0] = right_pose.orientation.w
        self.IK_data.mocap_quat[right_mocap_id][1] = right_pose.orientation.x
        self.IK_data.mocap_quat[right_mocap_id][2] = right_pose.orientation.y
        self.IK_data.mocap_quat[right_mocap_id][3] = right_pose.orientation.z
        
        # TODO: compare the two different ways of transform. Currently drifting is severe.
        self.IK_setup['configuration'].update()
        
        # left_target_transform1 = mink.SE3.from_mocap_name(self.IK_setup['model'],
        #                             self.IK_data,
        #                             "left_target")
        # right_target_transform1 = mink.SE3.from_mocap_name(self.IK_setup['model'],
        #                             self.IK_data,
        #                             "right_target")
        
        # test_left = self.IK_setup['configuration'].get_transform_frame_to_world(
        #     frame_name="left_target",
        #     frame_type="site"
        # )

        left_target_transform = self.IK_setup['configuration'].get_transform(
            source_name="left_target",
            source_type="body",
            dest_name=self.IK_setup['source_frame']['name'],
            dest_type=self.IK_setup['source_frame']['type']
        )
        right_target_transform = self.IK_setup['configuration'].get_transform(
            source_name="right_target",
            source_type="body",
            dest_name=self.IK_setup['source_frame']['name'],
            dest_type=self.IK_setup['source_frame']['type']
        ) 
        
        # print(f"left_target_transform from mocap_name: {left_target_transform1}")
        # print(f"left_target_transform wrt world: {test_left}")
        # print(f"left_target_transform from configuration: {left_target_transform}")
        
        self.IK_setup['hands_tasks'][0].set_target(left_target_transform)
        
        self.IK_setup['hands_tasks'][1].set_target(right_target_transform)
        
        #### Update the IK state
        qpos_swap = np.copy(self.IK_data.qpos) # swap or whatever
        qpos_curr = self.robot.get_state().position
        qpos_swap[7:22]  = np.copy(qpos_curr[0:15])  # wheel + torso + right arm, in rad
        qpos_swap[24:31] = np.copy(qpos_curr[15:22])  # left arm
        self.IK_setup['configuration'].update(qpos_swap) # update qpos to solve FK!!
        print(f"current qpos {qpos_curr}")
        
        #### Compute velocity and integrate into the next configuration.
        dt = 1.0 / self.ctrl_loop_rate
        vel = mink.solve_ik(
                self.IK_setup['configuration'],
                tasks=self.IK_setup['base_tasks'] + self.IK_setup["reg_tasks"] + self.IK_setup['hands_tasks'],
                dt=dt,
                solver=self.IK_setup['solver'],
                damping=1e-1,
                limits=self.IK_setup['limits'],
        )
        self.IK_setup['configuration'].integrate_inplace(vel, dt)
        
        #### Control the robot through RBY API ####
        q_waist = self.IK_data.qpos[9: 15] # data is the reference to the mink solver data
        q_right_arm = self.IK_data.qpos[15: 22]
        q_left_arm = self.IK_data.qpos[24: 31]
        
        if self.IK_setup['teleop_type'] == TeleopType.WHOLE_BODY:
            # For whole body control, we send the waist and arms joint commands
            self.send_joint_command_wbc(q_waist, q_left_arm, q_right_arm, min_time=dt)
        elif self.IK_setup['teleop_type'] == TeleopType.UPPER_BODY:
            # For upper body control, we only send the arms joint commands
            self.send_joint_command_arms(q_left_arm, q_right_arm, min_time=dt)
        else:
            raise ValueError(f"Unsupported teleop type: {self.IK_setup['teleop_type']}")

        # q_arm_test = np.concatenate([right_arm_joint, left_arm_joint])
        # self.send_joint_command(q_arm_test, min_time=dt) # NOTE: min time 0.05
        
    def continuous_action_check(self):
        """
        Check whether delta of current command and last command less than a threshold.
        """
        pass
    
    def send_joint_command_wbc(self, q_waist, q_left_arm, q_right_arm, min_time=0.05):
        
        q = np.concatenate([q_waist, q_right_arm, q_left_arm])
        
        # Add constraints!!!!!!!!!
        MINIMUM_TIME = min_time

        # Build command
        rc = RobotCommandBuilder().set_command(
            ComponentBasedCommandBuilder().set_body_command(
                BodyCommandBuilder().set_command(
                    JointPositionCommandBuilder().set_position(q).set_minimum_time(MINIMUM_TIME)
                )
            )
        )
        self.stream.send_command(rc)

    def send_joint_command_arms(self, q_left_arm, q_right_arm , min_time=0.05):
        # TODO: fix joint limit and contraints      
        target_position_right = np.clip(q_right_arm, self.q_lower_limit[:7], self.q_upper_limit[:7])
        target_position_left = np.clip(q_left_arm, self.q_lower_limit[7:14], self.q_upper_limit[7:14])
        
        acc_limit = np.full(7, 1200.0, dtype=np.float64)
        acc_limit = np.deg2rad(acc_limit) 

        vel_limit = np.array([160, 160, 160, 160, 330, 330, 330], dtype=np.float64)
        vel_limit = np.deg2rad(vel_limit)
        
        right_arm_minimum_time = min_time
        left_arm_minimum_time = min_time
        
        rc = RobotCommandBuilder().set_command(
            ComponentBasedCommandBuilder().set_body_command(
                BodyComponentBasedCommandBuilder()
                .set_right_arm_command(
                    JointPositionCommandBuilder()
                    .set_command_header(CommandHeaderBuilder().set_control_hold_time(4.0))
                    .set_minimum_time(right_arm_minimum_time)
                    .set_position(target_position_right)
                    .set_velocity_limit(vel_limit)
                    .set_acceleration_limit(acc_limit)
                )
                .set_left_arm_command(
                    JointPositionCommandBuilder()
                    .set_command_header(CommandHeaderBuilder().set_control_hold_time(4.0))
                    .set_minimum_time(left_arm_minimum_time)
                    .set_position(target_position_left)
                    .set_velocity_limit(vel_limit)
                    .set_acceleration_limit(acc_limit)
                )
            )
        )
        self.stream.send_command(rc)
                    
    def right_command_callback(self, msg, verbose=True):
        """
        Receiver right arm EEF pose command and setup the right hand IK task goal.

        """
        # if verbose:
        #     self.get_logger().info(f"Received quest state: {msg}")

        # Process the message as needed

        robot_pose = msg.robot_pose.cartesian_pose  # TODO: add support for both arms
        right_arm_position = robot_pose.position  # geometry_msgs/Point.msg
        right_arm_orientation = robot_pose.orientation  # geometry_msgs/Quaternion.msg
        
        self.right_last_command = robot_pose
        self.right_last_command_time = msg.header.stamp

        # if self.T_right is None:
            # print(f"First ever command received. right arm position: {right_arm_position}")

        # print(f"%%%%%%%%%%%Right arm pos: {right_arm_position}")
        # print(f"%%%%%%%%%%%%%Right arm ori: {right_arm_orientation}")

        # if self.continuous_action_check():
        #     return

        # T_base_2_ee_right = pose_to_matrix(robot_pose)

        # T_right = T_base_2_ee_right
        
        # print(f"right hand command received: {T_right}")

        # #### Test original code ####
        # T_torso = np.eye(4)
        # # T_right = np.eye(4)
        # T_left = np.eye(4)

        # # Define transformation matrices
        # T_torso[:3, :3] = np.eye(3)
        # T_torso[:3, 3] = [0, 0, 1]

        # angle = -np.pi / 4
        # # T_right[:3, :3] = np.array(
        # #     [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]
        # # )
        # # T_right[:3, 3] = [0.5, -0.3, 1.0]

        # T_left[:3, :3] = np.array(
        #     [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]
        # )
        # T_left[:3, 3] = [0.5, 0.3, 1.0]
        
        # TODO: add thread safe mechanism
        # TODO: check timestamped to avoid outdated command
        
        ##### IK testing #####
        # with self._lock:
        #     data = self.IK_setup['data']
        #     right_mocap_id = self.IK_setup['right_mocap_id']
        #     data.mocap_pos[right_mocap_id][0] = T_right[0, 3]
        #     data.mocap_pos[right_mocap_id][1] = T_right[1, 3]
        #     data.mocap_pos[right_mocap_id][2] = T_right[2, 3]
        #     data.mocap_quat[right_mocap_id][0] = robot_pose.orientation.w
        #     data.mocap_quat[right_mocap_id][1] = robot_pose.orientation.x
        #     data.mocap_quat[right_mocap_id][2] = robot_pose.orientation.y
        #     data.mocap_quat[right_mocap_id][3] = robot_pose.orientation.z
            
        #     self.IK_setup['hands_tasks'][1].set_target(
        #         mink.SE3.from_mocap_name(self.IK_setup['model'],
        #                                 self.IK_setup['data'],
        #                                 "right_target"))
        
        
    def left_command_callback(self, msg, verbose=True):
        """
        Callback function for /RBY_1/goal_pose topic.

        """

        # if verbose:
        #     self.get_logger().info(f"Received quest state: {msg}")

        # Process the message as needed

        robot_pose = msg.robot_pose.cartesian_pose
        left_arm_position = robot_pose.position  # geometry_msgs/Point.msg
        left_arm_orientation = robot_pose.orientation  # geometry_msgs/Quaternion.msg

        self.left_last_command = robot_pose
        self.left_last_command_time = msg.header.stamp

        # if self.T_left is None:
            # print(f"First ever command received. right arm position: {left_arm_position}")

        # print(f"%%%%%%%%%%%Right arm pos: {right_arm_position}")
        # print(f"%%%%%%%%%%%%%Right arm ori: {right_arm_orientation}")

        # if self.continuous_action_check():
        #     return

        # T_base_2_ee_left = pose_to_matrix(robot_pose)
        # T_left = T_base_2_ee_left
        
        # print(f"left hand command received: {T_left}")

        #### Test original code ####
        # T_torso = np.eye(4)
        # # T_right = np.eye(4)
        # T_left = np.eye(4)

        # # Define transformation matrices
        # T_torso[:3, :3] = np.eye(3)
        # T_torso[:3, 3] = [0, 0, 1]

        # angle = -np.pi / 4
        # # T_right[:3, :3] = np.array(
        # #     [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]
        # # )
        # # T_right[:3, 3] = [0.5, -0.3, 1.0]

        # T_left[:3, :3] = np.array(
        #     [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]
        # )
        # T_left[:3, 3] = [0.5, 0.3, 1.0]
        #############################
        
        ##### IK testing #####
        # with self._lock:
        #     data = self.IK_setup['data']
        #     left_mocap_id = self.IK_setup['left_mocap_id']
        #     data.mocap_pos[left_mocap_id][0] = T_left[0, 3]
        #     data.mocap_pos[left_mocap_id][1] = T_left[1, 3]
        #     data.mocap_pos[left_mocap_id][2] = T_left[2, 3]
        #     data.mocap_quat[left_mocap_id][0] = robot_pose.orientation.w
        #     data.mocap_quat[left_mocap_id][1] = robot_pose.orientation.x
        #     data.mocap_quat[left_mocap_id][2] = robot_pose.orientation.y
        #     data.mocap_quat[left_mocap_id][3] = robot_pose.orientation.z
            
        #     self.IK_setup['hands_tasks'][0].set_target(
        #         mink.SE3.from_mocap_name(self.IK_setup['model'],
        #                                 self.IK_setup['data'],
        #                                 "left_target"))
        
def main():
    rclpy.init()
    node = RBY1Controller(node_name="rby1_controller", teleop_type=TeleopType.WHOLE_BODY, robot_ip="localhost:50051")
    # node = RBY1Controller(node_name="rby1_controller", robot_ip="192.168.30.1:50051")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Keyboard interrupt received, shutting down...")
    finally:
        node.destory_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
