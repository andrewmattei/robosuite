"""Teleoperate robot with keyboard or SpaceMouse.

***Choose user input option with the --device argument***

***Kong added teleop for DualKinova3 hardware for the Safe Robotics Lab at Georgia Tech***

Keyboard:
    We use the keyboard to control the end-effector of the robot.
    The keyboard provides 6-DoF control commands through various keys.
    The commands are mapped to joint velocities through an inverse kinematics
    solver from Bullet physics.

    Note:
        To run this script with macOS, you must run it with root access.

SpaceMouse:

    We use the SpaceMouse 3D mouse to control the end-effector of the robot.
    The mouse provides 6-DoF control commands. The commands are mapped to joint
    velocities through an inverse kinematics solver from Bullet physics.

    The two side buttons of SpaceMouse are used for controlling the grippers.

    SpaceMouse Wireless from 3Dconnexion: https://www.3dconnexion.com/spacemouse_wireless/en/
    We used the SpaceMouse Wireless in our experiments. The paper below used the same device
    to collect human demonstrations for imitation learning.

    Reinforcement and Imitation Learning for Diverse Visuomotor Skills
    Yuke Zhu, Ziyu Wang, Josh Merel, Andrei Rusu, Tom Erez, Serkan Cabi, Saran Tunyasuvunakool,
    János Kramár, Raia Hadsell, Nando de Freitas, Nicolas Heess
    RSS 2018

    Note:
        This current implementation only supports macOS (Linux support can be added).
        Download and install the driver before running the script:
            https://www.3dconnexion.com/service/drivers.html

Additionally, --pos_sensitivity and --rot_sensitivity provide relative gains for increasing / decreasing the user input
device sensitivity


***Choose controller with the --controller argument***

Choice of using either inverse kinematics controller (ik) or operational space controller (osc):
Main difference is that user inputs with ik's rotations are always taken relative to eef coordinate frame, whereas
    user inputs with osc's rotations are taken relative to global frame (i.e.: static / camera frame of reference).


***Choose environment specifics with the following arguments***

    --environment: Task to perform, e.g.: "Lift", "TwoArmPegInHole", "NutAssembly", etc.

    --robots: Robot(s) with which to perform the task. Can be any in
        {"Panda", "Sawyer", "IIWA", "Jaco", "Kinova3", "UR5e", "Baxter"}. Note that the environments include sanity
        checks, such that a "TwoArm..." environment will only accept either a 2-tuple of robot names or a single
        bimanual robot name, according to the specified configuration (see below), and all other environments will
        only accept a single single-armed robot name

    --config: Exclusively applicable and only should be specified for "TwoArm..." environments. Specifies the robot
        configuration desired for the task. Options are {"parallel" and "opposed"}

            -"parallel": Sets up the environment such that two robots are stationed next to
                each other facing the same direction. Expects a 2-tuple of robot names to be specified
                in the --robots argument.

            -"opposed": Sets up the environment such that two robots are stationed opposed from
                each other, facing each other from opposite directions. Expects a 2-tuple of robot names
                to be specified in the --robots argument.

    --arm: Exclusively applicable and only should be specified for "TwoArm..." environments. Specifies which of the
        multiple arm eef's to control. The other (passive) arm will remain stationary. Options are {"right", "left"}
        (from the point of view of the robot(s) facing against the viewer direction)

    --switch-on-grasp: Exclusively applicable and only should be specified for "TwoArm..." environments. If enabled,
        will switch the current arm being controlled every time the gripper input is pressed

    --toggle-camera-on-grasp: If enabled, gripper input presses will cycle through the available camera angles

Examples:

    For normal single-arm environment:
        $ python demo_device_control.py --environment PickPlaceCan --robots Sawyer --controller osc

    For two-arm bimanual environment:
        $ python demo_device_control.py --environment TwoArmLift --robots Baxter --config bimanual --arm left --controller osc

    For two-arm multi single-arm robot environment:
        $ python demo_device_control.py --environment TwoArmLift --robots Sawyer Sawyer --config parallel --controller osc


"""

import argparse
import time

import numpy as np
import os

# get the path of robosuite
repo_path = os.path.abspath(os.path.join(os.path.abspath(__file__), os.pardir, os.pardir))
dual_kinova3_osc_config_path = os.path.join(repo_path, "controllers", "config", "robots", "dualkinova3_osc.json")

import robosuite as suite
from robosuite import load_composite_controller_config
from robosuite.controllers.composite.composite_controller import WholeBody
from robosuite.wrappers import VisualizationWrapper

## Import the Kinova API
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient

from kortex_api.autogen.messages import Base_pb2
import robosuite.utils.tool_box_no_ros as tb
import robosuite.utils.kortex_utilities as ku


class TimeKeeper:
    def __init__(self, desired_freq=60):
        self.period = 1.0 / desired_freq
        self.last_time = time.perf_counter()
        self.time_accumulator = 0
        self.frame_count = 0
        self.start_time = self.last_time

    def should_step(self):
        current_time = time.perf_counter()
        frame_time = current_time - self.last_time
        self.last_time = current_time
        self.time_accumulator += frame_time
        return self.time_accumulator >= self.period

    def consume_step(self):
        self.time_accumulator -= self.period
        self.frame_count += 1

    def get_fps(self):
        elapsed = time.perf_counter() - self.start_time
        return self.frame_count / elapsed if elapsed > 0 else 0
    

def sync_joint_pos_with_kortex(env, left_base, left_base_cyclic, right_base, right_base_cyclic):
    ## Retrieve the simulation robot states
    active_robot = env.robots[0]
    sim_q = env.sim.data.qpos  # simulation’s full joint state vector
    sim_qd = env.sim.data.qvel
    kp = 1.0
    left_qd_cmd = None
    right_qd_cmd = None

    #### Get simulation joint positions for left arm ####
    # It is assumed that _ref_arm_joint_pos_indexes contains 14 indices (7 for right, 7 for left)
    
    left_indices  = active_robot._ref_arm_joint_pos_indexes[7:]
    sim_left_q  = sim_q[left_indices]
    sim_left_qd = sim_qd[left_indices]

    ## Retrive the hardware robot states
    left_base_feedback = left_base_cyclic.RefreshFeedback()
    real_left_q, real_left_qd = tb.get_realtime_q_qdot(left_base_feedback)

    ## compute joint velocity control error
    left_qd_cmd = np.degrees(sim_left_qd + kp*(sim_left_q - real_left_q)) # kinova uses degress/sec
    left_joint_speeds = Base_pb2.JointSpeeds()


    #### Get simulation joint positions for right arm ####
    right_indices = active_robot._ref_arm_joint_pos_indexes[:7]
    sim_right_q = sim_q[right_indices]
    sim_right_qd = sim_qd[right_indices]

    ## Retrive the hardware robot states
    right_base_feedback = right_base_cyclic.RefreshFeedback()
    real_right_q, real_right_qd = tb.get_realtime_q_qdot(right_base_feedback)

    ## compute joint velocity control error
    right_qd_cmd = np.degrees(sim_right_qd + kp*(sim_right_q - real_right_q))
    right_joint_speeds = Base_pb2.JointSpeeds()

    #### Populate and send joint speed command ####
    for joint_id in range(7):
        left_js = left_joint_speeds.joint_speeds.add()
        left_js.joint_identifier = joint_id 
        left_js.value = left_qd_cmd[joint_id]
        left_js.duration = 0

        right_js = right_joint_speeds.joint_speeds.add()
        right_js.joint_identifier = joint_id 
        right_js.value = right_qd_cmd[joint_id]
        right_js.duration = 0

    left_base.SendJointSpeedsCommand(left_joint_speeds)
    right_base.SendJointSpeedsCommand(right_joint_speeds)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="DualKinova3SRLEnv")
    parser.add_argument("--robots", nargs="+", type=str, default="DualKinova3", help="Which robot(s) to use in the env")
    parser.add_argument(
        "--config", type=str, default="default", help="Specified environment configuration if necessary"
    )
    parser.add_argument("--arm", type=str, default="right", help="Which arm to control (eg bimanual) 'right' or 'left'")
    parser.add_argument("--switch-on-grasp", action="store_true", help="Switch gripper control on gripper action")
    parser.add_argument("--toggle-camera-on-grasp", action="store_true", help="Switch camera angle on gripper action")
    parser.add_argument(
        "--controller",
        type=str,
        default=dual_kinova3_osc_config_path,
        help="Choice of controller. Can be generic (eg. 'BASIC' or 'WHOLE_BODY_MINK_IK') or json file (see robosuite/controllers/config for examples) or None to get the robot's default controller if it exists",
    )
    parser.add_argument("--device", type=str, default="keyboard")
    parser.add_argument("--pos-sensitivity", type=float, default=1.0, help="How much to scale position user inputs")
    parser.add_argument("--rot-sensitivity", type=float, default=1.0, help="How much to scale rotation user inputs")
    parser.add_argument(
        "--max_fr",
        default=30,
        type=int,
        help="Sleep when simluation runs faster than specified frame rate; 30 fps is real time.",
    )
    args = parser.parse_args()

    # Get controller config
    controller_config = load_composite_controller_config(
        controller=args.controller,
        robot=args.robots[0],
    )

    # Create argument configuration
    config = {
        "env_name": args.environment,
        "robots": args.robots,
        "controller_configs": controller_config,
    }

    # Check if we're using a multi-armed environment and use env_configuration argument if so
    if "TwoArm" in args.environment:
        config["env_configuration"] = args.config
    else:
        args.config = None

    # Create environment
    env = suite.make(
        **config,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera="frontview",
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=30,
        hard_reset=False,
    )

    env_step_size = 0.0005 #

    # Wrap this environment in a visualization wrapper
    env = VisualizationWrapper(env, indicator_configs=None)

    # Setup printing options for numbers
    np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

    # initialize device
    if args.device == "keyboard":
        from robosuite.devices import Keyboard

        device = Keyboard(env=env, pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
        env.viewer.add_keypress_callback(device.on_press)
    elif args.device == "spacemouse":
        from robosuite.devices import SpaceMouse

        device = SpaceMouse(env=env, pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
    elif args.device == "mjgui":
        from robosuite.devices.mjgui import MJGUI

        device = MJGUI(env=env)
    elif args.device == "quest":
        from robosuite.devices.quest import Quest

        device = Quest(env=env,debug=True)
    elif args.device == "questdualkinova3":
        from robosuite.devices.quest_dualkinova3_teleop import QuestDualKinova3Teleop

        device = QuestDualKinova3Teleop(env=env, debug=False)
    else:
        raise Exception("Invalid device choice: choose either 'keyboard' or 'spacemouse'.")


    while True:
        # Reset the environment
        obs = env.reset()


        # Setup rendering
        cam_id = 0
        num_cam = len(env.sim.model.camera_names)
        env.render()

        # Initialize variables that should be maintained between resets
        last_grasp = 0

        # Initialize device control
        device.start_control()
        all_prev_gripper_actions = [
            {
                f"{robot_arm}_gripper": np.repeat([0], robot.gripper[robot_arm].dof)
                for robot_arm in robot.arms
                if robot.gripper[robot_arm].dof > 0
            }
            for robot in env.robots
        ]

        while True:
            start = time.time()

            # Set active robot
            active_robot = env.robots[device.active_robot]

            # Get the newest action
            input_ac_dict = device.input2action()
            # this sends our actions to the sim using the dictionary returned by input2action

            # If action is none, then this a reset so we should break
            if input_ac_dict is None:
                break

            from copy import deepcopy

            action_dict = deepcopy(input_ac_dict)  # {}
            # set arm actions
            for arm in active_robot.arms:
                if isinstance(active_robot.composite_controller, WholeBody):  # input type passed to joint_action_policy
                    controller_input_type = active_robot.composite_controller.joint_action_policy.input_type
                else:
                    controller_input_type = active_robot.part_controllers[arm].input_type

                if controller_input_type == "delta":
                    action_dict[arm] = input_ac_dict[f"{arm}_delta"]
                elif controller_input_type == "absolute":
                    action_dict[arm] = input_ac_dict[f"{arm}_abs"]
                else:
                    raise ValueError

            # Maintain gripper state for each robot but only update the active robot with action
            env_action = [robot.create_action_vector(all_prev_gripper_actions[i]) for i, robot in enumerate(env.robots)]
            env_action[device.active_robot] = active_robot.create_action_vector(action_dict)
            env_action = np.concatenate(env_action)
            for gripper_ac in all_prev_gripper_actions[device.active_robot]:
                all_prev_gripper_actions[device.active_robot][gripper_ac] = action_dict[gripper_ac]

            env.step(env_action)

            env.render()

            # limit frame rate if necessary
            if args.max_fr is not None:
                elapsed = time.time() - start
                diff = 1 / args.max_fr - elapsed
                if diff > 0:
                    time.sleep(diff)
