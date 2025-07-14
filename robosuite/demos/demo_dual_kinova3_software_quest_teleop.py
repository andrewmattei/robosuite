"""Teleoperate DualKinova3 robot in simulation with Quest controllers.

This is a software-only version that does not require any Kortex API or hardware connections.
Only simulation-based teleoperation using Quest controllers.

***Choose user input option with the --device argument***

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

Quest:
    We use Quest controllers to provide 6-DoF control commands for bimanual manipulation.
    The controllers provide position and orientation tracking for end-effector control.

Additionally, --pos_sensitivity and --rot_sensitivity provide relative gains for increasing / decreasing the user input
device sensitivity

***Choose controller with the --controller argument***

Choice of using either inverse kinematics controller (ik) or operational space controller (osc):
Main difference is that user inputs with ik's rotations are always taken relative to eef coordinate frame, whereas
    user inputs with osc's rotations are taken relative to global frame (i.e.: static / camera frame of reference).

Examples:

    For Quest teleoperation:
        $ python demo_dual_kinova3_software_quest_teleop.py --device questdualkinova3

    For keyboard teleoperation:
        $ python demo_dual_kinova3_software_quest_teleop.py --device keyboard

"""

import argparse
import time
from copy import deepcopy

import numpy as np
import os

# get the path of robosuite
repo_path = os.path.abspath(os.path.join(os.path.abspath(__file__), os.pardir, os.pardir))
# dual_kinova3_osc_config_path = os.path.join(repo_path, "controllers", "config", "robots", "dualkinova3_osc.json")
dual_kinova3_osc_config_path = os.path.join(repo_path, "controllers", "config", "robots", "dualkinova3_osc_geo.json")

import mujoco
import robosuite as suite
from robosuite import load_composite_controller_config
from robosuite.controllers.composite.composite_controller import WholeBody
from robosuite.wrappers import VisualizationWrapper


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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="DualKinova3SRLEnv", help="Which environment to use")
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
    parser.add_argument("--device", type=str, default="questdualkinova3", help="Control device: keyboard, spacemouse, quest, questdualkinova3")
    parser.add_argument("--pos-sensitivity", type=float, default=1.0, help="How much to scale position user inputs")
    parser.add_argument("--rot-sensitivity", type=float, default=1.0, help="How much to scale rotation user inputs")
    parser.add_argument(
        "--max_fr",
        default=30,
        type=int,
        help="Sleep when simulation runs faster than specified frame rate; 30 fps is real time.",
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
        has_renderer=False,
        has_offscreen_renderer=False,
        render_camera="frontview",
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=30,
        hard_reset=False,
    )

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

        device = Quest(env=env, debug=True)
    elif args.device == "questdualkinova3":
        from robosuite.devices.quest_dualkinova3_teleop import QuestDualKinova3Teleop

        device = QuestDualKinova3Teleop(env=env, debug=False, mirror_actions=False)
        # note that when mirroring actions, keep quest facing the same direction as the robot
    else:
        raise Exception("Invalid device choice: choose 'keyboard', 'spacemouse', 'quest', or 'questdualkinova3'.")

    # Initialize device control
    device.start_control()

    while True:
        # Reset the environment
        obs = env.reset()

        print("Environment reset complete. Starting teleoperation...")
        print("Use your selected device to control the robot.")
        print("Press Ctrl+C to exit.")


        model = env.sim.model._model
        data = env.sim.data._data

        with mujoco.viewer.launch_passive(model, data) as viewer:
            # Set initial camera parameters
            viewer.cam.distance = 3.0
            viewer.cam.azimuth = 0
            viewer.cam.elevation = -95
            viewer.cam.lookat[:] = np.array([-0.5, 0.0, 0.0])

            print("Simulation viewer launched. Ready for teleoperation!")

            while viewer.is_running() and not env.done:
                start = time.time()

                # Set active robot
                active_robot = env.robots[device.active_robot]

                # Get the newest action
                input_ac_dict = device.input2action()
                # this sends our actions to the sim using the dictionary returned by input2action

                # If action is none, then this a reset so we should break
                if input_ac_dict is None:
                    print("Reset signal received. Stopping simulation...")
                    break

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
                        raise ValueError(f"Unsupported controller input type: {controller_input_type}")

                # Directly create action vector from current action_dict
                env_action = active_robot.create_action_vector(action_dict)
                
                # Step the simulation
                env.step(env_action)

                # Sync the viewer
                viewer.sync()

                # Limit frame rate if necessary
                if args.max_fr is not None:
                    elapsed = time.time() - start
                    diff = 1 / args.max_fr - elapsed
                    if diff > 0:
                        time.sleep(diff)

