#!/usr/bin/env python3

"""
Demo script for dual Kinova3 robot teleoperation using human pose estimation.

This script demonstrates how to use MediaPipe human pose estimation to control
dual Kinova3 robots via SEW (Shoulder, Elbow, Wrist) mimicking. The human's arm
movements are captured via webcam and translated to robot arm movements using
body-centric coordinate extraction and inverse kinematics.

An optional VR headset integration can be enabled to control the simulation
camera's viewpoint.

Usage:
    python demo_dual_kinova3_software_human_pose_teleop.py [options]

Example:
    # Run with default settings
    python demo_dual_kinova3_software_human_pose_teleop.py --environment DualKinova3SRLEnv --robots DualKinova3

    # Run with VR-controlled camera
    python demo_dual_kinova3_software_human_pose_teleop.py --vr-camera

Requirements:
    - MediaPipe: pip install mediapipe
    - OpenCV: pip install opencv-python
    - Webcam or camera device
    - For VR camera: TODO

Controls:
    - Raise both arms to shoulder height: Start pose tracking
    - Lower arms: Stop pose tracking  
    - 'q' key in camera window: Quit
"""

import argparse
import time
import os
import numpy as np
import mujoco
from copy import deepcopy

import robosuite as suite
import robosuite.utils.transform_utils as T

# Get the path of robosuite
repo_path = os.path.abspath(os.path.join(os.path.abspath(__file__), os.pardir, os.pardir))
dual_kinova3_sew_config_path = os.path.join(repo_path, "controllers", "config", "robots", "dualkinova3_sew_mimic.json")

from robosuite import load_composite_controller_config
from robosuite.controllers.composite.composite_controller import WholeBody
from robosuite.wrappers import VisualizationWrapper


# VR Headset Camera Control Utilities
def get_calibration_matrix(offset_pos, offset_yaw_degrees):
    """
    Creates a fixed transformation matrix to map VR headset coordinates
    to MuJoCo world coordinates.
    """
    rotation = T.quat2mat(T.axisangle2quat([0, 0, 1], np.deg2rad(offset_yaw_degrees)))
    translation = np.array(offset_pos)
    
    calib_matrix = np.eye(4)
    calib_matrix[:3, :3] = rotation
    calib_matrix[:3, 3] = translation
    return calib_matrix


def update_viewer_camera_from_vr(viewer, vr_pose_matrix, calibration_matrix):
    """
    Updates the free camera's pose from the VR headset's pose matrix.
    
    Args:
        viewer: The mujoco.viewer instance.
        vr_pose_matrix (np.array): A 3x4 numpy array from OpenVR.
        calibration_matrix (np.array): A 4x4 matrix to align coordinate systems.
    """
    # Convert 3x4 matrix from OpenVR to a 4x4 homogeneous matrix
    hmd_pose_4x4 = np.vstack([vr_pose_matrix, [0, 0, 0, 1]])
    
    # Apply the calibration to transform from VR space to MuJoCo space
    mujoco_pose = calibration_matrix @ hmd_pose_4x4
    
    # MuJoCo's free camera uses 'pos' for position and 'mat' for the rotation matrix
    viewer.cam.pos = mujoco_pose[:3, 3]
    viewer.cam.mat = mujoco_pose[:3, :3]
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.fixedcamid = -1  # Ensure we are using the free camera


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Human pose teleoperation demo for dual Kinova3 robots")
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
        default=dual_kinova3_sew_config_path,
        help="Choice of controller. Can be generic (eg. 'BASIC' or 'WHOLE_BODY_MINK_IK') or json file (see robosuite/controllers/config for examples) or None to get the robot's default controller if it exists",
    )
    parser.add_argument("--device", type=str, default="humanposedualkinova3", help="Control device: humanposedualkinova3")
    parser.add_argument("--camera-id", type=int, default=0, help="Camera device ID for pose estimation")
    parser.add_argument("--mirror-actions", action="store_true", help="Mirror actions (right robot arm follows left human arm)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode for pose estimation device")
    parser.add_argument(
        "--max_fr",
        default=30,
        type=int,
        help="Sleep when simulation runs faster than specified frame rate; 30 fps is real time.",
    )
    parser.add_argument("--vr-camera", action="store_true", help="Enable VR headset for controlling the simulation camera.")
    args = parser.parse_args()

    # VR Camera Setup
    vr_device = None
    if args.vr_camera:
        try:
            from robosuite.devices.vr_headset_device import VRHeadsetDevice
            vr_device = VRHeadsetDevice()
            vr_device.start()
            # This matrix is a starting point. You may need to adjust it for your setup.
            # It moves the camera back, up, and rotates it to face the scene.
            CALIBRATION_MATRIX = get_calibration_matrix(offset_pos=[-0.5, 0, 1.5], offset_yaw_degrees=180)
            print("VR Camera Enabled. Make sure SteamVR is running.")
        except Exception as e:
            print(f"Failed to initialize VR headset device. Continuing without VR camera. Error: {e}")
            vr_device = None

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

    # Initialize device
    if args.device == "humanposedualkinova3":
        from robosuite.devices.human_pose_dualkinova3_teleop_device import HumanPoseDualKinova3Teleop

        device = HumanPoseDualKinova3Teleop(
            env=env, 
            debug=args.debug, 
            camera_id=args.camera_id,
            mirror_actions=args.mirror_actions
        )
    else:
        raise Exception("Invalid device choice: choose 'humanposedualkinova3'.")

    # Initialize device control
    device.start_control()

    print("=" * 80)
    print("Human Pose Teleoperation Demo for Dual Kinova3 Robots")
    print("=" * 80)
    print("This demo uses MediaPipe to track your arm movements and control the robots.")
    print("\nSetup Instructions:")
    print("1. Position yourself in front of the camera")
    print("2. Make sure your full upper body is visible")
    print("3. Raise both arms to shoulder height to start control")
    print("4. Lower arms to stop control")
    print("\nCamera Controls:")
    print("- 'q' key: Quit")
    print("\nController: SEW_MIMIC (Shoulder-Elbow-Wrist mimicking)")
    print(f"Mirror actions: {'Enabled' if args.mirror_actions else 'Disabled'}")
    print(f"Camera ID: {args.camera_id}")
    print("=" * 80)

    while True:
        # Check if device wants to quit before resetting environment
        if hasattr(device, 'should_quit') and device.should_quit():
            print("Quit signal detected before environment reset. Exiting...")
            device.stop()
            break
            
        # Reset the environment
        obs = env.reset()

        print("\nEnvironment reset complete. Starting teleoperation...")
        print("Waiting for pose tracking to engage...")
        print("Raise both arms to shoulder height to begin control.")

        model = env.sim.model._model
        data = env.sim.data._data

        with mujoco.viewer.launch_passive(model, data) as viewer:
            # Set initial camera parameters for good view of dual arms if not using VR
            if vr_device is None:
                viewer.cam.distance = 3.0
                viewer.cam.azimuth = 0
                viewer.cam.elevation = -95
                viewer.cam.lookat[:] = np.array([-0.5, 0.0, 0.0])

            print("Simulation viewer launched. Ready for human pose teleoperation!")
            
            step_count = 0
            last_status_print = 0

            while viewer.is_running() and not env.done:
                start = time.time()
                step_count += 1

                # Set active robot
                active_robot = env.robots[device.active_robot]

                # Check for quit signal first, regardless of engagement state
                if hasattr(device, 'should_quit') and device.should_quit():
                    print("Quit signal detected. Stopping simulation...")
                    break  # Break from viewer loop first

                # Print status every 100 steps to help with debugging
                if step_count - last_status_print > 100:
                    engaged_status = "ENGAGED" if device.engaged else "WAITING FOR ENGAGEMENT"
                    print(f"Status: {engaged_status} (Step {step_count})")
                    last_status_print = step_count

                # VR Camera Update
                if vr_device:
                    vr_pose = vr_device.get_pose()
                    if vr_pose is not None:
                        update_viewer_camera_from_vr(viewer, vr_pose, CALIBRATION_MATRIX)

                # Get the newest action from human pose
                try:
                    input_ac_dict = device.input2action()
                except Exception as e:
                    print(f"Error getting input action: {e}")
                    # Use neutral action on error
                    input_ac_dict = {}
                    for arm in active_robot.arms:
                        input_ac_dict[f"{arm}_sew"] = np.zeros(9)

                # If action is none, check if it's due to engagement or reset
                if input_ac_dict is None:
                    # Check if it's a reset signal (not quit, since quit is handled above)
                    if hasattr(device, '_reset_state') and device._reset_state == 1:
                        print("Reset signal received. Restarting environment...")
                        break
                    # Otherwise, it's just waiting for engagement - continue without breaking
                    else:
                        # Use neutral/hold action when not engaged
                        input_ac_dict = {}
                        for arm in active_robot.arms:
                            input_ac_dict[f"{arm}_sew"] = np.zeros(9)  # Neutral SEW positions

                action_dict = deepcopy(input_ac_dict)
                
                # Set arm actions - SEW_MIMIC controller expects SEW coordinates directly
                for arm in active_robot.arms:
                    # SEW_MIMIC controller takes absolute SEW coordinates (no input_type)
                    if f"{arm}_sew" in input_ac_dict:
                        action_dict[arm] = input_ac_dict[f"{arm}_sew"]
                    else:
                        # Fallback to neutral pose if no valid SEW action
                        action_dict[arm] = np.zeros(9)  # 9 DOF for SEW positions

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
        
        # Check for quit signal after exiting viewer loop
        if hasattr(device, 'should_quit') and device.should_quit():
            print("Quit signal detected after viewer loop. Exiting...")
            break  # Break from main while loop

    # Cleanup
    print("\nCleaning up...")
    if vr_device:
        vr_device.stop()
    env.close()  # Close environment first
    device.stop()  # Stop device and close OpenCV windows
    print("Demo completed successfully!")
