import argparse
import os
import time
from copy import deepcopy

import mujoco
import numpy as np

import robosuite as suite

# Get the path of robosuite
repo_path = os.path.abspath(os.path.join(os.path.abspath(__file__), os.pardir, os.pardir, os.pardir))
dual_kinova3_sew_config_path = os.path.join(repo_path, "controllers", "config", "robots", "dualkinova3_sew_mimic.json")

from robosuite import load_composite_controller_config
from robosuite.devices.webrtc_body_pose_device import Bone, WebRTCBodyPoseDevice, FullBodyBoneId
from robosuite.wrappers import VisualizationWrapper



def custom_process_bones_to_action(bones: list[Bone]) -> dict:
    """
    A custom function to demonstrate how to override the default action processing.
    """
    action_dict = {}

    # --- Get bone positions ---
    bone_positions = {b.id: np.array(b.position) for b in bones}
    left_wrist = bone_positions.get(FullBodyBoneId.FullBody_LeftHandWrist)
    right_wrist = bone_positions.get(FullBodyBoneId.FullBody_RightHandWrist)
    left_thumb_tip = bone_positions.get(FullBodyBoneId.FullBody_LeftHandThumbTip)
    left_index_tip = bone_positions.get(FullBodyBoneId.FullBody_LeftHandIndexTip)
    right_thumb_tip = bone_positions.get(FullBodyBoneId.FullBody_RightHandThumbTip)
    right_index_tip = bone_positions.get(FullBodyBoneId.FullBody_RightHandIndexTip)

    # --- Safety checks ---
    if left_wrist is None or right_wrist is None:
        print("Warning: Wrist bones not found. Skipping action.")
        return None

    # --- Gripper state ---
    left_gripper_dist = np.linalg.norm(left_thumb_tip - left_index_tip) if left_thumb_tip is not None and left_index_tip is not None else 0.1
    right_gripper_dist = np.linalg.norm(right_thumb_tip - right_index_tip) if right_thumb_tip is not None and right_index_tip is not None else 0.1
    left_gripper_action = np.array([1]) if left_gripper_dist > 0.05 else np.array([-1])
    right_gripper_action = np.array([1]) if right_gripper_dist > 0.05 else np.array([-1])

    # --- Arm control (absolute SEW) ---
    identity_rotation = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])
    # Note: This is a simplified mapping. You may need to scale and offset the
    # bone positions to match the robot's workspace.
    left_sew = np.concatenate([left_wrist, np.zeros(6), identity_rotation])
    right_sew = np.concatenate([right_wrist, np.zeros(6), identity_rotation])

    action_dict["left_sew"] = left_sew
    action_dict["right_sew"] = right_sew
    action_dict["left_gripper"] = left_gripper_action
    action_dict["right_gripper"] = right_gripper_action

    return action_dict



# --- Main Simulation Script ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Integrated WebRTC teleoperation demo for robosuite")
    parser.add_argument("--environment", type=str, default="DualKinova3SRLEnv")
    parser.add_argument("--robots", nargs="+", type=str, default="DualKinova3")
    parser.add_argument("--controller", type=str, default=dual_kinova3_sew_config_path)
    parser.add_argument("--device", type=str, default="keyboard")
    parser.add_argument("--max_fr", default=30, type=int)
    args = parser.parse_args()

    # 1. Create the robosuite environment.
    controller_config = load_composite_controller_config(
        controller=args.controller,
        robot=args.robots[0],
    )
    env = suite.make(
        args.environment,
        robots=args.robots,
        controller_configs=controller_config,
        has_renderer=False,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=30,
    )
    env = VisualizationWrapper(env)
    np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

    # 2. Set up the device.
    device = WebRTCBodyPoseDevice(env=env, process_bones_to_action_fn=custom_process_bones_to_action)


    # 3. Wait for the VR client to connect.
    print("\nWaiting for a VR client to connect...")
    while not device.is_connected:
        time.sleep(0.5)
    
    print("Client connected! Starting robosuite simulation.")

    # 4. Run the simulation loop.
    obs = env.reset()
    model = env.sim.model._model
    data = env.sim.data._data

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 0
        viewer.cam.elevation = -95
        viewer.cam.lookat[:] = np.array([-0.5, 0.0, 0.0])

        while viewer.is_running():
            start_time = time.time()
            
            # Get the latest pose action from the shared state
            action = device.input2action()

            if action is None:
                time.sleep(0.01) # Wait for the first pose to arrive
                continue

            env.step(action)
            viewer.sync()

            # Maintain target frame rate
            elapsed = time.time() - start_time
            if elapsed < 1 / args.max_fr:
                time.sleep(1 / args.max_fr - elapsed)

    # 5. Cleanup
    print("\nSimulation finished. Closing environment...")
    env.close()
    print("Demo completed.")
