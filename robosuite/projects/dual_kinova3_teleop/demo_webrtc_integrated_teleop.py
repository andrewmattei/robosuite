import argparse
import os
import time
from copy import deepcopy

import mujoco
import numpy as np

import robosuite as suite
from scipy.spatial.transform import Rotation

# Get the path of robosuite
repo_path = os.path.abspath(os.path.join(os.path.abspath(__file__), os.pardir, os.pardir, os.pardir))
dual_kinova3_sew_config_path = os.path.join(repo_path, "controllers", "config", "robots", "dualkinova3_sew_mimic.json")

from robosuite import load_composite_controller_config
from robosuite.devices.webrtc_body_pose_device import Bone, WebRTCBodyPoseDevice, FullBodyBoneId
from robosuite.wrappers import VisualizationWrapper



def _get_body_centric_coordinates(bones: list[Bone]) -> dict:
    """
    Convert bone positions to a body-centric coordinate system.
    """
    bone_positions = {b.id: np.array(b.position) for b in bones}
    bone_rotations = {b.id: np.array(b.rotation) for b in bones}

    # Get key body landmarks
    left_shoulder = bone_positions.get(FullBodyBoneId.FullBody_LeftShoulder)
    right_shoulder = bone_positions.get(FullBodyBoneId.FullBody_RightShoulder)
    hips = bone_positions.get(FullBodyBoneId.FullBody_Hips)

    if left_shoulder is None or right_shoulder is None or hips is None:
        return None

    # Calculate body center
    shoulder_center = (left_shoulder + right_shoulder) / 2
    hip_center = hips

    # Use shoulder center as origin for upper body tracking
    body_origin = shoulder_center

    # Create body-centric coordinate frame
    # Y-axis: right to left (shoulder line)
    y_axis = left_shoulder - right_shoulder
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-8)  # normalize

    # Z-axis: up direction (shoulder to hip, inverted)
    torso_vector = hip_center - shoulder_center
    z_axis = -torso_vector / (np.linalg.norm(torso_vector) + 1e-8)  # up is positive Z

    # X-axis: forward direction (cross product)
    x_axis = np.cross(y_axis, z_axis)
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-8)

    # Create transformation matrix from world to body-centric frame
    rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])

    def transform_to_body_frame(world_pos):
        """Transform a world position to body-centric coordinates."""
        # Translate to body origin
        translated = world_pos - body_origin
        # Rotate to body frame
        body_pos = rotation_matrix.T @ translated
        return body_pos

    # Extract SEW coordinates in body-centric frame
    sew_coordinates = {}

    for side in ['left', 'right']:
        side_key_pascal = side.capitalize()

        shoulder_id = getattr(FullBodyBoneId, f'FullBody_{side_key_pascal}Shoulder')
        elbow_id = getattr(FullBodyBoneId, f'FullBody_{side_key_pascal}ArmLower')
        wrist_id = getattr(FullBodyBoneId, f'FullBody_{side_key_pascal}HandWrist')

        shoulder_pos = bone_positions.get(shoulder_id)
        elbow_pos = bone_positions.get(elbow_id)
        wrist_pos = bone_positions.get(wrist_id)

        if shoulder_pos is None or elbow_pos is None or wrist_pos is None:
            sew_coordinates[side] = None
            continue

        # Transform to body-centric coordinates
        S_body = transform_to_body_frame(shoulder_pos)
        E_body = transform_to_body_frame(elbow_pos)
        W_body = transform_to_body_frame(wrist_pos)

        # Get wrist rotation
        wrist_rot = bone_rotations.get(wrist_id)

        sew_coordinates[side] = {
            'S': S_body,
            'E': E_body,
            'W': W_body,
            'wrist_rot': wrist_rot
        }

    return sew_coordinates


def custom_process_bones_to_action(bones: list[Bone]) -> dict:
    """
    A custom function to demonstrate how to override the default action processing.
    """
    action_dict = {}

    # --- Get bone positions ---
    # --- Get bone positions and rotations ---
    bone_positions = {b.id: np.array(b.position) for b in bones}
    bone_rotations = {b.id: np.array(b.rotation) for b in bones}
    left_thumb_tip = bone_positions.get(FullBodyBoneId.FullBody_LeftHandThumbTip)
    left_index_tip = bone_positions.get(FullBodyBoneId.FullBody_LeftHandIndexTip)
    right_thumb_tip = bone_positions.get(FullBodyBoneId.FullBody_RightHandThumbTip)
    right_index_tip = bone_positions.get(FullBodyBoneId.FullBody_RightHandIndexTip)

    # --- Gripper state ---
    left_gripper_dist = np.linalg.norm(left_thumb_tip - left_index_tip) if left_thumb_tip is not None and left_index_tip is not None else 0.1
    right_gripper_dist = np.linalg.norm(right_thumb_tip - right_index_tip) if right_thumb_tip is not None and right_index_tip is not None else 0.1
    left_gripper_action = np.array([1]) if left_gripper_dist > 0.05 else np.array([-1])
    right_gripper_action = np.array([1]) if right_gripper_dist > 0.05 else np.array([-1])

    # --- Arm control (absolute SEW) ---
    sew_coords = _get_body_centric_coordinates(bones)

    if sew_coords is None or sew_coords['left'] is None or sew_coords['right'] is None:
        print("Warning: Could not calculate SEW coordinates. Skipping action.")
        return None

    # --- Get wrist rotations ---
    left_wrist_rot_quat = sew_coords['left']['wrist_rot']
    right_wrist_rot_quat = sew_coords['right']['wrist_rot']

    if left_wrist_rot_quat is None or right_wrist_rot_quat is None:
        print("Warning: Could not get wrist rotation. Skipping action.")
        return None

    # Convert to scipy Rotation objects
    left_wrist_r = Rotation.from_quat(left_wrist_rot_quat)
    right_wrist_r = Rotation.from_quat(right_wrist_rot_quat)

    # --- Gripper Orientation Offset ---
    # The human hand's natural "forward" is different from the robot's gripper.
    # We apply a 90-degree rotation offset around the Z-axis to align them.
    z_offset_90_deg = Rotation.from_euler('z', 90, degrees=True)
    
    # Apply the offset
    left_wrist_r_oriented = left_wrist_r * z_offset_90_deg
    right_wrist_r_oriented = right_wrist_r * z_offset_90_deg

    # Convert final rotation to rotation matrix for the controller
    left_rot_matrix = left_wrist_r_oriented.as_matrix().flatten()
    right_rot_matrix = right_wrist_r_oriented.as_matrix().flatten()

    # --- Assemble final action ---
    left_sew_pos = np.concatenate([sew_coords['left']['S'], sew_coords['left']['E'], sew_coords['left']['W']])
    right_sew_pos = np.concatenate([sew_coords['right']['S'], sew_coords['right']['E'], sew_coords['right']['W']])

    left_sew = np.concatenate([left_sew_pos, left_rot_matrix])
    right_sew = np.concatenate([right_sew_pos, right_rot_matrix])


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
