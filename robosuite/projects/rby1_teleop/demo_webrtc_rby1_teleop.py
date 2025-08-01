"""
RBY1 robot teleoperation using WebRTC body pose estimation.
Integrates WebRTC device and SEW mimic controller for human pose teleoperation.
Author: Roo
Date created: 2025-07-29
"""

import argparse
import os
import time
from copy import deepcopy
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from robosuite.devices.webrtc_body_pose_device import (
    Bone,
    WebRTCBodyPoseDevice,
    FullBodyBoneId,
)
from sew_mimic_rby1 import SEWMimicRBY1
import sys
import traceback


class TeleopKeyCallback:
    """Key callback for teleoperation control"""

    def __init__(self):
        self.reset_requested = False
        self.teleop_enabled = True  # Enabled by default for WebRTC
        self.home_requested = False

    def __call__(self, key: int) -> None:
        if key == ord("r") or key == ord("R"):
            self.reset_requested = True
            print("Reset requested")
        elif key == ord("q") or key == ord("Q"):
            print("Quit requested")
            # This does not actually quit, the viewer does.
        elif key == ord("t") or key == ord("T"):
            self.teleop_enabled = not self.teleop_enabled
            status = "ENABLED" if self.teleop_enabled else "DISABLED"
            print(f"Teleoperation {status}")
        elif key == ord("h") or key == ord("H"):
            self.home_requested = True
            print("Home position requested")


# Get the path of robosuite
repo_path = os.path.abspath(
    os.path.join(os.path.abspath(__file__), os.pardir, os.pardir, os.pardir)
)

_HERE = Path(__file__).parent
_XML = _HERE / "rby1a" / "mujoco" / "model.xml"


def _get_body_centric_coordinates(bones: list[Bone]) -> dict:
    """
    Convert bone positions to a body-centric coordinate system for RBY1 robot.
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

    # Create body-centric coordinate frame
    # Y-axis: right to left (shoulder line)
    y_axis = left_shoulder - right_shoulder
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-8)

    # Z-axis: up direction (shoulder to hip, inverted)
    torso_vector = hip_center - shoulder_center
    z_axis = -torso_vector / (np.linalg.norm(torso_vector) + 1e-8)

    # X-axis: forward direction (cross product)
    x_axis = np.cross(y_axis, z_axis)
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-8)

    # Create transformation matrix from world to body-centric frame
    rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
    # normalize the rotation matrix
    U_rot, _, V_rot_T = np.linalg.svd(rotation_matrix)
    R_world_body = U_rot @ V_rot_T

    def transform_to_body_frame(world_pos):
        """Transform a world position to body-centric coordinates."""
        translated = world_pos - shoulder_center
        body_pos = R_world_body.T @ translated
        return body_pos

    # Extract SEW coordinates in body-centric frame
    sew_coordinates = {}

    for side in ["left", "right"]:
        side_key_pascal = side.capitalize()

        shoulder_id = getattr(FullBodyBoneId, f"FullBody_{side_key_pascal}Shoulder")
        elbow_id = getattr(FullBodyBoneId, f"FullBody_{side_key_pascal}ArmLower")
        wrist_id = getattr(FullBodyBoneId, f"FullBody_{side_key_pascal}HandWrist")

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
        wrist_rot = bone_rotations.get(wrist_id).reshape(3, 3) if wrist_id in bone_rotations else None
        if wrist_rot is not None:
            # Convert wrist rotation to body frame
            wrist_rot = R_world_body.T @ wrist_rot
            if side == 'left': # Z in palm, -X in thumb, Y in fingers pointing
                wrist_rot = wrist_rot @ Rotation.from_euler('zyx', [np.pi/2,-np.pi/2,0]).as_matrix() # some how lowercase is body frame...
            else: # right arm: -Z in palm, X in thumb, -Y in fingers pointing
                wrist_rot = wrist_rot @ Rotation.from_euler('zyx', [-np.pi/2,np.pi/2,0]).as_matrix()

        body_frame_wrist_rot = wrist_rot
        # print(f"Side: {side}, Wrist Rotation: \n{body_frame_wrist_rot}")
        # Note: after conversion, both hand pointing forward, palms facing each other, thumbs pointing upward should match
        # the x forward, y left, z up convention in robosuite

        sew_coordinates[side] = {
            'S': S_body,
            'E': E_body,
            'W': W_body,
            'wrist_rot': body_frame_wrist_rot.flatten(),
        }

    return sew_coordinates


def custom_process_bones_to_action(bones: list[Bone]) -> dict:
    """
    Custom function to process bones into actions for RBY1 robot.
    """
    action_dict = {}

    # Get bone positions and rotations
    bone_positions = {b.id: np.array(b.position) for b in bones}
    bone_rotations = {b.id: np.array(b.rotation) for b in bones}

    # Gripper state based on thumb and index finger distance
    left_thumb_tip = bone_positions.get(FullBodyBoneId.FullBody_LeftHandThumbTip)
    left_index_tip = bone_positions.get(FullBodyBoneId.FullBody_LeftHandIndexTip)
    right_thumb_tip = bone_positions.get(FullBodyBoneId.FullBody_RightHandThumbTip)
    right_index_tip = bone_positions.get(FullBodyBoneId.FullBody_RightHandIndexTip)

    left_gripper_dist = (
        np.linalg.norm(left_thumb_tip - left_index_tip)
        if left_thumb_tip is not None and left_index_tip is not None
        else 0.1
    )
    right_gripper_dist = (
        np.linalg.norm(right_thumb_tip - right_index_tip)
        if right_thumb_tip is not None and right_index_tip is not None
        else 0.1
    )

    left_gripper_action = np.array([1]) if left_gripper_dist > 0.05 else np.array([-1])
    right_gripper_action = (
        np.array([1]) if right_gripper_dist > 0.05 else np.array([-1])
    )

    # Arm control (absolute SEW)
    sew_coords = _get_body_centric_coordinates(bones)

    if sew_coords is None or sew_coords["left"] is None or sew_coords["right"] is None:
        print("Warning: Could not calculate SEW coordinates. Skipping action.")
        return None

    # Get wrist rotations
    left_rot_matrix = sew_coords["left"]["wrist_rot"]
    right_rot_matrix = sew_coords["right"]["wrist_rot"]

    if left_rot_matrix is None or right_rot_matrix is None:
        print("Warning: Could not get wrist rotation. Skipping action.")
        return None

    # Assemble final action
    left_sew_pos = np.concatenate(
        [sew_coords["left"]["S"], sew_coords["left"]["E"], sew_coords["left"]["W"]]
    )
    right_sew_pos = np.concatenate(
        [sew_coords["right"]["S"], sew_coords["right"]["E"], sew_coords["right"]["W"]]
    )

    left_sew = np.concatenate([left_sew_pos, left_rot_matrix])
    right_sew = np.concatenate([right_sew_pos, right_rot_matrix])

    action_dict["left_sew"] = left_sew
    action_dict["right_sew"] = right_sew
    action_dict["left_gripper"] = left_gripper_action
    action_dict["right_gripper"] = right_gripper_action

    return action_dict


def main():
    """Main function for RBY1 WebRTC teleoperation demo."""
    parser = argparse.ArgumentParser(
        description="WebRTC teleoperation demo for RBY1 robot"
    )
    parser.add_argument("--max_fr", default=1000, type=int, help="Maximum frame rate")
    args = parser.parse_args()

    # Check if XML file exists
    if not _XML.exists():
        print(f"Error: XML file not found at {_XML}")
        return

    try:
        print(f"Loading model from: {_XML}")
        model = mujoco.MjModel.from_xml_path(_XML.as_posix())
        data = mujoco.MjData(model)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        traceback.print_exc()
        return

    # Initialize teleoperation components
    print("Initializing teleoperation system...")
    try:
        device = WebRTCBodyPoseDevice(
            env=None,  # Not strictly needed for this script
            process_bones_to_action_fn=custom_process_bones_to_action,
        )
        controller = SEWMimicRBY1(model, data, debug=False)
        print("Teleoperation system initialized successfully!")
    except Exception as e:
        print(f"Error initializing teleoperation system: {e}")
        traceback.print_exc()
        return

    key_callback = TeleopKeyCallback()

    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=True,
        key_callback=key_callback,
    ) as viewer:
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 180
        viewer.cam.elevation = -15
        viewer.cam.lookat[:] = [0, 0, 1.0]

        print("\nWaiting for a WebRTC client to connect...")
        while not device.is_connected:
            time.sleep(0.5)
        print("Client connected! Starting simulation.")

        mujoco.mj_resetData(model, data)
        data.qpos[:] = 0
        mujoco.mj_forward(model, data)

        timestep = model.opt.timestep

        while viewer.is_running():
            start_time = time.time()

            if key_callback.reset_requested:
                mujoco.mj_resetData(model, data)
                data.qpos[:] = 0
                mujoco.mj_forward(model, data)
                key_callback.reset_requested = False
                if controller is not None:
                    controller.reset_to_home_position()

            if key_callback.home_requested:
                if controller is not None:
                    controller.reset_to_home_position()
                key_callback.home_requested = False

            if (
                device is not None
                and controller is not None
                and key_callback.teleop_enabled
            ):
                action = device.get_controller_state()
                if action:
                    sew_left = action.get("left_sew")
                    sew_right = action.get("right_sew")

                    if sew_left is not None and sew_right is not None:
                        # The controller expects SEW and wrist separately.
                        # We extract them from the action dict.
                        # Note: Gripper action is ignored in this implementation.
                        wrist_left_rot_matrix = sew_left[9:].reshape(3, 3)
                        wrist_right_rot_matrix = sew_right[9:].reshape(3, 3)

                        # True engagement since we are receiving data
                        sew_left_dict = {
                            "S": sew_left[0:3],
                            "E": sew_left[3:6],
                            "W": sew_left[6:9],
                        }
                        sew_right_dict = {
                            "S": sew_right[0:3],
                            "E": sew_right[3:6],
                            "W": sew_right[6:9],
                        }

                        controller.update_control(
                            sew_left_dict,
                            sew_right_dict,
                            wrist_left_rot_matrix,
                            wrist_right_rot_matrix,
                            engaged=True,
                        )

            # Step simulation and apply controller torques
            torques_right, torques_left, torques_torso = (
                controller.compute_control_torques()
            )
            controller.apply_torques(torques_right, torques_left, torques_torso)
            mujoco.mj_step(model, data)
            viewer.sync()

            elapsed = time.time() - start_time
            if elapsed < 1 / args.max_fr:
                time.sleep(1 / args.max_fr - elapsed)

    print("Shutting down...")


if __name__ == "__main__":
    main()
    print("Demo completed.")
