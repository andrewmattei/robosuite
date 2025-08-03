import argparse
import os
import time
from copy import deepcopy

import mujoco
import numpy as np

import robosuite as suite
from scipy.spatial.transform import Rotation
from xr_robot_teleop_server.schemas.body_pose import Bone
from xr_robot_teleop_server.schemas.openxr_skeletons import FullBodyBoneId

# Get the path of robosuite
repo_path = os.path.abspath(
    os.path.join(os.path.abspath(__file__), os.pardir, os.pardir, os.pardir)
)
dual_kinova3_sew_config_path = os.path.join(
    repo_path, "controllers", "config", "robots", "dualkinova3_sew_mimic.json"
)

from robosuite import load_composite_controller_config
from robosuite.devices.xr_robot_teleop_client import XRRTCBodyPoseDevice, q2R
from robosuite.wrappers import VisualizationWrapper


def _get_body_centric_coordinates(bones: list[Bone]) -> dict:
    """
    Convert bone positions to a body-centric coordinate system.
    """
    bone_positions = {b.id: np.array(b.position) for b in bones}
    bone_rotations = {b.id: np.array(b.rotation) for b in bones}

    # Get key body landmarks
    left_shoulder = bone_positions.get(FullBodyBoneId.FullBody_LeftArmUpper)
    right_shoulder = bone_positions.get(FullBodyBoneId.FullBody_RightArmUpper)
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

    # Create transformation matrix from world to body-centric frame R_world_body
    rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
    # normalize the rotation matrix
    U_rot, _, V_rot_T = np.linalg.svd(rotation_matrix)
    R_world_body = U_rot @ V_rot_T

    def transform_to_body_frame(world_pos):
        """Transform a world position to body-centric coordinates."""
        # Translate to body origin
        translated = world_pos - body_origin
        # Rotate to body frame
        body_pos = R_world_body.T @ translated
        return body_pos

    # Extract SEW coordinates in body-centric frame
    sew_coordinates = {}

    for side in ["left", "right"]:
        side_key_pascal = side.capitalize()

        shoulder_id = getattr(FullBodyBoneId, f"FullBody_{side_key_pascal}ArmUpper")
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
        wrist_rot = (
            q2R(bone_rotations.get(wrist_id)) if wrist_id in bone_rotations else None
        )
        if wrist_rot is not None:
            # Convert wrist rotation to body frame
            wrist_rot = R_world_body.T @ wrist_rot
            if side == "left":  # Z in palm, -X in thumb, Y in fingers pointing
                wrist_rot = (
                    wrist_rot
                    @ Rotation.from_euler("zyx", [np.pi / 2, -np.pi / 2, 0]).as_matrix()
                )  # some how lowercase is body frame...
            else:  # right arm: -Z in palm, X in thumb, -Y in fingers pointing
                wrist_rot = (
                    wrist_rot
                    @ Rotation.from_euler("zyx", [-np.pi / 2, np.pi / 2, 0]).as_matrix()
                )

        body_frame_wrist_rot = wrist_rot
        # print(f"Side: {side}, Wrist Rotation: \n{body_frame_wrist_rot}")
        # Note: after conversion, both hand pointing forward, palms facing each other, thumbs pointing upward should match
        # the x forward, y left, z up convention in robosuite

        sew_coordinates[side] = {
            "S": S_body,
            "E": E_body,
            "W": W_body,
            "wrist_rot": body_frame_wrist_rot.flatten(),
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

    left_gripper_action = np.array([-1]) if left_gripper_dist > 0.05 else np.array([1])
    right_gripper_action = (
        np.array([-1]) if right_gripper_dist > 0.05 else np.array([1])
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


# --- Main Simulation Script ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Integrated WebRTC teleoperation demo for robosuite"
    )
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
    device = XRRTCBodyPoseDevice(
        env=env, process_bones_to_action_fn=custom_process_bones_to_action
    )

    # 3. Wait for the VR client to connect.
    print("\nWaiting for a VR client to connect...")
    while not device.is_connected:
        time.sleep(0.5)

    print("Client connected! Starting robosuite simulation.")

    # 4. Run the simulation loop.
    obs = env.reset()
    model = env.sim.model._model
    data = env.sim.data._data

    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        # Set initial camera parameters for good view of dual arms
        # viewer.cam.distance = 3.0
        # viewer.cam.azimuth = 0
        # viewer.cam.elevation = -95
        # viewer.cam.lookat[:] = np.array([-0.5, 0.0, 0.0])

        viewer.cam.distance = 2.5
        viewer.cam.azimuth = 180
        viewer.cam.elevation = -15
        viewer.cam.lookat[:] = [0, 0, 1.0]

        while viewer.is_running():
            start_time = time.time()

            # Get the latest pose action from the shared state
            action = device.input2action()

            if action is None:
                time.sleep(0.01)  # Wait for the first pose to arrive
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
