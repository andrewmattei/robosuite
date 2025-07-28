#!/usr/bin/env python3
import argparse
import struct
import threading
import time
from copy import deepcopy

import mujoco
import numpy as np

import robosuite as suite
from robosuite.wrappers import VisualizationWrapper
from xr_360_camera_streamer.streaming import WebRTCServer

# --- Data Structures and Deserialization ---
# This section is adapted from the 360_server_unity.py example to process
# incoming data from a VR client.

class Bone:
    """A simple data structure to hold deserialized bone data."""
    def __init__(
        self,
        id: int,
        position: tuple[float, float, float],
        rotation: tuple[float, float, float, float],
    ):
        self.id = id
        self.position = position
        self.rotation = rotation

    def __repr__(self):
        return f"Bone(id={self.id}, pos={self.position}, rot={self.rotation})"

def deserialize_pose_data(data: bytes) -> list[Bone]:
    """
    Deserializes a binary pose data stream.
    Format: int32 (bone_count), then for each bone: int32 (id) + 7*float32 (pos/rot).
    """
    bones = []
    offset = 0
    try:
        (bone_count,) = struct.unpack_from("<i", data, offset)
        offset += 4
        for _ in range(bone_count):
            if offset + 32 > len(data):
                break  # Avoid reading past the buffer
            bone_data = struct.unpack_from("<i7f", data, offset)
            offset += 32
            bones.append(Bone(bone_data[0], tuple(bone_data[1:4]), tuple(bone_data[4:8])))
    except struct.error as e:
        print(f"Error deserializing pose data: {e}")
    return bones

# --- Robosuite and WebRTC Integration ---

class RobosuiteTeleopState:
    """
    Thread-safe state to share data from the WebRTC server thread
    to the main robosuite simulation thread.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self.pose_action = None
        self.is_connected = False

    def update_pose(self, action_dict):
        with self._lock:
            self.pose_action = action_dict

    def get_pose(self):
        with self._lock:
            return self.pose_action.copy() if self.pose_action else None

def process_bones_to_action(bones: list[Bone]) -> dict:
    """
    **Placeholder**: Converts raw bone data into a robosuite action dictionary.

    This is the core translation logic. You need to implement this based on
    your VR system's bone IDs and the specific robosuite controller's needs.
    For the 'WHOLE_BODY_MIMIC' controller, you need to provide absolute
    Shoulder-Elbow-Wrist (SEW) coordinates.

    Args:
        bones: A list of Bone objects from the VR client.

    Returns:
        A dictionary with actions for the robot arms and grippers.
    """
    # --- TODO: Implement your bone-to-action mapping here. ---
    # Example: Find the wrist bones, extract their positions, and map them
    # to the robot's workspace. Determine gripper state from finger bones.
    
    # For demonstration, we return randomized mock data.
    action_dict = {}
    identity_rotation = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])
    action_dict["left_sew"] = np.concatenate([np.random.rand(9) * 0.1, identity_rotation])
    action_dict["right_sew"] = np.concatenate([np.random.rand(9) * 0.1, identity_rotation])
    action_dict["left_gripper"] = np.random.rand(1)
    action_dict["right_gripper"] = np.random.rand(1)
    return action_dict

def on_body_pose_message(message: bytes, state: RobosuiteTeleopState):
    """
    Callback for the 'body_pose' data channel. This is called by the
    WebRTCServer whenever a message is received.
    """
    if not state.is_connected:
        state.is_connected = True
        print("\n[WebRTC] Body pose data channel connected and receiving data.")

    try:
        bones = deserialize_pose_data(message)
        if bones:
            action_dict = process_bones_to_action(bones)
            state.update_pose(action_dict)
    except Exception as e:
        print(f"Error processing body pose message: {e}")

class StateFactory:
    """A factory to create and hold a reference to the state object."""
    def __init__(self):
        self.instance = None
    def __call__(self):
        # Called by WebRTCServer to create a state for a new peer.
        # We store the instance so the main thread can access it.
        if self.instance is None:
            self.instance = RobosuiteTeleopState()
        return self.instance

# --- Main Simulation Script ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Integrated WebRTC teleoperation demo for robosuite")
    parser.add_argument("--environment", type=str, default="DualKinova3SRLEnv")
    parser.add_argument("--robots", nargs=":", type=str, default="DualKinova3")
    parser.add_argument("--controller", type=str, default="WHOLE_BODY_MIMIC")
    parser.add_argument("--max_fr", default=30, type=int)
    args = parser.parse_args()

    # 1. Create a state factory. The created state will be shared between threads.
    state_factory = StateFactory()

    # 2. Set up the WebRTC server with the data handler and state factory.
    datachannel_handlers = {"body_pose": on_body_pose_message}
    server = WebRTCServer(
        datachannel_handlers=datachannel_handlers,
        state_factory=state_factory,
        video_track_factory=None,  # No video stream from server to client
    )

    # 3. Run the WebRTC server in a background thread.
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()
    print("=" * 80)
    print("Integrated Robosuite WebRTC Teleoperation Server")
    print("The WebRTC server is running in the background.")
    print("Connect your VR client to this machine on port 8080.")
    print("=" * 80)

    # 4. Create the robosuite environment.
    controller_config = suite.load_controller_config(default_controller=args.controller)
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

    # 5. Wait for the VR client to connect.
    print("\nWaiting for a VR client to connect...")
    while state_factory.instance is None or not state_factory.instance.is_connected:
        time.sleep(0.5)
    
    shared_state = state_factory.instance
    print("Client connected! Starting robosuite simulation.")

    # 6. Run the simulation loop.
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
            input_ac_dict = shared_state.get_pose()

            if input_ac_dict is None:
                time.sleep(0.01) # Wait for the first pose to arrive
                continue

            # Create the action vector for robosuite
            active_robot = env.robots[0]
            action_dict = deepcopy(input_ac_dict)
            for arm in active_robot.arms:
                action_dict[arm] = input_ac_dict.get(f"{arm}_sew")
                action_dict[f"{arm}_gripper"] = input_ac_dict.get(f"{arm}_gripper")

            env_action = active_robot.create_action_vector(action_dict)
            env.step(env_action)
            viewer.sync()

            # Maintain target frame rate
            elapsed = time.time() - start_time
            if elapsed < 1 / args.max_fr:
                time.sleep(1 / args.max_fr - elapsed)

    # 7. Cleanup
    print("\nSimulation finished. Closing environment...")
    env.close()
    # The daemon server thread will exit automatically.
    print("Demo completed.")
