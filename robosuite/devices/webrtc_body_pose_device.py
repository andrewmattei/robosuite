import struct
import threading
import time
from copy import deepcopy
import enum

import numpy as np
from xr_360_camera_streamer.streaming import WebRTCServer

from robosuite.devices.device import Device
class FullBodyBoneId(enum.IntEnum):
    """Specifies the bone IDs for a full body skeleton, including legs and feet."""

    FullBody_Start = 0
    FullBody_Root = 0
    FullBody_Hips = 1
    FullBody_SpineLower = 2
    FullBody_SpineMiddle = 3
    FullBody_SpineUpper = 4
    FullBody_Chest = 5
    FullBody_Neck = 6
    FullBody_Head = 7
    FullBody_LeftShoulder = 8
    FullBody_LeftScapula = 9
    FullBody_LeftArmUpper = 10
    FullBody_LeftArmLower = 11
    FullBody_LeftHandWristTwist = 12
    FullBody_RightShoulder = 13
    FullBody_RightScapula = 14
    FullBody_RightArmUpper = 15
    FullBody_RightArmLower = 16
    FullBody_RightHandWristTwist = 17
    FullBody_LeftHandPalm = 18
    FullBody_LeftHandWrist = 19
    FullBody_LeftHandThumbMetacarpal = 20
    FullBody_LeftHandThumbProximal = 21
    FullBody_LeftHandThumbDistal = 22
    FullBody_LeftHandThumbTip = 23
    FullBody_LeftHandIndexMetacarpal = 24
    FullBody_LeftHandIndexProximal = 25
    FullBody_LeftHandIndexIntermediate = 26
    FullBody_LeftHandIndexDistal = 27
    FullBody_LeftHandIndexTip = 28
    FullBody_LeftHandMiddleMetacarpal = 29
    FullBody_LeftHandMiddleProximal = 30
    FullBody_LeftHandMiddleIntermediate = 31
    FullBody_LeftHandMiddleDistal = 32
    FullBody_LeftHandMiddleTip = 33
    FullBody_LeftHandRingMetacarpal = 34
    FullBody_LeftHandRingProximal = 35
    FullBody_LeftHandRingIntermediate = 36
    FullBody_LeftHandRingDistal = 37
    FullBody_LeftHandRingTip = 38
    FullBody_LeftHandLittleMetacarpal = 39
    FullBody_LeftHandLittleProximal = 40
    FullBody_LeftHandLittleIntermediate = 41
    FullBody_LeftHandLittleDistal = 42
    FullBody_LeftHandLittleTip = 43
    FullBody_RightHandPalm = 44
    FullBody_RightHandWrist = 45
    FullBody_RightHandThumbMetacarpal = 46
    FullBody_RightHandThumbProximal = 47
    FullBody_RightHandThumbDistal = 48
    FullBody_RightHandThumbTip = 49
    FullBody_RightHandIndexMetacarpal = 50
    FullBody_RightHandIndexProximal = 51
    FullBody_RightHandIndexIntermediate = 52
    FullBody_RightHandIndexDistal = 53
    FullBody_RightHandIndexTip = 54
    FullBody_RightHandMiddleMetacarpal = 55
    FullBody_RightHandMiddleProximal = 56
    FullBody_RightHandMiddleIntermediate = 57
    FullBody_RightHandMiddleDistal = 58
    FullBody_RightHandMiddleTip = 59
    FullBody_RightHandRingMetacarpal = 60
    FullBody_RightHandRingProximal = 61
    FullBody_RightHandRingIntermediate = 62
    FullBody_RightHandRingDistal = 63
    FullBody_RightHandRingTip = 64
    FullBody_RightHandLittleMetacarpal = 65
    FullBody_RightHandLittleProximal = 66
    FullBody_RightHandLittleIntermediate = 67
    FullBody_RightHandLittleDistal = 68
    FullBody_RightHandLittleTip = 69
    FullBody_LeftUpperLeg = 70
    FullBody_LeftLowerLeg = 71
    FullBody_LeftFootAnkleTwist = 72
    FullBody_LeftFootAnkle = 73
    FullBody_LeftFootSubtalar = 74
    FullBody_LeftFootTransverse = 75
    FullBody_LeftFootBall = 76
    FullBody_RightUpperLeg = 77
    FullBody_RightLowerLeg = 78
    FullBody_RightFootAnkleTwist = 79
    FullBody_RightFootAnkle = 80
    FullBody_RightFootSubtalar = 81
    FullBody_RightFootTransverse = 82
    FullBody_RightFootBall = 83
    FullBody_End = 84

    @classmethod
    def _missing_(cls, value):
        return "FullBody_Unknown"


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

class WebRTCBodyPoseDevice(Device):
    """
    A device to control a robot using body pose data from a WebRTC stream.
    """

    def __init__(self, env, process_bones_to_action_fn=None, **kwargs):
        super().__init__(env)
        self.state_factory = StateFactory()

        if process_bones_to_action_fn is None:
            self.process_bones_to_action_fn = self._default_process_bones_to_action
        else:
            self.process_bones_to_action_fn = process_bones_to_action_fn

        datachannel_handlers = {"body_pose": self.on_body_pose_message}
        self.server = WebRTCServer(
            datachannel_handlers=datachannel_handlers,
            state_factory=self.state_factory,
            video_track_factory=None,
        )

        self.server_thread = threading.Thread(target=self.server.run, daemon=True)
        self.server_thread.start()
        print("=" * 80)
        print("Integrated Robosuite WebRTC Teleoperation Server")
        print("The WebRTC server is running in the background.")
        print("Connect your VR client to this machine on port 8080.")
        print("=" * 80)

    @property
    def is_connected(self):
        """
        Returns true if the WebRTC client is connected.
        """
        return self.state_factory.instance is not None and self.state_factory.instance.is_connected

    @staticmethod
    def _default_process_bones_to_action(bones: list[Bone]) -> dict:
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

    def on_body_pose_message(self, message: bytes, state: RobosuiteTeleopState):
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
                action_dict = self.process_bones_to_action_fn(bones)
                state.update_pose(action_dict)
        except Exception as e:
            print(f"Error processing body pose message: {e}")

    def start_control(self):
        """
        Method that should be called externally before controller can
        start receiving commands.
        """
        # No-op for this device, as the server is already running.
        pass

    def get_controller_state(self):
        """
        Returns the current state of the device, a dictionary of pos, orn, grasp, and reset.
        """
        shared_state = self.state_factory.instance
        if not self.is_connected:
            return None

        pose_action = shared_state.get_pose()
        if pose_action is None:
            return None
        return pose_action

    def input2action(self, mirror_actions=False):
        """
        Converts an input from an active device into a valid action sequence that can be fed into an env.step() call
        If a reset is triggered from the device, immediately returns None. Else, returns the appropriate action
        Args:
            mirror_actions (bool): actions corresponding to viewing robot from behind.
                first axis: left/right. second axis: back/forward. third axis: down/up.
        Returns:
            Optional[Dict]: Dictionary of actions to be fed into env.step()
                            if reset is triggered, returns None
        """

        input_ac_dict = self.get_controller_state()

        if input_ac_dict is None:
            return None

        # Create the action vector for robosuite
        active_robot = self.env.robots[0]
        action_dict = deepcopy(input_ac_dict)
        for arm in active_robot.arms:
            action_dict[arm] = input_ac_dict.get(f"{arm}_sew")
            action_dict[f"{arm}_gripper"] = input_ac_dict.get(f"{arm}_gripper")

        return active_robot.create_action_vector(action_dict)
