import threading
from copy import deepcopy

import numpy as np
from xr_robot_teleop_server import configure_logging
from xr_robot_teleop_server.schemas.body_pose import (
    Bone,
    deserialize_pose_data,
)

# from xr_robot_teleop_server.schemas.openxr_skeletons import (
#     FULL_BODY_SKELETON_CONNECTIONS,
#     FullBodyBoneId,
# )
from xr_robot_teleop_server.streaming import WebRTCServer

from robosuite.devices.device import Device


def convert_unity_to_right_handed_z_up(
    position: tuple[float, float, float],
    rotation: tuple[float, float, float, float],
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float, float, float, float, float, float, float],
]:
    """
    Converts position and rotation from Unity's left-handed, (X-right,Y-up, Z-forward) coordinate system
    to a right-handed, Z-up coordinate system (+X Forward, +Y Left, +Z Up).
    Also from quaternion to rotation matrix.
    """
    # Position conversion: Unity (x,y,z) -> (z, -x, y)
    new_position = (position[2], -position[0], position[1])

    # Rotation quaternion conversion:
    # For quaternion conversion, we need to first convert the quaternion from left-handed to right-handed by flipping
    # one of the xyz components.
    # right_hand_quat = np.array([rotation[0], rotation[1], rotation[2], rotation[3]])
    # from testing, this seems to be the correct conversion
    right_hand_quat = np.array([-rotation[2], rotation[0], -rotation[1], rotation[3]])
    R_unity = q2R(right_hand_quat)  # Convert quaternion to rotation matrix

    return new_position, R_unity.flatten()


def q2R(q):
    """
    Converts a quaternion into a 3 x 3 rotation matrix according to the
    Euler-Rodrigues formula.

    :type    q: numpy.array
    :param   q: 4 x 1 vector representation of a quaternion q = [qv;q0] or [x, y, z, w]
    :rtype:  numpy.array
    :return: the 3x3 rotation matrix
    """

    I = np.identity(3)
    qhat = hat(q[0:3])
    qhat2 = qhat.dot(qhat)
    return I + 2 * q[-1] * qhat + 2 * qhat2


def hat(k):
    """
    Returns a 3 x 3 cross product matrix for a 3 x 1 vector

             [  0 -k3  k2]
     khat =  [ k3   0 -k1]
             [-k2  k1   0]

    :type    k: numpy.array
    :param   k: 3 x 1 vector
    :rtype:  numpy.array
    :return: the 3 x 3 cross product matrix
    """

    khat = np.zeros((3, 3))
    khat[0, 1] = -k[2]
    khat[0, 2] = k[1]
    khat[1, 0] = k[2]
    khat[1, 2] = -k[0]
    khat[2, 0] = -k[1]
    khat[2, 1] = k[0]
    return khat


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


class XRRTCBodyPoseDevice(Device):
    """
    A device to control a robot using body pose data from XR Robot Teleop Client.
    """

    def __init__(self, env, process_bones_to_action_fn=None, **kwargs):
        if env is not None:
            super().__init__(env)
        else:
            self.env = None
            self.all_robot_arms = []
            self.all_robot_grippers = []
            self.num_robot_arms = 0
            self.num_robot_grippers = 0
        self.state_factory = StateFactory()

        if process_bones_to_action_fn is None:
            self.process_bones_to_action_fn = self._default_process_bones_to_action
        else:
            self.process_bones_to_action_fn = process_bones_to_action_fn

        datachannel_handlers = {"body_pose": self.on_body_pose_message}

        configure_logging(level="INFO")  # set xr_robot_teleop's verbosity
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
        return (
            self.state_factory.instance is not None
            and self.state_factory.instance.is_connected
        )

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
        action_dict["left_sew"] = np.concatenate(
            [np.random.rand(9) * 0.1, identity_rotation]
        )
        action_dict["right_sew"] = np.concatenate(
            [np.random.rand(9) * 0.1, identity_rotation]
        )
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
        if self.env is None:
            return input_ac_dict

        # Create the action vector for robosuite environments
        active_robot = self.env.robots[0]
        action_dict = deepcopy(input_ac_dict)
        for arm in active_robot.arms:
            action_dict[arm] = input_ac_dict.get(f"{arm}_sew")
            action_dict[f"{arm}_gripper"] = input_ac_dict.get(f"{arm}_gripper")

        return active_robot.create_action_vector(action_dict)
