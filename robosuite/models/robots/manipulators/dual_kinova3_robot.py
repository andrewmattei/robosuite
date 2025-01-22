import numpy as np

from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.utils.mjcf_utils import xml_path_completion


class DualKinova3(ManipulatorModel):
    """
    The Gen3 robot is the sparkly newest addition to the Kinova line

    Args:
        idn (int or str): Number or some other unique identification string for this robot instance
    """

    arms = ["left", "right"]

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("robots/dual_kinova3/robot.xml"), idn=idn)

    @property
    def default_base(self):
        return "RethinkMount"

    @property
    def default_gripper(self):
        """
        Since this is bimanual robot, returns dict with `'left'`, `'right'` keywords corresponding to their respective
        values

        Returns:
            dict: Dictionary containing arm-specific gripper names
        """
        return {"left": "Robotiq85Gripper", "right": "Robotiq85Gripper"}

    @property
    def default_controller_config(self):
        return {"left": "osc_pose", "right": "osc_pose"}

    @property
    def init_qpos(self):
        return np.array([np.pi/2, -np.pi/2, np.pi/2, -np.pi/2, 0.0, -np.pi/6, -np.pi/2,
                         -np.pi/2, np.pi/2, -np.pi/2, np.pi/2, 0.0, np.pi/6, np.pi/2])

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.5, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 1.0))

    @property
    def _horizontal_radius(self):
        return 0.5

    @property
    def arm_type(self):
        return "bimanual"
    
    @property
    def _eef_name(self):
        """
        Since this is bimanual robot, returns dict with `'right'`, `'left'` keywords corresponding to their respective
        values

        Returns:
            dict: Dictionary containing arm-specific eef names
        """
        return {"left": "left_hand", "right": "right_hand"}
