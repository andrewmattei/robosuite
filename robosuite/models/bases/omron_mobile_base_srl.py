"""
Omron LD-60 Mobile Base.
"""
import numpy as np

from robosuite.models.bases.mobile_base_model import MobileBaseModel
from robosuite.utils.mjcf_utils import xml_path_completion


class OmronMobileBaseSRL(MobileBaseModel):
    """
    Omron LD-60 Mobile Base.
    For the usage in the Safe Robotics Lab, we thickened the columns 
    for temporary self-collision avoidance.
    First created by: Chuizheng Kong
    Date: 2025/04/06
    
    Args:
        idn (int or str): Number or some other unique identification string for this mount instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("bases/omron_mobile_base_srl.xml"), idn=idn)

    @property
    def top_offset(self):
        return np.array((0, 0, 0))

    @property
    def horizontal_radius(self):
        # TODO: This may be inaccurate; just a placeholder for now
        return 0.25
