import time
import threading
import numpy as np

from robosuite.devices.oculus_reader.oculus_reader import OculusReader
from robosuite.devices import *
from robosuite.models.robots import *
from robosuite.robots import *
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper
import robosuite.utils.transform_utils as T
import robosuite.utils.tool_box_no_ros as tb

### Example output from OculusReader.get_transformations_and_buttons() (Rail Version)
# (
#     {
#         'l': array([[-0.15662  , -0.0793369,  0.177977 ,  0.012368 ],
#         [ 0.0902732, -0.231901 , -0.0239341, -0.0251249],
#         [ 0.172688 ,  0.0492721,  0.173929 , -0.0560221],
#         [ 0.       ,  0.       ,  0.       ,  1.       ]]),
#         'r': array([[ 0.923063 , -0.0438108, -0.382145 ,  0.183798 ],
#         [-0.351833 ,  0.305362 , -0.884854 , -0.303325 ],
#         [ 0.155459 ,  0.951228 ,  0.266455 , -0.102936 ],
#         [ 0.       ,  0.       ,  0.       ,  1.       ]])
#     },
#     {'A': False, 'B': False, 'RThU': True, 'RJ': False, 'RG': True, 'RTr': False, 'X': False, 'Y': False, 'LThU': True, 'LJ': False, 'LG': False, 'LTr': False, 'leftJS': (0.0, 0.0), 'leftTrig': (0.0,), 'leftGrip': (0.0,), 'rightJS': (0.0, 0.0), 'rightTrig': (0.0,), 'rightGrip': (1.0,)}
# )

# With Nadun's version:
# {'LeftController': {'PrimaryButton': 0, 'SecondaryButton': 0, 'Trigger': 0, 'Grip': 0, 'Menu': 0, 'AxisClicked': 0, 'IsTracked': 0, 
# 'TriggerValue': 0.0, 'GripValue': 0.0, 'Thumbstick': {'x': 0.0, 'y': 0.0}, 'Position': {'x': -0.12119103968143463, 'y': 0.5454994440078735, 'z': 0.8644617795944214}, 
# 'Rotation': {'x': 0.5608005523681641, 'y': 0.119669109582901, 'z': 0.5367729663848877, 'w': 0.618915855884552}}, 
# 'RightController': {'PrimaryButton': 0, 'SecondaryButton': 0, 'Trigger': 0, 'Grip': 0, 'Menu': 0, 'AxisClicked': 0, 'IsTracked': 0, 
# 'TriggerValue': 0.0, 'GripValue': 0.0, 'Thumbstick': {'x': 0.0, 'y': 0.0}, 'Position': {'x': 0.24589267373085022, 'y': 0.4867749810218811, 'z': 0.863879919052124}, 
# 'Rotation': {'x': -0.07060858607292175, 'y': -0.5715464949607849, 'z': 0.6635912656784058, 'w': 0.4774887263774872}}}

class Quest(Device):
    def __init__(
            self,
            env=None,
            robot_type="franka",
            bimanual=False,
            debug=False,
        ):
        super().__init__(env)

        self.robot_interface = env
        self.robot_type = robot_type
        self.bimanual = bimanual

        self.oculus_reader = OculusReader()

        self.controller_state_lock = threading.Lock() # lock to ensure safe access
        self.controller_state = None
        self._reset_state = 0

        self._controller_names = ["LeftController", "RightController"] # NOTE if there is one robot, it will use the "l" controller
        self._arm2controller = {
            "left": "LeftController",
            "right": "RightController",
        }
        self._button_names = {
            "LeftController": {
                "trigger_val": "TriggerValue",
                "trigger_bool": "Trigger",
                "grip_val": "GripValue",
                "grip_bool": "Grip",
                "reset_bool": "SecondaryButton",
            },
            "RightController": {
                "trigger_val": "TriggerValue",
                "trigger_bool": "Trigger",
                "grip_val": "GripValue",
                "grip_bool": "Grip",
                "reset_bool": "SecondaryButton",
            },
        }

        self.grip_pressed = False  # this can be used to control engage
        self.trigger_pressed = dict([(name, False) for name in self._controller_names])
        self._reset_pressed = False # whether we should reset the transform
        self.engaged = False # TODO(VS) why? remove, not useful

        self.hand_grasp = [-1]
        # TODO above variables can perhaps be made local variables
        self.initialize_pose = True
        # self.pos_delta = [0, 0, 0]
        # self.last_pos = None

        # controller initial poses
        self.controller_init_pos = dict([(name, np.zeros(3)) for name in self._controller_names])
        self.controller_init_rot = dict([(name, np.zeros((3, 3))) for name in self._controller_names])
        self.controller_init_rpy = dict([(name, np.zeros(3)) for name in self._controller_names])

        self.ee_init_pos = dict([(name, np.zeros(3)) for name in self._controller_names])
        self.ee_init_rot = dict([(name, np.zeros((3, 3))) for name in self._controller_names])
        self.ee_init_rpy = dict([(name, np.zeros(3)) for name in self._controller_names])

        # # command poses
        # self.target_ori = self.ee_init_rot
        # self.target_pos = self.ee_init_pos
        # self.target_pose = (self.target_pos.tolist(), self.target_ori.tolist())

        # Golden offset for Grey Robot with Quest 3 # TODO(VS) why?
        self.R_rs_questwd = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])

        self.debug = debug

        # Set controller offset from robot base frame.
        self.approx_R_rs_questwd()

        self._display_controls()
        self._reset_internal_state()

    @staticmethod
    def _display_controls():
        """
        Method to pretty print controls.
        """

        def print_command(char, info):
            char += " " * (30 - len(char))
            print("{}\t{}".format(char, info))

        print("")
        print_command("Control", "Command")
        print_command("Front Grip", "control pose")
        print_command("Side Grip", "control gripper")
        print("")

    def _reset_internal_state(self):
        super()._reset_internal_state()

        self.rotation = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
        self.raw_drotation = np.zeros(3)  # immediate roll, pitch, yaw delta values from keyboard hits
        self.last_drotation = np.zeros(3)
        self.pos = np.zeros(3)  # (x, y, z)
        self.last_pos = np.zeros(3)

        self.trigger_pressed = dict([(name, False) for name in self._controller_names])
        self.initialize_pose = True

        self.controller_state = None
        if self.robot_interface is not None or self.debug:
            self.set_robot_transform() # intializes robot pose and controller_state
        self.engaged = False # TODO maybe remove, not useful

    def start_control(self):
        """
        Method that should be called externally before controller can
        start receiving commands.
        """
        self._reset_internal_state()
        self._reset_state = 0
        self.engaged = True

    def _postprocess_device_outputs(self, dpos, drotation):
        drotation = drotation * 1
        dpos = dpos * 30

        dpos = np.clip(dpos, -1, 1)
        drotation = np.clip(drotation, -1, 1)

        return dpos, drotation

    # With Nadun's version:
    # {'LeftController': {'PrimaryButton': 0, 'SecondaryButton': 0, 'Trigger': 0, 'Grip': 0, 'Menu': 0, 'AxisClicked': 0, 'IsTracked': 0, 
    # 'TriggerValue': 0.0, 'GripValue': 0.0, 'Thumbstick': {'x': 0.0, 'y': 0.0}, 'Position': {'x': -0.12119103968143463, 'y': 0.5454994440078735, 'z': 0.8644617795944214}, 
    # 'Rotation': {'x': 0.5608005523681641, 'y': 0.119669109582901, 'z': 0.5367729663848877, 'w': 0.618915855884552}}, 
    # 'RightController': {'PrimaryButton': 0, 'SecondaryButton': 0, 'Trigger': 0, 'Grip': 0, 'Menu': 0, 'AxisClicked': 0, 'IsTracked': 0, 
    # 'TriggerValue': 0.0, 'GripValue': 0.0, 'Thumbstick': {'x': 0.0, 'y': 0.0}, 'Position': {'x': 0.24589267373085022, 'y': 0.4867749810218811, 'z': 0.863879919052124}, 
    # 'Rotation': {'x': -0.07060858607292175, 'y': -0.5715464949607849, 'z': 0.6635912656784058, 'w': 0.4774887263774872}}}

    def get_controller_state(self):
        with self.controller_state_lock:
            new_controller_state = {}
            robot = self.env.robots[self.active_robot]

            # Get controller(s) data.
            quest_data = self.oculus_reader.get_controller_inputs()
            # print(quest_data)
            # When not in operation, no data will be generated so wait for the datastream
            while quest_data == None:  
                time.sleep(0.001)
                quest_data = self.oculus_reader.get_controller_inputs()
                # print(quest_data)

            # Parse data per controller.
            for controller in quest_data:
                new_controller_state[controller] = {}

                controller_state = quest_data[controller]

                # Trigger
                if controller_state[self._button_names[controller]["trigger_bool"]]:
                    if not self.trigger_pressed[controller]:
                        # trigger was just pressed, (re)initialize pose
                        self.initialize_pose = True
                    self.trigger_pressed[controller] = True
                else:
                    self.trigger_pressed[controller] = False
                    # self.pos_delta = [0, 0, 0]

                # Grip, used for gripper control
                if controller_state[self._button_names[controller]["grip_bool"]]:
                    if controller_state[self._button_names[controller]["grip_val"]] > 0.3:
                        # if self.debug:
                            # print(f"DEBUG get_controller_state(): grip_val: {controller_state[self._button_names[controller]["grip_val"]]}")
                            # print(f"DEBUG get_controller_state(): grasp_state: {self.grasp_states[self.active_robot][self.active_arm_index]}")
                            # print(controller_state[self._button_names[controller]["grip_val"]])
                        self.grasp_states[self.active_robot][self.active_arm_index] = not self.grasp_states[self.active_robot][self.active_arm_index] 
                        # self.hand_grasp = controller_state[self._button_names[controller]["grip_val"]]

                if controller_state[self._button_names[controller]["reset_bool"]]:
                    self._reset_state = 1
                    self._reset_pressed = True
                    new_controller_state[controller] = dict(
                        dpos=np.zeros(3),
                        rotation=np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]),
                        raw_drotation=np.array([0, 0, 0]),
                        grasp=[-1],
                        reset=self._reset_state,
                        base_mode=int(self.base_mode),
                    )
                    self.controller_state = self._nested_dict_update(self.controller_state, new_controller_state)
                    return self.controller_state[self._arm2controller[self.active_arm]]
                else:
                    self._reset_state = 0
                    self._reset_pressed = False

                # if buttons_data["RThU"] and self.debug:
                #     print("DEBUG: RThU")

                if self.trigger_pressed[controller]: # Teleop only works if the trigger is pressed
                    controller_curr_pos = np.array([controller_state["Position"]["x"], controller_state["Position"]["z"], -controller_state["Position"]["y"]]) 
                    controller_curr_rot = np.array([controller_state["Rotation"]["x"], controller_state["Rotation"]["y"], controller_state["Rotation"]["z"], controller_state["Rotation"]["w"]])
                    controller_curr_rot = T.quat2mat(controller_curr_rot)
                    controller_curr_rpy = T.mat2euler(controller_curr_rot)

                    # drift issue:
                    # how to convert rotation of joystick into the arm's worldframe?
                    robosuite_controller = robot.part_controllers[self.active_arm]

                    ee_current_pos = self.robot_interface.robots[0].recent_ee_pose[self.active_arm].last
                    # ee_current_quat = np.array([ee_current_pos[4], ee_current_pos[5], ee_current_pos[6], ee_current_pos[3]])
                    ee_curr_posit = np.array([ee_current_pos[0], ee_current_pos[1], ee_current_pos[2]])
                    ee_curr_rot = robosuite_controller.ref_ori_mat.copy()
                    # ee_curr_rpy = T.mat2euler(ee_curr_rot)

                    if self.initialize_pose:
                        # once trigger is (re)pressed, rebase controller's initial pose to current pose, and compute deltas using it
                        self.controller_init_pos[controller] = controller_curr_pos
                        self.controller_init_rot[controller] = controller_curr_rot
                        self.controller_init_rpy[controller] = controller_curr_rpy

                        self.ee_init_pos[controller] = ee_curr_posit
                        self.ee_init_rot[controller] = ee_curr_rot.copy()
                        # self.ee_init_rpy[controller] = ee_curr_rpy
                        if self.debug:
                            print("initialized pose")
                        self.initialize_pose = False

                    # Computing absolute action for the robot (in the robot's frame).

                    # r_controller_curr_from_init_rs = R_robosuit_from_quest @ r_controller_curr_from_init_quest
                    d_hand_posit = controller_curr_pos - self.controller_init_pos[controller] # difference in current hand position from previous
                    dR_controller = self.controller_init_rot[controller] @ controller_curr_rot.T
                    dR_controller_rs = self.R_rs_questRctrl @ dR_controller @ self.R_rs_questRctrl.T
                    dR_ee_rs = self.ee_init_rot[controller] @ ee_curr_rot.T
                    

                    # want the dR_ee to catch up to dR_controller
                    # ER_ee = dR_ee @ dR_controller_rs.T
                    ER_ee = dR_ee_rs @ dR_controller_rs.T
                    k, theta = tb.R2rot(ER_ee.T) # swapped ?? not sure why yet
                    k = np.array(k)
                    eR2 = -np.sin(theta/2)*k

                    k, theta = tb.R2rot(dR_controller_rs)
                    k = np.array(k)
                    eR2_controller = -np.sin(theta/2)*k

                    k, theta = tb.R2rot(dR_ee_rs)
                    k = np.array(k)
                    eR2_ee = -np.sin(theta/2)*k
                    # eR2_controller = self.R_rs_questRctrl @ eR2_controller 
                    # delta_rpy = eR2 
                    # d_hand_rpy = eR2

                    # d_hand_rpy[abs(d_hand_rpy) < np.pi/36] = 0.0
                    # d_hand_posit[abs(d_hand_posit) < 0.01] = 0.0

                    # d_hand_rpy = d_hand_rpy/10
                    if self.debug:
                        print("Initials")
                        print(f"DEBUG get_controller_state(): d_hand_pos vs controller_init_pos: {d_hand_posit} {self.controller_init_pos[controller]}")
                        # print(f"DEBUG get_controller_state(): d_hand_rpy: {d_hand_rpy}")
                        print(f"DEBUG get_controller_state(): eR2 vs eR2_controller vs eR2_ee: \n{eR2} \n{eR2_controller} \n {eR2_ee}")
                        # print(f"DEBUG get_controller_state(): controller_curr_rpy vs controller_init_rpy: {controller_curr_rpy} {self.controller_init_rpy[controller]}")
                    d_hand_rs_frame = self.R_rs_questwd @ d_hand_posit # I would hope that this turns it from the quest's frame to the robot's frame?
                    # d_hand_rot_rs_frame = self.R_rs_questwd @ ER_controller

                    target_posit = self.ee_init_pos[controller] + d_hand_rs_frame # desired change in position from the robot's initial position
                    # target_rot = self.ee_init_rot[controller] @ d_hand_rot_rs_frame  # desired change in orientation from the robot's initial orientation
                    # target_rpy = self.ee_init_rpy[controller] - d_hand_rpy

                    # target_rpy[target_rpy > 2 * np.pi] -= 2 * np.pi
                    # target_rpy[target_rpy < -2 * np.pi] += 2 * np.pi

                    # ER_EE_rot = ee_curr_rot @ self.ee_init_rot[controller].T
                    # k_ee, theta_ee = tb.R2rot(ER_EE_rot)
                    # k_ee = np.array(k_ee)
                    # eR2_ee = -np.sin(theta_ee/2)*k_ee
                    # d_ee_rpy = eR2_ee

                    if self.debug:
                        print("Targets")
                        print(f"DEBUG get_controller_state(): target_pos vs ee_init: {target_posit} {self.ee_init_pos[controller]}")
                        # print(f"DEBUG get_controller_state(): target_rpy vs ee_init_rpy: {target_rpy} {self.ee_init_rpy[controller]}")
                        print(f"DEBUG get_controller_state(): ee_curr_rot vs ee_init_rot: \n{ee_curr_rot} \n{self.ee_init_rot[controller]}")

                    # Computing delta action for the robot.
                    delta_pos = target_posit - ee_curr_posit # delta between desired EE pose and current EE pose
                    # delta_rpy = ee_curr_rpy - target_rpy # delta between desired EE orientation and current EE orientation
                    # delta_rot = ee_curr_rot @ target_rot.T
                    # k_ee, theta_ee = tb.R2rot(delta_rot)
                    # k_ee = np.array(k_ee)
                    # eR2_ee = -np.sin(theta_ee/2)*k_ee
                    # d_ee_w = eR2_ee

                    # delta_rpy[delta_rpy > 2 * np.pi] -= 2 * np.pi
                    # delta_rpy[delta_rpy < -2 * np.pi] += 2 * np.pi

                    # delta_rpy = -delta_rpy

                    if self.debug:
                        print("Deltas")
                        print(f"DEBUG get_controller_state(): delta_pos vs ee_curr: {delta_pos} {ee_curr_posit}")


                    self.engaged = True # TODO maybe remove, not useful
                    new_controller_state[controller] = dict(
                        dpos=delta_pos,
                        # dpos=np.zeros(3),
                        rotation=np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]),
                        raw_drotation=eR2,
                        # raw_drotation=np.zeros(3),
                        grasp=int(self.grasp),
                        reset=self._reset_state,
                        base_mode=int(self.base_mode),
                    )
                    # self.controller_init_rot[controller] = controller_curr_rot
                    # self.ee_init_rot[controller] = ee_curr_rpy
                else:
                    self.engaged = False # TODO maybe remove, not useful
                    # new_controller_state[controller]['engaged'] = self.engaged # TODO maybe remove, not useful

                    # self.ee_init_pos[controller] = self.robot_interface.robots[0]._hand_pos[self.active_arm]
                    # self.ee_init_rot[controller] = self.robot_interface.robots[0]._hand_pose[self.active_arm][:3, :3]
                    # self.ee_init_rpy[controller] = T.mat2euler(self.ee_init_rot[controller])
                    current_pos = self.robot_interface.robots[0].recent_ee_pose[self.active_arm].last
                    current_quat = np.array([current_pos[4], current_pos[5], current_pos[6], current_pos[3]])
                    self.ee_init_pos[controller] = np.array([current_pos[0], current_pos[1], current_pos[2]])
                    self.ee_init_rot[controller] = T.quat2mat(current_quat)
                    self.ee_init_rpy[controller] = T.mat2euler(self.ee_init_rot[controller])
                    # zero-ing delta actions; creates a minor gap b/w absolute and delta control
                    new_controller_state[controller] = dict(
                        dpos=np.zeros(3),
                        rotation=np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]),
                        raw_drotation=np.zeros(3),
                        grasp=int(self.grasp),
                        reset=self._reset_state,
                        base_mode=int(self.base_mode),
                    )

            self.controller_state = self._nested_dict_update(self.controller_state, new_controller_state)
            return self.controller_state[self._arm2controller[self.active_arm]]
        
    def _nested_dict_update(self, curr_dict, update_dict):
        for k, v in update_dict.items():
            if k in curr_dict and isinstance(v, dict) and isinstance(curr_dict[k], dict):
                curr_dict[k] = self._nested_dict_update(curr_dict[k], v)
            else:
                curr_dict[k] = v
        return curr_dict
    
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
        robot = self.env.robots[self.active_robot]
        active_arm = self.active_arm

        state = self.get_controller_state()
        # Note: Devices output rotation with x and z flipped to account for robots starting with gripper facing down
        #       Also note that the outputted rotation is an absolute rotation, while outputted dpos is delta pos
        #       Raw delta rotations from neutral user input is captured in raw_drotation (roll, pitch, yaw)
        dpos, rotation, raw_drotation, grasp, reset = (
            state["dpos"],
            state["rotation"],
            state["raw_drotation"],
            state["grasp"],
            state["reset"],
        )

        #### ensure that the rotation is in the robotsuite frame (+x in front, +y to the left, +z up) in robot's perspective!!!

        if mirror_actions:
            dpos[0] *= -1
            dpos[1] *= -1
            raw_drotation[0] *= -1
            raw_drotation[1] *= -1

        # If we're resetting, immediately return None
        if reset:
            return None

        # Get controller reference
        controller = robot.part_controllers[active_arm]
        gripper_dof = robot.gripper[active_arm].dof

        assert controller.name in ["OSC_POSE", "JOINT_POSITION"], "only supporting OSC_POSE and JOINT_POSITION for now"

        # # process raw device inputs
        # drotation = raw_drotation[[1, 0, 2]]
        # # Flip z
        # drotation[2] = -drotation[2]
        # drotation = self.R_rs_questwd @ raw_drotation
        drotation = raw_drotation

        # Scale rotation for teleoperation (tuned for OSC) -- gains tuned for each device
        dpos, drotation = self._postprocess_device_outputs(dpos, drotation)
        # map 0 to -1 (open) and map 1 to 1 (closed)
        grasp = 1 if grasp else -1

        ac_dict = {}
        # populate delta actions for the arms
        for arm in robot.arms:
            # OSC keys
            arm_action = self.get_arm_action(
                robot,
                arm,
                norm_delta=np.zeros(6),
            )
            ac_dict[f"{arm}_abs"] = arm_action["abs"]
            ac_dict[f"{arm}_delta"] = arm_action["delta"]
            ac_dict[f"{arm}_gripper"] = np.zeros(robot.gripper[arm].dof)

        if robot.is_mobile:
            base_mode = bool(state["base_mode"])
            if base_mode is True:
                arm_norm_delta = np.zeros(6)
                base_ac = np.array([dpos[0], dpos[1], drotation[2]])
                torso_ac = np.array([dpos[2]])
            else:
                arm_norm_delta = np.concatenate([dpos, drotation])
                base_ac = np.zeros(3)
                torso_ac = np.zeros(1)

            ac_dict["base"] = base_ac
            # ac_dict["torso"] = torso_ac
            ac_dict["base_mode"] = np.array([1 if base_mode is True else -1])
        else:
            arm_norm_delta = np.concatenate([dpos, drotation])

        # populate action dict items for arm and grippers
        arm_action = self.get_arm_action(
            robot,
            active_arm,
            norm_delta=arm_norm_delta,
        )
        ac_dict[f"{active_arm}_abs"] = arm_action["abs"]
        ac_dict[f"{active_arm}_delta"] = arm_action["delta"]
        ac_dict[f"{active_arm}_gripper"] = np.array([grasp] * gripper_dof)

        # clip actions between -1 and 1
        for (k, v) in ac_dict.items():
            if "abs" not in k:
                ac_dict[k] = np.clip(v, -1, 1)

        return ac_dict

    def set_robot_transform(self):
        """
        Uses robot current pose to initialize controller_state
        """

        for i_robot in range(len(self.robot_interface.robots)):
            # Get robot's current pose.
            # attempt to work for multiple arms
            for arm in self.robot_interface.robots[i_robot].arms:
                current_pos = self.robot_interface.robots[0].recent_ee_pose[arm].last
                current_quat = np.array([current_pos[4], current_pos[5], current_pos[6], current_pos[3]])

            
                # print(f"robot {i_robot} current_pos: {current_pos}")

                # Set eef pose to robot's current pose.
                controller_name = self._controller_names[i_robot]
                self.ee_init_pos[controller_name] = np.array([current_pos[0], current_pos[1], current_pos[2]])
                self.ee_init_rot[controller_name] = T.quat2mat(current_quat)
                self.ee_init_rpy[controller_name] = T.mat2euler(self.ee_init_rot[controller_name])
                if self.debug:
                    print("set_robot_transform(): current_pos: ", current_pos)
                # self.ee_init_rot[controller_name] = current_quat

                # When called in __init__(), set controller state to robot's current pose.
                if self.controller_state is None:
                    self.controller_state = dict([(name, None) for name in self._controller_names])
                if self.controller_state[controller_name] is None:
                    print("resetting controller states to robot pose.")
                    target_pose = current_pos
                    self.controller_state[controller_name] = dict(
                        dpos=[0,0,0],
                        rotation=np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]),
                        raw_drotation=np.array([0, 0, 0]),
                        grasp=int(self.grasp),
                        reset=self._reset_state,
                        base_mode=int(self.base_mode),
                    )
                # print(self.controller_state)

    def approx_R_rs_questwd(self):
        # hard-coding the headset offset for now
        self.R_rs_questwd = np.array([ # robot_T_headset
            [0, 0, -1],
            [-1, 0, 0],
            [0, 1, 0]
        ])
        
        # rotate around z-axis for -90 degrees
        self.R_rs_questRctrl = np.array([ # robot_T_right_controller
            [0, 1, 0],
            [-1, 0, 0],
            [0, 0, 1]
        ])