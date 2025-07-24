"""
Main IK task logic follows humanoid_apollo.py
Feedback interface is from mobile_kinova.py
"""

from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from dm_control.viewer import user_input
from loop_rate_limiters import RateLimiter

import mink

_HERE = Path(__file__).parent
_XML = _HERE / "rby1a" / "mujoco" / "model_act.xml"
# act model include the controller (pos for joint, vel for wheel

# fmt: off
joint_names = [
    # Base joints.
    "left_wheel", "right_wheel",
    # Arm joints.
    "torso_0", "torso_1", "torso_2", "torso_3", "torso_4", "torso_5",
    "right_arm_0", "right_arm_1", "right_arm_2", "right_arm_3", "right_arm_4", "right_arm_5", "right_arm_6",
    "gripper_finger_r1", "gripper_finger_r2",
    "left_arm_0",  "left_arm_1",  "left_arm_2",  "left_arm_3",  "left_arm_4",  "left_arm_5",  "left_arm_6",
    "gripper_finger_l1", "gripper_finger_l2",
    "head_0", "head_1",
]
actuator_names = [
    "left_wheel_act", "right_wheel_act",
    "link1_act", "link2_act", "link3_act", "link4_act", "link5_act", "link6_act",
    "right_arm_1_act", "right_arm_2_act", "right_arm_3_act", "right_arm_4_act", "right_arm_5_act", "right_arm_6_act", "right_arm_7_act",
    "left_arm_1_act", "left_arm_2_act", "left_arm_3_act", "left_arm_4_act", "left_arm_5_act", "left_arm_6_act", "left_arm_7_act",
    "head_0_act", "head_1_act",
    "right_finger_act", "left_finger_act",
]
# site name for control
hands = ["left_palm", "right_palm"] # site linked to end effector
# hands = ["left_target", "right_target"] # mocap site

@dataclass
class KeyCallback:
    fix_base: bool = False
    pause: bool = False
    data: mujoco.MjData = None

    def input_data(self, data):
        self.data = data

    def __call__(self, key: int) -> None:
        if key == user_input.KEY_ENTER:
            self.fix_base = not self.fix_base
        elif key == user_input.KEY_SPACE:
            self.pause = not self.pause

        elif key == 265:  # Up arrow
            data.mocap_pos[0, 2] += 0.01
        elif key == 264:  # Down arrow
            data.mocap_pos[0, 2] -= 0.01
        elif key == 263:  # Left arrow
            data.mocap_pos[0, 0] -= 0.01
        elif key == 262:  # Right arrow
            data.mocap_pos[0, 0] += 0.01
        elif key == 320:  # Numpad 0
            data.mocap_pos[0, 1] += 0.01
        elif key == 330:  # Numpad .
            data.mocap_pos[0, 1] -= 0.01
        elif key == 260:  # Insert
            data.mocap_quat[0] = rotate_quaternion(data.mocap_quat[0], [1, 0, 0], 10)
        elif key == 261:  # Home
            data.mocap_quat[0] = rotate_quaternion(data.mocap_quat[0], [1, 0, 0], -10)
        elif key == 268:  # Home
            data.mocap_quat[0] = rotate_quaternion(data.mocap_quat[0], [0, 1, 0], 10)
        elif key == 269:  # End
            data.mocap_quat[0] = rotate_quaternion(data.mocap_quat[0], [0, 1, 0], -10)
        elif key == 266:  # Page Up
            data.mocap_quat[0] = rotate_quaternion(data.mocap_quat[0], [0, 0, 1], 10)
        elif key == 267:  # Page Down
            data.mocap_quat[0] = rotate_quaternion(data.mocap_quat[0], [0, 0, 1], -10)
        else:
            print(key)


def rotate_quaternion(quat, axis, angle):
    """
    Rotate a quaternion by an angle around an axis
    """
    angle_rad = np.deg2rad(angle)
    axis = axis / np.linalg.norm(axis)
    q = pyq.Quaternion(quat)
    q = q * pyq.Quaternion(axis=axis, angle=angle_rad)
    return q.elements

# def key_callback_data(key, data):

# TODO: ************************
# 1. attach the end effector to the target
# 2. check why the robot is not following at all:
# target not set, or task setup not good, or weights are wrong.

if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())

    configuration = mink.Configuration(model)

    # fmt: on
    # dof_ids = np.array([model.joint(name).id for name in joint_names])
    # actuator_ids = np.array([model.actuator(name).id for name in actuator_names])

    #### Define all the IK related tasks ####
    # task for limiting the movement of base and torso
    # we prioritize the joint with less torque and intertia (for torso)
    tasks = [
        # Base task
        pelvis_orientation_task := mink.FrameTask(
            frame_name="base",
            frame_type="body",
            position_cost=0.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        ),
        # Main torso task
        torso_orientation_task := mink.FrameTask(
            frame_name="link_torso_5",
            frame_type="body",
            position_cost=0.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        ),
        posture_task := mink.PostureTask(model, cost=1e-1), # TODO: add different cost for different joints
        com_task := mink.ComTask(cost=1.0),
    ]

    # tracking the poses of both hands
    hands_tasks = [] 
    for hand in hands:
        hand_task = mink.FrameTask(
            frame_name=hand,
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
        )
        hands_tasks.append(hand_task)
    tasks.extend(hands_tasks)

    ### Define the constraints and limits ###
    # TODO: add self collision avoidance between hands and body
    # When move the base, mainly focus on the motion on xy plane, minimize the rotation.
    # posture_cost = np.zeros((model.nv,))
    # posture_cost[2] = 1e-3
    # posture_task = mink.PostureTask(model, cost=posture_cost)

    # immobile_base_cost = np.zeros((model.nv,))
    # immobile_base_cost[:2] = 100
    # immobile_base_cost[2] = 1e-3
    # damping_task = mink.DampingTask(model, immobile_base_cost)

    # tasks = [
    #     end_effector_task,
    #     posture_task,
    # ]

    limits = [
        mink.ConfigurationLimit(model),
    ]

    # IK settings.
    solver = "quadprog"
    pos_threshold = 1e-4
    ori_threshold = 1e-4
    max_iters = 20

    model = configuration.model 
    data = configuration.data # this create a link to the mink model data

    key_callback = KeyCallback()
    key_callback.input_data(data)

    left_mocap_id = model.body("left_target").mocapid[0]
    right_mocap_id = model.body("right_target").mocapid[0]
    com_mid = model.body("com_target").mocapid[0]


    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=True,
        show_right_ui=True,
        key_callback=key_callback,
    ) as viewer:
        
        # Set up the viewer and camera
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = 1  # Make mocap body visible
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = 1    # Show perturbation force
        viewer._pert.select = 1                                      # Enable selection
        viewer._pert.active = 1        
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Initialize to the home keyframe.
        # configuration.update_from_keyframe("teleop")
        posture_task.set_target_from_configuration(configuration)
        pelvis_orientation_task.set_target_from_configuration(configuration)
        torso_orientation_task.set_target_from_configuration(configuration)
        com_task.set_target(data.mocap_pos[com_mid]) # NOTE: make it changable

        # hands_tasks[0].set_target_from_configuration(configuration)
        # hands_tasks[1].set_target_from_configuration(configuration)
        
        # mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        # configuration.update(data.qpos)
        # posture_task.set_target_from_configuration(configuration)
        # mujoco.mj_forward(model, data)

        # # Initialize the mocap target at the end-effector site.
        # mink.move_mocap_to_frame(model, data, "pinch_site_target", "pinch_site", "site")

        rate = RateLimiter(frequency=200.0, warn=False)
        dt = rate.period
        t = 0.0
        while viewer.is_running():

            ## Dummy Mocap Data, adapt from flying_dual_arm_ur5e.py
            # print(f"EE body R: {data.body('EE_BODY_R')}")
            # print(f"EE body L: {data.body('EE_BODY_L')}")
            print(data.mocap_pos[left_mocap_id])

            pos_EE_R = data.body("EE_BODY_R").xpos
            pos_EE_L = data.body("EE_BODY_L").xpos

            data.mocap_pos[left_mocap_id][0] = pos_EE_L[0]
            data.mocap_pos[left_mocap_id][1] = pos_EE_L[1] + 0.1 * np.sin(2.0 * t)
            data.mocap_pos[left_mocap_id][2] = pos_EE_L[2]
            hands_tasks[0].set_target(mink.SE3.from_mocap_name(model, data, "left_target"))

            data.mocap_pos[right_mocap_id][0] = pos_EE_R[0]
            data.mocap_pos[right_mocap_id][1] = pos_EE_R[1] + 0.1 * np.sin(2.0 * t)
            data.mocap_pos[right_mocap_id][2] = pos_EE_R[2]
            hands_tasks[1].set_target(mink.SE3.from_mocap_name(model, data, "right_target"))

            # # Update task target.
            # T_wt = mink.SE3.from_mocap_name(model, data, "pinch_site_target")
            # # print(f"T_wt: {T_wt}")
            # end_effector_task.set_target(T_wt)

            # # Compute velocity and integrate into the next configuration.
            # for i in range(max_iters):
            #     if key_callback.fix_base:
            #         vel = mink.solve_ik(
            #             configuration, [*tasks, damping_task], rate.dt, solver, 1e-3
            #         )
            #     else:
            #         vel = mink.solve_ik(configuration, tasks, rate.dt, solver, 1e-3)
            #     configuration.integrate_inplace(vel, rate.dt)

            #     # Exit condition.
            #     pos_achieved = True
            #     ori_achieved = True
            #     err = end_effector_task.compute_error(configuration)
            #     pos_achieved &= bool(np.linalg.norm(err[:3]) <= pos_threshold)
            #     ori_achieved &= bool(np.linalg.norm(err[3:]) <= ori_threshold)
            #     if pos_achieved and ori_achieved:
            #         break

            # if not key_callback.pause:
            #     data.ctrl[actuator_ids] = configuration.q[dof_ids]
            #     mujoco.mj_step(model, data)
            # else:
            #     mujoco.mj_forward(model, data)

            # Compute velocity and integrate into the next configuration.
            vel = mink.solve_ik(
                configuration, tasks, rate.dt, solver, 1e-1, limits=limits
            )
            configuration.integrate_inplace(vel, rate.dt)

            # Visualize at fixed FPS.
            viewer.sync()
            rate.sleep()
            t += dt
