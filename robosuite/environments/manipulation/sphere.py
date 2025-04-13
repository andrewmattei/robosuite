from collections import OrderedDict

import numpy as np
np.set_printoptions(precision=4, suppress=True)
import robosuite as suite
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import LemonObject, SphereObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.transform_utils import convert_quat

from robosuite.controllers import load_part_controller_config
# control_config = load_part_controller_config(default_controller="JOINT_POSITION") # this doesn't even work...
from robosuite.controllers import load_composite_controller_config
from robosuite.kinematics.pinocchio_ik import compute_ik

import mujoco
import time
import matplotlib.pyplot as plt

class Sphere(ManipulationEnv):
    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="agentview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        lite_physics=True,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,
    ):
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        # self.table_offset = np.array((0, 0, 0.7))  # made changes
        self.table_offset = np.array((0, 0, 0.5))  # made changes
        # Omron LD-60 Mobile Base setting
        self.init_torso_height = 0.342

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            lite_physics=lite_physics,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
        )

    def reward(self, action):
        """
        Placeholder reward function
        Args:
            action (np.array): Action to execute within the environment
        Returns:
            float: Reward from environment
        """
        # For now, return a constant reward of 0 since we're just observing
        return 0.0

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        # Initialize sphere instead of lemon
        
        # self.ball = LemonObject(
        #     name="lemon",
        # )

        self.ball = SphereObject(
            name="sphere", # has to match the model="sphere" in the xml file
        )
        

        # No need to modify collision properties as they're set in the object initialization

        # Create placement initializer
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.ball)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.ball,
                x_range=[-0.56, -0.56],
                y_range=[1.245, 1.245],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.4,  # 1 meter above the table
            )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.ball,
        )

    def _setup_references(self):
        """
        Sets up references to important components
        """
        super()._setup_references()

        # Additional object references from this env
        self.ball_body_id = self.sim.model.body_name2id(self.ball.root_body)

    def _setup_observables(self):
        """
        Sets up observables
        """
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            # define observables modality
            modality = "object"

            # ball-related observables
            @sensor(modality=modality)
            def ball_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.ball_body_id])

            @sensor(modality=modality)
            def ball_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.ball_body_id]), to="xyzw")

            sensors = [ball_pos, ball_quat]
            names = [s.__name__ for s in sensors]

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        # set the mobilebase joint torso height if it exists
        self.deterministic_reset = True
        active_robot = self.robots[0]
        if active_robot.robot_model._torso_joints is not None:
            # dont need this since it's in super.reset()
            # torso_name = active_robot.robot_model._torso_joints[0]
            # self.sim.data.qpos[self.sim.model.get_joint_qpos_addr(torso_name)] = self.init_torso_height
            # # also set the initial torso height in the robot model
            active_robot.init_torso_qpos = np.array([self.init_torso_height,])

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:
            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))
        else:
            # Deterministic reset -- set all objects to their specified positions
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                # new_pos = np.array([0.1, 1.0, 1.0])
                # new_quat = np.array([1, 0, 0, 0])  # Keep fixed orientation
                # self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([new_pos, new_quat]))
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

        super()._reset_internal()


    def _apply_gravity_compensation(self):
        """
        Computes the control needed to compensate for gravity to hold the arm in place.
        and applies it to the robot.
        """
        # Zero out accelerations for the full simulation (ensure the state is appropriate)
        # self.sim.data.qacc[:] = 0

        # Get the total number of degrees of freedom from the raw MuJoCo model
        # n_dof = self.sim.model._model.nv

        # # Preallocate a 2D column vector for the computed torques with shape (n_dof, 1)
        # gravity_torques_full = np.zeros((n_dof, 1), dtype=np.float64)

        # # Use the underlying raw model and data objects (successful but not needed)
        # mujoco.mj_rne(self.sim.model._model, self.sim.data._data, 0, gravity_torques_full)

        # For each robot, extract the relevant torques and assign them as control inputs
        for robot in self.robots:
            indices = robot._ref_joint_pos_indexes
            gravity_compensation = self.sim.data.qfrc_bias[indices]
            
            control_indices = robot._ref_arm_joint_actuator_indexes
            self.sim.data.ctrl[control_indices] = gravity_compensation

    def _jog_robot_to_pose(self, desired_arm_pos, desired_torso_height=0.342):
        """
        Jog the robot to a desired arm position and torso height.
        """
        active_robot = self.robots[0]

        # Preparing Input for the default_dual_kinova3 controller (HybridMobileBase)
        action_dict = {}
        for arm in active_robot.arms:
            # got the following syntex from demo_sensor_corruption.py
            if arm == "right":
                action_dict[arm] = desired_arm_pos[:7]
            if arm == "left":
                action_dict[arm] = desired_arm_pos[7:]
            action_dict[f"{arm}_gripper"] = np.zeros(active_robot.gripper[arm].dof)

        action_dict["torso"] = np.array([desired_torso_height,])
        action_dict["base"] = np.array([0.0, 0.0, 0.0])

        env_action = active_robot.create_action_vector(action_dict)
        return env_action

    def _ik_left_arm_to_sphere_tangent(self, sphere_center, sphere_radius):
        # Compute the left-most point (assuming positive x is right)
        target_pos = sphere_center + np.array([-sphere_radius, 0, 0])
        target_pos = sphere_center
        # Define a desired tangent orientation.
        # Here, we form a desired 4x4 homogeneous transformation.
        # (You should replace with the specific orientation your task requires.)
        lhand_id = self.sim.model.body_name2id('gripper0_left_eef')
        R_wd_lhand = self.sim.data.body_xmat[lhand_id].reshape(3, 3)
        R_desired = R_wd_lhand  # placeholder: identity rotation
        
        body_names = ['world', 'table', 'left_eef_target', 'right_eef_target', 'robot0_base', 'mobilebase0_base', 'mobilebase0_fixed_support', 'mobilebase0_support', 'robot0_vention_upper_body', 'robot0_right_arm_fixed_base_link', 'robot0_right_shoulder_link', 'robot0_right_HalfArm1_Link', 'robot0_right_HalfArm2_Link', 'robot0_right_forearm_link', 'robot0_right_SphericalWrist1_Link', 'robot0_right_SphericalWrist2_Link', 'robot0_right_Bracelet_Link', 'robot0_right_hand', 'gripper0_right_robotiq_85_adapter_link', 'gripper0_right_eef', 'gripper0_right_left_outer_knuckle', 'gripper0_right_left_inner_finger', 'gripper0_right_left_inner_knuckle', 'gripper0_right_right_outer_knuckle', 'gripper0_right_right_inner_finger', 'gripper0_right_right_inner_knuckle', 'robot0_left_arm_fixed_base_link', 'robot0_left_shoulder_link', 'robot0_left_HalfArm1_Link', 'robot0_left_HalfArm2_Link', 'robot0_left_forearm_link', 'robot0_left_SphericalWrist1_Link', 'robot0_left_SphericalWrist2_Link', 'robot0_left_Bracelet_Link', 'robot0_left_hand', 'gripper0_left_robotiq_85_adapter_link', 'gripper0_left_eef', 'gripper0_left_left_outer_knuckle', 'gripper0_left_left_inner_finger', 'gripper0_left_left_inner_knuckle', 'gripper0_left_right_outer_knuckle', 'gripper0_left_right_inner_finger', 'gripper0_left_right_inner_knuckle', 'mobilebase0_wheeled_base', 'sphere_main']
        for name in body_names:
            body_id = self.sim.model.body_name2id(name)
            p_wd = self.sim.data.body_xpos[body_id]
            R_wd = self.sim.data.body_xmat[body_id].reshape(3, 3)
            print(f"Body: {name}, Position: {p_wd}, Rotation:\n{R_wd}")

        T_wd_target = np.eye(4)
        T_wd_target[:3, :3] = R_desired
        T_wd_target[:3, 3] = target_pos
        
        lbase_id = self.sim.model.body_name2id('robot0_left_arm_fixed_base_link')
        p_wd_lbase = self.sim.data.body_xpos[lbase_id]
        R_wd_lbase = self.sim.data.body_xmat[lbase_id].reshape(3, 3)
        # print("Left base position:", p_wd_lbase)
        # print("Left base rotation:\n", R_wd_lbase)
        T_wd_lbase = np.eye(4)
        T_wd_lbase[:3, :3] = R_wd_lbase
        T_wd_lbase[:3, 3] = p_wd_lbase
        
        print("T_wd_lbase:\n", T_wd_lbase)
        print("T_wd_target:\n", T_wd_target)
        # to get R_lbase_target in lbase frame, we need to do the following:
        R_lbase_lhand_in_wd = R_wd_lbase.T @ R_wd_lhand
        R_lbase_lhand = R_wd_lbase.T @ R_lbase_lhand_in_wd @ R_wd_lbase
        print("R_lbase_lhand:\n", R_lbase_lhand)
        T_lbase_target = np.linalg.inv(T_wd_lbase) @ T_wd_target # original
        # T_lbase_target = np.linalg.inv(T_wd_target) @ T_wd_lbase # reverse
        print("T_lbase_target:\n", T_lbase_target)


        # Path to the robot's URDF (update with your actual URDF file)
        urdf_path = "robosuite/models/assets/robots/dual_kinova3/leonardo.urdf"
        # Name of the left end-effector frame (adjust as needed)
        # Initial configuration for the left arm (using full robot qpos; adjust joint indices)
        # q0 = self.robots[0].init_qpos.copy()
        lq0 = self.sim.data.qpos[self.robots[0]._ref_joint_pos_indexes[7:14]]

        # Compute IK for the left arm using Pinocchio.
        left_ee = "end_effector"
        q_sol = compute_ik(urdf_path, left_ee, T_lbase_target, lq0)
        
        desired_arm_pos = self.robots[0].init_qpos.copy()
        # Assuming left arm joints are indexed from 7 to 14.
        desired_arm_pos[7:14] = q_sol
        
        return desired_arm_pos
            
class TimeKeeper:
    def __init__(self, desired_freq=60):
        self.period = 1.0 / desired_freq
        self.last_time = time.perf_counter()
        self.time_accumulator = 0
        self.frame_count = 0
        self.start_time = self.last_time

    def should_step(self):
        current_time = time.perf_counter()
        frame_time = current_time - self.last_time
        self.last_time = current_time
        self.time_accumulator += frame_time
        return self.time_accumulator >= self.period

    def consume_step(self):
        self.time_accumulator -= self.period
        self.frame_count += 1

    def get_fps(self):
        elapsed = time.perf_counter() - self.start_time
        return self.frame_count / elapsed if elapsed > 0 else 0

if __name__ == "__main__":

    simulation_time = 10.0 # seconds
    env_step_size = 0.0001 # seconds
    horizon = int(simulation_time / env_step_size)
    # Create environment
    # note default controller is in "robosuite/controllers/config/robots/default_dualkinova3.json"
    # which uses JOINT_POSITION part_controller for both arm in the HYBRID_MOBILE_BASE type.
    env = suite.make(
        env_name="Sphere",
        robots="DualKinova3",
        # controller_configs=load_composite_controller_config(controller="BASIC"), 
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
        horizon=horizon,
    )

    # Reset the environment
    env.reset()

    active_robot = env.robots[0]

    # Get initial joint positions for both arms
    # ways to retrieve joint positions
    # right_arm_joints = env.sim.data.qpos[active_robot._ref_arm_joint_pos_indexes[:7]]  # First 7 joints for right arm
    # left_arm_joints = env.sim.data.qpos[active_robot._ref_arm_joint_pos_indexes[7:]]   # Next 7 joints for left arm
    
    desired_arm_positions = active_robot.init_qpos
    # desired_arm_positions[7:14] = [1.5707963267948966, -1.5707963267948966, 1.5707963267948966, -1.5707963267948966, 0.0, -0.5235987755982988, -1.5707963267948966]
    desired_torso_height = env.init_torso_height

    # Preparing Input for the default_dual_kinova3 controller (HybridMobileBase)
    action_dict = {}
    for arm in active_robot.arms:
        # got the following syntex from demo_sensor_corruption.py
        if arm == "right":
            action_dict[arm] = desired_arm_positions[:7]
        if arm == "left":
            action_dict[arm] = desired_arm_positions[7:]
        action_dict[f"{arm}_gripper"] = np.zeros(active_robot.gripper[arm].dof)

    action_dict["torso"] = np.array([desired_torso_height,])
    action_dict["base"] = np.array([0.0, 0.0, 0.0])

    env_action = active_robot.create_action_vector(action_dict)
    # assess action dimension
    # to inspect use
    # print(active_robot.composite_controller._action_split_indexes)


    # Get model and data
    model = env.sim.model._model
    data = env.sim.data._data
    
    # Set smaller timestep for more accurate physics simulation
    model.opt.timestep = env_step_size  # commented out to use the default timestep


    # Lists to store time, force and position data
    times = []
    forces = []
    z_positions = []
    contact_object = 'sphere_g0'

    # ball_body_id = env.sim.model.body_name2id('sphere_main')
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
        # Set initial camera parameters
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 120
        viewer.cam.elevation = -45
        viewer.cam.lookat[:] = np.array([0.0, -0.25, 0.824])

        time_keeper = TimeKeeper(desired_freq=1/model.opt.timestep)
        
        # Set the targeting pose to be just a bit front of the initial robot hand position
        left_hand_body_id = env.sim.model.body_name2id('robot0_left_hand')
        left_hand_pos = data.xpos[left_hand_body_id]

        ball_body_id = env.sim.model.body_name2id('sphere_main')
        p_wd_ball = env.sim.data.body_xpos[ball_body_id]
        R_wd_ball = env.sim.data.body_xmat[ball_body_id].reshape(3, 3)

        print("Ball position:", p_wd_ball)
        print("Ball rotation:\n", R_wd_ball)

        p_wd_lhand = env.sim.data.body_xpos[left_hand_body_id]
        
        print("Bool: ", left_hand_pos == p_wd_lhand)
        R_wd_lhand = env.sim.data.body_xmat[left_hand_body_id].reshape(3, 3)
        print("Left hand position:", p_wd_lhand)
        print("Left hand rotation:\n", R_wd_lhand)
        # sphere_center = data.xpos[env.sim.model.body_name2id('sphere_main')]
        sphere_radius = 0.05
        desired_joint = env._ik_left_arm_to_sphere_tangent(left_hand_pos, sphere_radius)
        
        while viewer.is_running() and not env.done and data.time < simulation_time:
            if time_keeper.should_step():
                # Simulation step
                
                # Record data and update viewer
                
                # data.ctrl[:] = 0  # Disable controller
                
                # env._apply_gravity_compensation()

                ####Controlling the ball ######
                # Apply a force to the ball
                
                # Step the simulation
                # env.sim.step()
                # mujoco.mj_step(model, data)

                # # jog both arm to zero configuration 
                # zeros_config = np.zeros(14)
                # env_action = env._jog_robot_to_pose(zeros_config, desired_torso_height)
                
                env_action = env._jog_robot_to_pose(desired_joint)
                # env_action = np.zeros_like(env_action)
                env.step(env_action)

                total_force = 0
                # Iterate over all detected contacts
                for i in range(data.ncon):
                    contact = data.contact[i]
                    # Check if contact involves the table and the object of interest
                    if ((contact.geom1 == env.sim.model.geom_name2id('table_collision') and 
                        contact.geom2 == env.sim.model.geom_name2id(contact_object)) or
                        (contact.geom2 == env.sim.model.geom_name2id('table_collision') and 
                        contact.geom1 == env.sim.model.geom_name2id(contact_object))):
                        
                        # Compute contact force (6D: 3D force + 3D torque)
                        force_vector = np.zeros(6)
                        mujoco.mj_contactForce(model, data, i, force_vector)
                        
                        # Extract normal force (first component in the contact frame)
                        normal_force = force_vector[0]
                        total_force += normal_force
                
                # Record positions, times, and forces
                ball_body_id = env.sim.model.body_name2id('sphere_main')
                z_positions.append(data.xpos[ball_body_id][2])
                times.append(data.time)
                forces.append(total_force)  # This now includes the spike
                
                # Viewer updates (unchanged)
                with viewer.lock():
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1

                viewer.sync()
                time_keeper.consume_step()
                
                # # Optional: Monitor performance
                # if time_keeper.frame_count % 60 == 0:
                #     print(f"Current FPS: {time_keeper.get_fps():.2f}")

    # # Create subplots for force and position
    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # # Plot forces
    # ax1.plot(times, forces)
    # ax1.set_xlabel('Time (s)')
    # ax1.set_ylabel('Impact Force (N)')
    # ax1.set_title('Ball-Table Impact Force over Time')
    # ax1.grid(True)
    
    # # Plot z position
    # ax2.plot(times, z_positions)
    # ax2.set_xlabel('Time (s)')
    # ax2.set_ylabel('Z Position (m)')
    # ax2.set_title('Ball Z Position over Time')
    # ax2.grid(True)
    
    # plt.tight_layout()
    # plt.show()
