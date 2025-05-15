from collections import OrderedDict

import numpy as np
import robosuite as suite
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.transform_utils import convert_quat

from robosuite.controllers import load_part_controller_config
# control_config = load_part_controller_config(default_controller="JOINT_POSITION") # this doesn't even work...
from robosuite.controllers import load_composite_controller_config

import mujoco
import time
import matplotlib.pyplot as plt

import pinocchio as pin
import pinocchio.casadi as cpin
import robosuite.demos.optimizing_gen3_arm as opt

class Kinova3ContactControl(ManipulationEnv):
    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types=None,
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
        self.table_offset = np.array((0, 0, 0.7))  # made changes

        # Omron LD-60 Mobile Base setting
        self.init_torso_height = 0.342

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        # cpin functions
        # import cpin model
        self.pin_model = None
        self.pin_data = None
        self.fk_fun = None
        self.pos_fun = None
        self.jac_fun = None
        self.M_fun = None
        self.C_fun = None
        self.G_fun = None
        self.pin_model, self.pin_data = opt.load_kinova_model()
        self.fk_fun, self.pos_fun, self.jac_fun, self.M_fun, self.C_fun, self.G_fun = opt.build_casadi_kinematics_dynamics(self.pin_model)

        self.init_jogging = True

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
    
    def step(self, action, ball_action=None):
        """
        step with the original step function and the ball_action
        """
        ret = super().step(action)

        return ret


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
        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # initialize objects of interest
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        self.cube = BoxObject(
            name="cube",
            size_min=[0.020, 0.020, 0.020],  # [0.015, 0.015, 0.015],
            size_max=[0.022, 0.022, 0.022],  # [0.018, 0.018, 0.018])
            rgba=[1, 0, 0, 1],
            material=redwood,
        )

        # Create placement initializer
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.cube)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.cube,
                x_range=[-0.03, 0.03],
                y_range=[-0.03, 0.03],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.cube,
        )

        print("Model loaded")


    def _setup_references(self):
        """
        Sets up references to important components
        """
        super()._setup_references()

         # Additional object references from this env
        self.cube_body_id = self.sim.model.body_name2id(self.cube.root_body)

    def _setup_observables(self):
        """
        Sets up observables
        """
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            # define observables modality
            modality = "object"

            # cube-related observables
            @sensor(modality=modality)
            def cube_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.cube_body_id])

            @sensor(modality=modality)
            def cube_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.cube_body_id]), to="xyzw")

            sensors = [cube_pos, cube_quat]

            arm_prefixes = self._get_arm_prefixes(self.robots[0], include_robot_name=False)
            full_prefixes = self._get_arm_prefixes(self.robots[0])

            # # gripper to cube position sensor; one for each arm
            # sensors += [
            #     self._get_obj_eef_sensor(full_pf, "cube_pos", f"{arm_pf}gripper_to_cube_pos", modality)
            #     for arm_pf, full_pf in zip(arm_prefixes, full_prefixes)
            # ]
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
        super()._reset_internal()

        # set the mobilebase joint torso height if it exists
        self.deterministic_reset = True
        active_robot = self.robots[0]
        if active_robot.robot_model._torso_joints is not None:
            # dont need this since it's in super.reset()
            # torso_name = active_robot.robot_model._torso_joints[0]
            # self.sim.data.qpos[self.sim.model.get_joint_qpos_addr(torso_name)] = self.init_torso_height
            # # also set the initial torso height in the robot model
            active_robot.init_torso_qpos = np.array([self.init_torso_height,])

        ## Reset gripper positions to initial values
        # for arm in active_robot.arms:
        #     gripper_idx = active_robot._ref_gripper_joint_pos_indexes[arm]
        #     init_gripper_pos = active_robot.gripper[arm].init_qpos
        #     self.sim.data.qpos[gripper_idx] = init_gripper_pos

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
                obj_pos = np.array([0,0,obj_pos[2]]) # remove the randomness in x,y of the ball
                obj_quat = np.array([1,0,0,0]) # remove the randomness in the orientation of the ball
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

    def _compute_mass_matrix(self):
        """Compute mass matrix using mj_fullM."""
        M = np.zeros((self.sim.model.nv, self.sim.model.nv))
        mujoco.mj_fullM(self.sim.model, M, self.sim.data.qM)
        return M
    

    def _ee_pose_in_base(self, robot):
        """
        Computes the end-effector pose in the base frame.
        Args:
            robot (Robot): The robot object
        Returns:
            np.ndarray: The end-effector pose in the base frame
        """
        # Get the end-effector position and orientation
        ee_body_id = self.sim.model.body_name2id(robot.robot_model.bodies[-1])
        p_wd_ee = self.sim.data.body_xpos[ee_body_id]
        R_wd_ee = self.sim.data.body_xmat[ee_body_id].reshape(3, 3)
        H_wd_ee = np.eye(4)
        H_wd_ee[:3, :3] = R_wd_ee
        H_wd_ee[:3, 3] = p_wd_ee

        # Get the base position and orientation
        p_wd_base = robot.base_pos
        R_wd_base = robot.base_ori
        H_wd_base = np.eye(4)
        H_wd_base[:3, :3] = R_wd_base
        H_wd_base[:3, 3] = p_wd_base
        
        return np.linalg.inv(H_wd_base) @ H_wd_ee


    def _apply_gravity_compensation(self):
        """
        Computes the control needed to compensate for gravity to hold the arm in place.
        and applies it to the robot.
        """
        
        action_dict = OrderedDict()
        # using pinocchio-casadi
        active_robot = self.robots[0]
        indices = active_robot._ref_joint_pos_indexes
        q_curr = self.sim.data.qpos[indices]
        qd_curr = self.sim.data.qvel[indices]
        G_curr = self.G_fun(q_curr).full().flatten()
        C_curr = self.C_fun(q_curr, qd_curr).full()
        # M_curr = self.M_fun(q_curr).full()
        # M_mujoco = self.sim.data.qM

        action_dict[active_robot.arms[0]] = G_curr + C_curr @ qd_curr

        # control_indices = robot._ref_arm_joint_actuator_indexes
        # pin_bias = G_curr + C_curr @ qd_curr
        # mujoco_bias = self.sim.data.qfrc_bias[indices]
        # self.sim.data.ctrl[:] = G_curr + C_curr @ qd_curr
        # self.sim.data.ctrl[control_indices] = self.sim.data.qfrc_bias[indices]
        
        # H_base_ee = self._ee_pose_in_base(robot)
        # H_base_ee_cpin = opt.forward_kinematics_homogeneous(q_curr, self.fk_fun).full()
        #debug
        # if self.sim.data.time > 0.5:
        #     print("G_curr", G_curr)
        #     print("C_curr", C_curr)
        #     pass
        return action_dict
    
    def inverse_dynamics(self, ddq_des):
        """
        Computes the inverse dynamics for the given joint accelerations.
        Args:
            ddq_des (np.ndarray): Desired joint accelerations
        Returns:
            np.ndarray: Joint torques
        """
        active_robot = self.robots[0]
        indices = active_robot._ref_joint_pos_indexes
        q_curr = self.sim.data.qpos[indices]
        qd_curr = self.sim.data.qvel[indices]
        G_curr = self.G_fun(q_curr).full().flatten()
        C_curr = self.C_fun(q_curr, qd_curr).full()
        M_curr = self.M_fun(q_curr).full()

        tau = M_curr @ ddq_des + C_curr @ qd_curr + G_curr
        return tau
    

    def pid_joint_jog(self, q_desired, Kp=80, Kd=30, tol = 0.001):
        """
        PID control for joint jogging.
        Args:
            q_desired (np.ndarray): Desired joint positions
            q_current (np.ndarray): Current joint positions
            qd_current (np.ndarray): Current joint velocities
            Kp (float): Proportional gain
            Kd (float): Derivative gain
        Returns:
            np.ndarray: Control action
        """
        active_robot = self.robots[0]
        indices = active_robot._ref_joint_pos_indexes
        q_curr = self.sim.data.qpos[indices]
        qd_curr = self.sim.data.qvel[indices]
        error = q_desired - q_curr
        ddq_des = Kp * error - Kd * qd_curr
        
        if np.linalg.norm(error) < tol:
            self.init_jogging = False
       
        if self.init_jogging:
            torque = self.inverse_dynamics(ddq_des)
            # print("pid error:", np.linalg.norm(error))
        else:
            torque = self.inverse_dynamics(np.zeros_like(ddq_des))
        return torque

            
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
    env_step_size = 0.001 # seconds
    horizon = int(simulation_time / env_step_size)
    # Create environment
    # note default controller is in "robosuite/controllers/config/robots/default_dualkinova3.json"
    # which uses JOINT_POSITION part_controller for both arm in the HYBRID_MOBILE_BASE type.
    env = suite.make(
        env_name="Kinova3ContactControl",
        robots="Kinova3SRL",
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

    desired_torso_height = env.init_torso_height


    # Get model and data
    model = env.sim.model._model
    data = env.sim.data._data
    
    # Set smaller timestep for more accurate physics simulation
    model.opt.timestep = env_step_size  # commented out to use the default timestep


    # Lists to store time, force and position data
    times = []
    forces = []
    z_positions = []
    contact_object = 'bounceball_g0'

    # ball_body_id = env.sim.model.body_name2id('bounceball_main')

    ### compute inverse kinematics for desired impact location 
    # example impact location
    p_f = np.array([0.4, 0.0, 0.1])    # meters
    v_f = np.array([0, 0, -0.5])   # m/s

    q_init = env.robots[0].init_qpos

    H_base_ee_f = opt.forward_kinematics_homogeneous(q_init, env.fk_fun).full()
    H_base_ee_f[0:3, 3] = p_f
    q_f = opt.inverse_kinematics_casadi(H_base_ee_f, env.fk_fun, q_init).full().flatten()

    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Set initial camera parameters
        # viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE  # visulizing site
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 120
        viewer.cam.elevation = -45
        viewer.cam.lookat[:] = np.array([0.0, -0.25, 0.824])
        viewer.opt.geomgroup = np.array([0,1,1,0,0,0]) # disabled the geom1 collision body visual

        time_keeper = TimeKeeper(desired_freq=1/model.opt.timestep)

        while viewer.is_running() and not env.done and data.time < simulation_time:
            if time_keeper.should_step():
                # Simulation step
                
                # Record data and update viewer
                
                # data.ctrl[:] = 0  # Disable controller
                
                torque = env.pid_joint_jog(q_f)
                env_action = np.clip(torque, -env.pin_model.effortLimit, env.pin_model.effortLimit)

                ####Controlling the ball ######
                # obtaining free_joint_pose
                # free_joint_pose = env.sim.data.get_joint_qpos("bounceball_free_joint")
                # Apply a force to the ball
                
                # Step the simulation
                env.sim.data.ctrl[:] = env_action
                env.sim.step()
                # mujoco.mj_step(model, data)
                # env.step(env_action)

                # total_force = 0
                # # Iterate over all detected contacts
                # for i in range(data.ncon):
                #     contact = data.contact[i]
                #     # Check if contact involves the table and the object of interest
                #     if ((contact.geom1 == env.sim.model.geom_name2id('table_collision') and 
                #         contact.geom2 == env.sim.model.geom_name2id(contact_object)) or
                #         (contact.geom2 == env.sim.model.geom_name2id('table_collision') and 
                #         contact.geom1 == env.sim.model.geom_name2id(contact_object))):
                        
                #         # Compute contact force (6D: 3D force + 3D torque)
                #         force_vector = np.zeros(6)
                #         mujoco.mj_contactForce(model, data, i, force_vector)
                        
                #         # Extract normal force (first component in the contact frame)
                #         normal_force = force_vector[0]
                #         total_force += normal_force
                
                # # Record positions, times, and forces
                # ball_body_id = env.sim.model.body_name2id('bounceball_main')
                # z_positions.append(data.xpos[ball_body_id][2])
                # times.append(data.time)
                # forces.append(total_force)  # This now includes the spike
                
                # Viewer updates (unchanged)
                with viewer.lock():
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1

                viewer.sync()
                time_keeper.consume_step()
                
                # # Optional: Monitor performance
                # if time_keeper.frame_count % 60 == 0:
                #     print(f"Current FPS: {time_keeper.get_fps():.2f}")

    # Create subplots for force and position
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot forces
    ax1.plot(times, forces)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Impact Force (N)')
    ax1.set_title('Ball-Table Impact Force over Time')
    ax1.grid(True)
    
    # Plot z position
    ax2.plot(times, z_positions)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Z Position (m)')
    ax2.set_title('Ball Z Position over Time')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
