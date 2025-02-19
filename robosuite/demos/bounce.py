import numpy as np
import robosuite as suite
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import LemonObject, BounceballObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.transform_utils import convert_quat

from robosuite.controllers import load_composite_controller_config
import mujoco
import time
import matplotlib.pyplot as plt

class Bounce(ManipulationEnv):
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
        self.table_offset = np.array((0, 0, 0.8))

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

        # Make table surface more elastic
        # table_collision = mujoco_arena.table_body.find(".//geom[@name='table_collision']")
        # table_collision.set('solimp', '0.9 0.95 0.001')  # Make contact more elastic
        # table_collision.set('solref', '-1 0')  # Faster contact response
        # table_collision.set('condim', '3')  # Enable rotation in contact
        # table_collision.set('priority', '1')  # Higher priority for table contacts
        ## TODO: the table is being penetrated by the lemon
        # Initialize bounceball instead of lemon
        
        # self.lemon = LemonObject(
        #     name="lemon",
        # )

        self.ball = BounceballObject(
            name="bounceball", # has to match the model="bounceball" in the xml file
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
                x_range=[-0.03, 0.03],
                y_range=[-0.03, 0.03],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=1.0,  # 1 meter above the table
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
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:
            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

if __name__ == "__main__":
    # Create environment
    env = suite.make(
        env_name="Bounce",
        robots="DualKinova3",
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
        render_camera="frontview",
    )

    # Reset the environment
    env.reset()

    # Get model and data
    model = env.sim.model._model
    data = env.sim.data._data
    
    # Set smaller timestep for more accurate physics simulation
    model.opt.timestep = 0.0005  # 0.5ms timestep

    # Disable the physics of the robot by setting the mass to zero
    # for body_id in range(model.nbody):
    #     body_name = env.sim.model.body_id2name(body_id)
    #     body_is_robot = "robot" in body_name or "arm" in body_name or "gripper" in body_name or "finger" in body_name
    #     if body_is_robot:
    #         # Make the body static by setting its mass to zero
    #         model.body_mass[body_id] = 0
    #         # Set the body's inertia to zero
    #         model.body_inertia[body_id] = [0, 0, 0]

    # Lists to store time, force and position data
    times = []
    forces = []
    z_positions = []
    contact_object = 'bounceball_g0'
    simulation_time = 10
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Set initial camera parameters
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -45
        viewer.cam.lookat[:] = np.array([0.0, -0.25, 0.824])

        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_time:
            step_start = time.time()
            
            # Step the simulation
            data.ctrl[:] = 0  # Disable controller
            mujoco.mj_step(model, data)
            
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
            ball_body_id = env.sim.model.body_name2id('bounceball_main')
            z_positions.append(data.xpos[ball_body_id][2])
            times.append(data.time)
            forces.append(total_force)  # This now includes the spike
            
            # Viewer updates (unchanged)
            with viewer.lock():
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
            viewer.sync()
            
            # Maintain real-time simulation
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

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