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
import time, os
import matplotlib.pyplot as plt

import casadi as cs
import pinocchio as pin
import pinocchio.casadi as cpin
import robosuite.demos.optimizing_gen3_arm as opt
import robosuite.utils.tool_box_no_ros as tb

# Load reference trajectory
current_path = os.path.dirname(os.path.realpath(__file__))
traj_names = ["kinova_gen3_opt_trajectory_flex_pose.npy"]

##########################
#######ADJUST HERE########
play_traj = 0
##########################
opt_save_path = os.path.join(current_path, traj_names[play_traj])
opt_trajectory_data = np.load(opt_save_path, allow_pickle=True).item()

plot_save_path = os.path.join(current_path, 'plots', traj_names[play_traj].replace('.npy', '.png'))
# Create plots directory if it doesn't exist
if not os.path.exists(os.path.dirname(plot_save_path)):
    os.makedirs(os.path.dirname(plot_save_path))




class Kinova3ContactControl(ManipulationEnv):
    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        # gripper_types=None,
        gripper_types="ImpactBody",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=False,
        use_object_obs=False,
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
        self.table_offset = np.array((0, 0, 0.912))  # made changes

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
        self.jdot_pos_fun = opt.build_position_jacobian_derivative_function(self.jac_fun)
        self.jdot_fun = opt.build_jacobian_derivative_function_efficient(self.jac_fun)

        rev_lim = np.pi * 2
        self.q_lower   =  np.array([-rev_lim, -2.41, -rev_lim, -2.66, -rev_lim, -2.23, -rev_lim])
        self.q_upper   =  np.array([ rev_lim,  2.41,  rev_lim,  2.66,  rev_lim,  2.23,  rev_lim])
        self.dq_lower  =  -self.pin_model.velocityLimit
        self.dq_upper  =  self.pin_model.velocityLimit
        self.tau_lower =  -self.pin_model.effortLimit
        self.tau_upper =  self.pin_model.effortLimit

        self.init_jogging = True
        self.viewer = None

        self.v_f = None
        self.p_f = None
        self.v_p_mag = None
        self.R_f = None
        
        # for LQR controller
        self.reversed_traj = None
        self.linearization_cache = None
        self.reversed_linearization_cache = None

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

        self.gui_on = True

    def _configure_viewer(self):
        """Configure the viewer camera."""
        self.viewer.cam.distance = 3.0
        self.viewer.cam.azimuth = 120
        self.viewer.cam.elevation = -45
        self.viewer.cam.lookat[:] = np.array([0.0, -0.25, 0.824])
        self.viewer.opt.geomgroup = np.array([0,1,1,0,0,0]) # disabled the geom1 collision body visual

    

    def _configure_camera(self):
        self.cam.distance = 4.0
        self.cam.azimuth = 90
        self.cam.elevation = -90
        self.cam.lookat[:] = np.array([0.2, 0.2, 0.0])
        # self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED


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
        # if self.placement_initializer is not None:
        #     self.placement_initializer.reset()
        #     self.placement_initializer.add_objects(self.cube)
        # else:
        #     self.placement_initializer = UniformRandomSampler(
        #         name="ObjectSampler",
        #         mujoco_objects=self.cube,
        #         x_range=[-0.03, 0.03],
        #         y_range=[-0.03, 0.03],
        #         rotation=None,
        #         ensure_object_boundary_in_range=False,
        #         ensure_valid_placement=True,
        #         reference_pos=self.table_offset,
        #         z_offset=0.01,
        #     )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            # mujoco_objects=self.cube,
        )

        print("Model loaded")


    def _setup_references(self):
        """
        Sets up references to important components
        """
        super()._setup_references()

         # Additional object references from this env
        # self.cube_body_id = self.sim.model.body_name2id(self.cube.root_body)

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
        # if not self.deterministic_reset:
        #     # Sample from the placement initializer for all objects
        #     object_placements = self.placement_initializer.sample()

        #     # Loop through all objects and reset their positions
        #     for obj_pos, obj_quat, obj in object_placements.values():
        #         self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))
        # else:
        #     # Deterministic reset -- set all objects to their specified positions
        #     object_placements = self.placement_initializer.sample()

        #     # Loop through all objects and reset their positions
        #     for obj_pos, obj_quat, obj in object_placements.values():
        #         obj_pos = np.array([0,0,obj_pos[2]]) # remove the randomness in x,y of the ball
        #         obj_quat = np.array([1,0,0,0]) # remove the randomness in the orientation of the ball
        #         self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

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
    

    def _ee_twist_in_base(self, robot):
        """
        Computes the end-effector velocity in the base frame.
        Args:
            robot (Robot): The robot object
        Returns:
            np.ndarray: The end-effector velocity in the base frame
        """
        model = self.sim.model
        data = self.sim.data
        # Get the end-effector velocity
        site_id = model.site_name2id(robot.robot_model.sites[-1])
        # note that we don't know if the jacobian is in the base frame or not
        # Get end-effector velocity
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model._model, data._data, jacp, jacr, site_id)
        indices = robot._ref_joint_pos_indexes
        jacp = jacp[:, indices]
        jacr = jacr[:, indices]
        ee_vel = jacp @ data.qvel[indices]
        ee_omega = jacr @ data.qvel[indices]
        
        return ee_vel, ee_omega


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
    
    
    def optimal_trajectory_tracking_ctrl(self, current_time, T_opt, U_opt, Z_opt):
        """
        Compute torque commands to track a precomputed Gen3 trajectory.
        Uses feedforward torques from U_opt and simple PD feedback on joint error.
        """
        # 1. Interpolate feedforward torque and reference state
        tau_ref, z_ref = opt.match_trajectories(current_time, T_opt, U_opt, T_opt, Z_opt)
        tau_ref = tau_ref.flatten()
        
        # Split z_ref into positions and velocities
        n = tau_ref.size
        q_ref  = z_ref[:n].flatten()
        dq_ref = z_ref[n:].flatten()
        
        # 2. Read current joint state
        robot    = self.robots[0]
        idxs     = robot._ref_joint_pos_indexes
        q_curr   = self.sim.data.qpos[idxs]
        dq_curr  = self.sim.data.qvel[idxs]
        
        # 3. PD gains (tweak as needed)
        # Kp, Kd = 80.0, 5.0
        Kp, Kd = 80.0, 5.0  # for when tau_fb is used
        
        # 4. Compute feedback
        q_err  = q_ref  - q_curr
        dq_err = dq_ref - dq_curr
        ddq_des = Kp * q_err + Kd * dq_err
        tau_fb = self.inverse_dynamics(ddq_des)
        
        # 5. Total command
        tau = tau_fb
        
        # 6. Zero out after trajectory ends
        if current_time > T_opt[-1]:
            # print("Trajectory ended, zeroing out torques")
            tau = self.inverse_dynamics(np.zeros_like(ddq_des))
        
        return tau
    

    def lqr_tracking_controller(self, current_time, T_opt, U_opt, Z_opt, horizon=100, cache=None):
        """
        Real-time LQR controller that interpolates Z_opt and U_opt.
        Assumes self.linearization_cache holds (A_list, B_list).
        """
        # Make sure we've already called linearize_dynamics_along_trajectory
        if cache is None:
            cache = self.linearization_cache

        A_list, B_list = cache

        # 1) Interpolate feed-forward torque and reference state
        u_ff, z_ref = opt.match_trajectories(current_time, T_opt, U_opt, T_opt, Z_opt)

        # 2) Find index for selecting local linearization
        idx = np.searchsorted(T_opt, current_time)
        idx = min(max(idx, 0), len(A_list) - 1)

        # 3) Build cost weights (tune these as needed)
        state_dim   = Z_opt.shape[0]    # e.g. 14 for Gen3 (7 q + 7 dq)
        control_dim = U_opt.shape[0]    # e.g. 7 joints
        Q = np.eye(state_dim)
        Q[:control_dim,:control_dim] *= 1e4
        
        R = np.eye(control_dim) * 1e-5
        

        # 4) Backward Riccati recursion over a short finite horizon
        P = Q.copy()
        K_seq = []
        for k in reversed(range(horizon)):
            if k == horizon - 1:
                P = 10 * Q
            j = min(idx + k, len(A_list) - 1)
            A = A_list[j]
            B = B_list[j]
            K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
            P = Q + A.T @ P @ (A - B @ K)
            K_seq.insert(0, K)

        # 5) First feedback gain
        K0 = K_seq[0]

        # 6) Read current joint state
        robot   = self.robots[0]
        idxs    = robot._ref_joint_pos_indexes
        q_curr  = self.sim.data.qpos[idxs]
        dq_curr = self.sim.data.qvel[idxs]

        q_error_raw = q_curr - z_ref[:control_dim].flatten()
        q_error = (q_error_raw + np.pi) % (2 * np.pi) - np.pi
        dq_error = dq_curr - z_ref[control_dim:].flatten()
        z_error = np.concatenate((q_error, dq_error)).flatten()
        
        # Compute LQR command
        tau = u_ff.flatten() - K0 @ z_error

        # 8) Zero out after trajectory end
        if current_time > T_opt[-1]:
            tau = self.inverse_dynamics(np.zeros_like(dq_curr))

        return tau
    

    def post_contact_joint_ctrl(self, q_end, Kp=100, Kd=10):
        """
        Compute control torques for post-impact joint position control.
        
        Args:
            q_end: Target joint position [q1, q2, q3]
        Returns:
            tau: Control torques
        """
        data  = self.sim.data._data
        robot = self.robots[0]
        arm_indices = robot._ref_joint_pos_indexes

        q_curr = data.qpos[arm_indices]
        dq_curr = data.qvel[arm_indices]
        # Compute position error
        q_err = q_end - q_curr
        
        # PD control in joint space
        ddq_des = (Kp * q_err - 
                  Kd * dq_curr)
        
        # Compute required torques using inverse dynamics
        return self.inverse_dynamics(ddq_des)
    

    def lyapunov_ee_retreat_controller(self, v_f, d_up, R_des=None, current_time=0.0, 
                                 Kp=100.0, Kd=20.0, alpha=1.0,
                                 start_pos=None, max_time=5.0):
        """
        Lyapunov-based controller to move end-effector in opposite direction of v_f
        for a distance d_up with guaranteed convergence.
        
        Args:
            v_f: Final velocity vector (3,) - direction to move away from
            d_up: Distance to move in opposite direction (scalar)
            current_time: Current time for initialization
            Kp: Position gain (scalar or 3x3 matrix)
            Kd: Damping gain (scalar or 3x3 matrix)  
            alpha: Convergence rate parameter
            start_pos: Starting end-effector position (if None, uses current)
            max_time: Maximum time for the motion
            
        Returns:
            tau: Control torques (7,)
            finished: Boolean indicating if target reached
        """
        # Get current robot state
        robot = self.robots[0]
        indices = robot._ref_joint_pos_indexes
        q_curr = self.sim.data.qpos[indices]
        dq_curr = self.sim.data.qvel[indices]
        
        # Get current end-effector position and velocity
        H_base_ee = self._ee_pose_in_base(robot)
        ee_pos_curr = H_base_ee[:3, 3]
        R_ee_curr = H_base_ee[:3, :3]
        ee_vel_curr, ee_omega_curr = self._ee_twist_in_base(robot)
        
        # Initialize starting position if not provided
        if not hasattr(self, '_retreat_start_pos') or start_pos is not None:
            self._retreat_start_pos = start_pos if start_pos is not None else ee_pos_curr.copy()
            self._retreat_start_time = current_time

            # set target orientation
            if R_des is not None:
                self._R_des = R_des.copy()
            else:
                # Use current orientation if no desired orientation provided
                self._R_des = R_ee_curr.copy()
        
        # Calculate retreat direction (opposite to v_f)
        v_f_norm = np.linalg.norm(v_f)
        if v_f_norm < 1e-8:
            print("Warning: v_f magnitude too small, using default z-direction")
            retreat_direction = np.array([0, 0, 1])
        else:
            retreat_direction = -v_f / v_f_norm  # Opposite direction
        
        # Target position: start_pos + d_up * retreat_direction
        p_target = self._retreat_start_pos + d_up * retreat_direction
        
        # Position error in task space
        p_error = p_target - ee_pos_curr
        p_error_norm = np.linalg.norm(p_error)

        # Orientation error in task space
        ER = R_ee_curr @ self._R_des.T # Desired orientation relative to current
        k, theta = tb.R2rot(ER)  # Axis-angle representation
        k = np.array(k)
        rot_error = -np.sin(theta/2) * k  # Error vector for orientation
        # print(theta)
        print(ER[0, 0], ER[1, 1], ER[2, 2], ER[0, 1], ER[0, 2], ER[1, 2])
    
        # Check if target reached
        position_tolerance = 0.05  # 5mm
        velocity_tolerance = 0.01   # 1cm/s
        ee_vel_norm = np.linalg.norm(ee_vel_curr)

        # print(f"Retreat: p_error={p_error_norm:.4f}m, ee_vel={ee_vel_norm:.4f}m/s, target={p_target}, current={ee_pos_curr}")
        
        if p_error_norm < position_tolerance and ee_vel_norm < velocity_tolerance:
            # Target reached - apply zero torque
            print(f"Retreat target reached: error={p_error_norm:.4f}m, vel={ee_vel_norm:.4f}m/s")
            return self.inverse_dynamics(np.zeros_like(dq_curr)), True
        
        # Check timeout
        if current_time - self._retreat_start_time > max_time:
            print(f"Retreat timeout reached: error={p_error_norm:.4f}m")
            return self.inverse_dynamics(np.zeros_like(dq_curr)), True
        
        # Convert gains to matrices if scalars
        rad_mm_error_scale = 30.0
        if np.isscalar(Kp):
            Kp_pos = Kp * np.eye(3)
            Kp_orient = Kp * rad_mm_error_scale * np.eye(3)  # Lower gain for orientation
        else:
            Kp_pos = np.array(Kp)
            Kp_orient = np.array(Kp) * rad_mm_error_scale

        if np.isscalar(Kd):
            Kd_pos = Kd * np.eye(3)
            Kd_orient = Kd * rad_mm_error_scale * np.eye(3)  # Lower damping for orientation
        else:
            Kd_pos = np.array(Kd)
            Kd_orient = np.array(Kd) * rad_mm_error_scale
        
        # Lyapunov-based control law in task space
        # V = 0.5 * p_error^T * Kp * p_error + 0.5 * ee_vel^T * ee_vel
        # For V_dot < 0, we choose: ee_accel_des = Kp * p_error - Kd * ee_vel - alpha * p_error
        
        # Desired end-effector acceleration
        ee_accel_pos_des = Kp_pos @ p_error - Kd_pos @ ee_vel_curr - alpha * p_error
        ee_accel_ori_des = Kp_orient @ rot_error - Kd_orient @ ee_omega_curr - alpha * rot_error
        ee_accel_des = np.concatenate((ee_accel_pos_des, ee_accel_ori_des))

        # Get current Jacobian
        J6 = self.jac_fun(q_curr)
        
        # Compute Jacobian time derivative for acceleration mapping
        J_dot = self.jdot_fun(q_curr, dq_curr).full()

        # Map desired task-space acceleration to joint space
        # ee_accel = J * ddq + J_dot * dq
        # Solve for ddq: ddq = J_pinv @ (ee_accel_des - J_dot @ dq)
        
        # Use damped pseudo-inverse for numerical stability
        J_damped = cs.DM(opt.damping_pseudoinverse(J6)).full()
        
        ddq_des = J_damped @ (ee_accel_des - J_dot @ dq_curr)
        
        # Add null-space control to maintain joint limits and comfort
        # Null-space projection: N = I - J_pinv @ J
        N = np.eye(robot.dof) - J_damped @ J6.full()

        # Null-space objective: move towards comfortable joint configuration
        q_nominal = np.radians([0, 15, 180, -130, 0, -35, 90])  # Comfortable Gen3 pose
        q_diff = q_nominal - q_curr
        q_diff = (q_diff + np.pi) % (2 * np.pi) - np.pi  # Normalize to [-pi, pi]
        ddq_null = 10.0 * q_diff - 5.0 * dq_curr
        
        # Combine task-space and null-space control
        ddq_total = ddq_des + N @ ddq_null
        
        # Compute required torques using inverse dynamics
        tau = self.inverse_dynamics(ddq_total)

        # # Additional damping for stability
        # tau_damping = -2.0 * dq_curr
        # tau += tau_damping
        
        return tau, False
        
    

    def run_pid_jogging_simulation(self, q_desired, 
                             Kp=80, Kd=30, 
                             tol=0.001,
                             sim_dt=1e-3,
                             record_dt=1e-3,
                             max_time=50.0,
                             slow_factor=1.0):
        """
        Run PID joint jogging simulation until reaching desired joint positions.
        """
        model = self.sim.model._model
        data  = self.sim.data._data
        model.opt.timestep = sim_dt
        robot = self.robots[0]

        # Initialize viewer if GUI is on
        if self.gui_on:
            self.viewer = mujoco.viewer.launch_passive(model, data)
            self._configure_viewer()

        # Compute record steps
        record_steps = max(1, int(record_dt / sim_dt))
        step_count = 0

        # Storage for logs
        times = []
        q_pos = []
        q_vel = []
        ee_pos = []
        ee_vel = []
        tau_log = []
        contact_forces = []

        # Reset simulation time
        data.time = 0.0
        self.init_jogging = True  # Reset jogging flag

        # Run until reaching target or timeout
        while data.time < max_time:
            mujoco.mj_forward(model, data)

            if step_count % record_steps == 0:
                # Get current state
                indices = robot._ref_joint_pos_indexes


                # Compute control
                tau = self.pid_joint_jog(q_desired, Kp=Kp, Kd=Kd, tol=tol)
                
                # Clip torques
                tau = np.clip(tau, robot.torque_limits[0], robot.torque_limits[1])

                if self.init_jogging:
                    # Log data
                    q_pos.append(data.qpos[indices].copy())
                    q_vel.append(data.qvel[indices].copy())

                    # Get end-effector state
                    H_base_ee = self._ee_pose_in_base(robot)
                    ee_pos.append(H_base_ee[:3, 3].copy())

                    ee_v, ee_w = self._ee_twist_in_base(robot)
                    ee_vel.append(ee_v.copy())

                    # Store torques and time
                    tau_log.append(tau.copy())
                    times.append(data.time)
                
                # Apply control
                data.ctrl[:] = tau

                # GUI update
                if self.gui_on:
                    with self.viewer.lock():
                        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
                    self.viewer.sync()
                    time.sleep(record_dt * slow_factor)

            # Step simulation
            mujoco.mj_step(model, data)
            step_count += 1

        if self.gui_on:
            self.viewer.close()

        # Create target trajectory for plotting
        T_opt = np.array(times)
        ee_pos_opt = np.tile(ee_pos[-1], (len(times), 1))  # Use final EE pos as target
        ee_vel_opt = np.zeros_like(ee_pos_opt)  # Zero target velocity
        U_opt = np.zeros((robot.dof, len(times)))  # Zero target torques
        q_opt = np.tile(q_desired, (len(times), 1)).T  # Use desired joint pos as target
        dq_opt = np.zeros((robot.dof, len(times)))  # Zero target velocities
        Z_opt = np.vstack((q_opt, dq_opt))  # Combine positions and velocities

        return {
            'times':           np.array(times),
            'q_pos':           np.array(q_pos),
            'q_vel':           np.array(q_vel),
            'ee_pos':          np.array(ee_pos),
            'ee_vel':          np.array(ee_vel),
            'tau':             np.array(tau_log),
            'contact_forces':  np.array(contact_forces),
            'times_opt':       T_opt,
            'tau_opt':         U_opt,
            'Z_opt':          Z_opt,    
            'ee_pos_opt':      ee_pos_opt,
            'ee_vel_opt':      ee_vel_opt,
        }
    

    def run_simulation_offscreen(self, T_opt, U_opt, Z_opt,
                                slow_factor=1.0,
                                sim_dt=1e-4, 
                                record_dt=1e-3):
        """
        Headless rollout following (T_opt, U_opt, Z_opt).
        Logs at intervals of record_dt using sim_dt to compute record_steps.
        """
        model = self.sim.model._model
        data  = self.sim.data._data
        model.opt.timestep = sim_dt
        robot = self.robots[0]

        # Initialize viewer
        if self.gui_on:
            self.viewer = mujoco.viewer.launch_passive(self.sim.model._model, self.sim.data._data)
            self._configure_viewer()


        # compute how many sim steps between records
        record_steps = max(1, int(record_dt / sim_dt))
        step_count   = 0

        # prepare storage
        times, q_pos, q_vel, ee_pos, ee_vel, tau_log, contact_forces = [], [], [], [], [], [], []
        impact_time = None
        impact_pos = None
        impact_vel = None

        # precompute desired EE states
        num = len(T_opt)
        ee_pos_opt = np.zeros((num, 3))
        ee_vel_opt = np.zeros((num, 3))
        for k in range(num):
            qk  = Z_opt[:robot.dof, k]
            dqk = Z_opt[robot.dof:, k]
            ee_pos_opt[k] = self.pos_fun(qk).full().flatten()
            jac_pos = self.jac_fun(qk)[0:3, :]
            ee_vel_opt[k] = (jac_pos @ dqk).full().flatten()

        # reset sim to initial state
        arm_indices = robot._ref_joint_pos_indexes
        data.qpos[arm_indices] = Z_opt[:robot.dof, 0]
        data.qvel[arm_indices] = Z_opt[robot.dof:, 0]
        data.time    = 0.0

        # set phase: goes "tracking" -> "post_impact"
        phase = "tracking"
        # impact flag to enable recording at impact just once
        impact_flag = False
        # choose 0.80/1 point of the trajectory to be the q_end for post impact
        q_end = Z_opt[:robot.dof, int(len(T_opt) *0.80)]
        T_rev = self.reversed_traj['T']
        Z_rev = self.reversed_traj['Z']
        U_rev = self.reversed_traj['U']

        # run until just past final time
        while data.time < T_opt[-1] * 2.5:
            mujoco.mj_forward(model, data)

            total_force = 0.0
            for i in range(data.ncon):
                force = np.zeros(6)
                mujoco.mj_contactForce(model, data, i, force)
                total_force += np.linalg.norm(force[:3])

            # only record every record_steps
            if step_count % record_steps == 0:
                
                # reading at recording freq
                q_curr = data.qpos[arm_indices]
                dq_curr = data.qvel[arm_indices]

                # control
                if phase == "tracking":
                    if self.linearization_cache is not None:
                        tau = self.lqr_tracking_controller(
                            data.time, T_opt, U_opt, Z_opt)
                    else:
                        tau = self.optimal_trajectory_tracking_ctrl(
                            data.time, T_opt, U_opt, Z_opt)
                    if data.ncon > 0 and not impact_flag:
                        impact_flag = True
                        impact_time = times[-1]
                        impact_pos = ee_pos[-1]
                        impact_vel = ee_vel[-1]
                        if self.gui_on:
                            print(f"Impact detected at time {impact_time:.3f}s, position: {impact_pos}, velocity: {impact_vel}")
                        phase = "post_impact"

                elif phase == "post_impact":
                    ### method 1
                    # tau = self.post_contact_joint_ctrl(q_end)
                    # error = np.linalg.norm(data.qpos - q_end)
                    # if error < 0.04:
                    #     print(f"End-effector reached target position at time {data.time:.3f}s")
                    #     break
                    # reverse tracking the trajectory
                    ### method 2
                    # tau = self.lqr_tracking_controller(
                    #     data.time-impact_time,T_rev, U_rev, Z_rev, horizon=100,
                    #     cache=self.reversed_linearization_cache)
                    # # if data.time > impact_time + T_rev[-1]:
                    # q_err = Z_rev[:7, -1].flatten() - q_curr
                    # q_err = (q_err + np.pi) % (2 * np.pi) - np.pi
                    # q_err_norm = np.linalg.norm(q_err)
                    # print(f"Post-impact control: error = {q_err_norm:.4f}")
                    # if q_err_norm < 0.05:
                    #     print(f"Post-impact control completed at time {data.time:.3f}s")
                    #     break
                    ### method 3
                    tau, finished = self.lyapunov_ee_retreat_controller(
                        self.v_f, 0.2, R_des=self.R_f, current_time=data.time,
                        Kp=100.0, Kd=20, alpha=1.0)
                    if finished:
                        print(f"Post-impact retreat completed at time {data.time:.3f}s")
                        break

               
                # bound torques
                tau = np.clip(tau, self.tau_lower, self.tau_upper)

                if impact_time is None or data.time < impact_time + T_opt[-1]:
                    # joint log
                    q_pos.append(q_curr.copy())
                    q_vel.append(dq_curr.copy())

                    # end-effector position
                    H_base_ee = self._ee_pose_in_base(robot)
                    ee_pos.append(H_base_ee[:3, 3].copy())

                    # end-effector velocity via Jacobian
                    ee_v, ee_w = self._ee_twist_in_base(robot)
                    ee_vel.append(ee_v.copy())

                    # torques and summed contact force magnitude
                    tau_log.append(tau.copy())
                    contact_forces.append(total_force)
                    times.append(data.time)
                elif not self.gui_on:
                    break
                
                data.ctrl[:] = tau

            # step
            mujoco.mj_step(model, data)
            # model.opt.timestep = sim_dt
            step_count += 1
            if self.gui_on and step_count %record_steps == 0:
                with self.viewer.lock():
                    self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
                self.viewer.sync()
                time.sleep(record_dt * slow_factor)

        if self.gui_on:
            self.viewer.close()

        return {
            'times':           np.array(times),
            'q_pos':           np.array(q_pos),
            'q_vel':           np.array(q_vel),
            'ee_pos':          np.array(ee_pos),
            'ee_vel':          np.array(ee_vel),
            'tau':             np.array(tau_log),
            'contact_forces':  np.array(contact_forces),
            'times_opt':       T_opt,
            'tau_opt':         U_opt,
            'Z_opt':           Z_opt,
            'ee_pos_opt':      ee_pos_opt,
            'ee_vel_opt':      ee_vel_opt,
            'impact_time':     impact_time,
            'impact_pos':      impact_pos,
            'impact_vel':      impact_vel
        }
    

    def plot_joints(self, sim_data):
        """
        Plot individual joint positions and velocities in a 7x2 grid.
        Left column: Position (desired vs actual)
        Right column: Velocity (desired vs actual)
        """
        # Color definitions for joints
        colors = {
            'joint1': '#1f77b4',    # bright blue
            'joint2': '#ff7f0e',    # bright orange
            'joint3': '#2ca02c',    # bright green
            'joint4': '#d62728',    # bright red
            'joint5': '#9467bd',    # purple
            'joint6': '#8c564b',    # brown
            'joint7': '#e377c2',    # pink
        }

        # unpack
        t        = sim_data['times']
        q_pos    = sim_data['q_pos']
        q_vel    = sim_data['q_vel']
        T_opt    = sim_data['times_opt']
        Z_opt    = sim_data['Z_opt']
        dof      = q_pos.shape[1]

        # Create figure with 7x2 subplots
        fig, axes = plt.subplots(7, 2, figsize=(15, 20))
        plt.subplots_adjust(hspace=0.3)

        for j in range(dof):
            # Position trajectory for joint j
            q_des = np.interp(t, T_opt, Z_opt[j,:])
            axes[j,0].plot(t, q_pos[:,j], color=colors[f'joint{j+1}'], 
                        label='actual', linewidth=2)
            axes[j,0].plot(t, q_des, '--', color=colors[f'joint{j+1}'], 
                        label='desired', linewidth=2)
            axes[j,0].set_ylabel(f'J{j+1} Pos (rad)')
            axes[j,0].grid(True)
            if j == 0:  # Only add legend to first subplot
                axes[j,0].legend()
            
            # Add RMS tracking error to title
            rms_err = np.sqrt(np.mean((q_des - q_pos[:,j])**2))
            axes[j,0].set_title(f'Position (RMS error: {rms_err:.3f} rad)')

            # Velocity trajectory for joint j
            dq_des = np.interp(t, T_opt, Z_opt[j+dof,:])
            axes[j,1].plot(t, q_vel[:,j], color=colors[f'joint{j+1}'], 
                        label='actual', linewidth=2)
            axes[j,1].plot(t, dq_des, '--', color=colors[f'joint{j+1}'], 
                        label='desired', linewidth=2)
            axes[j,1].set_ylabel(f'J{j+1} Vel (rad/s)')
            axes[j,1].grid(True)
            if j == 0:  # Only add legend to first subplot
                axes[j,1].legend()
            
            # Add RMS tracking error to title
            rms_err = np.sqrt(np.mean((dq_des - q_vel[:,j])**2))
            axes[j,1].set_title(f'Velocity (RMS error: {rms_err:.3f} rad/s)')

        # Add common x-label
        for ax in axes[-1,:]:
            ax.set_xlabel('Time (s)')

        plt.suptitle('Joint Positions and Velocities', fontsize=16)
        plt.tight_layout()
        plt.show()

    
    def plot_joint_errors(self, sim_data):
        """
        Plot individual joint position and velocity errors in a 7x2 grid.
        Left column: Position errors
        Right column: Velocity errors
        """
        # Color definitions for joints
        colors = {
            'joint1': '#1f77b4',    # bright blue
            'joint2': '#ff7f0e',    # bright orange
            'joint3': '#2ca02c',    # bright green
            'joint4': '#d62728',    # bright red
            'joint5': '#9467bd',    # purple
            'joint6': '#8c564b',    # brown
            'joint7': '#e377c2',    # pink
        }

        # unpack
        t        = sim_data['times']
        q_pos    = sim_data['q_pos']
        q_vel    = sim_data['q_vel']
        T_opt    = sim_data['times_opt']
        dof      = q_pos.shape[1]

        # Create figure with 7x2 subplots
        fig, axes = plt.subplots(7, 2, figsize=(15, 20))
        plt.subplots_adjust(hspace=0.3)

        for j in range(dof):
            # Position error for joint j
            q_des = np.interp(t, T_opt, Z_opt[j,:])
            q_err = q_des - q_pos[:,j]
            axes[j,0].plot(t, q_err, color=colors[f'joint{j+1}'], linewidth=2)
            axes[j,0].set_ylabel(f'J{j+1} Pos Err (rad)')
            axes[j,0].grid(True)
            
            # Add RMS error to title
            rms_err = np.sqrt(np.mean(q_err**2))
            axes[j,0].set_title(f'Position Error (RMS: {rms_err:.3f} rad)')

            # Velocity error for joint j
            dq_des = np.interp(t, T_opt, Z_opt[j+dof,:])
            dq_err = dq_des - q_vel[:,j]
            axes[j,1].plot(t, dq_err, color=colors[f'joint{j+1}'], linewidth=2)
            axes[j,1].set_ylabel(f'J{j+1} Vel Err (rad/s)')
            axes[j,1].grid(True)
            
            # Add RMS error to title
            rms_err = np.sqrt(np.mean(dq_err**2))
            axes[j,1].set_title(f'Velocity Error (RMS: {rms_err:.3f} rad/s)')

        # Add common x-label
        for ax in axes[-1,:]:
            ax.set_xlabel('Time (s)')
        plt.tight_layout()
        # plt.suptitle('Joint-wise Position and Velocity Errors', fontsize=16)
        plt.show()


    def plot_results(self, sim_data, save_path=None):
        """
        Plot tracking results including joint errors and velocities.
        """
        # Color definitions
        colors = {
            'joint1': '#1f77b4',    # bright blue
            'joint2': '#ff7f0e',    # bright orange
            'joint3': '#2ca02c',    # bright green
            'joint4': '#d62728',    # bright red
            'joint5': '#9467bd',    # purple
            'joint6': '#8c564b',    # brown
            'joint7': '#e377c2',    # pink
            'actual': '#7f7f7f',    # gray
            'desired':'#bcbd22',    # olive
            'impact': '#9467bd',      # purple
            'target': '#8c564b',      # brown
        }

        # unpack
        t        = sim_data['times']
        q_pos   = sim_data['q_pos']
        q_vel   = sim_data['q_vel']
        ee_pos   = sim_data['ee_pos']
        ee_vel   = sim_data['ee_vel']
        tau      = sim_data['tau']
        T_opt    = sim_data['times_opt']
        ee_pos_o = sim_data['ee_pos_opt']
        ee_vel_o = sim_data['ee_vel_opt']
        U_opt    = sim_data['tau_opt']
        contact_forces = sim_data['contact_forces']
        impact_time = sim_data['impact_time']
        impact_vel = sim_data['impact_vel']


        dof = tau.shape[1]

        # Create figure with 7 subplots
        fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

        # 1) End-effector position error norm
        pos_interp = np.vstack([
            np.interp(t, T_opt, ee_pos_o[:,i]) for i in range(3)
        ]).T
        err = np.linalg.norm(ee_pos - pos_interp, axis=1)
        axes[0].plot(t, err, color=colors['actual'], label='EE pos error', linewidth=2)
        axes[0].set_ylabel('Error (m)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 2) End-effector velocity magnitude
        axes[1].plot(t, np.linalg.norm(ee_vel,axis=1), 
                    color=colors['actual'], label='actual', linewidth=2)
        axes[1].plot(T_opt, np.linalg.norm(ee_vel_o,axis=1),
                    '--', color=colors['desired'], label='desired', linewidth=2)
        # Add desired impact velocity line
        imp_vel_des = np.linalg.norm(ee_vel_o[-1])
        axes[1].axhline(y=imp_vel_des, color=colors['target'], linestyle='--', 
                    label=f'imp_vel_des={imp_vel_des:.2f}')
        if impact_time is not None:
            axes[1].axvline(x=T_opt[-1], color=colors['target'], 
                        linestyle='--', label='Desired Impact Time')
            axes[1].annotate(f"Impact vel: {np.linalg.norm(impact_vel):.4f} m/s", 
                        xy=(impact_time, np.linalg.norm(impact_vel)),
                        xytext=(impact_time, np.linalg.norm(impact_vel)+0.5),
                        arrowprops=dict(facecolor=colors['impact'], shrink=0.05))
        axes[1].set_ylabel('EE Speed (m/s)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # 3) Joint torques
        for j in range(dof):
            axes[2].plot(t, tau[:,j], color=colors[f'joint{j+1}'],
                        label=f'τ{j+1}', linewidth=2)
            axes[2].plot(T_opt, U_opt[j], '--', color=colors[f'joint{j+1}'],
                        label=f'τ{j+1} des', linewidth=2)
        axes[2].set_ylabel('Torque (Nm)')
        axes[2].legend(ncol=4, fontsize='small')
        axes[2].grid(True, alpha=0.3)

        # 4) contact forces
        axes[3].plot(t, contact_forces, color='red', label='Contact Force', linewidth=2)
        axes[3].set_ylabel('Force (N)')
        axes[3].grid(True, alpha=0.3)
        if impact_time is not None:
            axes[3].axvline(x=impact_time, color=colors['impact'], 
                        linestyle='--', label='Impact')
            peak_force = max(contact_forces)
            peak_time = t[np.argmax(contact_forces)]
            axes[3].annotate(f"Peak: {peak_force:.2f}N", 
                            xy=(peak_time, peak_force), 
                            xytext=(peak_time, peak_force+0.05),
                            arrowprops=dict(facecolor='black', shrink=0.05))
        axes[3].legend(frameon=True)

        # # 6) EE position components
        # for i, comp in enumerate(['x', 'y', 'z']):
        #     axes[3].plot(t, ee_pos[:,i], color=colors[f'joint{i+1}'],
        #                 label=f'p_{comp}', linewidth=2)
        #     axes[3].plot(t, pos_interp[:,i], '--', color=colors[f'joint{i+1}'],
        #                 label=f'p_{comp} des', linewidth=2)
        # axes[3].set_ylabel('EE Position (m)')
        # axes[3].legend(ncol=3)
        # axes[3].grid(True)

        # # 7) EE velocity components
        # for i, comp in enumerate(['x', 'y', 'z']):
        #     axes[4].plot(t, ee_vel[:,i], color=colors[f'joint{i+1}'],
        #                 label=f'v_{comp}', linewidth=2)
        #     axes[4].plot(T_opt, ee_vel_o[:,i], '--', color=colors[f'joint{i+1}'],
        #                 label=f'v_{comp} des', linewidth=2)
        # axes[4].set_ylabel('EE Velocity (m/s)')
        # axes[4].legend(ncol=3)
        # axes[4].grid(True)

        axes[-1].set_xlabel('Time (s)')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


            
if __name__ == "__main__":

    T_opt = opt_trajectory_data['T_opt']
    U_opt = opt_trajectory_data['U_opt']
    Z_opt = opt_trajectory_data['Z_opt']


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
    # env.gui_on = False
    active_robot = env.robots[0]


    ## generate an initialization trajectory
    # example optimization
    # p_ee_to_ball_buttom = 0.05 + 0.025
    # p_f = np.array([0.5, 0.0, p_ee_to_ball_buttom])    # meters
    # v_f = np.array([0, 0, -0.05])   # m/s
    # v_p_mag = 1.0 # m/s

    p_ee_to_ball_buttom = 0.05 + 0.025
    p_f = np.array([0.5, 0.05, p_ee_to_ball_buttom])    # meters
    v_f = np.array([0, 0, -0.4111])   # m/s
    v_p_mag = np.linalg.norm(v_f)+0.5
    q_init = active_robot.init_qpos
    target_pose = env.fk_fun(q_init).full()
    target_pose[:3, 3] = p_f
    q_sol = opt.inverse_kinematics_casadi(
        target_pose,
        env.fk_fun,
        q_init, env.q_lower, env.q_upper
        ).full().flatten()
    
    # obtain elbow pos function
    # elbow_link_name = "forearm_link"
    # _, elbow_pos_fun,_,_,_,_ = opt.build_casadi_kinematics_dynamics(env.pin_model, elbow_link_name)
    # # q_sol_el = opt.inverse_kinematics_casadi_elbow_above(
    #     target_pose,
    #     env.fk_fun,
    #     q_init, env.q_lower, env.q_upper,
    #     elbow_pos_fun=elbow_pos_fun
    # ).full().flatten()

    # TODO (DONE) fix inverse kinematics error 
    # TODO (DONE) change LQR gain to get better position tracking (0.05 is kinda large)
    # TODO (DONE) get the contact force and plotting ready for the full running pipeline?
    # TODO see if it's possible to sense collsion from joint discountinuity? for torque feedback. do later
    # TODO Figure out a control method to converge the tracking error since no feedforward - tune more gain or integral term added to iLQR?
    
    # back propagated trajectory data
    T = 1.0  # seconds
    N = 150  # number of points
    dt = T / (N)  # time step
    traj  = opt.back_propagate_traj_using_manip_ellipsoid(
        v_f, q_sol, env.fk_fun, env.jac_fun, N=N, dt=dt, v_p_mag=v_p_mag
    )

    env.v_f= v_f
    env.p_f = p_f
    env.v_p_mag = v_p_mag
    env.R_f = target_pose[:3, :3]  # desired end-effector orientation
    

    ## or using the max inertial direction
    # traj = opt.back_propagate_traj_using_max_inertial_direction(
    #     v_f, q_sol, opt.build_trace_grad_fun(env.M_fun), env.jac_fun,
    #     N=80, dt=0.02
    # )

    # add ff torques to the trajectory
    Z_init = traj['Z']
    T_init = traj['T']
    U_init = traj['U']

    env.linearization_cache = opt.linearize_dynamics_along_trajectory(
        T_init, U_init, Z_init, env.M_fun, env.C_fun, env.G_fun
    )

    traj_reversed = opt.back_trace_from_traj(traj, env.jac_fun, ratio = 0.6)
    env.reversed_linearization_cache = opt.linearize_dynamics_along_trajectory(
        traj_reversed['T'], traj_reversed['U'], traj_reversed['Z'],
        env.M_fun, env.C_fun, env.G_fun
    )
    env.reversed_traj = traj_reversed


    sim_data = env.run_simulation_offscreen(
                T_init,
                U_init,
                Z_init,
                slow_factor=1.0,
                sim_dt = 1e-3, record_dt=1e-3
    )


    # sim_data = env.run_simulation_offscreen(
    #             T_opt,
    #             U_opt,
    #             Z_opt,
    #             slow_factor=1.0)
    
    # sim_data = env.run_pid_jogging_simulation(
    #             q_desired=Z_opt[:active_robot.dof,-1],
    #             Kp=80,
    #             Kd=30,
    #             tol=0.001,
    #             sim_dt=1e-3,
    #             record_dt=1e-3,
    #             max_time=5.0,
    #             slow_factor=1.0)
    
    # Plot results
    env.plot_joints(sim_data)
    env.plot_results(sim_data, plot_save_path)