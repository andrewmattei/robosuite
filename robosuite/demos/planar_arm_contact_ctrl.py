import mujoco
from mujoco import viewer
import numpy as np
import time
import matplotlib.pyplot as plt
import os

import casadi as cs
from robosuite.demos.optimizing_max_jacobian import formulate_symbolic_dynamic_matrices, optimize_trajectory
from robosuite.demos.optimizing_max_jacobian import match_trajectories, compute_jacobian
import robosuite.demos.optimizing_max_jacobian as omj
import imageio, cv2

# Robot parameters (defined at the top of the file)
LINK_MASS = 0.1/3  # kg
LINK_LENGTH = 0.5  # m
LINK_RADIUS = LINK_LENGTH / 20  # m
INERTIA_XY = (1/12) * LINK_MASS * (LINK_LENGTH**2 + 3*(LINK_RADIUS**2))
INERTIA_Z = (1/2) * LINK_MASS * (LINK_RADIUS**2)  # Small radius cylinder assumption

# Load reference trajectory
current_path = os.path.dirname(os.path.realpath(__file__))
save_path = os.path.join(current_path, 'ball_trajectory.npy')
trajectory_data = np.load(save_path, allow_pickle=True).item()
REF_TIMES = trajectory_data['times']
REF_POSITIONS = trajectory_data['positions']
REF_VELOCITIES = trajectory_data['velocities']

NO_CONTROL = False

traj_names = ["opt_trajectory.npy", 
              "opt_trajectory_max_cartesian_accel.npy",
              "time_opt_trajectory_max_cartesian_accel.npy",
              "opt_trajectory_max_cartesian_accel_flex_pose.npy",
              "../results/2025-04-20_19-10-20/sol_729.npy"]
##########################
#######ADJUST HERE########
play_traj = 3
##########################
opt_save_path = os.path.join(current_path, traj_names[play_traj])
opt_trajectory_data = np.load(opt_save_path, allow_pickle=True).item()
T_opt = opt_trajectory_data['T_opt']
U_opt = opt_trajectory_data['U_opt']
Z_opt = opt_trajectory_data['Z_opt']

plot_save_path = os.path.join(current_path, 'plots', traj_names[play_traj].replace('.npy', '.png'))
# Create plots directory if it doesn't exist
if not os.path.exists(os.path.dirname(plot_save_path)):
    os.makedirs(os.path.dirname(plot_save_path))

# save to gif
record_gif = False
# TODO figure out how to render
if record_gif:
    gif_save_path = os.path.join(current_path, 'videos', traj_names[play_traj].replace('.npy', '.gif'))
    if not os.path.exists(os.path.dirname(gif_save_path)):
        os.makedirs(os.path.dirname(gif_save_path))

robot_xml = f"""
<mujoco model="3DOF_Planar_Robot">
  <compiler inertiafromgeom="false" angle="radian"/>
  <option gravity="0 0 -9.81"/>
  
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3" rgb2=".2 .3 .4" 
        width="512" height="512" mark="edge" markrgb=".2 .3 .4"/>
    <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance="0.2"/>
    <material name="BaseColor" rgba="0.2 0.5 0.8 1" />
    <material name="JointColor" rgba="0.8 0.3 0.3 1" />
    <material name="LimbColor" rgba="0.1 0.1 0.1 1" />
    <material name="TableColor" rgba="0.5 0.3 0.2 1" />
  </asset>
  
  <worldbody>
    <!-- Add table surface -->
    <!-- table surface is 0.05 below robot base-->
    <body name="table" pos="0 -0.075 0" euler="0 0 0">
      <geom name="table_surface" type="box" size="1.5 0.025 0.5" material="TableColor" />
    </body>

    <geom name="ground" type="plane" pos="0 0 -0.5" size="4 4 0.1" material="grid"/>
    <light pos="0 0 3" dir="0 0 -1" directional="true"/>
    
    <!-- Base -->
    <body name='body1' pos='0 0 0'>
      <geom name='base' type='cylinder' pos='0 0 0' material='BaseColor' size='{LINK_RADIUS*1.1} {LINK_RADIUS*1.1}'/>
      <joint name='joint1' type='hinge' stiffness='0' pos='0 0 0' axis='0 0 1' />
      
      <inertial pos='{LINK_LENGTH/2} 0 0' mass='{LINK_MASS}' diaginertia='{INERTIA_Z} {INERTIA_XY} {INERTIA_XY}'/>
      
      <geom type='capsule' fromto='0 0 0 {LINK_LENGTH} 0 0' material='LimbColor' size='{LINK_RADIUS}'/>

      <body name='body2' pos='{LINK_LENGTH} 0 0'>
        <joint name='joint2' type='hinge' stiffness='0' pos='0 0 0' axis='0 0 1'/>
        <inertial pos='{LINK_LENGTH/2} 0 0' mass='{LINK_MASS}' diaginertia='{INERTIA_Z} {INERTIA_XY} {INERTIA_XY}'/>
        <geom name='geom2' type='cylinder' pos='0 0 0' material='JointColor' size='{LINK_RADIUS*1.1} {LINK_RADIUS*1.1}'/>
        <geom type='capsule' fromto='0 0 0 {LINK_LENGTH} 0 0' material='LimbColor' size='{LINK_RADIUS}'/>
        
        <body name='body3' pos='{LINK_LENGTH} 0 0'>
          <joint name='joint3' type='hinge' stiffness='0' pos='0 0 0' axis='0 0 1'/>
          <inertial pos='{LINK_LENGTH/2} 0 0' mass='{LINK_MASS}' diaginertia='{INERTIA_Z} {INERTIA_XY} {INERTIA_XY}'/>
          <geom name='geom3' type='cylinder' pos='0 0 0' material='JointColor' size='{LINK_RADIUS*1.1} {LINK_RADIUS*1.1}'/>
          <geom type='capsule' fromto='0 0 0 {LINK_LENGTH} 0 0' material='LimbColor' size='{LINK_RADIUS}'/>
          <site name="site_end_effector" pos='{LINK_LENGTH} 0 0'/>
          <geom name="end_effector" type="sphere" pos="{LINK_LENGTH} 0 0" size="0.05" material="LimbColor" solref="0.01 0.1" condim="4" priority="1"/>
        </body>
      </body>
    </body>
  </worldbody>
  
  <actuator>
    <motor joint='joint1' name='motor_joint1' forcelimited='false'/>
    <motor joint='joint2' name='motor_joint2' forcelimited='false'/>
    <motor joint='joint3' name='motor_joint3' forcelimited='false'/>
  </actuator>
</mujoco>
"""

def get_state(model, data, nbatch=1):
  full_physics = mujoco.mjtState.mjSTATE_FULLPHYSICS
  state = np.zeros((mujoco.mj_stateSize(model, full_physics),))
  mujoco.mj_getState(model, data, state, full_physics)
  return np.tile(state, (nbatch, 1))


class RobotSystem:
    def __init__(self, model, data, gui=True, record_video=True, video_speed=0.2):
        self.model = model
        self.data = data
        self.nv = model.nv

        self.gui_on = gui
        
        # Initialize viewer
        if self.gui_on:
            self.viewer = viewer.launch_passive(model, data)
            self._configure_viewer()

        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "site_end_effector")
        
        # Control gains for trajectory tracking
        self.kp_pose = 200.0  # proportional gain for velocity tracking

        self.kp_joint = 5.0  # proportional gain for joint position tracking
        self.kd_joint = 0.5   # derivative gain for joint velocity tracking
        self.ki_joint = 5.0  # integral gain for joint position tracking
        self.error_buffer_size = 10  # buffer for integral error
        self.error_buffer = np.zeros((self.error_buffer_size, self.nv))
        self.buffer_index = 0  # Current position in circular buffer
        self.buffer_full = False  # Flag to track if buffer has been filled once

        self.kp_post = 300.0
        self.kd_post = 30.0

        #to fix a bug that if ee pass the y=0 line, calculation blow up
        self.u_prev = None

        # inertia mag
        self.M00_max = None

        self.m = [LINK_MASS, LINK_MASS, LINK_MASS]
        self.l = [LINK_LENGTH, LINK_LENGTH, LINK_LENGTH]
        self.r = [LINK_RADIUS, LINK_RADIUS, LINK_RADIUS]

        # video rendering 
        # Initialize offscreen renderer for video recording
        if record_video:
            # Create shared camera settings
            self.cam = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(self.cam)
            self._configure_camera()

            self.renderer = mujoco.Renderer(model)
            self._configure_camera()
            self.record_fps = 30
            self.video_speed = video_speed  # 0.1 means 10x slower than real-time
            self.frame_interval = (1.0 / self.record_fps) * self.video_speed
            self.last_frame_time = 0.0
            video_save_path = os.path.join(current_path, 'videos', 
                                         traj_names[play_traj].replace('.npy', '.mp4'))
            if not os.path.exists(os.path.dirname(video_save_path)):
                os.makedirs(os.path.dirname(video_save_path))
            self.video_writer = imageio.get_writer(video_save_path, fps=self.record_fps)
        else:
            self.renderer = None
            self.video_writer = None

        

    def _configure_viewer(self):
        """Configure the viewer camera."""
        self.viewer.cam.distance = 8.0
        self.viewer.cam.azimuth = 90
        self.viewer.cam.elevation = -90
        self.viewer.cam.lookat[:] = np.array([0.5, 0.0, 0.0])

        # Configure light to match camera position
        # Get light ID (assuming there's only one light)
        light_id = 0
        # Position light relative to camera
        self.model.light_pos[light_id] = np.array([0.5, 4.0, 4.0])
        # Set light direction
        self.model.light_dir[light_id] = np.array([0.0, -0.1, -0.9])
        # Make sure light is directional
        self.model.light_directional[light_id] = 1
    

    def _configure_camera(self):
        self.cam.distance = 4.0
        self.cam.azimuth = 90
        self.cam.elevation = -90
        self.cam.lookat[:] = np.array([0.2, 0.2, 0.0])
        # self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED

        

    def _compute_mass_matrix(self):
        """Compute mass matrix using mj_fullM."""
        M = np.zeros((self.nv, self.nv))
        mujoco.mj_fullM(self.model, M, self.data.qM)
        return M

    def inverse_dynamics(self, ddq_des):
        """Compute required torques for desired acceleration."""
        M = self._compute_mass_matrix()
        bias = self.data.qfrc_bias.copy()
        tau = M.dot(ddq_des) + bias
        return tau

    def get_reference_state(self, current_time):
        """Get reference position and velocity at current time"""
        # Find the closest time index
        idx = np.searchsorted(REF_TIMES, current_time)
        if idx >= len(REF_TIMES):
            idx = -1
        
        return REF_POSITIONS[idx], REF_VELOCITIES[idx]
    

    def consistent_sign_svd(self, u_new, u_old):
        """Ensure that the sign of the SVD vectors are consistent"""
        if u_old is not None and u_old.ndim == 2:
            # u_old is the principal axes of the manipulability ellipsoid
            for i in range(u_new.shape[1]):
                if np.dot(u_new[:, i], u_old[:, i]) < 0:
                    u_new[:, i] *= -1
        elif u_old is not None and u_old.ndim == 1:
            # u_old is the direction vector of increasing inertia
            # choose u_new that is closest to u_old
            u_new = -u_new * np.sign(np.dot(u_new[:,0], u_old))
        else:
            # on the first iter, ensure that v point counterclockwise
            max_manip_dir = u_new[:, 0]
            r_ori_ee = self.data.site_xpos[self.ee_site_id].copy()
            init_dir = np.cross(max_manip_dir, r_ori_ee)
            u_new = -np.sign(init_dir[2]) * u_new  # ensure that v points counterclockwise
        return u_new
    

    def update_error_buffer(self, error):
        """
        Update error buffer using circular buffer pattern.
        
        Args:
            error: Current error vector (shape: nv,)
        Returns:
            Weighted sum of integrated error
        """
        # Add new error to buffer
        self.error_buffer[self.buffer_index] = error
        
        # Update buffer index
        self.buffer_index = (self.buffer_index + 1) % self.error_buffer_size
        self.buffer_full |= (self.buffer_index == 0)
        
        # Calculate integral term with temporal weighting
        if self.buffer_full:
            # Use exponential weighting for older errors
            weights = np.exp(-np.arange(self.error_buffer_size)[::-1] * 0.1)
            weights /= np.sum(weights)  # Normalize weights
            integral_error = np.sum(self.error_buffer * weights[:, None], axis=0)
        else:
            # Use only filled portion of buffer
            filled_size = self.buffer_index
            weights = np.exp(-np.arange(filled_size)[::-1] * 0.1)
            weights /= np.sum(weights)
            integral_error = np.sum(self.error_buffer[:filled_size] * weights[:, None], axis=0)
            
        return integral_error
    

    def should_record_frame(self, current_time):
        """Check if enough time has passed to record next frame"""
        return current_time >= self.last_frame_time + self.frame_interval

    def render_frame(self, text=None):
        """Render current frame with playback speed indicator"""
        if not self.video_writer or not self.renderer:
            return

        current_time = self.data.time
        if self.should_record_frame(current_time):
            self.renderer.update_scene(self.data, camera=self.cam)
            img = self.renderer.render()
            
            if text is not None:
                font = cv2.FONT_HERSHEY_SIMPLEX
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
                # Add main text at top left
                cv2.putText(img, text, (10, 30), font, 0.7, (255, 255, 255), 2)
                
                # Add speed text at bottom right
                speed_text = f"{self.video_speed:.1f}x"
                text_size = cv2.getTextSize(speed_text, font, 0.7, 2)[0]
                text_x = img.shape[1] - text_size[0] - 10
                text_y = img.shape[0] - 10
                cv2.putText(img, speed_text, (text_x, text_y), 
                           font, 0.7, (255, 255, 255), 2)
                
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            self.video_writer.append_data(img)
            self.last_frame_time = current_time



    def plot_results(self, sim_data, track_traj=True, save_path=None):
        """
        Plot simulation results.
        
        Args:
            sim_data: Dictionary containing simulation data
            track_traj: Boolean indicating if tracking optimal trajectory
            save_path: Path to save the plot (optional)
        """
        times = sim_data['times']
        ee_positions = sim_data['ee_pos']
        ee_velocities = sim_data['ee_vel']
        applied_torques = sim_data['tau']
        contact_forces = sim_data['contact_forces']
        impact_time = sim_data.get('impact_time', None)
        impact_vel = sim_data.get('impact_vel', None)

        if track_traj:
            T_opt = sim_data['times_opt']
            U_opt = sim_data['tau_opt']
            ee_pos_opt = sim_data['ee_pos_opt']
            ee_vel_opt = sim_data['ee_vel_opt']
            ee_vx = ee_vel_opt[:, 0]
            ee_vy = ee_vel_opt[:, 1]
            fig, axes = plt.subplots(6, 1, figsize=(8, 14))
        else:
            fig, axes = plt.subplots(4, 1, figsize=(6, 10))
        
        # Color definitions
        colors = {
            'actual': '#1f77b4',      # bright blue
            'desired': '#ff7f0e',     # bright orange
            'error_x': '#2ca02c',     # bright green
            'error_y': '#d62728',     # bright red
            'impact': '#9467bd',      # purple
            'target': '#8c564b',      # brown
            'joint1': '#e377c2',      # pink
            'joint2': '#7f7f7f',      # gray
            'joint3': '#bcbd22'       # olive
        }

        # Contact Force plot (new)
        axes[0].plot(times, contact_forces, color='red', label='Contact Force', linewidth=2)
        # axes[0].set_title('Contact Force (y-direction)', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Force (N)')
        axes[0].grid(True, alpha=0.3)
        if impact_time is not None:
            axes[0].axvline(x=impact_time, color=colors['impact'], 
                        linestyle='--', label='Impact')
            peak_force = max(contact_forces)
            peak_time = times[np.argmax(contact_forces)]
            axes[0].annotate(f"Peak: {peak_force:.2f}N", 
                            xy=(peak_time, peak_force), 
                            xytext=(peak_time, peak_force+0.05),
                            arrowprops=dict(facecolor='black', shrink=0.05))
        axes[0].legend(frameon=True)


        # Position error plot
        ee_pos_opt_dense = omj.match_trajectories(times, T_opt, ee_pos_opt.T)[0]
        ee_pos_err = ee_positions[:,:2] - ee_pos_opt_dense.T
        axes[1].plot(times, ee_pos_err[:, 0], color=colors['error_x'], 
                    label='x', linewidth=2)
        axes[1].plot(times, ee_pos_err[:, 1], color=colors['error_y'], 
                    label='y', linewidth=2)
        # axes[1].set_title('End-Effector Position Error', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Position Error (m)')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(frameon=True)

        # Velocity magnitude plot
        ee_vel_mag = np.linalg.norm(ee_velocities, axis=1)
        axes[2].plot(times, ee_vel_mag, color=colors['actual'], 
                    label='Actual', linewidth=2)
        if track_traj:
            ee_vel_opt_mag = np.linalg.norm(ee_vel_opt, axis=1)
            axes[2].plot(T_opt, ee_vel_opt_mag, '--', color=colors['desired'], 
                        label='Desired', linewidth=2)
            axes[2].set_ylabel('Velocity Magnitude (m/s)')
            
            # Add desired impact velocity line
            imp_vel_des = np.linalg.norm(ee_vel_opt[-1])
            axes[2].axhline(y=imp_vel_des, color=colors['target'], linestyle='--', 
                        label=f'imp_vel_des={imp_vel_des:.2f}')
            
            if impact_time is not None:
                axes[2].axvline(x=impact_time, color=colors['impact'], 
                            linestyle='--', label='Impact')
                axes[2].annotate(f"Impact vel: {np.linalg.norm(impact_vel):.4f} m/s", 
                            xy=(impact_time, np.linalg.norm(impact_vel)),
                            xytext=(impact_time, np.linalg.norm(impact_vel)+0.5),
                            arrowprops=dict(facecolor=colors['impact'], shrink=0.05))

        # Torque plot
        for i, color in enumerate([colors['joint1'], colors['joint2'], colors['joint3']]):
            axes[3].plot(times, applied_torques[:, i], color=color, 
                        label=f'Joint {i+1}', linewidth=2)
            axes[3].set_ylabel('Torque (Nm)')
            if track_traj:
                axes[3].plot(T_opt, U_opt[i, :], '--', color=color, alpha=0.5)

        if track_traj:
            # Velocity components plots
            axes[4].plot(times, ee_velocities[:, 0], color=colors['actual'], 
                        label='Actual', linewidth=2)
            axes[4].plot(T_opt, ee_vx, '--', color=colors['desired'], 
                        label='Optimal', linewidth=2)
            axes[4].set_ylabel('Velocity X (m/s)')
            
            axes[5].plot(times, ee_velocities[:, 1], color=colors['actual'], 
                        label='Actual', linewidth=2)
            axes[5].plot(T_opt, ee_vy, '--', color=colors['desired'], 
                        label='Optimal', linewidth=2)
            axes[5].set_ylabel('Velocity Y (m/s)')

        # Update all subplot properties
        for ax in axes:
            ax.grid(True, alpha=0.3)
            ax.legend(frameon=True)
            # ax.set_xlabel('Time (s)')
            ax.tick_params(labelsize=10)
        axes[-1].set_xlabel('Time (s)')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to: {save_path}")
        plt.show()
    
    
    def max_manip_vel_tracking_ctrl(self, current_time):
        """
        Compute control torques for trajectory tracking
        The strategy here is to not use the desired position.
        Instead, we take the desired velocity and 
        aim it towards the direction of maximum manipulability at that pose of the arm.
        """
        
        # Get reference state
        pos_ref, vel_ref = self.get_reference_state(current_time)
        
        # Obtain the jacobian of the end effector
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "site_end_effector")
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, site_id)
        # use svd to get the principal axes of the manipulability ellipsoid
        U, S, V = np.linalg.svd(jacp)
        U = self.consistent_sign_svd(U, self.u_prev)
        self.u_prev = U
        # get the direction of maximum manipulability
        max_manip_dir = U[:, 0]
        
        # Compute desired acceleration purely based on desired velocity
        vel_des = vel_ref * max_manip_dir
        vel_ee_current = jacp @ self.data.qvel
        acc_des = self.kp_pose * (vel_des - vel_ee_current)
        lambda_sq = 0.01  # damping factor squared (tune as needed)
        J_inv_dls = jacp.T @ np.linalg.inv(jacp @ jacp.T + lambda_sq * np.eye(jacp.shape[0]))
        ddq_des = J_inv_dls @ acc_des
        
        # Compute required torques using inverse dynamics
        tau = self.inverse_dynamics(ddq_des)
        return tau, max_manip_dir
    

    def max_inertia_vel_tracking_ctrl(self, current_time, dM00_dq_fun):
        """
        Compute control torques for trajectory tracking"
        Use Casadi to compute the direction of increasing inertia
        """
        
        # Get reference state
        pos_ref, vel_ref = self.get_reference_state(current_time)

        # Obtain the jacobian of the end effector
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "site_end_effector")
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, site_id)

        # obtain the increasing inertia direction
        dM00_dq_val = dM00_dq_fun(self.data.qpos)
        q_increasing_inertia_dir = np.array(dM00_dq_val).flatten()
        u_increasing_inertia_dir = jacp @ q_increasing_inertia_dir
        u_increasing_inertia_dir = u_increasing_inertia_dir / np.linalg.norm(u_increasing_inertia_dir)

        vel_des = -vel_ref * u_increasing_inertia_dir # negative sign to move towards increasing inertia
        vel_ee_current = jacp @ self.data.qvel
        acc_des = self.kp_pose * (vel_des - vel_ee_current)
        lambda_sq = 0.01  # damping factor squared (tune as needed)
        J_inv_dls = jacp.T @ np.linalg.inv(jacp @ jacp.T + lambda_sq * np.eye(jacp.shape[0]))
        ddq_des = J_inv_dls @ acc_des

        # Compute required torques using inverse dynamics
        tau = self.inverse_dynamics(ddq_des.flatten())
        return tau, u_increasing_inertia_dir
    

    def optimal_trajectory_tracking_ctrl(self, current_time, T_opt, U_opt, Z_opt):
        """
        Compute control torques for trajectory tracking
        The strategy here is to use the desired position and velocity
        to compute the desired acceleration.
        """
        
        # Get reference state
        [tau_ref, z_ref] = match_trajectories(current_time, T_opt, U_opt, T_opt, Z_opt)
        q_ref = z_ref[:3].flatten()
        dq_ref = z_ref[3:].flatten()
        tau_ref = tau_ref.flatten()
        
        
        # tau = tau_ref # TODO pure feedforward from opt based is not working due to lack of step size
        # TODO P on dq better than P on both q and dq
        q_err = q_ref - self.data.qpos
        dq_err = dq_ref - self.data.qvel
        integral_err = self.update_error_buffer(q_err)

        # PID control
        tau = (tau_ref +
                self.kp_joint * q_err +
                self.ki_joint * integral_err +
                self.kd_joint * dq_err)

        # ddq_feedback = K_q * q_err + K_dq * dq_err
        # tau_feedback = self.inverse_dynamics(ddq_feedback)
        # tau = tau_ref + tau_feedback
        
        # TODO meth1: inverse dynamics pd control
        # TODO meth2: optimization extension on the tail 0.01(same vel, panerate the table to where ever)
        # TODO meth3: replace the feedback portion with LQR / MPC like control
        # TODO meth4: use robust control from Umich
        # TODO meth0: tune the gains
        # no control (for now) if current time is greater than the trajectory time
        if current_time > T_opt[-1]:
            ## no control
            tau = np.zeros(self.model.nv)
            
        return tau
    

    def lqr_tracking_controller(self, current_time, T_opt, U_opt, Z_opt, horizon=50):
        """
        Real-time LQR controller that interpolates Z_opt and U_opt.
        Assumes precomputed A_list, B_list with matching indices.
        """
        if self.linearization_cache is None:
            raise RuntimeError("Call linearize_dynamics_along_trajectory before using this controller.")

        A_list, B_list = self.linearization_cache

        # Interpolate reference at current time
        [u_ff, z_ref] = match_trajectories(current_time, T_opt, U_opt, T_opt, Z_opt)

        # Find closest index in T_opt (for selecting linearization)
        idx = np.searchsorted(T_opt, current_time)
        idx = min(max(idx, 0), len(A_list) - 1)

        # LQR backward recursion (short horizon)
        # Q = np.diag([1e-2, 1e-2, 1e-2, 1e1, 1e1, 1e1])
        # Q = np.diag([1e2, 1e2, 1e2, 1e0, 1e0, 1e0])
        Q = np.eye(6)
        R = np.eye(3) * 1e-5
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

        K0 = K_seq[0]
        current_z = np.hstack((self.data.qpos, self.data.qvel))
        tau = u_ff.flatten() - K0 @ (current_z - z_ref.flatten())

        if current_time > T_opt[-1]:
            ## no control
            tau = np.zeros(self.model.nv)

        return tau
    

    def post_contact_pose_ctrl(self, p_end):
        """
        Compute control torques for post-impact position control.
        
        Args:
            p_end: Target end-effector position [x, y]
            ee_pos: Current end-effector position
            ee_vel: Current end-effector velocity
            jacp: Current end-effector Jacobian
        Returns:
            tau: Control torques
        """
        # Obtain the jacobian of the end effector
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "site_end_effector")
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, site_id)
        lambda_sq = 0.01  # damping factor squared (tune as needed)
        J_inv_dls = jacp.T @ np.linalg.inv(jacp @ jacp.T + lambda_sq * np.eye(jacp.shape[0]))
        
        # Compute position error (only x,y components)
        ee_pos = self.data.site_xpos[site_id].copy()
        # padding the z component if needed
        if len(p_end) == 2:
            p_end = np.array([p_end[0], p_end[1], ee_pos[2]])
        pos_err = p_end - ee_pos

        # compute ee_vel
        ee_vel = jacp @ self.data.qvel
        
        # PD control in operational space
        acc_des = (self.kp_post * pos_err - 
                  self.kd_post * ee_vel)
        
        
        # Convert to joint accelerations
        ddq_des = J_inv_dls @ acc_des
        
        # Compute required torques using inverse dynamics
        tau = self.inverse_dynamics(ddq_des)
        
        # Clip torques to limits
        # tau = np.clip(tau, self.model.actuator_ctrlrange[:, 0], 
        #              self.model.actuator_ctrlrange[:, 1])
        tau = np.clip(tau, -1, 1)
        
        return tau
    

    def post_contact_joint_ctrl(self, q_end):
        """
        Compute control torques for post-impact joint position control.
        
        Args:
            q_end: Target joint position [q1, q2, q3]
        Returns:
            tau: Control torques
        """
        # Compute position error
        q_err = q_end - self.data.qpos
        
        # PD control in joint space
        ddq_des = (self.kp_post * q_err - 
                  self.kd_post * self.data.qvel)
        
        # Compute required torques using inverse dynamics
        tau = self.inverse_dynamics(ddq_des)
        
        # Clip torques to limits
        # tau = np.clip(tau, self.model.actuator_ctrlrange[:, 0], 
        #              self.model.actuator_ctrlrange[:, 1])
        tau = np.clip(tau, -1, 1)
        
        return tau
        

    def run_simulation_offscreen(self, solution, sim_dt = 5e-5):
        """
        Run the simulation headlessly using the optimal trajectory.
        Returns a dict of logged data suitable for saving to HDF5 or .mat.
        """
        model = self.model
        data = self.data

        num_steps = solution['q'].shape[1]
        T = solution['t_f']
        Z_opt = np.vstack((solution['q'], solution['dq']))
        # U_opt = np.hstack((solution['tau'], np.zeros((3,1)))) # !!bad shouldn't pad with zero
        U_opt = np.hstack((solution['tau'], solution['tau'][:, -1:]))  # pad with last value
        T_opt = np.linspace(0, T, num_steps)
        # compute the lqr linearization list
        self.linearization_cache = omj.linearize_dynamics_along_trajectory(
            T_opt, U_opt, Z_opt, self.m, self.l, self.r)

        # rollout the trajectories in p and v of the end effector
        ee_pos_opt = np.zeros((num_steps, 2))
        ee_vel_opt = np.zeros((num_steps, 2))
        ee_vel_mag_opt = np.zeros(num_steps)

        ee_vel = np.zeros
        for k in range(num_steps):
            q_opt = Z_opt[:3, k]
            dq_opt = Z_opt[3:, k]
            ee_pos_opt[k, :] = omj.end_effector_position(q_opt, self.l).full().flatten() # convert to numpy array
            jacp_opt = omj.compute_jacobian(q_opt, self.l)
            ee_vel_opt[k, :] = cs.DM(jacp_opt @ dq_opt).full().flatten()
            ee_vel_mag_opt[k] = np.linalg.norm(ee_vel_opt[k, :])


        # Set initial conditions
        data.qpos[:] = Z_opt[:3, 0]
        data.qvel[:] = Z_opt[3:, 0]
        data.time = 0.0
        self.error_buffer = np.zeros((self.error_buffer_size, self.nv))
        self.buffer_index = 0  # Current position in circular buffer
        self.buffer_full = False  # Flag to track if buffer has been filled once


        model.opt.timestep = sim_dt
        sim_duration = T_opt[-1]

        times, q_log, dq_log, tau_log = [], [], [], []
        ee_pos_log, ee_vel_log, xdd_log = [], [], []

        impact_flag = False

        impact_time = None
        impact_pos = None
        impact_vel = None

        self.u_prev = None

        while not impact_flag and data.time < sim_duration*1.5:
            mujoco.mj_forward(model, data)

            site_id = self.ee_site_id
            ee_pos = data.site_xpos[site_id].copy()
            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
            ee_vel = jacp @ data.qvel

            # tau = self.optimal_trajectory_tracking_ctrl(data.time, T_opt, U_opt, Z_opt)
            tau = self.lqr_tracking_controller(data.time, T_opt, U_opt, Z_opt)

            tau = np.asarray(tau).flatten()

            # log impact
            if data.ncon > 0 and not impact_flag:
                impact_flag = True
                impact_time = times[-1]
                impact_pos = ee_pos_log[-1]
                impact_vel = ee_vel_log[-1]

            # Record logs
            times.append(data.time)
            q_log.append(data.qpos.copy())
            dq_log.append(data.qvel.copy())
            tau_log.append(tau.copy())
            ee_pos_log.append(ee_pos[:2])  # only X-Y plane
            ee_vel_log.append(ee_vel[:2])

            data.ctrl[:] = tau
            mujoco.mj_step(model, data)
            if self.gui_on:
                self.viewer.sync()
                time.sleep(model.opt.timestep * 5)

        q_log = np.array(q_log)
        dq_log = np.array(dq_log)
        tau_log = np.array(tau_log)
        ee_pos_log = np.array(ee_pos_log)
        ee_vel_log = np.array(ee_vel_log)
        xdd_log = np.array(xdd_log)
        times = np.array(times)

        if self.gui_on:
            self.viewer.close()

        return {
            'times': times,
            'q': q_log,
            'dq': dq_log,
            'tau': tau_log,
            'ee_pos': ee_pos_log,
            'ee_vel': ee_vel_log,
            'q_opt': Z_opt[:3, :],
            'dq_opt': Z_opt[3:, :],
            'tau_opt': U_opt,
            'times_opt': T_opt,
            'ee_pos_opt': ee_pos_opt,
            'ee_vel_opt': ee_vel_opt,
            'ee_vel_mag_opt': ee_vel_mag_opt,
            'final_pose': Z_opt[:3, -1],
            'final_position': ee_pos_opt[-1],
            'final_velocity': ee_vel_opt[-1],
            'impact_time': impact_time,
            'impact_pos': impact_pos,
            'impact_vel': impact_vel,
        }

        


def main():
    if NO_CONTROL:
        track_traj = False
    else:
        track_traj = True
    # Initialize simulation
    model = mujoco.MjModel.from_xml_string(robot_xml)
    data = mujoco.MjData(model)
    robot = RobotSystem(model, data)

    # Setup video recording


    
    # Data recording
    times = []
    joint_positions = []
    joint_velocities = []
    applied_torques = []
    ee_positions = []
    ee_velocities = []
    jaco_pos = []

    motion_vector = []
    accel_vector = []
    contact_forces = []

    impact_time = None
    impact_pos = None
    impact_vel = None

    # CasaDi formulations
    m = [LINK_MASS, LINK_MASS, LINK_MASS]
    l = [LINK_LENGTH, LINK_LENGTH, LINK_LENGTH]
    r = [LINK_RADIUS, LINK_RADIUS, LINK_RADIUS]
    q_sym = cs.SX.sym('q', 3)
    dq_sym = cs.SX.sym('dq', 3)
    tau_sym = cs.SX.sym('tau', 3)
    M, C, M_fun, C_fun = formulate_symbolic_dynamic_matrices(m, l, r, q_sym, dq_sym)
    dM00_dq = cs.jacobian(M[0, 0], q_sym)
    dM00_dq_fun = cs.Function('dM00_dq', [q_sym], [dM00_dq])
    J_p3 = compute_jacobian(q_sym, l)
    J_p3_fun = cs.Function('J_p3', [q_sym], [J_p3])

    # compute cartesian acceleration expression
    xdd_sym = omj.compute_symbolic_cartesian_acceleration(M, C, J_p3, q_sym, dq_sym, tau_sym)
    xdd_fun = cs.Function('xdd', [q_sym, dq_sym, tau_sym], [xdd_sym])

    # compute maximum acceleration direction
    xdd_mag_sq = cs.dot(xdd_sym, xdd_sym)
    dxdd_mag_dq = cs.jacobian(xdd_mag_sq, q_sym)
    dxdd_mag_dq_fun = cs.Function('dxdd_dq', [q_sym, dq_sym, tau_sym], [dxdd_mag_dq])

    # maximum base joint inertia magnitude
    M_max = M_fun(np.zeros(3))
    robot.M00_max = M_max[0, 0]

    impact_flag = False

    if track_traj:
        # Set initial joint positions based on reference trajectory
        q_start = Z_opt[:3, 0]
        dq_start = Z_opt[3:, 0]
    else:
        q_start = np.array([np.pi*0.5, -np.pi*0.5, -np.pi*0.5])
        dq_start = np.zeros(model.nv)
    data.qpos[:] = q_start
    data.qvel[:] = dq_start
    
    # Desired joint accelerations (example: trying to hold position)
    # ddq_des = np.zeros(model.nv)


     # Simulation parameters
    if track_traj:
        simulation_time = T_opt[-1]
        # have different time step for simulation and recording
        sim_dt = 0.00005  # simulation timestep: 0.05ms (20kHz)
        record_dt = 0.001  # recording timestep: 1ms (1kHz)
        record_steps = int(record_dt / sim_dt)  # record every N simulation steps
        model.opt.timestep = sim_dt
        phase = "optimal"
        ee_vel_opt = np.zeros((len(T_opt), 2))
        ee_pos_opt = np.zeros((len(T_opt), 2))
        ee_vx = []
        ee_vy = []
        for k in range(len(T_opt)):
            q_opt = Z_opt[:3, k]
            dq_opt = Z_opt[3:, k]
            ee_pos_opt[k, :] = omj.end_effector_position(q_opt, l).full().flatten() # convert to numpy array
            jacp_opt = omj.compute_jacobian(q_opt, l)
            ee_vel_opt[k, :] = cs.DM(jacp_opt @ dq_opt).full().flatten()
            ee_vx.append(ee_vel_opt[k, 0])
            ee_vy.append(ee_vel_opt[k, 1])
        ee_vx = np.array(ee_vx).ravel()
        ee_vy = np.array(ee_vy).ravel()

        # compute the lqr linearization list
        robot.linearization_cache = omj.linearize_dynamics_along_trajectory(
            T_opt, U_opt, Z_opt, m, l, r)
        
        # choose 0.80/1 point of the trajectory to be the q_end for post impact
        q_end = Z_opt[:3, int(len(T_opt) *0.80)]
        

    else:
        if NO_CONTROL:
            simulation_time = 30  #sec
            model.opt.timestep = 0.001 # 1ms
            phase = "no_control"
        else:
            simulation_time = REF_TIMES[-1]
            model.opt.timestep = REF_TIMES[1] - REF_TIMES[0] # follow the trajectory time step
            # phase management: "inertia" -> "velocity"
            phase = "inertia"
    
    try:
        step_count = 0
        while robot.viewer.is_running() and data.time < simulation_time*2:

            mujoco.mj_forward(robot.model, robot.data)
            # compute the contact forces
            # Record contact force on the end-effector
            total_force = 0.0
            for i in range(data.ncon):
                force = np.zeros(6)
                mujoco.mj_contactForce(model, data, i, force)
                total_force += np.linalg.norm(force[:3])

            if step_count % record_steps == 0:
                # Get end-effector info
                ee_pos = data.site_xpos[robot.ee_site_id].copy()
                
                # Get end-effector velocity
                jacp = np.zeros((3, model.nv))
                jacr = np.zeros((3, model.nv))
                mujoco.mj_jacSite(model, data, jacp, jacr, robot.ee_site_id)
                ee_vel = jacp @ data.qvel


                jacp_cs = J_p3_fun(data.qpos)
                ee_vel_cs = jacp_cs @ data.qvel
                # print(f"ee_vel: {ee_vel}, ee_vel_cs: {ee_vel_cs}")
                
                # # Compute tracking control torques
                # if track_traj:
                #     # tau = robot.max_manip_vel_tracking_ctrl(data.time)
                #     tau, u_inc_iner = robot.max_inertia_vel_tracking_ctrl(data.time, dM00_dq_fun)
                # else:
                #     tau = np.zeros(model.nv)

                # M_eval, C_eval = compute_dynamics_matrices(M_fun, C_fun, data.qpos, data.qvel)
                
                if phase == "inertia":
                    tau, u_inc_iner = robot.max_inertia_vel_tracking_ctrl(data.time, dM00_dq_fun)
                    motion_vector.append(u_inc_iner)
                    M_curr = M_fun(data.qpos)
                    if M_curr[0, 0] > 0.98 * robot.M00_max:
                        phase = "velocity"
                        robot.u_prev = u_inc_iner # inform velocity phase about the direction of increasing inertia
                elif phase == "velocity":
                    tau, max_manip_dir = robot.max_manip_vel_tracking_ctrl(data.time)
                    motion_vector.append(max_manip_dir)
                elif phase == "optimal":
                    # tau = robot.optimal_trajectory_tracking_ctrl(data.time, T_opt, U_opt, Z_opt)
                    tau = robot.lqr_tracking_controller(data.time, T_opt, U_opt, Z_opt)
                    # catching the moment of impact
                    if data.ncon > 0 and not impact_flag:
                        impact_flag = True
                        impact_time = times[-1]
                        impact_pos = ee_positions[-1]
                        impact_vel = ee_velocities[-1]
                        print(f"Impact detected at time {impact_time:.3f}s, position: {impact_pos}, velocity: {impact_vel}")
                        phase = "post_impact"
                elif phase == "post_impact":
                    # Apply post-impact control
                    # p_end = impact_pos.copy()
                    # p_end[1] += 0.3
                    # tau = robot.post_contact_pose_ctrl(p_end)
                    # # Check if the end-effector is close to the target position
                    # if np.linalg.norm(ee_pos - p_end) < 0.01:
                    #     print(f"End-effector reached target position at time {data.time:.3f}s")
                    #     break

                    # Apply post-impact control
                    tau = robot.post_contact_joint_ctrl(q_end)
                    error = np.linalg.norm(data.qpos - q_end)
                    # print(f"error: {error}")
                    # Check if the end-effector is close to the target position
                    if error < 0.2:
                        print(f"End-effector reached target position at time {data.time:.3f}s")
                        break


                # compute the cartisian acceleration
                if phase != "no_control":
                    xdd = xdd_fun(data.qpos, data.qvel, tau)
                    accel_vector.append(xdd)
                    applied_torques.append(tau.copy())

                # Record data
                times.append(data.time)
                joint_positions.append(data.qpos.copy())
                joint_velocities.append(data.qvel.copy())
                ee_positions.append(ee_pos)
                ee_velocities.append(ee_vel)
                jaco_pos.append(jacp)
                contact_forces.append(total_force)

                # Apply torques and step simulation
                if not NO_CONTROL:
                    # Apply control torques
                    data.ctrl[:] = tau.flatten()

            mujoco.mj_step(model, data)
            step_count += 1

            if step_count % record_steps == 0:
                # visualize the contact points
                with robot.viewer.lock():
                    robot.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
                robot.viewer.sync()

                # Add video recording
                text = f"Time: {data.time:.3f}s"
                if impact_flag:
                    text += f" | Impact!"
                robot.render_frame(text)
                
                # Sleep to roughly match real time
                time.sleep(record_dt * 5) # 10x slow motion
            
    except KeyboardInterrupt:
        pass
    
    finally:
        robot.viewer.close()
        # Close video writer
        if robot.video_writer:
            robot.video_writer.close()
        
    # Plot results
    joint_positions = np.array(joint_positions)
    joint_velocities = np.array(joint_velocities)
    applied_torques = np.array(applied_torques)
    
    # Convert lists to arrays for plotting
    times = np.array(times)
    ee_positions = np.array(ee_positions)
    ee_velocities = np.array(ee_velocities)
    applied_torques = np.array(applied_torques)
    sim_data = {
        'times': times,
        'joint_positions': joint_positions,
        'joint_velocities': joint_velocities,
        'tau': applied_torques,
        'ee_pos': ee_positions,
        'ee_vel': ee_velocities,
        'contact_forces': np.array(contact_forces),
        'impact_time': impact_time,
        'impact_pos': impact_pos,
        'impact_vel': impact_vel,
        'times_opt': T_opt,
        'tau_opt': U_opt,
        'ee_pos_opt': ee_pos_opt,
        'ee_vel_opt': ee_vel_opt,
    }
    
    robot.plot_results(sim_data, track_traj=track_traj, save_path=plot_save_path)


    if not track_traj:
        # Process acceleration vectors
        accel_vector = np.array(accel_vector)  # Shape: (N, 2)
        x = ee_positions[:, 0]
        y = ee_positions[:, 1]
        u = motion_vector[:, 0]
        v = motion_vector[:, 1]
        a_x = accel_vector[:, 0]
        a_y = accel_vector[:, 1]

        # Downsample based on simulation time: select one sample per 0.01 sec
        selected_indices = []
        last_time = -np.inf
        for i, t in enumerate(times):
            if t - last_time >= 0.01:
                selected_indices.append(i)
                last_time = t

        # Apply downsampling to both motion and acceleration vectors
        x_down = x[selected_indices]
        y_down = y[selected_indices]
        u_down = u[selected_indices]
        v_down = v[selected_indices]
        a_x_down = a_x[selected_indices]
        a_y_down = a_y[selected_indices]

        # Compute arrow length based on x-axis range
        x_min, x_max = np.min(x), np.max(x)
        arrow_length = (x_max - x_min) / 20.0

        # Normalize motion vectors
        motion_mag = np.sqrt(u_down**2 + v_down**2)
        motion_mag[motion_mag == 0] = 1.0
        u_norm = u_down / motion_mag
        v_norm = v_down / motion_mag

        # Normalize acceleration vectors
        accel_mag = np.sqrt(a_x_down**2 + a_y_down**2)
        accel_mag[accel_mag == 0] = 1.0
        a_x_norm = a_x_down / accel_mag
        a_y_norm = a_y_down / accel_mag

        # Scale the normalized vectors
        u_plot = u_norm * arrow_length
        v_plot = v_norm * arrow_length
        a_x_plot = a_x_norm * arrow_length
        a_y_plot = a_y_norm * arrow_length

        # Create side-by-side plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Plot motion vectors
        ax1.quiver(x_down, y_down, u_plot, v_plot, color='r', angles='xy',
                scale_units='xy', scale=1, width=0.005)
        ax1.plot(x, y, 'bo-', label="End-Effector Trajectory", markersize=arrow_length/5)
        ax1.set_xlabel('X Position (m)')
        ax1.set_ylabel('Y Position (m)')
        ax1.set_title('Motion Vector (u_inc_iner)')
        ax1.grid(True)
        ax1.legend()

        # Plot acceleration vectors
        ax2.quiver(x_down, y_down, a_x_plot, a_y_plot, color='g', angles='xy',
                scale_units='xy', scale=1, width=0.005)
        ax2.plot(x, y, 'bo-', label="End-Effector Trajectory", markersize=arrow_length/5)
        ax2.set_xlabel('X Position (m)')
        ax2.set_ylabel('Y Position (m)')
        ax2.set_title('Acceleration Vector')
        ax2.grid(True)
        ax2.legend()

        # Make the plots have the same scale
        ax1.set_aspect('equal')
        ax2.set_aspect('equal')

        plt.tight_layout()
        # plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
        # print(f"Saved plot to: {plot_save_path}")
        plt.show()

if __name__ == "__main__":
    main()
