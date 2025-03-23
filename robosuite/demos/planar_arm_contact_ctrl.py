import mujoco
from mujoco import viewer
import numpy as np
import time
import matplotlib.pyplot as plt
import os

import casadi as cs
from optimizing_max_jacobian import formulate_symbolic_dynamic_matrices, compute_dynamics_matrices

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
  </asset>
  
  <worldbody>
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

class RobotSystem:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.nv = model.nv
        
        # Initialize viewer
        self.viewer = viewer.launch_passive(model, data)
        self._configure_viewer()

        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "site_end_effector")
        
        # Control gains for trajectory tracking
        self.Kp = 200.0  # proportional gain for velocity tracking
        self.Kd = 0.5   # derivative gain for velocity tracking

        #to fix a bug that if ee pass the y=0 line, calculation blow up
        self.u_prev = None

        # inertia mag
        self.M00_max = None
        

    def _configure_viewer(self):
        """Configure the viewer camera."""
        self.viewer.cam.distance = 8.0
        self.viewer.cam.azimuth = 90
        self.viewer.cam.elevation = -90
        self.viewer.cam.lookat[:] = np.array([0.5, 0.0, 0.0])

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
        acc_des = self.Kp * (vel_des - vel_ee_current)
        lambda_sq = 0.01  # damping factor squared (tune as needed)
        J_inv_dls = jacp.T @ np.linalg.inv(jacp @ jacp.T + lambda_sq * np.eye(jacp.shape[0]))
        ddq_des = J_inv_dls @ acc_des
        
        # Compute required torques using inverse dynamics
        tau = self.inverse_dynamics(ddq_des)
        return tau
    

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
        acc_des = self.Kp * (vel_des - vel_ee_current)
        lambda_sq = 0.01  # damping factor squared (tune as needed)
        J_inv_dls = jacp.T @ np.linalg.inv(jacp @ jacp.T + lambda_sq * np.eye(jacp.shape[0]))
        ddq_des = J_inv_dls @ acc_des

        # Compute required torques using inverse dynamics
        tau = self.inverse_dynamics(ddq_des.flatten())
        return tau, u_increasing_inertia_dir
        


def main():
    track_traj = True
    # Initialize simulation
    model = mujoco.MjModel.from_xml_string(robot_xml)
    data = mujoco.MjData(model)
    robot = RobotSystem(model, data)
    
    # Simulation parameters
    if track_traj:
        simulation_time = REF_TIMES[-1]
        model.opt.timestep = REF_TIMES[1] - REF_TIMES[0] # follow the trajectory time step
    else:
        simulation_time = 30.0  # seconds
    
    # Data recording
    times = []
    joint_positions = []
    joint_velocities = []
    applied_torques = []
    ee_positions = []
    ee_velocities = []
    jaco_pos = []

    motion_vector = []

    # CasaDi formulations
    m = [LINK_MASS, LINK_MASS, LINK_MASS]
    l = [LINK_LENGTH, LINK_LENGTH, LINK_LENGTH]
    r = [LINK_RADIUS, LINK_RADIUS, LINK_RADIUS]
    q_sym = cs.MX.sym('q', 3)
    dq_sym = cs.MX.sym('dq', 3)
    M, C, M_fun, C_fun = formulate_symbolic_dynamic_matrices(m, l, r, q_sym, dq_sym)
    dM00_dq = cs.jacobian(M[0, 0], q_sym)
    dM00_dq_fun = cs.Function('dM00_dq', [q_sym], [dM00_dq])

    # maximum base joint inertia magnitude
    M_max = M_fun(np.zeros(3))
    robot.M00_max = M_max[0, 0]
    
    
    # Set initial joint positions (optional)
    data.qpos[:] = [np.pi, np.pi*0.3, np.pi*0.5]  # Example initial joint angles
    
    # Desired joint accelerations (example: trying to hold position)
    # ddq_des = np.zeros(model.nv)

    # phase management: "inertia" -> "velocity"
    phase = "inertia"
    
    try:
        while robot.viewer.is_running() and data.time < simulation_time:

            mujoco.mj_forward(robot.model, robot.data)

            # Get end-effector info
            ee_pos = data.site_xpos[robot.ee_site_id].copy()
            
            # Get end-effector velocity
            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jacp, jacr, robot.ee_site_id)
            ee_vel = jacp @ data.qvel
            
            # # Compute tracking control torques
            # if track_traj:
            #     # tau = robot.max_manip_vel_tracking_ctrl(data.time)
            #     tau, u_inc_iner = robot.max_inertia_vel_tracking_ctrl(data.time, dM00_dq_fun)
            # else:
            #     tau = np.zeros(model.nv)

            # M_eval, C_eval = compute_dynamics_matrices(M_fun, C_fun, data.qpos, data.qvel)
            
            if phase == "inertia":
                tau, u_inc_iner = robot.max_inertia_vel_tracking_ctrl(data.time, dM00_dq_fun)
                M_curr = M_fun(data.qpos)
                if M_curr[0, 0] > 0.98 * robot.M00_max:
                    phase = "velocity"
                    robot.u_prev = u_inc_iner # inform velocity phase about the direction of increasing inertia
            elif phase == "velocity":
                tau = robot.max_manip_vel_tracking_ctrl(data.time)

            # Record data
            times.append(data.time)
            joint_positions.append(data.qpos.copy())
            joint_velocities.append(data.qvel.copy())
            ee_positions.append(ee_pos)
            ee_velocities.append(np.linalg.norm(ee_vel))
            applied_torques.append(tau.copy())
            jaco_pos.append(jacp)
            motion_vector.append(u_inc_iner)
            
            # Apply torques and step simulation
            data.ctrl[:] = tau
            mujoco.mj_step(model, data)
            robot.viewer.sync()
            
            # Sleep to roughly match real time
            time.sleep(model.opt.timestep)
            
    except KeyboardInterrupt:
        pass
    
    finally:
        robot.viewer.close()
        
    # Plot results
    joint_positions = np.array(joint_positions)
    joint_velocities = np.array(joint_velocities)
    applied_torques = np.array(applied_torques)
    
    # Convert lists to arrays for plotting
    times = np.array(times)
    ee_positions = np.array(ee_positions)
    ee_velocities = np.array(ee_velocities)
    applied_torques = np.array(applied_torques)
    motion_vector = np.array(motion_vector)  # Shape: (N, 3)
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    
    # End-effector position
    axes[0].plot(times, ee_positions[:, 0], label='x', linewidth=2)
    axes[0].plot(times, ee_positions[:, 1], label='y', linewidth=2)
    axes[0].plot(times, ee_positions[:, 2], label='z', linewidth=2)
    axes[0].set_title('End-Effector Position')
    axes[0].set_ylabel('Position (m)')
    axes[0].grid(True)
    axes[0].legend()
    
    # End-effector velocity magnitude
    axes[1].plot(times, ee_velocities, label='Actual', linewidth=2)
    if track_traj:
        # Add reference velocity if tracking trajectory
        ref_velocities = [np.linalg.norm(vel) for vel in REF_VELOCITIES]
        axes[1].plot(REF_TIMES, ref_velocities, '--', label='Desired', linewidth=2)
    axes[1].set_title('End-Effector Velocity Magnitude')
    axes[1].set_ylabel('Velocity (m/s)')
    axes[1].grid(True)
    axes[1].legend()
    
    # Applied torques
    for i in range(3):
        axes[2].plot(times, applied_torques[:, i], label=f'Joint {i+1}')
    axes[2].set_title('Applied Torques')
    axes[2].set_ylabel('Torque (N⋅m)')
    axes[2].set_xlabel('Time (s)')
    axes[2].grid(True)
    axes[2].legend()
    
    plt.tight_layout()
    plt.show()


    # Downsample based on simulation time: select one sample per 0.01 sec.
    selected_indices = []
    last_time = -np.inf
    for i, t in enumerate(times):
        if t - last_time >= 0.01:
            selected_indices.append(i)
            last_time = t

    # Extract x and y positions from the end–effector positions
    x = ee_positions[:, 0]
    y = ee_positions[:, 1]
    u = motion_vector[:, 0]
    v = motion_vector[:, 1]

    # Apply downsampling
    x_down = x[selected_indices]
    y_down = y[selected_indices]
    u_down = u[selected_indices]
    v_down = v[selected_indices]

    # Compute the horizontal axis range from the full trajectory
    x_min, x_max = np.min(x), np.max(x)
    arrow_length = (x_max - x_min) / 20.0  # Arrow length is 1/20 of the x–axis range

    # Normalize the downsampled motion vectors
    motion_mag = np.sqrt(u_down**2 + v_down**2)
    motion_mag[motion_mag == 0] = 1.0  # Avoid division by zero
    u_norm = u_down / motion_mag
    v_norm = v_down / motion_mag

    # Scale the normalized vectors by arrow_length
    u_plot = u_norm * arrow_length
    v_plot = v_norm * arrow_length

    plt.figure(figsize=(8, 8))
    plt.quiver(x_down, y_down, u_plot, v_plot, color='r', angles='xy',
               scale_units='xy', scale=1, width=0.005)
    plt.plot(x, y, 'bo-', label="End-Effector Trajectory", markersize=arrow_length/5)
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('Motion Vector (u_inc_iner) at End-Effector')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
