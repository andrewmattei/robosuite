import time, os
import numpy as np
import mujoco
from mujoco import viewer
import matplotlib.pyplot as plt

# Updated XML with a higher starting height for the ball (1.15 m instead of 1.0 m)
XML = """
<mujoco>
  <option timestep="0.0005"/>
  <asset>
    <!-- A skybox texture for ambient lighting -->
    <texture type="skybox" builtin="gradient" rgb1="0.6 0.8 1.0" rgb2="1 1 1" width="256" height="256"/>
  </asset>
  
  <worldbody>
    <!-- Table -->
    <body name="table" pos="0 0 0.8">
      <geom type="box" size="0.4 0.4 0.05" rgba=".6 .6 .6 1" friction="1 0.005 0.0001"/>
    </body>

    <!-- Directional light similar to robosuite room -->
    <light name="room_light" pos="0 3 4" dir="0 -1 -1" diffuse="1 1 1" specular="0.2 0.2 0.2" active="true"/>

    <!-- Actuated Ball -->
    <body name="ball" pos="0 0 1.15">
      <freejoint name="free_joint"/>
      <geom type="sphere" mass="0.1" size="0.025" solref="0.01 0.1" condim="4" priority="1"/>
      <!-- <geom type="sphere" mass="0.05" size="0.025" solref="-70000 -0.1" condim="4" priority="1"/> -->
    </body>
  </worldbody>

  <actuator>
    <motor name="z_motor" joint="free_joint" gear="0 0 1 0 0 0" ctrlrange="-100 100"/>
  </actuator>
</mujoco>
"""

class ContactCtrlEnv:
    def __init__(self, init_height=1.15):
        self.model = mujoco.MjModel.from_xml_string(XML)
        self.data = mujoco.MjData(self.model)
        self.init_height = init_height
        
        # Get actuator ID for the z_motor
        self.actuator_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "z_motor"
        )
        
        # Initialize viewer
        self.viewer = viewer.launch_passive(self.model, self.data)
        self._configure_viewer()

    def _configure_viewer(self):
        self.viewer.cam.distance = 2.0
        self.viewer.cam.azimuth = 120
        self.viewer.cam.elevation = -30
        self.viewer.cam.lookat[:] = np.array([0.0, 0.0, 1.0])

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        # Set initial ball position: x, y, z, then quaternion (w, x, y, z)
        self.data.qpos[:] = [0, 0, self.init_height, 1, 0, 0, 0]  

    def step(self, control_signal):
        self.data.ctrl[self.actuator_id] = control_signal
        mujoco.mj_step(self.model, self.data)
        self.viewer.sync()

if __name__ == "__main__":
    debug = False
    # === Parameters for Bang–Bang control & Suspension ===
    m = 0.1            # mass (kg)
    g = 9.81            # gravity (m/s^2)
    table_z = 0.85      # table top height (m)
    opt_timestep = 0.00005  # timestep (s)
    suspend_height = 1.15  # height to suspend the ball after impact
    suspend_tol = 0.1
    simulation_time = 2.0 # total simulation time (s)

    # Control force settings for bang–bang phase (units in Newtons)
    # Note: In our coordinate frame (z positive upward),
    # a negative control pushes downward.
    Fz_acc_control = -10  # force during acceleration phase (downward)
    Fz_decc_control = 20.0   # force during deceleration phase (upward)

    # Desired speed magnitudes (m/s) for bang–bang
    v_des = 20.0  # speed to reach before deceleration (magnitude)
    v_f = 0.5   # desired impact speed (magnitude), with v_f < v_des

    # compute minimum acceleration required to reach v_des
    a_acc_net = (Fz_acc_control / m) - g  # must be negative for effective acceleration
    acc_distance = np.abs((v_des**2 - 0) / (2 * a_acc_net))
    print(f"Acceleration distance: {acc_distance:.3f} m")

    # Compute net deceleration during deceleration phase:
    # In deceleration phase, the net upward acceleration is:
    a_decc_net = (Fz_decc_control / m) - g  # must be positive for effective deceleration
    decel_distance = (v_des**2 - v_f**2) / (2 * a_decc_net)
    print(f"Deceleration distance: {decel_distance:.3f} m")

    # Compute starting height
    ball_radius = 0.025
    start_z = table_z + acc_distance + decel_distance + ball_radius
    print(f"Starting height: {start_z:.3f} m")

    # max distance that ball can travel between opt.timestep
    max_timestep_travel_dist = np.abs(0.5 * a_acc_net * opt_timestep**2) + v_des * opt_timestep

    # === create an expected trajectory for the ball to be used in debug ===
    times_expected = np.arange(0, simulation_time, opt_timestep)
    z_dot_expected = np.zeros_like(times_expected)
    z_expected = np.zeros_like(times_expected)
    z_dot_expected[0] = 0.0
    z_expected[0] = start_z - table_z - ball_radius
    for i in range(1, len(times_expected)):
        if z_expected[i-1] > decel_distance:
            z_dot_expected[i] = z_dot_expected[i-1] + a_acc_net * opt_timestep
        elif z_expected[i-1] <= 0:
            z_dot_expected[i] = 0.0
        else:
            z_dot_expected[i] = z_dot_expected[i-1] + a_decc_net * opt_timestep
        z_expected[i] = z_expected[i-1] + z_dot_expected[i] * opt_timestep


    # Initialize environment
    env = ContactCtrlEnv(init_height=start_z)
    env.reset()
    env.model.opt.timestep = opt_timestep

    # === estimating the impact force ===
    timeconst, dampratio = env.model.geom_solref[-1]
    # F_imp_expacted = m*v_f/timeconst + 2*m*dampratio*v_f/timeconst # chatgpt estimate
    k_est = 1108030
    b_est = 210.53
    panetration_depth = np.sqrt(m*v_f**2/k_est)
    # panetration_depth = v_f*timeconst # chatgpt estimate
    a1_impact_expected = -b_est* v_f - k_est*panetration_depth
    F_imp_expacted = m*a1_impact_expected
    print(f"Expected impact force: {np.abs(F_imp_expacted):.3f} N")
    #TODO the expected impact force is always 4 times smaller than the actual impact force
    #TODO use the impact force paper to test the impact force estimate

    # === PD Controller parameters for suspension phase ===
    Kp = 50.0  # proportional gain
    Kd = 10.0  # derivative gain

    # Phase management: "bangbang" -> "impact" -> "suspend"
    phase = "bangbang"

    # Data recording lists
    times = []
    forces = []
    positions = []
    velocities = []
    ctrl_inputs = []

    impact_time = None
    impact_velocity = None

    decc_but_passed_vel = False

    try:
        # Run simulation for a sufficient time to see the full cycle
        while env.viewer.is_running() and env.data.time < simulation_time:
            current_time = env.data.time
            # Get current ball position and vertical velocity.
            # qpos[2] is the z-position.
            current_z = env.data.qpos[2]
            z_bot_to_table = current_z - table_z - ball_radius
            # qvel[2] is vertical velocity (positive upward, negative downward)
            current_v = env.data.qvel[2]  # use as-is for PD control
            # For bang–bang phase, we compute downward speed as a positive number.
            v_down = -current_v if current_v < 0 else 0.0

            # Determine the current number of contacts (impact detection)
            ncon = env.data.ncon

            # --- Phase Switching Logic ---
            if phase == "bangbang":
                # Use bang-bang control until impact is detected.
                # note we want to stop accelerating before the ball move past the max_timestep_travel_dist at the fastist speed
                in_acceleration = current_z - table_z - ball_radius > decel_distance + max_timestep_travel_dist
                if in_acceleration:
                    control = Fz_acc_control
                else:
                    control = Fz_decc_control

                # If contact is detected, mark impact phase.
                if (not in_acceleration) and v_down < v_f:
                    if not decc_but_passed_vel:
                        print(f"Deccel ended early at t={current_time:.5f}s, v={v_down:.5f}m/s. z={current_z - table_z - ball_radius:.5f}m.")
                    decc_but_passed_vel = True
                
                if decc_but_passed_vel:
                    # control = 0.0
                    control = m * g  # cancel gravity
                    

                if ncon > 0:
                    phase = "impact"
                    impact_time = times[-1]
                    impact_velocity = velocities[-1]
                    print(f"Impact detected at t={impact_time:.5f}s, v={impact_velocity:.5f}m/s. Switching to impact.")
            
            elif phase == "impact":
                # During impact, we can hold the current control (or simply set control to 0).
                control = 0.0
                # Wait for contacts to end (i.e. impact is over).
                if ncon == 0:
                    phase = "suspend"
                    print(f"Impact ended at t={current_time:.5f}s. Switching to suspend.")
            
            elif phase == "suspend":
                # PD controller to drive the ball to start_z.
                error = suspend_height - current_z
                if np.abs(error) < suspend_tol:
                    print(f"Suspension height reached at t={current_time:.5f}s. Simulation ended.")
                    break
                # Use vertical velocity (qvel[2]) directly; positive means upward.
                u_des = Kp * error - Kd * current_v  
                # The required control force (to overcome gravity as well)
                control = np.clip(u_des + m * g, Fz_acc_control, Fz_decc_control)
            
            # Record contact force on the ball in the z-direction (for plotting)
            total_force = 0.0
            for i in range(env.data.ncon):
                force = np.zeros(6)
                mujoco.mj_contactForce(env.model, env.data, i, force)
                total_force += force[0]

            # Record data
            times.append(current_time)
            forces.append(total_force)
            positions.append(current_z-table_z-ball_radius)
            velocities.append(current_v)
            ctrl_inputs.append(control)

            # examine physical properties of the ball
            # control = 0.0

            # Step simulation with the computed control
            env.step(control)
            
            # Sleep to roughly match real time
            time.sleep(env.model.opt.timestep)

    except KeyboardInterrupt:
        pass

    finally:
        env.viewer.close()

    # Updated fig1: add expected trajectory curves into the position and velocity plots
    fig, axs = plt.subplots(4, 1, figsize=(10, 12))
    
    # Force plot remains unchanged
    axs[0].plot(times, forces, label='Contact Force')
    axs[0].set_title('Contact Force (z-direction)')
    axs[0].set_ylabel('Force (N)')
    axs[0].grid(True)
    peak_force = max(forces)
    peak_time = times[forces.index(peak_force)]
    axs[0].annotate(f"Peak: {peak_force:.2f}N", 
                    xy=(peak_time, peak_force), 
                    xytext=(peak_time, peak_force+0.5),
                    arrowprops=dict(facecolor='black', shrink=0.05))
    axs[0].legend()
    
    # Z-position plot: overlay expected vs actual
    axs[1].plot(times, positions, 'b-', label='Actual Z')
    if debug:
        axs[1].plot(times_expected, z_expected, 'r--', label='Expected Z')
    axs[1].set_title('Ball Height (z)')
    axs[1].set_ylabel('Z Position (m)')
    axs[1].grid(True)
    axs[1].legend()

    # Velocity plot: overlay expected vs actual vertical velocity
    axs[2].plot(times, velocities, 'b-', label='Actual Velocity')
    if debug:
        axs[2].plot(times_expected, z_dot_expected, 'r--', label='Expected Velocity')
    axs[2].set_title('Ball Vertical Velocity')
    axs[2].set_ylabel('Velocity (m/s)')
    # Draw horizontal lines for desired impact velocity v_f and max velocity v_des
    axs[2].axhline(y=-v_f, color='purple', linestyle='--', label=f"v_f ({-v_f} m/s)")
    axs[2].axhline(y=-v_des, color='orange', linestyle='--', label=f"v_des ({-v_des} m/s)")
    # Mark expected impact velocity: the velocity right before z_expected becomes <= 0
    if debug:
        impact_idxs = np.where(z_expected <= 0)[0]
        if len(impact_idxs) > 0:
            idx = impact_idxs[0] - 1 if impact_idxs[0] > 0 else 0
            expected_impact_time = times_expected[idx]
            expected_impact_velocity = z_dot_expected[idx]
            axs[2].axvline(x=expected_impact_time, color='magenta', linestyle='--', label='Expected Impact Time')
            axs[2].annotate(f"Exp Impact vel: {expected_impact_velocity:.2f} m/s", 
                            xy=(expected_impact_time, expected_impact_velocity),
                            xytext=(expected_impact_time, expected_impact_velocity+0.5),
                            arrowprops=dict(facecolor='magenta', shrink=0.05))
    if impact_time is not None:
        axs[2].axvline(x=impact_time, color='red', linestyle='--', label='Impact time')
        axs[2].annotate(f"Impact vel: {impact_velocity:.2f} m/s", 
                        xy=(impact_time, impact_velocity), 
                        xytext=(impact_time, impact_velocity+0.5),
                        arrowprops=dict(facecolor='red', shrink=0.05))
    axs[2].grid(True)
    axs[2].legend()
    
    # Control input plot remains unchanged with horizontal lines
    axs[3].plot(times, ctrl_inputs, label='Control Input')
    axs[3].set_title('Control Inputs')
    axs[3].set_ylabel('Force (N)')
    axs[3].set_xlabel('Time (s)')
    axs[3].grid(True)
    axs[3].axhline(y=Fz_acc_control, color='blue', linestyle='--', label=f"Fz_acc ({Fz_acc_control}N)")
    axs[3].axhline(y=Fz_decc_control, color='green', linestyle='--', label=f"Fz_decc ({Fz_decc_control}N)")
    axs[3].legend()
    
    plt.tight_layout()
    plt.show()

    # Save the trajectory data
    trajectory_data = {
        'times': np.array(times),
        'positions': np.array(positions),
        'velocities': np.array(velocities)
    }
    # get current file location
    current_path = os.path.dirname(os.path.realpath(__file__))
    save_path = os.path.join(current_path, 'ball_trajectory.npy')
    np.save(save_path, trajectory_data)

