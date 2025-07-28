import argparse
import time
from copy import deepcopy

import mujoco
import numpy as np

import robosuite as suite
from robosuite.devices.webrtc_body_pose_device import WebRTCBodyPoseDevice
from robosuite.wrappers import VisualizationWrapper

# --- Main Simulation Script ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Integrated WebRTC teleoperation demo for robosuite")
    parser.add_argument("--environment", type=str, default="DualKinova3SRLEnv")
    parser.add_argument("--robots", nargs=":", type=str, default="DualKinova3")
    parser.add_argument("--controller", type=str, default="WHOLE_BODY_MIMIC")
    parser.add_argument("--device", type=str, default="keyboard")
    parser.add_argument("--max_fr", default=30, type=int)
    args = parser.parse_args()

    # 1. Create the robosuite environment.
    controller_config = suite.load_controller_config(default_controller=args.controller)
    env = suite.make(
        args.environment,
        robots=args.robots,
        controller_configs=controller_config,
        has_renderer=False,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=30,
    )
    env = VisualizationWrapper(env)
    np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

    # 2. Set up the device.
    device = WebRTCBodyPoseDevice(env=env)
    env.sim.add_render_callback("controller_logging", device.render_callback)
    device.start_control()


    # 3. Wait for the VR client to connect.
    print("\nWaiting for a VR client to connect...")
    while not device.state_factory.instance or not device.state_factory.instance.is_connected:
        time.sleep(0.5)
    
    print("Client connected! Starting robosuite simulation.")

    # 4. Run the simulation loop.
    obs = env.reset()
    model = env.sim.model._model
    data = env.sim.data._data

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 0
        viewer.cam.elevation = -95
        viewer.cam.lookat[:] = np.array([-0.5, 0.0, 0.0])

        while viewer.is_running():
            start_time = time.time()
            
            # Get the latest pose action from the shared state
            action = device.input2action()

            if action is None:
                time.sleep(0.01) # Wait for the first pose to arrive
                continue

            env.step(action)
            viewer.sync()

            # Maintain target frame rate
            elapsed = time.time() - start_time
            if elapsed < 1 / args.max_fr:
                time.sleep(1 / args.max_fr - elapsed)

    # 5. Cleanup
    print("\nSimulation finished. Closing environment...")
    env.close()
    print("Demo completed.")
