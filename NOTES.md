# System Diagram for RBY1 XR Robot Teleoperation

This diagram illustrates the interaction between the different components of the `demo_rby1_xr_robot_teleop.py` script.

```mermaid
graph TD
    subgraph "External Inputs"
        direction LR
        A[User's Body Pose <br> via WebRTC Client]
        B[User's Keyboard Input]
    end

    subgraph "Python Application (demo_rby1_xr_robot_teleop.py)"
        direction TB
        C[Main Simulation Loop]

        subgraph "Input Handling"
            direction LR
            D[XRRTCBodyPoseDevice]
            E[custom_process_bones_to_action]
            F[TeleopKeyCallback]
        end

        subgraph "Robot Control"
            direction LR
            G[SEWMimicRBY1 Controller]
        end
    end

    subgraph "Simulation & Rendering"
        direction LR
        H[MuJoCo Physics Engine]
        I[MuJoCo Viewer]
    end

    %% Data Flow & Interactions
    A -- "Streams Raw Bone Data" --> D
    B -- "Sends Key Presses" --> I
    I -- "Invokes Callback" --> F
    F -- "Sets Control Flags (e.g., reset, home)" --> C

    C -- "Calls get_controller_state()" --> D
    D -- "Processes Raw Bones" --> E
    E -- "Returns Action Dictionary <br> (SEW Coords, Wrist Rot, Gripper)" --> D
    D -- "Returns Action Dictionary" --> C

    C -- "Updates Controller with Action" --> G
    C -- "Computes Control Torques" --> G
    G -- "Returns Joint Torques" --> C

    C -- "Applies Torques & Steps Simulation" --> H
    H -- "Provides Updated Sim Data" --> I
    I -- "Renders Robot Pose" --> B
```

## Component Descriptions

-   **User's Body Pose (WebRTC)**: The source of the human motion data, captured by a camera and processed by a WebRTC client in a browser.
-   **User's Keyboard Input**: Provides manual control over the simulation (e.g., resetting, quitting, toggling teleop).
-   **XRRTCBodyPoseDevice**: A Robosuite device client that receives the body pose data (bones) from the WebRTC server.
-   **custom_process_bones_to_action**: A function that converts the raw bone data from world coordinates into a robot-centric action dictionary (containing SEW coordinates, wrist rotations, and gripper states).
-   **TeleopKeyCallback**: A callback class that handles keyboard inputs from the MuJoCo viewer to control the simulation state.
-   **Main Simulation Loop**: The core loop that orchestrates the entire process: fetching actions, updating the controller, stepping the physics simulation, and rendering.
-   **SEWMimicRBY1 Controller**: The controller that takes the desired SEW (Shoulder-Elbow-Wrist) coordinates and computes the necessary joint torques to make the RBY1 robot mimic the motion.
-   **MuJoCo Physics Engine**: The underlying engine that simulates the robot's dynamics and physics.
-   **MuJoCo Viewer**: The passive viewer that renders the simulation and captures keyboard inputs.
