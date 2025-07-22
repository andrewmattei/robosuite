import threading
import time
import numpy as np

try:
    import openvr
except ImportError:
    print("Could not import OpenVR. Please install it with 'pip install openvr'")
    openvr = None

class VRHeadsetDevice:
    """
    A device class to handle polling a VR headset (via OpenVR) for its pose
    in a separate, non-blocking thread.
    """
    def __init__(self):
        if openvr is None:
            raise ImportError("OpenVR library not found. Cannot initialize VRHeadsetDevice.")

        try:
            openvr.init(openvr.VRApplication_Scene)
            self.vr_system = openvr.VRSystem()
            print("OpenVR initialized successfully.")
        except openvr.OpenVRError as e:
            print(f"Error initializing OpenVR. Is SteamVR running? Error: {e}")
            raise

        self.pose_lock = threading.Lock()
        self.stop_event = threading.Event()
        
        # Shared data: position and orientation (as a 3x4 matrix)
        self.headset_pose = None 
        
        self.polling_thread = threading.Thread(target=self._vr_polling_loop)
        self.polling_thread.daemon = True

    def start(self):
        """Starts the background polling thread."""
        self.polling_thread.start()
        print("VR headset polling thread started.")

    def stop(self):
        """Stops the background thread and cleans up OpenVR."""
        print("Stopping VR headset polling thread...")
        self.stop_event.set()
        self.polling_thread.join(timeout=1.0) # Wait for thread to finish
        openvr.shutdown()
        print("OpenVR shut down.")

    def _vr_polling_loop(self):
        """The background thread's main loop."""
        while not self.stop_event.is_set():
            poses = []
            try:
                # Poll OpenVR for the latest poses
                openvr.VRCompositor().waitGetPoses(poses, None)
                
                # The headset is always device index 0
                hmd_pose_matrix = poses[openvr.k_unTrackedDeviceIndex_Hmd].mDeviceToAbsoluteTracking
                
                # Acquire lock to update the shared data
                with self.pose_lock:
                    self.headset_pose = hmd_pose_matrix
            
            except openvr.OpenVRError as e:
                print(f"OpenVR error while polling: {e}")
                time.sleep(1) # Wait before retrying
                continue

            # Sleep briefly to prevent busy-waiting (e.g., for a 90Hz headset)
            time.sleep(1/100) 

    def get_pose(self):
        """
        Public method for the main thread to get the latest headset pose.
        Returns a 3x4 numpy array representing the headset's pose matrix.
        """
        with self.pose_lock:
            if self.headset_pose is None:
                return None
            # Return a copy to prevent race conditions after the lock is released
            return np.copy(self.headset_pose)
