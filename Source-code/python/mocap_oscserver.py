from pythonosc import dispatcher
from pythonosc import osc_server
import threading
from skeleton_limbs import SUBSET_LIMBS  # Get Unity limb names

class MocapOSCServer:
    def __init__(self, ip="0.0.0.0", port=5005):
        """
        Initialize the OSC server and motion capture data storage.
        """
        self.ip = ip
        self.port = port

        # initialize data for skeleton. if no data received, all values are None by default so that the main system can still run
        self.skeleton_rotation = {limb: {"x": None, "y": None, "z": None} for limb in SUBSET_LIMBS}
        self.skeleton_position = {limb: {"x": None, "y": None, "z": None} for limb in SUBSET_LIMBS}
        self.neutral_pose      = {limb: {"x": 0.0, "y": 0.0, "z": 0.0} for limb in SUBSET_LIMBS}

        self.skeleton_direction = {limb: {"direction": None} for limb in SUBSET_LIMBS}

        self.joint_angles = {
            "left": {
                "shoulder_flexion": None, 
                "shoulder_abduction": None,
                "shoulder_rotation": None,
                "elbow_flexion": None, 
                "elbow_pronation": None,
                "wrist_flexion": 0,
                "wrist_deviation": None,
                "wrist_pronation": None
            },
            "right": {
                "shoulder_flexion": 90,
                "shoulder_abduction": None,
                "shoulder_rotation": None,
                "elbow_flexion": 10,
                "elbow_pronation": None,
                "wrist_flexion": 0,
                "wrist_deviation": None,
                "wrist_pronation": None
            }
        }

        self.server_thread = None
        self.server = None


    def store_rotation(self, address, *args):
        """
        Store rotation of each limb
        """
        # Extract the limb name from the OSC address
        limb_name = address.split("/")[-1]

        # Check if the limb is in the predefined list
        if limb_name in self.skeleton_rotation:
            # Ensure the data has exactly three values (x, y, z)
            if len(args) == 3:
                # Convert values to the range -180 to 180
                x = (args[0] + 180) % 360 - 180
                y = (args[1] + 180) % 360 - 180
                z = (args[2] + 180) % 360 - 180

                # Apply the offset and store the rotation
                self.skeleton_rotation[limb_name] = {
                    "x": round(x - self.neutral_pose[limb_name]["x"], 2),
                    "y": round(y - self.neutral_pose[limb_name]["y"], 2),
                    "z": round(z - self.neutral_pose[limb_name]["z"], 2),
                }
                # print(f"Updated data for {limb_name}: {self.mocap_data[limb_name]}")
            else:
                print(f"Invalid data format for {address}: {args}")
        # else:
        #     print(f"Unexpected limb: {limb_name}. Ignoring data.")

    def store_position(self, address, *args):
        """
        Store position of each limb
        """
        # Extract the limb name from the OSC address
        limb_name = address.split("/")[-1]

        # Check if the limb is in the predefined list
        if limb_name in self.skeleton_position:
            # Ensure the data has exactly three values (x, y, z)
            if len(args) == 3:
                self.skeleton_position[limb_name] = {
                    "x": round(args[0], 2),
                    "y": round(args[1], 2),
                    "z": round(args[2], 2),
                }
                # print(f"Updated data for {limb_name}: {self.mocap_data[limb_name]}")
            else:
                print(f"Invalid data format for {address}: {args}")
        # else:
        #     print(f"Unexpected limb: {limb_name}. Ignoring data.")
    
    def store_direction(self, address, *args):
        """
        Store direction of each limb in relation to the avatar
        """
        # Extract the limb name from the OSC address
        limb_name = address.split("/")[-1]

        # Check if the limb is in the predefined list
        if limb_name in self.skeleton_direction:
            # Ensure the data has exactly three values (x, y, z)
            self.skeleton_direction[limb_name] = args[0]
            # print(f'{limb_name} {self.skeleton_direction[limb_name]}')
        # else:
        #     print(f"Unexpected limb: {limb_name}. Ignoring data.")

    def store_angles(self, address, *args):
        side = address.split("/")[-1]  # 'left' or 'right'

        if side not in self.joint_angles:
            print(f"Unknown side for angles: {side}")
            return

        if len(args) != 8:
            print(f"Invalid number of angles for {side}: {args}")
            return

        self.joint_angles[side] = {
            "shoulder_flexion": round(args[0], 2),
            "shoulder_abduction": round(args[1], 2),
            "shoulder_rotation": round(args[2], 2),
            "elbow_flexion": round(args[3], 2),
            "elbow_pronation": round(args[4], 2),
            "wrist_flexion": round(args[5], 2),
            "wrist_deviation": round(args[6], 2),
            "wrist_pronation": round(args[7], 2)
        }
        # print(f"[{side.upper()}] {self.joint_angles[side]}")


    def start_server(self):
        """
        Start the OSC server in a separate thread.
        """
        disp = dispatcher.Dispatcher()

        disp.map("/skeleton/rotation/*", self.store_rotation)  # Map OSC addresses to the store_rotation function
        disp.map("/skeleton/position/*", self.store_position)  # Map OSC addresses to the store_rotation function
        disp.map("/skeleton/direction/*", self.store_direction)  # Map OSC addresses to the store_rotation function
        disp.map("/skeleton/angles/*", self.store_angles)  # Map OSC addresses to the store_rotation function

        self.server = osc_server.ThreadingOSCUDPServer((self.ip, self.port), disp)
        self.server_thread = threading.Thread(target=self.server.serve_forever)
        self.server_thread.daemon = True  # Ensure the thread stops when the main program exits
        self.server_thread.start()
        print(f"OSC server running at {self.ip}:{self.port}")

    def stop_server(self):
        """
        Stop the OSC server and its thread.
        """
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            print("OSC server stopped.")

    def get_limb_rotation(self, limb):
        """
        Retrieve the motion capture data for a specific limb.
        """
        return self.skeleton_rotation.get(limb, {"x": None, "y": None, "z": None})
    
    def get_limb_position(self, limb):
        """
        Retrieve the motion capture data for a specific limb.
        """
        return self.skeleton_position.get(limb, {"x": None, "y": None, "z": None})
    
    def get_limb_direction(self, limb):
        """
        Retrieve the motion capture data for a specific limb.
        """
        return self.skeleton_direction.get(limb, {"direction": None})   
    
    def get_joint_angles(self, handedness, limb, movement):
        """
        Retrieve the joint angle for a specific handedness (left or right), limb, and movement.
        Returns the angle as a float or None if the joint is not found.
        """
        # Map handedness to the corresponding side
        side = "left" if handedness.lower() == "left" else "right"

        # Construct the joint key based on the limb and movement
        joint_key = f"{limb}_{movement}"

        # Retrieve the joint angle
        joint_data = self.joint_angles.get(side, {})
        return joint_data.get(joint_key, None)
