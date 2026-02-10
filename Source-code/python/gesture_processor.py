import re
import pandas as pd
from speech_engine import SpeechEngine
import time
import pandas as pd
from mocap_oscserver import MocapOSCServer
from oscservers import OSCServers

speech_engine = SpeechEngine()

# import your EMS device library here, e.g., https://github.com/orgs/ScienceMode/repositories
# and populate stimulate_ems() to work with your EMS device

class GestureProcessor:
    def __init__(self, calibrated_ems_csv='gesture_lists/ems-joint-limits.csv', skeleton=None, instruction_pause=False, debug=False):
        self.debug = debug

        self.ems_csv_path = calibrated_ems_csv
        self.ems_joint_limits_csv = pd.read_csv(self.ems_csv_path, skiprows=[0])

        self.ems_power = 1 # overall power of EMS in percentage of calibrated intensity (0-1)

        # setting default EMS parameters
        self.default_pulse_count = 40
        self.default_delay = 0.0098
        self.EMS_ch = [[0, 200, 10], # default values for 8 channels EMS device
                        [1, 200, 6],
                        [2, 200, 6],
                        [3, 200, 6],
                        [4, 200, 6],
                        [5, 200, 6],
                        [6, 200, 6],
                        [7, 200, 6],
                        [8, 200, 6]]
        
        self.load_ems_calibration()
        self.instruction_pause = instruction_pause

        if skeleton is None:
            self.skeleton = MocapOSCServer()
        else:
            self.skeleton = skeleton

        # Initialize OSC servers with error handling
        try:
            self.osc_servers = OSCServers()
        except OSError as e:
            print(f"OSC server error: {e}")
            self.osc_servers = None

    def update_ems_calibration(self):
        # read csv again
        self.ems_joint_limits_csv = pd.read_csv(self.ems_csv_path)
        print("Updated EMS calibration from", self.ems_csv_path)

    def load_ems_calibration(self):
        try:
            with open("ems_calibration.txt", "r") as file:
                lines = file.readlines()
                for i in range(len(lines)):
                    line = lines[i].strip().split(",")
                    self.EMS_ch[i] = [int(line[0]), int(line[1]), int(line[2])]
            print("EMS_ch", self.EMS_ch)
        except FileNotFoundError:
            print("ems_cali.txt not found, using default values")

    def set_ems_power(self, power):
        if not (0 < power < 1):
            raise("EMS power out of bound")
        else:
            self.ems_power = power
            print(f"Set EMS power to {self.ems_power*100}%")

    def parse_instruction(self, instruction):
        """
        Parse the instruction to extract joint, direction, and target angle.
        Example: Maintain a firm grip around the bottle to keep it stable. <left>[wrist][rotation][counterclockwise][180]
        Returns: handedness (str), joint (str), direction (str), target_angle (int)
        """
        # trim the words before the first '<'
        instruction = instruction[instruction.find("<"):]
        instruction = instruction.strip()
        full_match = re.match(r"<(.*?)>\[(.*?)\]\[(.*?)\]\[(.*?)\]\[(.*?)\]", instruction)
        if full_match:
            handedness      = full_match.group(1)
            joint           = full_match.group(2)
            movement        = full_match.group(3)
            direction       = full_match.group(4)
            target_angle    = int(full_match.group(5))
            # translate wrist rotation to match joint table
            if joint == 'wrist' and (movement == 'rotation' or movement == 'pronation' or movement == 'supination'):
                if movement == 'pronation':
                    movement = 'pronation (rotation)'
                    direction = 'outward'
                elif movement == 'supination':
                    movement = 'supination (rotation)'
                    direction = 'inward'
                if direction == 'counterclockwise':
                    movement = 'pronation (rotation)'
                    direction = 'outward' # update name to match the table
                elif direction == 'clockwise':
                    movement = 'supination (rotation)'
                    direction = 'inward'
            elif joint == 'shoulder':
                if movement == 'abduction':
                    direction = 'outward'
                elif movement == 'adduction':
                    direction = 'inward'


            return handedness, joint, movement, direction, target_angle

        simplified_match = re.match(r"<(.*?)>\[(.*?)\]", instruction)
        if simplified_match:
            handedness = simplified_match.group(1)
            joint = simplified_match.group(2)
            if joint == "grip":
                movement = "flexion"
                direction = "inward"
                target_angle = int(45)
            else:
                movement = None
                direction = None
                target_angle = None
            return handedness, joint, movement, direction, target_angle
    
        raise ValueError(f"Invalid instruction format: {instruction}")
    
    def get_parent_joint(self, joint, movement, direction):
        """
        Get the parent joint for the given joint, movement, and direction.
        Returns: Parent joint name (str).
        """
        parent = self.ems_joint_limits_csv[
            (self.ems_joint_limits_csv["joint"]     == joint) &
            (self.ems_joint_limits_csv["movement"]  == movement) &
            (self.ems_joint_limits_csv["direction"] == direction)
        ]
        if parent.empty:
            raise ValueError(f"No parent joint found for {joint} in {movement}")
        
        parent_joint        = parent.iloc[0]["parent joint"]
        parent_movement     = parent.iloc[0]["parent movement"]
        parent_direction    = parent.iloc[0]["parent direction"]

        return parent_joint, parent_movement, parent_direction
    
    def get_child_joint(self, joint, movement, direction):
        """
        Get the child joint for the given joint, movement, and direction.
        Returns: Parent joint name (str).
        """
        child = self.ems_joint_limits_csv[
            (self.ems_joint_limits_csv["parent joint"]     == joint) &
            (self.ems_joint_limits_csv["parent movement"]  == movement) &
            (self.ems_joint_limits_csv["parent direction"] == direction)
        ]
        if child.empty:
            raise ValueError(f"No child joint found for {joint} in {movement}")
        
        child_joint        = child.iloc[0]["joint"]
        child_movement     = child.iloc[0]["movement"]
        child_direction    = child.iloc[0]["direction"]

        print(child)

        return child_joint, child_movement, child_direction
        
    def get_joint_limits(self, joint, movement, direction):
        """
        Get the joint limits for the given joint, movement, and direction.
        Returns: Dictionary of joint limits.
        """
        joint_limits = self.ems_joint_limits_csv[
            (self.ems_joint_limits_csv["joint"]     == joint) &
            (self.ems_joint_limits_csv["movement"]  == movement) &
            (self.ems_joint_limits_csv["direction"] == direction)
        ]
        if joint_limits.empty:
            raise ValueError(f"No joint limits found for {joint} in {movement}")

        range_min = 0
        range_max = joint_limits["range_max"].values[0]

        return {
            "range_min": range_min,
            "range_max": range_max
        }
    
    def get_ems_channel(self, handedness, joint, movement, direction):
        """
        Validate the EMS channel is calibrated for the given joint, movement, and direction.
        Returns channel if calibrated, None otherwise.
        """
        ems_params = self.ems_joint_limits_csv[
            (self.ems_joint_limits_csv["joint"]     == joint) &
            (self.ems_joint_limits_csv["movement"]  == movement) &
            (self.ems_joint_limits_csv["direction"] == direction)
        ]
        if ems_params.empty:
            return None

        try:
            channel = int(ems_params.iloc[0]["channel-right"] if handedness == "right" else ems_params.iloc[0]["channel-left"]) 
            return channel

        except ValueError as e:
            return None    

    def validate_joint_limits(self, handedness, joint, movement, direction, target_angle):
        """
        Validate the target angle against the joint limits.
        Returns: True if valid, False otherwise.
        """
        if movement is None or direction is None:
            return True
        
        joint_limits = self.ems_joint_limits_csv[
            (self.ems_joint_limits_csv["joint"]     == joint) &
            (self.ems_joint_limits_csv["movement"]  == movement) &
            (self.ems_joint_limits_csv["direction"] == direction)
        ]
        if joint_limits.empty:
            raise ValueError(f"No joint limits found for {joint} in {movement}")

        range_min = 0
        range_max = joint_limits["range_max"].values[0]

        current_joint_angles = self.skeleton.get_joint_angles(handedness, joint, movement)
        if current_joint_angles is None:
            current_joint_angles = 0
        
        print(f"Current joint angles for {joint}: {current_joint_angles}")

        if not (range_min <= target_angle + current_joint_angles <= abs(range_max)):
            print(f"Target angle {target_angle} for {joint} is out of range. Current angle: {current_joint_angles}, Range: [{range_min}, {range_max}]")
            return False
    
        else: 
            print(f"Target angle {target_angle} for {joint} is within range. Current angle: {current_joint_angles}, Range: [{range_min}, {range_max}]")
            return True

    def get_full_ems_tree(self, handedness, joint, movement, direction, target_angle):
        """
        Recursively retrieve EMS parameters for the given joint and its parent joints.
        Returns: List of EMS parameters for the full joint tree.
        """
        ems_params_tree = []
        current_ems_params = None
        
        if self.debug: print(f"\nProcessing joint: {joint}, movement: {movement}, direction: {direction}, target angle: {target_angle}")

        if self.validate_joint_limits(handedness, joint, movement, direction, target_angle) and self.get_ems_channel(handedness, joint, movement, direction):
            if self.debug: print(f"Target angle {target_angle} is within range for {joint} in {direction}.")

            current_ems_params = self.get_ems_parameters(handedness, joint, movement, direction, target_angle)
            current_ems_params["handedness"]    = handedness
            current_ems_params["joint"]         = joint
            current_ems_params["movement"]      = movement
            current_ems_params["direction"]     = direction
            current_ems_params["target_angle"]  = target_angle
            ems_params_tree.append(current_ems_params)

        else:
            print(f"EMS exist: {self.get_ems_channel(handedness, joint, movement, direction)} ")
            if self.get_ems_channel(handedness, joint, movement, direction):
                if self.debug: print(f"Target angle {target_angle} is not within range for {joint} in {direction}. Looking for parent joint.")
                joint_limits = self.get_joint_limits(joint, movement, direction)
                current_target_angle = int(joint_limits["range_max"] * 0.67) 
                parent_target_angle = int(target_angle - current_target_angle)
                if self.debug: print(f"Parent target angle: {parent_target_angle} and current target angle: {current_target_angle} for {joint} in {movement}, {direction}")

                current_ems_params = self.get_ems_parameters(handedness, joint, movement, direction, target_angle)
                current_ems_params["handedness"]    = handedness
                current_ems_params["joint"]         = joint
                current_ems_params["movement"]      = movement
                current_ems_params["direction"]     = direction
                current_ems_params["target_angle"]  = current_target_angle
                ems_params_tree.append(current_ems_params)
            else:
                if self.debug: print(f"EMS channel not calibrated for {joint} in {direction}. Skipping this joint.")
                parent_target_angle = target_angle

            try:
                parent_joint, parent_movement, parent_direction = self.get_parent_joint(joint, movement, direction)
                if self.debug: print(f"Parent joint: {parent_joint}, movement: {parent_movement}, direction: {parent_direction}")

                parent_ems_params_tree = self.get_full_ems_tree(handedness, parent_joint, parent_movement, parent_direction, parent_target_angle)
               
                if not parent_ems_params_tree:
                    if self.debug: print(f"No parent joint found for {joint}. Updating target angle to {current_target_angle}.")
                    current_ems_params["target_angle"] = current_target_angle
                    

                else:
                    ems_params_tree.extend(parent_ems_params_tree)
                
            except ValueError as e:
                if current_ems_params != None:
                    joint_limits = self.get_joint_limits(joint, movement, direction)
                    current_ems_params["target_angle"] = joint_limits["range_max"]
                    # current_ems_params["target_angle"] = target_angle
                    ems_params_tree.append(current_ems_params)
                    if self.debug: print(f"No parent joint found for {joint} in {movement}, updating target angle to {current_target_angle}.. Error: {e}")
                else:
                    if self.debug: print(f"No parent joint found for {joint} in {movement}. Error: {e}")
            
        return ems_params_tree

    def get_ems_parameters(self, handedness, joint, movement, direction, target_angle):
        """
        Retrieve EMS parameters for the given joint and direction.
        Returns: Dictionary of EMS parameters.
        """
        ems_params = self.ems_joint_limits_csv[
            (self.ems_joint_limits_csv["joint"]     == joint) &
            (self.ems_joint_limits_csv["movement"]  == movement)  &
            (self.ems_joint_limits_csv["direction"] == direction) 
        ]
        if ems_params.empty:
            raise ValueError(f"No EMS gesture found for {joint} in {direction}")
        
        channel         = int(ems_params.iloc[0]["channel-right"] if handedness == "right" else ems_params.iloc[0]["channel-left"])
        pulse_width     = int(ems_params.iloc[0]["pulse_width"] if not pd.isna(ems_params.iloc[0]["pulse_width"]) else self.EMS_ch[channel][1])
        if handedness == 'right':
            fetch_handedness = 'intensity-right'
        else:
            fetch_handedness = 'intensity-left'
        intensity       = int(ems_params.iloc[0][fetch_handedness] if not pd.isna(ems_params.iloc[0][fetch_handedness]) else self.EMS_ch[channel][2])
        pulse_count     = int(ems_params.iloc[0]["pulse_count"] if not pd.isna(ems_params.iloc[0]["pulse_count"]) else self.default_pulse_count)
        delay           = float(ems_params.iloc[0]["delay"] if not pd.isna(ems_params.iloc[0]["delay"]) else self.default_delay)

        if self.debug: print(f'{channel} {pulse_width} {intensity} {pulse_count} {delay}')
        return {
            "channel"       : channel,
            "pulse_width"   : pulse_width,
            "intensity"     : intensity,
            "pulse_count"   : pulse_count,
            "delay"         : delay,
        }

    def process_instructions(self, instructions, timeout_duration=5):
        """
        Process instructions to validate and execute EMS stimulation.
        Includes parent joints in the EMS parameters tree.
        """
        output_log = ""
        for instruction in instructions.split("\n"):
            try:
                # find multiple EMS instructions in one line (<>[][][][][])
                instruction = instruction[instruction.find("<"):]
                instruction = instruction.strip()
                # split by spaces but keep the parts within <>
                parts = re.split(r'\s+(?=<)', instruction)
                for part in parts:
                    delay_intra_step = 0.2 # tune the delay (e.g., extent, flex between a shake)
                    handedness, joint, movement, direction, target_angle = self.parse_instruction(part)
                    print(f"Parsed Instruction: Handedness={handedness}, Joint={joint}, Direction={direction}, Movement={movement}, Target Angle={target_angle}")
                    output_log += f"Parsed Instruction: Handedness={handedness}, Joint={joint}, Direction={direction}, Movement={movement}, Target Angle={target_angle}\n"

                    ems_params = self.get_full_ems_tree(handedness, joint, movement, direction, target_angle)
                    ems_params.reverse()
                    
                    ems_params_df = pd.DataFrame(ems_params)
                    print(f"EMS Parameters Tree: \n{ems_params_df}")
                    output_log += f"EMS Parameters Tree: {ems_params}\n"

                    self.osc_servers.send_stim_status("stim_on")
                    
                    for params in ems_params:
                        if self.osc_servers is not None:
                            self.osc_servers.send_ems_movements(params["handedness"], params["joint"], params["direction"])

                        adjusted_intensity = int(params["intensity"] * self.ems_power)

                        if adjusted_intensity <= 0:
                            adjusted_intensity = 1
                        elif adjusted_intensity > params["intensity"]:
                            adjusted_intensity = params["intensity"]

                        current_joint_angle = self.skeleton.get_joint_angles(params["handedness"], params["joint"], params["movement"])
                        
                        if current_joint_angle is None:
                            print(f"No current angle available for {params['joint']}, performing single stimulation")
                            try:
                                for i in range(params["pulse_count"]):
                                    self.stimulate_ems(
                                        channel         = params["channel"]-1, # if csv has channel = 0, error
                                        pulse_width     = params["pulse_width"],
                                        intensity       = adjusted_intensity,
                                    )
                                    time.sleep(params["delay"]) 
                                print(f"Stimulated {params['joint']} {params['movement']} with target angle {params['target_angle']}.")
                                output_log += f"Stimulated {params['joint']} {params['movement']} with target angle {params['target_angle']}.\n\n"
                            except:
                                print("electrode error")

                        else:
                            start_time = time.time()
                            deadzone_threshold = 2
                            timeout_occurred = False
                            
                            while True:
                                current_angle = self.skeleton.get_joint_angles(params["handedness"], params["joint"], params["movement"])
                                if current_angle is None:
                                    current_angle = 0
                                
                                angle_difference = abs(current_angle - params["target_angle"])
                                
                                if angle_difference <= deadzone_threshold:
                                    print(f"Target angle reached for {params['joint']}. Current: {current_angle}, Target: {params['target_angle']}")
                                    output_log += f"Target angle reached for {params['joint']}. Current: {current_angle}, Target: {params['target_angle']}\n"
                                    break
                                
                                if time.time() - start_time >= timeout_duration:
                                    print(f"Timeout occurred for {params['joint']}. Current: {current_angle}, Target: {params['target_angle']}")
                                    output_log += f"Timeout occurred for {params['joint']}. Current: {current_angle}, Target: {params['target_angle']}\n"
                                    timeout_occurred = True
                                    break
                                
                                try:
                                    for i in range(params["pulse_count"]):
                                        self.stimulate_ems(
                                            channel         = params["channel"]-1, 
                                            pulse_width     = params["pulse_width"],
                                            intensity       = adjusted_intensity,
                                        )
                                        time.sleep(params["delay"]) 
                                except:
                                    print("electrode error")
                                
                                time.sleep(0.05)
                            
                            if timeout_occurred:
                                output_log += f"Timeout for {params['joint']} {params['movement']}.\n\n"
                            else:
                                output_log += f"Successfully reached target angle for {params['joint']} {params['movement']}.\n\n"
                    
                    if self.osc_servers is not None:
                        self.osc_servers.send_stim_status("stim_off")
                    time.sleep(delay_intra_step)

            except ValueError as e:
                print(f"Error processing instruction: {e}")
                output_log += f"Error processing instruction: {e}\n"

        return output_log
    

    # your EMS stimulation function
    def stimulate_ems(self, channel, pulse_width, intensity):
        # stimulate here
        print(f"Stimulating channel {channel} with intensity {intensity}, pulse width {pulse_width}")
