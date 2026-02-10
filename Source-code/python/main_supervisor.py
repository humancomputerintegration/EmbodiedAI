'''
this script is used to control the main system (main.py), allowing to start/stop the system at anytime based on voice commands
it also uses skeleton data to check if movements are congruent with system instructions, otherwise stops the main system
'''

import time
import subprocess
import threading

from skeleton_limbs import SUBSET_LIMBS
from speech_engine import SpeechEngine

from pythonosc import dispatcher
from pythonosc import osc_server
from pythonosc import udp_client

UNITY_IP = "0.0.0.0"    # inbound from Unity
UNITY_PORT = 5006

GPT_IN_IP = "0.0.0.0"   # inbound from main.py
GPT_IN_PORT = 5007

GPT_OUT_IP = "0.0.0.0"  # outbound to main.py
GPT_OUT_PORT = 5008
gpt_out_address = "/user_command"

# Global variable to store the subprocess
main_gpt_process = None
stim_on = False         # stim status
gpt_handedness, gpt_limb, gpt_direction = None, None, None

skeleton_movement = {limb: {"direction": None} for limb in SUBSET_LIMBS}

def start_main_gpt():
    global main_gpt_process
    print("[watchdog] Starting main.py...")
    main_gpt_process = subprocess.Popen(["python", "main.py"])
    print(f"[watchdog] main.py started with PID {main_gpt_process.pid}.")

def kill_main_gpt():
    global main_gpt_process
    if main_gpt_process and main_gpt_process.poll() is None:  # Check if the process is running
        print(f"[watchdog] Terminating main.py with PID {main_gpt_process.pid}...")
        main_gpt_process.terminate()  # Send SIGTERM
        main_gpt_process.wait(timeout=5)  # Wait for the process to terminate
        print("[watchdog] main.py terminated.")
    else:
        print("[watchdog] main.py is not running.")

def watchdog_main_gpt():
    global main_gpt_process
    while True:
        if main_gpt_process is None:
            # Start main.py
            start_main_gpt()
        try:
            # Listen for audio input
            spoken_text = speech_engine.get_audio(True)
            # print(f"[watchdog] Heard: {spoken_text}")
            
            # Check for exit keywords
            if "ems exit" in spoken_text or "ems cancel" in spoken_text or "ems stop" in spoken_text or "ems done" in spoken_text or "ems finish" in spoken_text:
                print("[watchdog] Exit keyword detected. Terminating main.py...")
                kill_main_gpt()
                main_gpt_process = None  # Reset the process variable
                # break  # Exit the watchdog loop
        except Exception as e:
            print(f"[watchdog] Error: {e}")
            time.sleep(1)

def get_GPT_direction(address, *args):
    global gpt_handedness, gpt_limb, gpt_direction
    # Process the OSC message from Unity
    print(f"Received from Unity: {address} {args}")
    handedness = args[0]
    limb = args[1]
    direction = args[2]
    return gpt_handedness, gpt_limb, gpt_direction

def get_stim_status(address, *args):
    global stim_on
    stim_on = args[0]
    print(f"Stim status: {stim_on}")


def get_unity_directions(address, *args):
    # Extract the limb name from the OSC address
    limb_name = str(address.split("/")[-1])
    # if args[0] != "still":
    #     print(f"Received from Unity: {address} {args}")

    if limb_name in skeleton_movement:
        limb = ""
        handedness = ""

        if limb_name.startswith("Right"):
            handedness = "right"
            limb = limb_name[5:].lower()  # Remove "Right" and convert the rest to lowercase
        elif limb_name.startswith("Left"):
            handedness = "left"
            limb = limb_name[4:].lower()  # Remove "Left" and convert the rest to lowercase
        
        if limb == "hand":
            limb = "wrist"

        elif limb == "forearm":
            limb = "elbow"
        elif limb == "upperarm" or limb == "arm":
            limb = "shoulder"
        else:
            limb = limb_name.lower()  # Convert the rest to lowercase



        # if still, ignore the data
        if args[0] != "still":
            skeleton_movement[limb] = args[0]
            if limb == "wrist":
                print(f"{handedness} {limb} {skeleton_movement[limb]}")
        elif limb_name in skeleton_movement:
            skeleton_movement[limb] = ""
            # print(f"No movement: {limb_name}. {handedness} {limb} {args[0]}")


def main():
    # Connect to the main server
    client = udp_client.SimpleUDPClient(GPT_OUT_IP, GPT_OUT_PORT)
    print("connected to main server at "+GPT_OUT_IP+":"+str(GPT_OUT_PORT))

    while True:
        if stim_on:
            # Compare the skeleton_movement with the gpt_handedness, gpt_limb, gpt_direction
            if gpt_handedness and gpt_limb and gpt_direction:
                # Get the current movement direction from Unity for the specified limb
                unity_direction = skeleton_movement.get(gpt_limb, {}).get("direction", None)

                # Check congruence
                if unity_direction != gpt_direction:
                    print(f"Halt: Mismatch detected! Main instructed {gpt_handedness} {gpt_limb} to move {gpt_direction}, but Unity reports {unity_direction}.")
                    client.send_message(gpt_out_address, ["halt", 0])  # Send a halt message to GPT
                else:
                    print(f"Match: Main and Unity agree on {gpt_handedness} {gpt_limb} moving {gpt_direction}.")
            else:
                print("Waiting for valid Main instructions...")

if __name__ == "__main__":
    # Initialize the SpeechEngine
    speech_engine = SpeechEngine()

    # # Start watchdog thread
    # watchdog_thread = threading.Thread(target=watchdog_main_gpt.serve_forever)
    # watchdog_thread.daemon = True  # Ensure the thread stops when the main program exits
    # watchdog_thread.start()
    # print("Watchdog started!")

    # Start the OSC server
    disp_unity = dispatcher.Dispatcher()
    disp_unity.map("/skeleton/movement/*", get_unity_directions) 
    osc_unity = osc_server.ThreadingOSCUDPServer((UNITY_IP, UNITY_PORT), disp_unity)
    osc_unity = threading.Thread(target=osc_unity.serve_forever)
    osc_unity.daemon = True  # Ensure the thread stops when the main program exits
    osc_unity.start()
    print("Unity OSC started!")

    # Start the main server
    disp_gpt = dispatcher.Dispatcher()
    disp_gpt.map("/instruction_direction", get_GPT_direction)  # Map OSC addresses to the store_rotation function
    disp_gpt.map("/stim_status", get_stim_status)  # Map OSC addresses to the store_rotation function
    osc_gpt = osc_server.ThreadingOSCUDPServer((GPT_IN_IP, GPT_IN_PORT), disp_gpt)
    osc_gpt = threading.Thread(target=osc_gpt.serve_forever)
    osc_gpt.daemon = True  # Ensure the thread stops when the main program exits
    osc_gpt.start()
    print("main OSC started!")


    # Start main user interface supervisor
    main()