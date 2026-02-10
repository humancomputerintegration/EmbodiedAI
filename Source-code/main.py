'''
this is the main script to run the Generative Muscle Stimulation system
it captures user input (image + audio), processes them using LLM to generate stimulation commands,
and sends the commands to the stimulation device
'''


import cv2
import time
from pynput import keyboard
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import pandas as pd
import base64
from speech_engine import SpeechEngine
from llm_gesture import LLMGesture
from camera_capture import CaptureCamera

handedness_detection = True
if handedness_detection:
    from hand_detector import HandDetector

import sys

USE_OFFLINE = False
LIVE_MIC    = True
HANDEDNESS_DETECTION_W_OBJECT = True

# ---- offline data for testing ---- #
# files to upload
image_file_offline = 'test_image_path.jpg'

# offline user prompt
offline_user_prompt = "help me"
# ---- offline data for testing ---- #


# alternative way to record user input with a keyboard press in case always listening live microphone does not work
def record_to_mp3(audio_file_name, sample_rate=44100):
    print("Recording... Press 'q' to stop.")
    
    stop_recording = False

    def on_press(key):
        nonlocal stop_recording
        try:
            if key.char == 'q':  # Press 'q' to stop recording
                stop_recording = True
                return False  # Stop listener
        except AttributeError:
            pass

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    audio_data = []

    def callback(indata, frames, time, status):
        if stop_recording:
            raise sd.CallbackStop
        audio_data.extend(indata.copy())

    with sd.InputStream(callback=callback, channels=1, samplerate=sample_rate, dtype='int16'):
        while not stop_recording:
            sd.sleep(100)

    listener.stop()

    print("Recording finished. Saving as WAV...")
    write(audio_file_name, sample_rate, np.array(audio_data))

def main():
    # init speech engine
    speech_engine = SpeechEngine()

    # initialize LLM gesture
    llm_gesture = LLMGesture()

    if handedness_detection:
        # init hand detector
        detector = HandDetector()

    # user_input = input("Press 'Enter' to continue with scene understanding module, '0' to exit: ")
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    user_prompt = ""

    # decide which image to use
    if USE_OFFLINE:
        if len(sys.argv) > 1:
            image_file_offline  = f'images/{sys.argv[1]}'
            user_prompt         = sys.argv[2]

        else:
            output_image_path   = image_file_offline 
            user_prompt = offline_user_prompt
        

    else:

        output_image_path   = f'user_inputs/{timestamp}_img.jpg'
        audio_file_path     = f'user_inputs/{timestamp}_audio.wav' # maybe text instead?

        # listen to audio and handle EMS settings
        if LIVE_MIC:  # always listening
            while user_prompt == "":
                heard = speech_engine.live_listening()
                if not heard:
                    continue
                h = heard.strip().lower()

                # Handle EMS settings commands
                if h == "exit":
                    return
                elif h.startswith("stimulation"):
                    try:
                        stimulation_mode = h.split("stimulation mode ", 1)[1].strip()
                        llm_gesture.set_stimulation_mode(stimulation_mode)
                        speech_engine.speak(f"Stimulation mode set to {llm_gesture.stimulation_mode}")
                    except Exception as e:
                        print(f"Error setting stimulation mode: {e}")
                    continue
                elif h.startswith("completion"):
                    try:
                        llm_gesture.completion_mode = h.split("completion mode ", 1)[1].strip()
                        speech_engine.speak(f"Completion mode set to {llm_gesture.completion_mode}")
                    except Exception as e:
                        print(f"Error setting completion mode: {e}")
                    continue
                elif h.startswith("user settings"):
                    try:
                        settings_text = h.split("user settings ", 1)[1].strip()
                        # Append to user profile consistently
                        if isinstance(llm_gesture.user_profile, list):
                            llm_gesture.user_profile.append(f"User settings: {settings_text}\n")
                        else:
                            llm_gesture.user_profile += f"\nUser settings: {settings_text}\n"
                            speech_engine.speak("User settings updated")
                    except Exception as e:
                        print(f"Error setting user settings: {e}")
                    continue
                # Otherwise treat as the actual task prompt
                else:
                    user_prompt = heard

        else: # this waits for keyboard press for recording
            record_to_mp3(audio_file_path)  # record audio from microphone
            user_prompt = speech_engine.mp3_to_text(audio_file_path) 

        camera = CaptureCamera('webcam')
        camera.capture_image(timestamp)
        if camera.camera is not None:
            camera.camera.release()
            cv2.destroyAllWindows()
        
    print(f"user prompt: {user_prompt}")

    # Path to files to upload
    image_path = output_image_path

    # detect handedness
    if handedness_detection:
        # handedness detection
        left_in_frame, right_in_frame = detector.detect_hands()
        print(f"Left hand in frame: {left_in_frame}, Right hand in frame: {right_in_frame}")

        if (left_in_frame == True) and (right_in_frame == False): 
            handedness = "left hand"
        elif (left_in_frame == False) and (right_in_frame == True):
            handedness = "right hand"
        else:
            handedness = "Analyze the image to determine the handedness."
    else:
        handedness = "Analyze the image to determine the handedness." # if not using mediapipe, let gpt analyze the handedness

    if HANDEDNESS_DETECTION_W_OBJECT:
        # detect object in hands
        if handedness_detection and ("left" in handedness or "right" in handedness):
            # handedness detection
            left_obj_in_hand, right_obj_in_hand = detector.detect_objects_in_hands()
            print(f"Left hand has: {left_obj_in_hand}, Right hand has object: {right_obj_in_hand}")

            if (left_obj_in_hand == True) and (right_obj_in_hand == False): # right hand is free; hence handedness is right hand (to manipulate the object)
                hands_obj = "object in right hand"
            elif (left_obj_in_hand == False) and (right_obj_in_hand == True):
                hands_obj = "object in left hand"
            else:
                hands_obj = "Analyze the image to determine if each hand holds an object."
        else:
            hands_obj = "object in right hand"
    else:
        hands_obj = None


    # convert image to base64
    with open(image_path, "rb") as img_file:
        image = base64.b64encode(img_file.read()).decode('utf-8')

    # LLM reasoning, constraining and EMS stimuation (nested function calls)
    gestures_response = llm_gesture.process_image_and_task(
        image_path,
        image,
        user_prompt,
        handedness,
        hands_obj
    )


if __name__ == "__main__":
    main()