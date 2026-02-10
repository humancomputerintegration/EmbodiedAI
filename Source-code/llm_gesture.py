import base64
import time
from openai import OpenAI
import pandas as pd
import cv2
import geocoder
from speech_engine import SpeechEngine
from gesture_processor import GestureProcessor
from mocap_oscserver import MocapOSCServer
import os
import re
from prompts import get_checkpoints_prompts, get_gestures_oneshot_prompt, get_recognition_prompt, get_movements_prompt, get_gestures_prompt

client = OpenAI(api_key="YOUR-API-KEY-HERE")
llm_model = "gpt-4.1"

class LLMGesture:
    def __init__(self):
        self.ems_joint_limits = pd.read_csv('gesture_lists/ems-joint-limits.csv', skiprows=[0])
        self.joint_list = self.ems_joint_limits.iloc[:, :3].values.tolist()

        # Load user profile lines -- user_profile_1.txt is a made-up profile for demo purposes.
        # We used this to showcase how location allows Multimodal-AI to find the type of window from context clues. 
        # We chose a city unrelated to any author or institution, but we needed a real city here, where these windows are common.
        with open("user_profile/user_profile_1.txt", "r", encoding="utf-8") as f:
            self.user_profile = f.readlines()

        # check if location is in user profile, otherwise use geocoder to get location based on IP
        try:
            has_location = any(re.search(r'^location\s*:', line.strip(), re.IGNORECASE) for line in self.user_profile)
            if not has_location:
                self.user_profile.append(self.get_location())
        except Exception:
            pass  

        self.filename = "user_inputs/llm_output.txt" # for logs

        self.speech_engine = SpeechEngine()
        self.skeleton = MocapOSCServer()
        self.gesture_processor = GestureProcessor()

        # set default settings
        self.stimulation_mode = "actuate"
        self.completion_mode = "full"

    def process_image_and_task(self, path, image, task, handedness, hands_obj, skeleton_direction=None, user_profile=None, checkpoints=False, load_cache=False):
        # if no user profile is provided, use the default one
        if user_profile is None:
            user_profile = self.user_profile

        if skeleton_direction is None:
            skeleton_direction = self.skeleton.skeleton_direction

        with open(self.filename, 'a', encoding='utf-8') as f :
            f.write(f"Input user request: \n {task}\n\n")

        with open(self.filename, 'a', encoding='utf-8') as f :
            f.write(f"Handedness: \n {handedness}\n\n")

        with open(self.filename, 'a', encoding='utf-8') as f :
            f.write(f"Stimulation mode: \n {self.stimulation_mode}\n\n")

        with open(self.filename, 'a', encoding='utf-8') as f :
            f.write(f"completion mode: \n {self.completion_mode}\n\n")

        with open(self.filename, 'a', encoding='utf-8') as f :
            f.write(f"user settings: \n {user_profile}\n\n")

         
        if load_cache:
            last_cached_output = self.load_cache_output("test-results/full-demo/")
            print("Try again with prior failed reasoning")
        else:
            last_cached_output = "No prior failed reasoning result. Start fresh."
            print(last_cached_output)

        with open(self.filename, 'a', encoding='utf-8') as f :
            f.write("LLM output")


        # Calling each LLM modules here
        print("Recognition:")
        recognition_response = self.recognize_object(image, task, handedness, hands_obj, user_profile, last_cached_output)
        print(recognition_response)
        with open(self.filename, 'a', encoding='utf-8') as f :
            f.write(f"LLM output (recognition): \n {recognition_response}\n\n")

        print("Movements:")
        movements_response = self.generate_movements(recognition_response, skeleton_direction, user_profile, task, last_cached_output)
        print(movements_response)
        with open(self.filename, 'a', encoding='utf-8') as f :
            f.write(f"LLM output (movement): \n {movements_response}\n\n")

        print("Gesture:")
        gestures_response = self.generate_gestures_oneshot(movements_response)
        print(gestures_response)

        with open(self.filename, 'a', encoding='utf-8') as f :
            f.write(f"input pose (skeleton directions): \n {skeleton_direction}\n\n")

        with open(self.filename, 'a', encoding='utf-8') as f :
            f.write(f"LLM output (gesture without checkpoint): \n {gestures_response}\n\n")

        if checkpoints:
            user_comfirmation = ""
            control_state = "running"
            
            while user_comfirmation == "":
                user_comfirmation = self.speech_engine.live_listening()
            
            if "continue" in user_comfirmation:
                control = True
                count = 2
            elif "stop" in user_comfirmation:
                control = False
                control_state = "stopped"
                self.speech_engine.speak("Stopping EMS assistance")
                with open(self.filename, 'a', encoding='utf-8') as f:
                    f.write(f"User command: STOP - EMS assistance terminated\n\n")
                return False
            elif "pause" in user_comfirmation:
                control = True
                control_state = "paused"
                self.speech_engine.speak("EMS assistance paused. Say continue to resume")
            elif "restart" in user_comfirmation:
                control = True
                control_state = "restarting"
                self.speech_engine.speak("Restarting EMS assistance from the beginning")
                return self.process_image_and_task(path, image, task, handedness, hands_obj, 
                                                    skeleton_direction, checkpoints=checkpoints, 
                                                    load_cache=False)

            prev = ""
            
            steps = list(map(str.strip, movements_response.strip().split("\n"))) 
            steps = [step.split(". ")[-1] for step in steps]

            repeat_times = 0

            while control:
                if control_state == "paused":
                    self.speech_engine.speak("System paused. Say ems continue to resume, ems stop to exit, or ems restart to begin again")
                    pause_command = ""
                    while pause_command == "":
                        pause_command = self.speech_engine.live_listening()
                    
                    if "continue" in pause_command:
                        control_state = "running"
                        self.speech_engine.speak("Resuming EMS assistance")
                    elif "stop" in pause_command:
                        control = False
                        self.speech_engine.speak("Stopping EMS assistance")
                        with open(self.filename, 'a', encoding='utf-8') as f:
                            f.write(f"User command: STOP (from pause) - EMS assistance terminated\n\n")
                        return False
                    elif "restart" in pause_command:
                        self.speech_engine.speak("Restarting EMS assistance from the beginning")
                        return self.process_image_and_task(path, image, task, handedness, hands_obj, 
                                                            skeleton_direction, checkpoints=checkpoints, 
                                                            load_cache=False)
                    elif "pause" in pause_command:
                        continue
                    
                if control_state == "running":
                    self.speech_engine.speak("Checkpoint in 3")
                    time.sleep(1)
                    self.speech_engine.speak("2")
                    time.sleep(1)
                    self.speech_engine.speak("1")
                    time.sleep(1)
                    
                    if path != "images/image1.jpg":
                        path = "images/image" + str(count) + ".jpg"
                        path = self.capture_image_oneshot(path)
                        count += 1
                        with open(path, "rb") as img_file:
                            image = base64.b64encode(img_file.read()).decode('utf-8')
                    else:
                        with open(path, "rb") as img_file:
                            image = base64.b64encode(img_file.read()).decode('utf-8')
                        path = "images/image2.jpg"

                    print("Checkpoints:")

                    checkpoints_response = self.generate_checkpoints(image, task, movements_response, handedness)
                    print(checkpoints_response)

                    if repeat_times >= 5:
                        if self.completion_mode == "partial":
                                step_index = steps.index(checkpoints_response.strip().split("\n")[-1])
                                print("step index: ", step_index)
                                if step_index + 1 == len(steps):
                                    checkpoints_response = "done"
                                elif not steps[step_index + 1]:
                                    checkpoints_response = steps[step_index + 5]
                                else:
                                    checkpoints_response = steps[step_index + 1]
                            
                        elif self.completion_mode == "full":
                            self.speech_engine.speak("EMS assistance requires all instructions to be followed. Please perform the gesture as instructed" + checkpoints_response)
                            return False
                    else:
                        if prev == checkpoints_response:
                            repeat_times += 1
                        else:
                            repeat_times = 0
                            prev = checkpoints_response

                    with open(self.filename, 'a', encoding='utf-8') as f:
                        f.write(f"LLM output (checkpoint): \n {checkpoints_response}\n\n")      
                    
                    checkpoints_response = checkpoints_response.lower()
                    if (checkpoints_response.strip() == "done") or "done" in checkpoints_response:
                        control = False
                        return False
                    
                    print("Gesture:")
                    gestures_response = self.generate_gestures(checkpoints_response)
                    print(gestures_response)

                    with open(self.filename, 'a', encoding='utf-8') as f:
                        f.write(f"LLM output (gesture): \n {gestures_response}\n\n")
                        
                    if "skip" in gestures_response:
                        if self.completion_mode == "partial":
                            while "skip" in gestures_response:
                                self.speech_engine.speak("please do the following gesture by yourself" + checkpoints_response)

                                step_index = steps.index(checkpoints_response.strip().split("\n")[-1])
                                print("step index: ", step_index)
                                if step_index + 1 == len(steps):
                                    checkpoints_response = "done"
                                    control = False
                                    return False
                                elif not steps[step_index + 1]:
                                    checkpoints_response = steps[step_index + 5]
                                else:
                                    checkpoints_response = steps[step_index + 1]
                                
                                print("Gesture:")
                                gestures_response = self.generate_gestures(checkpoints_response)
                                print(gestures_response)

                        elif self.completion_mode == "full":
                            self.speech_engine.speak("Please perform the gesture as instructed" + checkpoints_response)

                    self.speech_engine.speak("Ready to stimulate. Say ems continue to proceed, ems pause to pause, ems stop to stop, or ems restart to begin again")

                    control_command = ""
                    start_time = time.time()
                    timeout = 5
                    
                    while control_command == "" and (time.time() - start_time) < timeout:
                        control_command = self.speech_engine.live_listening()
                    
                    if control_command == "":
                        control_command = "continue"
                    
                    if "pause" in control_command:
                        control_state = "paused"
                        with open(self.filename, 'a', encoding='utf-8') as f:
                            f.write(f"User command: PAUSE - System paused before stimulation\n")
                        continue
                    elif "stop" in control_command:
                        control = False
                        self.speech_engine.speak("Stopping EMS assistance")
                        with open(self.filename, 'a', encoding='utf-8') as f:
                            f.write(f"User command: STOP - EMS assistance terminated before stimulation\n\n")
                        return False
                    elif "restart" in control_command:
                        self.speech_engine.speak("Restarting EMS assistance from the beginning")
                        return self.process_image_and_task(path, image, task, handedness, hands_obj, 
                                                            skeleton_direction, checkpoints=checkpoints, 
                                                            load_cache=False)
                    elif "continue" in control_command or control_command == "continue":
                        ems_params = self.gesture_processor.process_instructions(gestures_response)
                        print("EMS params", ems_params)
                        with open(self.filename, 'a', encoding='utf-8') as f:
                            f.write("END OF LLM")
                        with open(self.filename, 'a', encoding='utf-8') as f:
                            f.write(f"EMS params: \n {ems_params}\n\n")

                        if "Timeout" in ems_params:
                            self.speech_engine.speak("Gesture timeout occurred")
                            if self.completion_mode == "partial":
                                while "Timeout" in ems_params:
                                    self.speech_engine.speak("please do the following gesture by yourself" + checkpoints_response)

                                    step_index = steps.index(checkpoints_response.strip().split("\n")[-1])
                                    print("step index: ", step_index)
                                    if step_index + 1 == len(steps):
                                        checkpoints_response = "done"
                                        control = False
                                        return False
                                    elif not steps[step_index + 1]:
                                        checkpoints_response = steps[step_index + 5]
                                    else:
                                        checkpoints_response = steps[step_index + 1]
                                    
                                    print("Gesture:")
                                    gestures_response = self.generate_gestures(checkpoints_response)
                                    print(gestures_response)

                                    # constraining and EMS stimuation
                                    ems_params = self.gesture_processor.process_instructions(gestures_response)
                                    print("EMS params", ems_params)
                                    with open(self.filename, 'a', encoding='utf-8') as f:
                                        f.write("END OF LLM")
                                    with open(self.filename, 'a', encoding='utf-8') as f:
                                        f.write(f"EMS params: \n {ems_params}\n\n")
                                        
                            elif self.completion_mode == "full":
                                self.speech_engine.speak("EMS assistance requires all instructions to be followed. Please perform the gesture as instructed" + checkpoints_response)

        else:
            with open(self.filename, 'a') as f:
                f.write("END OF LLM \n\n")

            # constraining and EMS stimuation
            ems_params = self.gesture_processor.process_instructions(gestures_response)
            print("EMS params", ems_params)

            with open(self.filename, 'a') as f:
                f.write(f"EMS params: \n {ems_params}\n\n")

        return recognition_response, movements_response, gestures_response
        
    
    def capture_image_oneshot(self, output_path="image.jpg"):
        """Capture an image from the webcam and save it to a file."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise Exception("Could not open webcam")
    
        print("Capturing image...")
        ret, frame = cap.read()
        if not ret:
            raise Exception("Failed to capture frame from webcam")

        cv2.imwrite(output_path, frame)
        print(f"Image captured and saved to {output_path}")
    
        cap.release()
        cv2.destroyAllWindows()
        return output_path
    
    def capture_image(self, output_path="image.jpg"):
        """Capture an image from the webcam and save it to a file."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise Exception("Could not open webcam")
 
        print("Press 'Space' to capture an image or 'Esc' to exit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                raise Exception("Failed to capture frame from webcam")
 
            cv2.imshow("Webcam", frame)
            key = cv2.waitKey(1)
 
            if key == 27:
                print("Exiting without capturing.")
                cap.release()
                cv2.destroyAllWindows()
                return None
            elif key == 32:
                cv2.imwrite(output_path, frame)
                print(f"Image captured and saved to {output_path}")
                break
 
        cap.release()
        cv2.destroyAllWindows()
        return output_path

    def recognize_object(self, image, task, handedness, hands_obj, user_profile, cached_output=""):
        print("LLM recognizing object...")
        
        message_prompt = get_recognition_prompt(image, task, handedness, hands_obj, user_profile, cached_output)

        recognition = client.chat.completions.create(
            model=llm_model,
            messages=message_prompt
        )

        return recognition.choices[0].message.content

    def generate_movements(self, recognition_response, limb_directions, user_profile, task, cached_output= ""):
        print("LLM planning movement...")

        message_prompt = get_movements_prompt(recognition_response, limb_directions, user_profile, cached_output, task=task)

        movements = client.chat.completions.create(
            model=llm_model,
            messages = message_prompt

        )

        return movements.choices[0].message.content
    
    def generate_checkpoints(self, image, task, movements_response, handedness):
        print("LLM generating checkpoint...")
        
        message_prompt = get_checkpoints_prompts(image, task, movements_response, handedness)
        
        checkpoints =  client.chat.completions.create(
            model=llm_model,
            messages = message_prompt
        )
      
        return "checkpoint: \n" + checkpoints.choices[0].message.content

    def generate_gestures(self, checkpoints_response, limb_directions=None):
        print("LLM analyzing gesture...")

        message_prompt = get_gestures_prompt(checkpoints_response, limb_directions, self.joint_list)

        gestures = client.chat.completions.create(
            model=llm_model,
            messages= message_prompt 
        )
        
        return "llm output gestures: \n" + gestures.choices[0].message.content

    def generate_gestures_oneshot(self, movement_response, limb_directions=None):
        print("LLM analyzing gesture (without checkpoint)...")

        message_prompt = get_gestures_oneshot_prompt(movement_response, self.joint_list, limb_directions)

        gestures = client.chat.completions.create(
            model=llm_model,
            messages= message_prompt
        )
        
        return "llm output gestures (without checkpoint): \n" + gestures.choices[0].message.content
    
    def load_cache_output(self, path):
        files = os.listdir(path)
        files.sort()
        latest_file = files[-2]
        print("latest file: ", latest_file)
        with open(os.path.join(path, latest_file), 'r') as file:
            lines = file.readlines()
            if "END OF LLM \n" in lines:
                end_of_llm_index = lines.index("END OF LLM \n")
                llm_cache_ouput = lines[:end_of_llm_index]
                llm_cache_ouput = '\n'.join(llm_cache_ouput)
            else:
                llm_cache_ouput = '\n'.join(lines)
        return llm_cache_ouput
    
    def set_stimulation_mode(self, mode):
        if mode in ["actuate", "nudge", "tactile"]:
            self.stimulation_mode = mode

            if mode == "actuate":
                self.gesture_processor.set_ems_power(1)
            elif mode == "nudge": 
                self.gesture_processor.set_ems_power(0.75) # 75% power -- tune accordingly
            elif mode == "tactile":
                self.gesture_processor.set_ems_power(0.5) # 50% power -- tune accordingly
        else:
            raise ValueError("Invalid stimulation mode. Choose from 'actuate', 'nudge', or 'tactile'.")

    def get_location(self):
        try:
            g = geocoder.ip('me') 
            parts = []
            for attr in ('city', 'state', 'country'):
                val = getattr(g, attr, None)
                if val:
                    parts.append(val)
            if parts:
                location_line = "location: " + ", ".join(parts) + "\n"
            else:
                location_line = "location: Unknown\n"
        except Exception:
            location_line = "location: Unknown\n"
        return location_line
