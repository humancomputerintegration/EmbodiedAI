import sounddevice as sd
import numpy as np
import pyttsx3
import re
import speech_recognition as sr
import threading

class SpeechEngine:
    def __init__(self):
        # Initialize TTS engine
        self.engine = pyttsx3.init()

        self.recorder = sr.Recognizer()
        # self.recorder.energy_threshold = args.energy_threshold
        self.recorder.dynamic_energy_threshold = False

        self.mic = sr.Microphone()


    """Convert text to speech using pyttsx3"""
    def speak(self, text, blocking=True):
        if blocking:
            self.engine.say(text)
            # self.engine.save_to_file(text, "system-voice/user_confirm_tts.wav") # to save the TTS output
            self.engine.runAndWait()
        else:
            # Run the TTS operation in a separate thread, a bit finnicky
            threading.Thread(target=self._speak_non_blocking, args=(text,)).start()

    def _speak_non_blocking(self, text):
        """Helper method to handle non-blocking TTS"""
        self.engine.say(text)
        self.engine.runAndWait()
    
    def get_audio(self, print_status=True):
        # r = sr.Recognizer()
        if print_status: print("Listening...")
        with self.mic as source:
            audio = self.recorder.listen(source)
            said = ""

            try:
                said = self.recorder.recognize_google(audio)
                print(said)
            except Exception as e:
                print("Exception: " + str(e))

        return said.lower()
    
    """Real-time listening using speech recognition"""
    def live_listening(self):
        pattern = r"ems (.*)" # wake up word is "EMS"

        while True:
            user_prompt = self.get_audio()  # get_audio()
            up = user_prompt.strip().lower()

            if "exit" in up:
                self.speak("Goodbye!")
                return "exit"
            if "hello" in up:
                self.speak("Hello! How can I assist you?")
                continue

            stim = re.search(r"\b(?:ems[\s,]*)?stimulation\s*mode\s*:?\s*(actuate|nudge|tactile)\b", up, re.I)
            if stim:
                return f"stimulation mode {stim.group(1).lower()}"

            comp = re.search(r"\b(?:ems[\s,]*)?completion\s*mode\s*:?\s*(full|partial)\b", up, re.I)
            if comp:
                return f"completion mode {comp.group(1).lower()}"

            settings = re.search(r"\b(?:ems[\s,]*)?user\s*settings\s*:?\s*(.+)$", up, re.I | re.DOTALL)
            if settings:
                val = settings.group(1).strip().strip(" .,!?:;-")
                return f"user settings {val}"

            # EMS wake word: return payload after 'ems'
            match = re.search(pattern, up, re.DOTALL)
            if match:
                return match.group(1).strip()

    def mp3_to_text(self, audio_file_name):
        audio_file = sr.AudioFile(audio_file_name)

        with audio_file as source:
            audio = self.recorder.record(source)
        
        try:
            text = self.recorder.recognize_google(audio)
        except sr.UnknownValueError:
            print("Could not understand audio.")
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
        
        return text
