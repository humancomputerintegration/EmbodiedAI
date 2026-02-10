import subprocess
from pynput import keyboard
import cv2

class CaptureCamera:
    def __init__(self, cameratype):
        self.capture = False
        self.cameratype = cameratype
        self.camera = None
        if self.cameratype == 'webcam':
            self.camera = cv2.VideoCapture(0)

    def capture_image(self, timestamp):
        if self.cameratype == 'rayban':
            self.capture_rayban()
        elif self.cameratype == 'webcam':
            self.webcam_capture_image(timestamp)
    
    def capture_image_oneshot(self, output_path="image.jpg"):
        if self.cameratype == 'rayban':
            self.capture_rayban()
        elif self.cameratype == 'webcam':
            self.webcam_capture_image_oneshot(output_path=output_path)

    
    def webcam_capture_image(self, timestamp):
        """Capture an image from the webcam and save it to a file."""
        import time
        if self.camera is None or not self.camera.isOpened():
            # Attempt one re-open
            self.camera = cv2.VideoCapture(0)
            if self.camera is None or not self.camera.isOpened():
                raise Exception("Could not open webcam")

        warmup_frames = 2
        last_frame = None
        for _ in range(warmup_frames):
            ret, frame = self.camera.read()
            if ret:
                last_frame = frame
            time.sleep(0.005)

        ret, frame = self.camera.read()
        if not ret:
            if last_frame is not None:
                frame = last_frame
            else:
                raise Exception("Failed to capture frame from webcam")

        out_path = f"user_inputs/{timestamp}_img.jpg"
        img = cv2.imwrite(out_path, frame)
        if not img:
            raise Exception(f"Failed to write image to {out_path}")
        print("Image captured.")
    
    def webcam_capture_image_oneshot(self, output_path):
        """Capture an image from the webcam and save it to a file."""
        import time
        if self.camera is None or not self.camera.isOpened():
            self.camera = cv2.VideoCapture(0)
            if self.camera is None or not self.camera.isOpened():
                raise Exception("Could not open webcam")

        print("Capturing image (oneshot)...")
        # Warm-up
        for _ in range(5):
            self.camera.read()
            time.sleep(0.04)

        ret, frame = self.camera.read()
        if not ret:
            raise Exception("Failed to capture frame from webcam")

        # Brightness retry
        tries = 0
        while frame.mean() < 5 and tries < 5:
            time.sleep(0.08)
            ret, frame2 = self.camera.read()
            if ret:
                frame = frame2
            tries += 1

        ok = cv2.imwrite(output_path, frame)
        if not ok:
            raise Exception(f"Failed to write image to {output_path}")
        print(f"Image captured and saved to {output_path}")
        return output_path



    def capture_rayban(self):
        outfile = 'user_inputs/rayban_stream.jpg'
        try:
            subprocess.run(['screencapture', '-R450,64,600,966', f'{outfile}'])
        except subprocess.CalledProcessError as e:
            print('Python error: [%d]\n{}\n'.format(e.returncode, e.output))

    def on_press(self, key, injected):
        try:
            print('alphanumeric key {} pressed; it was {}'.format(
                key.char, 'faked' if injected else 'not faked'))
            print(type(key.char))

        except AttributeError:
            print('special key {} pressed'.format(
                key))
        

    def on_release(self, key, injected):
        print('{} released; it was {}'.format(
            key, 'faked' if injected else 'not faked'))
        if key == keyboard.Key.space:
            self.capture = True
            # Stop listener
            return False

if __name__ == "__main__":
    camera = CaptureCamera('webcam')
    camera.capture_image("test")