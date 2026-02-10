import cv2
import mediapipe as mp
import os
import logging
import absl.logging
from ultralytics import YOLO
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from roboflow import Roboflow
import supervision as sv


# # Suppress TensorFlow and MediaPipe warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
# logging.getLogger('tensorflow').setLevel(logging.ERROR)
# absl.logging.set_verbosity(absl.logging.ERROR)

class HandDetector:
    def __init__(self, min_detection_confidence=0.75, min_tracking_confidence=0.75):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=min_detection_confidence,
                                         min_tracking_confidence=min_tracking_confidence)
        self.mp_draw = mp.solutions.drawing_utils

        # load YOLO model
        self.yolo = YOLO("yolo12s.pt", verbose = False)  # "yolo12l.pt" for larger model

        # load handle model
        rf = Roboflow(api_key="YOUR-API-KEY-HERE")  # API for detecting door handles
        project = rf.workspace().project("door-handle-m41sa")
        self.roboflow_model = project.version(4).model

    # Function to get class colors for YOLO bounding boxes
    def getColours(self, cls_num):
        base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        color_index = cls_num % len(base_colors)
        increments = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]
        color = [base_colors[color_index][i] + increments[color_index][i] * 
                (cls_num // len(base_colors)) % 256 for i in range(3)]
        return tuple(color)

    # detect handedness from a static image
    def detect_hands_static(self, image_path):
        left_in_frame = False
        right_in_frame = False
        
        # Read and invert the image
        image = cv2.imread(image_path)
        image = cv2.flip(image, 1)
        
        # Convert the image color format (OpenCV uses BGR, MediaPipe expects RGB)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_image)
        
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handedness.classification[0].label  # 'Left' or 'Right'
                if label == 'Left':
                    left_in_frame = True
                elif label == 'Right':
                    right_in_frame = True
        
        return left_in_frame, right_in_frame
    
    # detect handedness from a CV frame
    def detect_hands_frame(self, frame):
        left_in_frame = False
        right_in_frame = False
        
        # Invert the image
        frame = cv2.flip(frame, 1)
        
        # Convert the image color format (OpenCV uses BGR, MediaPipe expects RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handedness.classification[0].label  # 'Left' or 'Right'
                if label == 'Left':
                    left_in_frame = True
                elif label == 'Right':
                    right_in_frame = True
        
        return left_in_frame, right_in_frame
    
    # detect handedness from a live video feed for a specified number of frames
    def detect_hands(self, max_frame = 10):
        cap = cv2.VideoCapture(0)  # Open webcam
        frame_count = 0
        left_hand_frames = []
        right_hand_frames = []

        while frame_count < max_frame:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame from webcam.")
                break

            left_in_frame, right_in_frame = self.detect_hands_frame(frame)
            left_hand_frames.append(left_in_frame)
            right_hand_frames.append(right_in_frame)
            frame_count += 1

        cap.release()

        # Calculate average handedness
        avg_left = sum(left_hand_frames) / len(left_hand_frames)
        avg_right = sum(right_hand_frames) / len(right_hand_frames)

        # Determine if the hand is in the frame based on the average
        left_in_frame = avg_left > 0.75  # More than X% of the frames detect the left hand
        right_in_frame = avg_right > 0.75  # More than X% of the frames detect the right hand

        return left_in_frame, right_in_frame

    def detect_handle_onetime(self, display_camera=False, confidence=50, overlap=50):
        cap = cv2.VideoCapture(0)  # Open webcam

        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from webcam.")
            return

        cap.release()

        result = self.roboflow_model.predict(frame, confidence=confidence, overlap=overlap).json()

        if display_camera:
            labels = [item["class"] for item in result["predictions"]]

            detections = sv.Detections.from_inference(result)

            label_annotator = sv.LabelAnnotator()
            bounding_box_annotator = sv.BoxAnnotator()

            image = cv2.imread(frame)

            annotated_image = bounding_box_annotator.annotate(
                scene=image, detections=detections)
            annotated_image = label_annotator.annotate(
                scene=annotated_image, detections=detections, labels=labels)

            sv.plot_image(image=annotated_image, size=(16, 16))

        return result

    def detect_objects_in_hands(self, max_frames=10, display_camera=False):
        cap = cv2.VideoCapture(0)  # Open webcam
        frame_count = 0
        detection_confidence = 0.6

        left_h_obj_frames = []
        right_h_obj_frames = []

        while frame_count < max_frames:
            frame_count += 1
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame from webcam.")
                break

            frame = cv2.flip(frame, 1)  # Flip for mirror effect
            h, w, _ = frame.shape

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)  # Process hands with MediaPipe
            yolo_results = self.yolo(frame, verbose = False)[0]  # verbose displays speed and other info

            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    label = handedness.classification[0].label  # 'Left' or 'Right'

                    # Get bounding box of the hand
                    landmark_x = [int(landmark.x * w) for landmark in hand_landmarks.landmark]
                    landmark_y = [int(landmark.y * h) for landmark in hand_landmarks.landmark]
                    
                    min_x, max_x = min(landmark_x), max(landmark_x)
                    min_y, max_y = min(landmark_y), max(landmark_y)
                    
                    hand_bbox = (min_x, min_y, max_x, max_y)  # Hand bounding box

                    # Find the best matching object
                    closest_iou = 0
                    closest_box = None

                    for box in yolo_results.boxes:
                        if box.conf[0] > detection_confidence and box.cls[0] != 0:  # Exclude 'person'
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            obj_bbox = (int(x1), int(y1), int(x2), int(y2))

                            iou = self.compute_iou(hand_bbox, obj_bbox)  # Calculate IoU

                            if iou > closest_iou:
                                closest_iou = iou
                                closest_box = box

                    # If IoU is above threshold, consider object "held"
                    iou_threshold = 0.2  # Adjust based on testing
                    if closest_iou > iou_threshold:
                        if label == 'Left':
                            left_h_obj_frames.append(True)
                        elif label == 'Right':
                            right_h_obj_frames.append(True)
                    else:
                        if label == 'Left':
                            left_h_obj_frames.append(False)
                        elif label == 'Right':
                            right_h_obj_frames.append(False)

                    # Visualization
                    if display_camera:
                        # Draw hand landmarks
                        self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                        if closest_box:
                            x1, y1, x2, y2 = closest_box.xyxy[0].cpu().numpy()
                            class_name = self.yolo.model.names[int(closest_box.cls[0])]
                            color = self.getColours(int(closest_box.cls[0]))

                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
                            cv2.putText(frame, f'{class_name} {closest_box.conf[0]:.2f}', (int(x1), int(y1) - 10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            cv2.putText(frame, f"IoU: {closest_iou:.2f}", (int(x1), int(y2) + 15), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            if display_camera:
                cv2.imshow('Hand-Object Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        print(f"Left hand object frames: {left_h_obj_frames}")
        print(f"Right hand object frames: {right_h_obj_frames}")

        # Compute averages for hand presence in frames
        left_avg = sum(left_h_obj_frames) / len(left_h_obj_frames) if left_h_obj_frames else 0
        right_avg = sum(right_h_obj_frames) / len(right_h_obj_frames) if right_h_obj_frames else 0

        # Determine if the hand is in the frame based on the average
        left_in_frame = left_avg > 0.75  # More than X% of the frames detect the left hand
        right_in_frame = right_avg > 0.75  # More than X% of the frames detect the right hand

        return left_in_frame, right_in_frame # Threshold for determining object possession

    def detect_handle_in_hands(self, max_frames=10, display_camera=False):
        cap = cv2.VideoCapture(0)  # Open webcam
        frame_count = 0
        detection_threshold = 60  # Using percentage threshold for Roboflow

        left_h_obj_frames = []
        right_h_obj_frames = []

        while frame_count < max_frames:
            frame_count += 1
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame from webcam.")
                break

            frame = cv2.flip(frame, 1)  # Flip for mirror effect
            h, w, _ = frame.shape

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)  # Process hands with MediaPipe

            # Get predictions from Roboflow's model (assumes self.roboflow_model is set up)
            # overlap threshold can change
            rf_result = self.roboflow_model.predict(frame, confidence=detection_threshold, overlap=60).json()
            predictions = rf_result.get("predictions", [])
            
            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    label = handedness.classification[0].label  # 'Left' or 'Right'

                    # Get bounding box of the hand using landmarks
                    landmark_x = [int(landmark.x * w) for landmark in hand_landmarks.landmark]
                    landmark_y = [int(landmark.y * h) for landmark in hand_landmarks.landmark]
                    min_x, max_x = min(landmark_x), max(landmark_x)
                    min_y, max_y = min(landmark_y), max(landmark_y)
                    hand_bbox = (min_x, min_y, max_x, max_y)  # Hand bounding box

                    closest_iou = 0
                    closest_detection = None

                    # Iterate through each prediction from Roboflow
                    for detection in predictions:
                        if detection["class"].lower() == "person":
                            continue

                        # Convert center-based bbox (from Roboflow) to corner format
                        center_x = detection["x"]
                        center_y = detection["y"]
                        box_w = detection["width"]
                        box_h = detection["height"]
                        x1 = int(center_x - box_w / 2)
                        y1 = int(center_y - box_h / 2)
                        x2 = int(center_x + box_w / 2)
                        y2 = int(center_y + box_h / 2)
                        obj_bbox = (x1, y1, x2, y2)

                        # Calculate the Intersection over Union (IoU)
                        iou = self.compute_iou(hand_bbox, obj_bbox)
                        if iou > closest_iou:
                            closest_iou = iou
                            closest_detection = detection

                    # Determine if the object held is significant based on IoU threshold
                    iou_threshold = 0.2  # Adjust as needed based on your testing
                    if closest_iou > iou_threshold:
                        if label == 'Left':
                            left_h_obj_frames.append(True)
                        elif label == 'Right':
                            right_h_obj_frames.append(True)
                    else:
                        if label == 'Left':
                            left_h_obj_frames.append(False)
                        elif label == 'Right':
                            right_h_obj_frames.append(False)

                    # Visualization if display_camera is True
                    if display_camera:
                        # Draw hand landmarks
                        self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                        if closest_detection:
                            center_x = closest_detection["x"]
                            center_y = closest_detection["y"]
                            box_w = closest_detection["width"]
                            box_h = closest_detection["height"]
                            x1 = int(center_x - box_w / 2)
                            y1 = int(center_y - box_h / 2)
                            x2 = int(center_x + box_w / 2)
                            y2 = int(center_y + box_h / 2)
                            class_name = closest_detection["class"]
                            conf_val = closest_detection["confidence"]
                            # Get a color based on the class or any other logic
                            # (Here, passing 0 to getColours; modify if you have a mapping for class names)
                            color = self.getColours(0)

                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                            cv2.putText(frame, f'{class_name} {conf_val:.2f}', (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            cv2.putText(frame, f"IoU: {closest_iou:.2f}", (x1, y2 + 15),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            if display_camera:
                cv2.imshow('Hand-Object Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()

        print(f"Left hand object frames: {left_h_obj_frames}")
        print(f"Right hand object frames: {right_h_obj_frames}")

        # Compute averages for hand presence in frames
        left_avg = sum(left_h_obj_frames) / len(left_h_obj_frames) if left_h_obj_frames else 0
        right_avg = sum(right_h_obj_frames) / len(right_h_obj_frames) if right_h_obj_frames else 0

        # Determine if the hand is in the frame based on the average detection rate
        left_in_frame = left_avg > 0.75  # More than 75% of the frames detect an object with left hand
        right_in_frame = right_avg > 0.75  # More than 75% of the frames detect an object with right hand

        return left_in_frame, right_in_frame
        
    def detect_objects_in_hands_mediapipe(self, max_frames=10, display_camera=False):
        # Set desired dimensions for the webcam frame.
        DESIRED_HEIGHT = 480
        DESIRED_WIDTH = 480
        
        # Define the colormap for segmentation (BGR format). Class 5 corresponds to objects.
        colormap = np.array([
            [192, 192, 192],  # 0 - background (gray)
            [255, 0, 0],      # 1 - hair (blue)
            [0, 255, 0],      # 2 - body-skin (green)
            [0, 0, 255],      # 3 - face-skin (red)
            [0, 255, 255],    # 4 - clothes (yellow)
            [255, 0, 255]     # 5 - others (accessories / object) (pink)
        ], dtype=np.uint8)
        
        # Setup the MediaPipe segmentation model.
        with open('selfie_multiclass_256x256.tflite', 'rb') as f:
            model = f.read()

        base_options = python.BaseOptions(model_asset_buffer=model)
        options = vision.ImageSegmenterOptions(
            base_options=base_options,
            output_category_mask=True
        )
        segmenter = vision.ImageSegmenter.create_from_options(options)
        
        # Setup the Mediapipe Hands detector.
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5
        )
        
        # Open the webcam. (Change the index if necessary)
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return None, None
        
        # Arrays to store 1/0 scores for each frame.
        left_h_obj_frames = []
        right_h_obj_frames = []
        
        frame_count = 0
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame from webcam.")
                break
            
            # Resize the frame to the desired dimensions.
            frame = cv2.resize(frame, (DESIRED_WIDTH, DESIRED_HEIGHT))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run the segmentation model
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            segmentation_result = segmenter.segment(image)
            # Obtain the segmentation mask as an integer array.
            mask = segmentation_result.category_mask.numpy_view().astype(np.int32)
            
            # Run the hand detection model
            results = hands.process(frame_rgb)
            
            # Initialize the interaction flags for this frame.
            interacting_left = False
            interacting_right = False
            
            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):
                    hand_label = hand_info.classification[0].label
                    # Since the detected "Left" corresponds to the actual right hand, swap the labels.
                    if hand_label == "Left":
                        corrected_label = "Right"
                    elif hand_label == "Right":
                        corrected_label = "Left"
                    else:
                        corrected_label = hand_label
                    
                    # Check each hand landmark. If any landmark falls on a segmentation pixel of class 5 ("object"),
                    # mark that hand as interacting.
                    for lm in hand_landmarks.landmark:
                        x_coord = int(lm.x * DESIRED_WIDTH)
                        y_coord = int(lm.y * DESIRED_HEIGHT)
                        if x_coord < 0 or x_coord >= DESIRED_WIDTH or y_coord < 0 or y_coord >= DESIRED_HEIGHT:
                            continue
                        if mask[y_coord, x_coord] == 5:
                            if corrected_label == "Left":
                                interacting_left = True
                            elif corrected_label == "Right":
                                interacting_right = True
                            # Once an interaction is found for a hand, stop further checks.
                            break
            
            # Append the results for this frame
            # For each frame, append 1 if interaction is detected; otherwise, 0.
            left_h_obj_frames.append(1 if interacting_left else 0)
            right_h_obj_frames.append(1 if interacting_right else 0)
            
            # Optionally display the segmentation with overlaid status text.
            if display_camera:
                if interacting_left and interacting_right:
                    text = "both hands are interacting with object"
                elif interacting_left:
                    text = "left hand is interacting with object"
                elif interacting_right:
                    text = "right hand is interacting with object"
                else:
                    text = "no hands are interacting with object"
                
                # Create an output image by mapping the segmentation mask to the colormap.
                output_image = colormap[mask]
                cv2.putText(output_image, text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.imshow("Webcam Segmentation", output_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_count += 1
        
        # Release resources.
        cap.release()
        hands.close()
        cv2.destroyAllWindows()
        
        print(f"Left hand object frames: {left_h_obj_frames}")
        print(f"Right hand object frames: {right_h_obj_frames}")
        
        # Compute average scores and determine interactions
        left_avg = sum(left_h_obj_frames) / len(left_h_obj_frames) if left_h_obj_frames else 0
        right_avg = sum(right_h_obj_frames) / len(right_h_obj_frames) if right_h_obj_frames else 0
        
        left_in_frame = left_avg > 0.75  # e.g. if more than 75% of the frames have left-hand interaction.
        right_in_frame = right_avg > 0.75  # e.g. if more than 75% of the frames have right-hand interaction.
        
        return left_in_frame, right_in_frame
    
    def compute_iou(self, box1, box2):
        """
        Computes Intersection over Union (IoU) between two bounding boxes.
        
        box1, box2: (x1, y1, x2, y2) - Bounding boxes in pixel coordinates.
        Returns: IoU value (0 to 1).
        """
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])

        # Compute intersection area
        inter_width = max(0, x2_inter - x1_inter)
        inter_height = max(0, y2_inter - y1_inter)
        intersection = inter_width * inter_height

        # Compute areas of both boxes
        area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        # Compute union
        union = area_box1 + area_box2 - intersection

        # Avoid division by zero
        if union == 0:
            return 0

        return intersection / union  # IoU value (between 0 and 1)
    
