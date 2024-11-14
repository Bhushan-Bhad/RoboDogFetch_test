#!/usr/bin/env python3

import rospy
import os
import sys
import platform
import ctypes
import numpy as np
import cv2
import mediapipe as mp
from ultralytics import YOLO
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from distance_calc_MonocularDepthEstimation import estimate_depth

# Known real-world sizes (in meters)
HUMAN_HEIGHT = 1.7  # Approximate average height of a human in meters
BALL_DIAMETER = 0.24  # Diameter of a soccer ball in meters

# Camera calibration parameters (adjust according to your camera calibration)
FOCAL_LENGTH = 600  # This is a placeholder, adjust based on your camera calibration

# Function to initialize the YOLO model
def initialize_model():
    return YOLO("yolo-Weights/yolov8n.pt")  # Ensure the correct path to your YOLOv8 weights

# Function to estimate distance using the bounding box height
def estimate_distance(focal_length, real_height, pixel_height):
    if pixel_height == 0:  # To avoid division by zero
        return float('inf')
    return (focal_length * real_height) / pixel_height

# Main function for human, sports ball detection with hand gesture recognition and distance calculation
def human_detection():
    # Force libgomp to be loaded before other libraries consuming dynamic TLS
    if platform.system() == "Linux":
        try:
            ctypes.cdll.LoadLibrary("/usr/lib/aarch64-linux-gnu/libgomp.so.1")
        except OSError as e:
            rospy.logerr(f"Error loading libgomp: {e}")

    # Initialize ROS node
    rospy.init_node('object_detection', anonymous=True)

    # Initialize the YOLO model
    model = initialize_model()

    # Initialize MediaPipe Hands module
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    # Initialize CvBridge for converting ROS Image messages to OpenCV images
    bridge = CvBridge()

    # Callback function to process each frame from the ROS camera topic
    def process_frame(ros_image):
        try:
            # Convert ROS Image message to OpenCV format
            frame = bridge.imgmsg_to_cv2(ros_image, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return

        # Run YOLOv8 to detect objects (including humans and sports balls)
        results = model(frame)

        # Convert the frame to RGB for MediaPipe processing
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Initialize MediaPipe Hands for gesture recognition
        with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:

            # Handle YOLO results safely
            if results is None or len(results) == 0 or not results[0].boxes:
                rospy.logwarn("No detections found.")
                cv2.imshow('Human and Sports Ball Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    rospy.signal_shutdown("User exit.")
                return
            # Get frame dimensions
            frame_height, frame_width, _ = frame.shape
            # Filter YOLO results for humans (class label 'person' = 0) and sports ball (class label '32')
            humans = [r for r in results if r.boxes and r.boxes.cls[0].item() == 0]  # Assuming class '0' is 'person'
            sports_balls = [r for r in results if r.boxes and r.boxes.cls[0].item() == 32]  # Assuming class '32' is 'sports ball'

            # Loop through detected humans
            for human in humans:
                box = human.boxes.xyxy[0].cpu().numpy()  # Get bounding box
                x1, y1, x2, y2 = map(int, box)  # Bounding box coordinates
                pixel_height = y2 - y1  # Height of bounding box in pixels

                # Estimate distance to the human
                #distance_to_human = estimate_distance(FOCAL_LENGTH, HUMAN_HEIGHT, pixel_height)
                #rospy.loginfo(f"Estimated distance to human: {distance_to_human:.2f} meters")

                # Draw bounding box around detected human
                #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                #cv2.putText(frame, f"Distance: {distance_to_human:.2f}m", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Process hand gesture recognition within the detected human's bounding box
                roi = image_rgb[y1:y2, x1:x2]  # Crop region of interest
                roi_contiguous = np.ascontiguousarray(roi)  # Ensure C-contiguous
                # Ensure the bounding box is within frame bounds
                x1 = max(0, min(x1, frame_width - 1))
                x2 = max(0, min(x2, frame_width - 1))
                y1 = max(0, min(y1, frame_height - 1))
                y2 = max(0, min(y2, frame_height - 1))

                # Calculate the center point of the bounding box
                x_center = (x1 + x2) // 2
                y_center = (y1 + y2) // 2

                # Estimate depth using the extracted region
                try:
                    depth_map = estimate_depth(frame)  # Assuming estimate_depth uses the full frame

                    # Get depth map dimensions
                    depth_map_height, depth_map_width = depth_map.shape

                    # Scale center point to match depth map resolution
                    x_center_scaled = int((x_center / frame_width) * depth_map_width)
                    y_center_scaled = int((y_center / frame_height) * depth_map_height)

                    # Clamp to valid indices
                    x_center_scaled = min(max(0, x_center_scaled), depth_map_width - 1)
                    y_center_scaled = min(max(0, y_center_scaled), depth_map_height - 1)

                    # Get depth at the scaled coordinates
                    depth_at_person = depth_map[y_center_scaled, x_center_scaled]
                    rospy.loginfo(f"Estimated distance to human: {depth_at_person} meters")

                    # Draw bounding box around detected human
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Distance: {depth_at_person:.2f}m", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                except Exception as e:
                    rospy.logerr(f"Error in depth estimation: {e}")
                    continue


                # Detect hands in the cropped ROI
                hand_results = hands.process(roi_contiguous)

                if hand_results.multi_hand_landmarks:
                    for hand_landmarks, hand_handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
                        # Draw hand landmarks
                        mp_drawing.draw_landmarks(frame[y1:y2, x1:x2], hand_landmarks, mp_hands.HAND_CONNECTIONS)

                        # Extract hand landmarks for gesture recognition
                        landmarks = hand_landmarks.landmark
                        wrist = landmarks[0]
                        middle_finger_tip = landmarks[12]
                        thumb_tip = landmarks[4]
                        pinky_tip = landmarks[20]

                        hand_label = hand_handedness.classification[0].label  # 'Left' or 'Right'
                        vertical = abs(wrist.x - middle_finger_tip.x) < 0.1

                        if hand_label == "Right":
                            palm_open = thumb_tip.x < wrist.x < pinky_tip.x
                        else:
                            palm_open = pinky_tip.x < wrist.x < thumb_tip.x

                        fingers_open = all(landmarks[tip].y < landmarks[tip - 2].y for tip in [8, 12, 16, 20])

                        if vertical and palm_open and fingers_open:
                            cv2.putText(frame, f"Hi Gesture Detected! ({hand_label} hand)", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Loop through detected sports balls
            for ball in sports_balls:
                box = ball.boxes.xyxy[0].cpu().numpy()  # Get bounding box
                x1, y1, x2, y2 = map(int, box)  # Bounding box coordinates
                pixel_height = y2 - y1  # Height of bounding box in pixels

                # Estimate distance to the sports ball
                #distance_to_ball = estimate_distance(FOCAL_LENGTH, BALL_DIAMETER, pixel_height)
                #rospy.loginfo(f"Estimated distance to sports ball: {distance_to_ball:.2f} meters")

                
                #depth_to_ball = estimate_depth(sports_balls)
                #x_center = (x1 + x2)//2
                #y_center = (y1 + y2)// 2
                #rospy.loginfo(f"Estimated distance to sports ball: {depth_to_ball}meters")


                # Draw bounding box around detected sports ball
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"Ball Distance: {distance_to_ball:.2f}m", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the frame with detected humans, sports balls, hand gestures, and distances
        cv2.imshow('Human, Sports Ball and Hand Gesture Detection', frame)

        # Wait for 'q' key press to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            rospy.signal_shutdown("User exit.")

    # Subscribe to the camera image topic (replace '/camera/image_raw' with your topic name)
    rospy.Subscriber('/camera/image_raw', Image, process_frame)

    # Keep the node running
    rospy.spin()

    # Release resources when done
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        human_detection()
    except rospy.ROSInterruptException:
        pass