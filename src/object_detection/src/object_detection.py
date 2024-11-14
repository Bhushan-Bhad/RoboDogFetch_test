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
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PoseStamped, PoseArray, Pose
from distance_calc_MonocularDepthEstimation import estimate_depth, estimate_distance, pixel_to_world
import time
import math
import torch
    

# Function to initialize the YOLO model
def initialize_model():
    return YOLO("yolo-Weights/yolov8n-seg.pt")

# Main function for human and sports ball detection with hand gesture recognition
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
    target_position_pub = rospy.Publisher('/target_position', PoseStamped, queue_size=10)
    target_command_pub = rospy.Publisher('/target_command', String, queue_size=10)
    all_objects_pub = rospy.Publisher('/detected_objects', PoseArray, queue_size=10)  # New publisher for all detected objects


    global last_detected_position
    global frame_counter
    last_detected_position = None
    frame_counter = 0  # To process every N-th frame
    frame_rate_limit = 3  # Process 1 frame out of every 3
    
    def detect_all_objects(frame):
        """Detect all objects and publish their positions."""
        results = model(frame)
        object_positions = PoseArray()
        object_positions.header.frame_id = "map"

        if results is None or len(results) == 0 or not results[0].boxes:
            rospy.logwarn("No objects detected.")
            return object_positions

        frame_height, frame_width, _ = frame.shape

        # Webcam specifications
        image_width = 1280  # FHD resolution
        fov = 90  # Estimated field of view in degrees (horizontal)

        # Calculate focal length
        focal_length = image_width / (2 * math.tan(math.radians(fov / 2)))
        real_human_height = 1.7  
        for obj in results[0].boxes:
            box = obj.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, box)

            # Calculate center of the detected object
            x_center = (x1 + x2) // 2
            y_center = (y1 + y2) // 2

            # Here, you might want to implement distance estimation as done for humans, if required
            # For example:
            # BBox_distance = estimate_distance(focal_length, object_height, pixel_height)

            # Convert bounding box center to world coordinates
            # For this example, we assume a constant distance; you can adjust this logic as needed
            dummy_distance = 1.0  # Replace with actual distance estimation logic
            world_x, world_y = pixel_to_world(x_center, y_center, focal_length, dummy_distance, frame_width, frame_height)

            # Create a Pose for the detected object
            pose = Pose()
           # pose.header.stamp = rospy.Time.now()
            #pose.header.frame_id = "map"
            pose.position.x = world_x
            pose.position.y = world_y
            pose.position.z = dummy_distance  # Set Z to the distance (if needed)
            object_positions.poses.append(pose)

        all_objects_pub.publish(object_positions)  # Publish all detected objects' positions

        return object_positions


    def process_frame(ros_image):
        global last_detected_position
        global frame_counter

        # Increment frame counter and skip frames to limit processing rate
        frame_counter += 1
        if frame_counter % frame_rate_limit != 0:
            return  # Skip frame

        start_time = time.time()  # Track time for performance measurement

        # Convert ROS Image message to OpenCV format
        try:
            frame = bridge.imgmsg_to_cv2(ros_image, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return

        # Resize frame to reduce computational load
        frame = cv2.resize(frame, (1280, 720))  # Adjust resolution to reduce processing

       # Detect all objects and publish their positions
        detect_all_objects(frame)

        # YOLOv8 to detect objects
        results = model(frame)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
            if results is None or len(results) == 0 or not results[0].boxes:
                rospy.logwarn("No detections found.")
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    rospy.signal_shutdown("User exit.")
                return

            frame_height, frame_width, _ = frame.shape
            humans = [r for r in results if r.boxes and r.boxes.cls[0].item() == 0]

            # Webcam specifications
            image_width = 1280  # FHD resolution
            fov = 90  # Estimated field of view in degrees (horizontal)

            # Calculate focal length
            focal_length = image_width / (2 * math.tan(math.radians(fov / 2)))
            real_human_height = 1.7  # Average human height in meters

            for human in humans:
                box = human.boxes.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, box)
                pixel_height = y2 - y1

                x_center = (x1 + x2) // 2
                y_center = (y1 + y2) // 2

                # Gesture detection within cropped region of interest (ROI)
                roi = image_rgb[y1:y2, x1:x2]
                roi_contiguous = np.ascontiguousarray(roi)
                hand_results = hands.process(roi_contiguous)

                if hand_results.multi_hand_landmarks:
                    for hand_landmarks, hand_handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
                        landmarks = hand_landmarks.landmark
                        wrist = landmarks[0]
                        middle_finger_tip = landmarks[12]
                        thumb_tip = landmarks[4]
                        pinky_tip = landmarks[20]

                        hand_label = hand_handedness.classification[0].label
                        vertical = abs(wrist.x - middle_finger_tip.x) < 0.1

                        if hand_label == "Right":
                            palm_open = thumb_tip.x < wrist.x < pinky_tip.x
                        else:
                            palm_open = pinky_tip.x < wrist.x < thumb_tip.x

                        fingers_open = all(landmarks[tip].y < landmarks[tip - 2].y for tip in [8, 12, 16, 20])

                        # If "Hi" gesture detected
                        if vertical and palm_open and fingers_open:
                            cv2.putText(frame, f"Hi Gesture Detected! ({hand_label} hand)", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            
                            # Bounding Box distance calculation
                            BBox_distance = estimate_distance(focal_length, real_human_height, pixel_height)   
                            rospy.loginfo(f"Estimated distance to human: {BBox_distance:.2f} meters")
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                            if not np.isnan(BBox_distance) and BBox_distance > 0:
                                final_distance = BBox_distance
                                rospy.loginfo(f"Using bounding box distance: {final_distance:.2f} meters")
                                
                                # Convert bounding box center to world coordinates using the final distance
                                world_x, world_y = pixel_to_world(x_center, y_center, focal_length, final_distance, frame_width, frame_height)

                                # Publish the target position
                                target_msg = PoseStamped()
                                target_msg.header.stamp = rospy.Time.now()
                                target_msg.header.frame_id = "map"  # Adjust based on your setup
                                target_msg.pose.position.x = world_x
                                target_msg.pose.position.y = world_y
                                target_msg.pose.position.z = final_distance  # Set Z to the final distance
                                target_msg.pose.orientation.w = 1.0  # Default orientation (can be adjusted)

                                target_position_pub.publish(target_msg)
                                rospy.loginfo(f"Target position published: {target_msg}")

                                # Publish the command
                                target_command_pub.publish("go_to_human")

                        else:
                            cv2.putText(frame, "Human detected", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Show the frame with detections
            cv2.imshow('Human Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                rospy.signal_shutdown("User exit.")

        end_time = time.time()  # For performance tracking
        # rospy.loginfo(f"Frame processing time: {end_time - start_time:.2f} seconds")

    # Subscribe to the camera topic
    rospy.Subscriber("/camera/image_raw", Image, process_frame)

    # Keep the node running
    rospy.spin()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        human_detection()
    except rospy.ROSInterruptException:
        pass
