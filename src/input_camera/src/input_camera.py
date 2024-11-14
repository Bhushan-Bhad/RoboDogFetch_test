#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge

def input_camera_node():
    # Initialize the ROS node
    rospy.init_node('input_camera', anonymous=True)
    rospy.loginfo("Input camera node started.")


    # Create a publisher to publish images on the `/camera/image_raw` topic
    pub = rospy.Publisher('/camera/image_raw', Image, queue_size=10)

    # Set the publishing rate (in Hz)
    rate = rospy.Rate(50)

    # Initialize the CvBridge
    bridge = CvBridge()

    # Open the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        rospy.logerr("Cannot open webcam")
        return

    while not rospy.is_shutdown():
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            rospy.logerr("Failed to capture image")
            continue

        # Optionally resize the frame
        frame = cv2.resize(frame, (1280, 720))  # Resize to 640x480
        # Convert OpenCV image (numpy array) to ROS Image message
        ros_image = bridge.cv2_to_imgmsg(frame, "bgr8")

        # Publish the image
        pub.publish(ros_image)

        # Sleep to maintain the rate
        rate.sleep()

    # When everything is done, release the capture
    cap.release()

if __name__ == '__main__':
    try:
        input_camera_node()
    except rospy.ROSInterruptException:
        pass
