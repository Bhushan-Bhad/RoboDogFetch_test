#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
import cv2
import sys
import math
import numpy as np
from geometry_msgs.msg import PoseStamped, PoseArray, Point
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import amd64.robot_interface as sdk

class RobotControlNode:
    def __init__(self):
        rospy.init_node('robot_control_node')

        # Subscribe to waypoints from path planning
        rospy.Subscriber('/path_planning/waypoints', Point, self.waypoint_callback)
        
        # Initialize UDP and command interface for robot control
        HIGHLEVEL = 0xee
        self.udp = sdk.UDP(HIGHLEVEL, 8080, "192.168.123.161", 8082)
        self.cmd = sdk.HighCmd()
        self.state = sdk.HighState()
        self.udp.InitCmdData(self.cmd)

        # Robot movement parameters
        self.current_waypoint = None
        self.robot_position = [0, 0]
        self.robot_yaw = 0  # Heading angle (in radians)
        self.waypoint_reached = False

        rospy.loginfo("Robot control node initialized and ready to move to waypoints.")

    def waypoint_callback(self, msg):
        # Update the current waypoint when a new one is received
        self.current_waypoint = msg
        self.waypoint_reached = False

    def calculate_yaw_error(self, current_x, current_y, target_x, target_y):
        # Calculate the yaw error to the target point
        target_angle = math.atan2(target_y - current_x, target_x - current_y)
        yaw_error = target_angle - self.robot_yaw
        return yaw_error * 0.5  # Gain factor for smoother control

    def move_to_waypoint(self, waypoint):
        current_x, current_y = self.robot_position
        target_x, target_y = waypoint.x, waypoint.y
        distance_threshold = 0.3  # Distance within which the waypoint is considered reached
        distance = math.sqrt((target_x - current_x) ** 2 + (target_y - current_y) ** 2)

        # Control commands to reach the waypoint
        if distance > distance_threshold:
            self.cmd.mode = 2  # Walking mode
            self.cmd.gaitType = 1  # Set walking gait
            self.cmd.velocity[0] = 0.3  # Move forward
            self.cmd.rotateSpeed = self.calculate_yaw_error(current_x, current_y, target_x, target_y)  # Adjust yaw

            # Keep the body level
            self.cmd.euler = [0.0, 0.0, self.robot_yaw]

            rospy.loginfo("Moving towards waypoint: ({}, {})".format(target_x, target_y))
        else:
            # Stop if waypoint is reached
            self.cmd.mode = 0  # Idle/stop mode
            self.cmd.velocity = [0, 0]
            self.cmd.yawSpeed = 0.0
            self.waypoint_reached = True
            rospy.loginfo("Reached waypoint: ({}, {})".format(target_x, target_y))

        # Send commands via UDP
        self.udp.SetSend(self.cmd)
        self.udp.Send()

    def control_loop(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.current_waypoint and not self.waypoint_reached:
                # Move towards the waypoint if not reached
                self.move_to_waypoint(self.current_waypoint)
            else:
                # Stop the robot when no waypoint is set or waypoint is reached
                self.cmd.mode = 0  # Stop mode
                self.cmd.velocity = [0, 0]
                self.cmd.yawSpeed = 0.0
                self.udp.SetSend(self.cmd)
                self.udp.Send()

            rate.sleep()

if __name__ == '__main__':
    try:
        controller = RobotControlNode()
        controller.control_loop()
    except rospy.ROSInterruptException:
        pass
