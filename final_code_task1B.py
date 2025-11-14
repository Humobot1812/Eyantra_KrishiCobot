#!/usr/bin/python3
# -*- coding: utf-8 -*-

'''
*****************************************************************************************
*
*        		===============================================
*           		    Krishi coBot (KC) Theme (eYRC 2025-26)
*        		===============================================
*
*  This script should be used to implement Task 1B of Krishi coBot (KC) Theme (eYRC 2025-26).
*
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or
*  breach of the terms of this agreement.
*
*****************************************************************************************
'''

# Team ID:          1732
# Author List:		[ Akash Dhyani, Shirshendu Ranjana Tripathi, Abhinav Goel, Harshit Kumar Saxena ]
# Filename:		    final_code_task1B.py
# Functions:
#			        [ bad_fruit_detection, get_accurate_depth, pixel_to_3d, transform_optical_to_base_frame, create_transform_stamped, detect_aruco_markers ]
# Nodes:		    Publishing Topics  - [ /tf ]
#                   Subscribing Topics - [ /camera/image_raw, /camera/depth/image_raw ]

import sys
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from sensor_msgs.msg import Image
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge
from std_srvs.srv import Trigger
import cv2
import numpy as np
from tf2_ros import TransformBroadcaster
from typing import Optional

SHOW_IMAGE = True
DISABLE_MULTITHREADING = False

class FruitsTF(Node):
    """
    ROS2 Node for fruit detection and TF publishing.
    """

    def __init__(self):
        super().__init__('fruits_tf')
        self.team_id = 1732
        self.bridge = CvBridge()
        self.cv_image = None
        self.depth_image = None

        if DISABLE_MULTITHREADING:
            self.cb_group = MutuallyExclusiveCallbackGroup()
        else:
            self.cb_group = ReentrantCallbackGroup()

        self.create_subscription(Image, '/camera/image_raw', self.colorimagecb, 10, callback_group=self.cb_group)
        self.create_subscription(Image, '/camera/depth/image_raw', self.depthimagecb, 10, callback_group=self.cb_group)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.create_timer(0.2, self.process_image, callback_group=self.cb_group)

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        if SHOW_IMAGE:
            cv2.namedWindow('fruits_tf_view', cv2.WINDOW_NORMAL)

        self.get_logger().info(f"FruitsTF node started - Team ID: {self.team_id}")

    def depthimagecb(self, data):
        '''
        Description:    Callback function for aligned depth camera topic.
                        Use this function to receive image depth data and convert to CV2 image.

        Args:
            data (Image): Input depth image frame received from aligned depth camera topic

        Returns:
            None
        '''
        try:
            self.original_depth_msg = data
            self.get_logger().info(f"Depth image encoding: {data.encoding}")

            if data.encoding == '32FC1':
                depth_data = self.bridge.imgmsg_to_cv2(data, desired_encoding="32FC1")
                self.depth_image = depth_data
                self.get_logger().info("Using 32FC1 encoding (meters)")

            elif data.encoding == '16UC1':
                depth_data = self.bridge.imgmsg_to_cv2(data, desired_encoding="16UC1")
                depth_meters = depth_data.astype(np.float32) / 1000.0
                self.depth_image = depth_meters
                self.get_logger().info("Using 16UC1 encoding (millimeters to meters)")

            else:
                self.get_logger().warn(f"Unknown encoding: {data.encoding}, trying raw conversion")
                depth_data = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
                self.depth_image = depth_data.astype(np.float32)

            if self.depth_image is not None:
                self.depth_image[self.depth_image <= 0] = np.nan
                self.depth_image[np.isinf(self.depth_image)] = np.nan

        except Exception as e:
            self.get_logger().error(f"Depth conversion error: {e}")
            self.depth_image = None

    def colorimagecb(self, data):
        '''
        Description:    Callback function for colour camera raw topic.
                        Use this function to receive raw image data and convert to CV2 image.

        Args:
            data (Image): Input coloured raw image frame received from image_raw camera topic

        Returns:
            None
        '''
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"RGB conversion error: {e}")

    def detect_aruco_markers(self, image):
        '''
        Description:    Function to detect ArUco markers in the image and draw pose estimation axes.

        Args:
            image (cv2 image): Input color image

        Returns:
            image_with_aruco (cv2 image): Image with ArUco markers detected and pose axes drawn
        '''
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = self.aruco_detector.detectMarkers(gray)
        image_with_aruco = image.copy()

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(image_with_aruco, corners, ids)

            camera_matrix = np.array([[915.3003540039062, 0.0, 642.724365234375],
                                    [0.0, 914.0320434570312, 361.9780578613281],
                                    [0.0, 0.0, 1.0]], dtype=np.float64)
            dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
            marker_size = 0.05

            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, dist_coeffs)

            if rvecs is not None and tvecs is not None:
                for i in range(len(ids)):
                    cv2.drawFrameAxes(image_with_aruco, camera_matrix, dist_coeffs,
                                    rvecs[i], tvecs[i], marker_size * 0.7)

        return image_with_aruco

    def bad_fruit_detection(self, rgb_image):
        '''
        Description:    Function to detect bad fruits in the image frame.
                        Use this function to detect bad fruits and return their center coordinates, distance from camera, angle, width and ids list.

        Args:
            rgb_image (cv2 image): Input coloured raw image frame received from image_raw camera topic

        Returns:
            list: A list of detected bad fruit information, where each entry is a dictionary containing:
                - 'center': (x, y) coordinates of the fruit center
                - 'distance': distance from the camera in meters
                - 'angle': angle of the fruit in degrees
                - 'width': width of the fruit in pixels
                - 'id': unique identifier for the fruit
        '''
        bad_fruits = []
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)

        # Green top detection
        lower_green = np.array([35, 100, 80])
        upper_green = np.array([85, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)

        # Grey shell detection
        lower_grey = np.array([0, 0, 50])
        upper_grey = np.array([180, 50, 200])
        mask_grey = cv2.inRange(hsv, lower_grey, upper_grey)

        # Overlap mask for bad fruit detection
        mask_green_dilated = cv2.dilate(mask_green, None, iterations=3)
        mask_grey_dilated = cv2.dilate(mask_grey, None, iterations=3)
        overlap_mask = cv2.bitwise_and(mask_green_dilated, mask_grey_dilated)

        contours, _ = cv2.findContours(overlap_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        fruit_id = 1
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 800:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = float(w) / h
                    if 0.4 < aspect_ratio < 2.5:
                        distance = None
                        if self.depth_image is not None:
                            distance = self.get_accurate_depth(cX, cY, self.depth_image)

                        fruit_info = {
                            'center': (cX, cY),
                            'distance': distance,
                            'angle': 0,
                            'width': w,
                            'id': fruit_id,
                            'area': area,
                            'contour': contour
                        }
                        bad_fruits.append(fruit_info)
                        fruit_id += 1

        return bad_fruits

    def get_accurate_depth(self, x: int, y: int, depth_image: np.ndarray) -> Optional[float]:
        """Get accurate depth value with proper validation"""
        if depth_image is None:
            return None

        height, width = depth_image.shape

        if x < 0 or x >= width or y < 0 or y >= height:
            return None

        depth_samples = []
        valid_samples = 0

        for dy in range(-3, 4):
            for dx in range(-3, 4):
                sample_x = x + dx
                sample_y = y + dy

                if 0 <= sample_x < width and 0 <= sample_y < height:
                    depth_val = depth_image[sample_y, sample_x]

                    if not np.isnan(depth_val) and 0.1 < depth_val < 3.0:
                        depth_samples.append(depth_val)
                        valid_samples += 1

        if valid_samples < 5:
            return None

        depth_samples = np.array(depth_samples)

        Q1 = np.percentile(depth_samples, 25)
        Q3 = np.percentile(depth_samples, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        filtered_depths = depth_samples[(depth_samples >= lower_bound) & (depth_samples <= upper_bound)]

        if len(filtered_depths) == 0:
            return None

        accurate_depth = np.median(filtered_depths)
        return accurate_depth

    def pixel_to_3d(self, u: int, v: int, depth: float, focalX: float, focalY: float, centerCamX: float, centerCamY: float) -> tuple:
        """Convert pixel coordinates to 3D point in camera optical frame"""
        z = depth
        x = (u - centerCamX) * z / focalX
        y = (v - centerCamY) * z / focalY
        return (x, y, z)

    def transform_optical_to_base_frame(self, point_optical: tuple) -> tuple:
        """Transform point from camera optical frame to base_link frame"""
        x_opt, y_opt, z_opt = point_optical

        # Transform from optical frame to camera_link frame
        x_cam = z_opt
        y_cam = -x_opt
        z_cam = -y_opt

        # Apply camera to base transformation
        pitch = -0.733  # -42° in radians

        cos_p = np.cos(pitch)
        sin_p = np.sin(pitch)

        x_base = (x_cam * cos_p - z_cam * sin_p) - 1.095239
        y_base = y_cam
        z_base = (x_cam * sin_p + z_cam * cos_p) + 1.10058

        return (x_base, y_base, z_base)

    def create_transform_stamped(self, parent_frame: str, child_frame: str, translation: tuple) -> TransformStamped:
        """Create TransformStamped message"""
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = parent_frame
        t.child_frame_id = child_frame
        t.transform.translation.x = float(translation[0])
        t.transform.translation.y = float(translation[1])
        t.transform.translation.z = float(translation[2])
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0
        return t

    def process_image(self):
        '''
        Description:    Timer-driven loop for periodic image processing.

        Returns:
            None
        '''
        sizeCamX = 1280
        sizeCamY = 720
        centerCamX = 642.724365234375
        centerCamY = 361.9780578613281
        focalX = 915.3003540039062
        focalY = 914.0320434570312

        if self.cv_image is None or self.depth_image is None:
            return

        try:
            valid_mask = ~np.isnan(self.depth_image)
            if np.any(valid_mask):
                valid_depths = self.depth_image[valid_mask]
                self.get_logger().info(f"Depth stats - Min: {valid_depths.min():.3f}m, Max: {valid_depths.max():.3f}m, Mean: {valid_depths.mean():.3f}m")
            else:
                self.get_logger().warn("No valid depth values found in image!")

            vis_image = self.detect_aruco_markers(self.cv_image)
            detections = self.bad_fruit_detection(self.cv_image)

            if not detections:
                if SHOW_IMAGE:
                    cv2.imshow("fruits_tf_view", vis_image)
                    cv2.waitKey(1)
                return

            published_count = 0

            for fruit_info in detections:
                cX, cY = fruit_info['center']
                distance = fruit_info['distance']
                fruit_id = fruit_info['id']
                contour = fruit_info['contour']

                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(vis_image, (cX, cY), 5, (0, 255, 0), -1)
                cv2.putText(vis_image, "bad fruit", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                if distance is None:
                    self.get_logger().warn(f"No valid depth at fruit {fruit_id} ({cX}, {cY})")
                    cv2.putText(vis_image, "d: invalid", (cX + 10, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    continue

                self.get_logger().info(f"Fruit {fruit_id} at ({cX}, {cY}) - Depth: {distance:.3f}m")

                optical_pos = self.pixel_to_3d(cX, cY, distance, focalX, focalY, centerCamX, centerCamY)
                base_link_pos = self.transform_optical_to_base_frame(optical_pos)

                frame_name = f"{self.team_id}_bad_fruit_{fruit_id}"
                transform = self.create_transform_stamped('base_link', frame_name, base_link_pos)
                self.tf_broadcaster.sendTransform(transform)

                self.get_logger().info(f"Published TF for {frame_name} at ({base_link_pos[0]:.3f}, {base_link_pos[1]:.3f}, {base_link_pos[2]:.3f})")
                published_count += 1

            if published_count == 0:
                self.get_logger().warn("No valid fruit detections with proper depth data")

            if SHOW_IMAGE:
                cv2.imshow("fruits_tf_view", vis_image)
                cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Processing error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = FruitsTF()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutting down FruitsTF")
        node.destroy_node()
        rclpy.shutdown()
        if SHOW_IMAGE:
            cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
