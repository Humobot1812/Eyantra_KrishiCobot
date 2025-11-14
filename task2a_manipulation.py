#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
*****************************************************************************************
* eYRC Krishi CoBot 2025-26 | Team ID: 1732
* UR5 Pick & Place Extended Task:
*
* MODIFIED (v8):
* - Created a new FRUIT_PICK_ORIENTATION (horizontal + 90-deg Z roll).
* - This is to match the orientation of the fruit on the tray.
*****************************************************************************************
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from tf2_ros import Buffer, TransformListener
import tf2_ros
import numpy as np
import time
from scipy.spatial.transform import Rotation
from linkattacher_msgs.srv import AttachLink, DetachLink

BASE_FRAME = "base_link"
EEF_FRAME = "wrist_3_link"

# --- Motion ---
LINEAR_KP = 1.5
MAX_LINEAR_VEL = 30.0
POSITION_TOLERANCE = 0.04
ANGULAR_KP = 0.8
ORIENTATION_TOLERANCE = 0.1
WAIT_AT_WAYPOINT = 0.8

# --- Waypoint Definitions ---
# Horizontal, for fertilizer shelf
PICK_ORIENTATION = Rotation.from_euler('x', 90, degrees=True)
# Vertical (Down), for all drops
DROP_ORIENTATION = Rotation.from_euler('y', 180, degrees=True)
# NEW: Horizontal + 90-deg roll, for fruit
FRUIT_PICK_ORIENTATION = Rotation.from_euler('x', 180, degrees=True) * Rotation.from_euler('z', 90, degrees=True) 

# --- ADD THIS LINE ---
INTERMEDIATE_P2_ORN = Rotation.from_quat(np.array([0.029, 0.997, 0.045, 0.033]))
# --- END ADD ---


TRASH_BIN_POS = np.array([-0.806, 0.010, 0.182])
INTERMEDIATE_1_POS = np.array([-0.159, 0.501, 0.600])

# Offsets
HOVER_OFFSET = np.array([0.0, 0.0, 0.2])
FERTILIZER_PICK_OFFSET = np.array([0.0, 0.0, 0.02])
FRUIT_PICK_OFFSET = np.array([0.0, 0.0, -0.04])

# --- Object Names for Gripper ---
FERTILIZER_MODEL = "fertiliser_can"
BAD_FRUIT_MODEL = "bad_fruit"
ROBOT_MODEL = "ur5"
ROBOT_GRIP_LINK = "wrist_3_link"
OBJECT_LINK = "body"


class UR5PickPlace(Node):
    def __init__(self):
        super().__init__("ur5_pick_place")
        self.get_logger().info("=== UR5 Pick & Place Node Started (Team 1732) ===")
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.cmd_pub = self.create_publisher(Twist, "/delta_twist_cmds", 10)

        self.attach_client = self.create_client(AttachLink, '/attach_link')
        self.detach_client = self.create_client(DetachLink, '/detach_link')

        while not self.attach_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Attach service not available, waiting...')
        while not self.detach_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Detach service not available, waiting...')
        self.get_logger().info('✅ Gripper services are ready.')

        self.timer = self.create_timer(0.1, self.control_loop)
        
        self.team_id = "1732"
        self.state = "WAIT_FOR_TFS"
        self.tf_positions = {}
        self.sequence = []
        self.current_index = 0
        self.reached_target = False
        self.last_reach_time = None
        self.service_call_in_progress = False
        self.service_future = None

    def get_tf_pos(self, frame):
        try:
            tf = self.tf_buffer.lookup_transform(BASE_FRAME, frame, rclpy.time.Time())
            pos = np.array([
                tf.transform.translation.x,
                tf.transform.translation.y,
                tf.transform.translation.z
            ])
            return pos
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            return None

    def stop(self):
        self.cmd_pub.publish(Twist())

    def get_eef_pose(self):
        try:
            tf = self.tf_buffer.lookup_transform(BASE_FRAME, EEF_FRAME, rclpy.time.Time())
            pos = np.array([tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z])
            orn = Rotation.from_quat([tf.transform.rotation.x, tf.transform.rotation.y, tf.transform.rotation.z, tf.transform.rotation.w])
            return pos, orn
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            return None, None

    def move_to_pose(self, target_pos, target_orn):
        current_pos, current_orn = self.get_eef_pose()
        if current_pos is None: return False
        linear_error = target_pos - current_pos
        pos_dist = np.linalg.norm(linear_error)
        if pos_dist > POSITION_TOLERANCE:
            linear_cmd = LINEAR_KP * linear_error
            if np.linalg.norm(linear_cmd) > MAX_LINEAR_VEL:
                linear_cmd = linear_cmd / np.linalg.norm(linear_cmd) * MAX_LINEAR_VEL
        else: linear_cmd = np.array([0.0, 0.0, 0.0])
        error_rotation = target_orn * current_orn.inv()
        angular_error = error_rotation.as_rotvec()
        orn_dist = np.linalg.norm(angular_error)
        if orn_dist > ORIENTATION_TOLERANCE:
            angular_cmd = ANGULAR_KP * angular_error
        else: angular_cmd = np.array([0.0, 0.0, 0.0])
        if pos_dist < POSITION_TOLERANCE and orn_dist < ORIENTATION_TOLERANCE:
            self.stop()
            return True
        msg = Twist()
        msg.linear.x, msg.linear.y, msg.linear.z = linear_cmd
        msg.angular.x, msg.angular.y, msg.angular.z = angular_cmd
        self.cmd_pub.publish(msg)
        return False

    def call_gripper_service(self, action, model_name):
        if action == "attach":
            self.get_logger().info(f"Attaching to {model_name}...")
            req = AttachLink.Request()
            req.model1_name = model_name
            req.link1_name = OBJECT_LINK
            req.model2_name = ROBOT_MODEL
            req.link2_name = ROBOT_GRIP_LINK
            self.service_future = self.attach_client.call_async(req)
        elif action == "detach":
            self.get_logger().info(f"Detaching from {model_name}...")
            req = DetachLink.Request()
            req.model1_name = model_name
            req.link1_name = OBJECT_LINK
            req.model2_name = ROBOT_MODEL
            req.link2_name = ROBOT_GRIP_LINK
            self.service_future = self.detach_client.call_async(req)
        self.service_call_in_progress = True

    def control_loop(self):
        if self.service_call_in_progress:
            if self.service_future.done():
                self.get_logger().info("Service call finished successfully.")
                self.service_call_in_progress = False
                self.service_future = None
                self.current_index += 1
                self.reached_target = False
                if self.current_index < len(self.sequence):
                     self.get_logger().info(f"➡️ Moving to next waypoint ({self.current_index + 1}/{len(self.sequence)})")
            else:
                return

        if self.state == "WAIT_FOR_TFS":
            self.collect_all_tfs()
        elif self.state == "MOVE_SEQUENCE":
            self.follow_sequence()
        elif self.state == "DONE":
            self.stop()

    # ---------------------------------------
    # THIS FUNCTION IS MODIFIED
    # ---------------------------------------
    def collect_all_tfs(self):
        frames_needed = [
            f"{self.team_id}_fertiliser_can", f"{self.team_id}_aruco_6",
            f"{self.team_id}_bad_fruit_1", f"{self.team_id}_bad_fruit_2", f"{self.team_id}_bad_fruit_3",
        ]
        for frame in frames_needed:
            pos = self.get_tf_pos(frame) 
            if pos is None:
                self.get_logger().info(f"Waiting for TF: {frame} ...")
                return
            self.tf_positions[frame] = pos 
        self.tf_positions["trash_bin"] = TRASH_BIN_POS
        self.tf_positions["intermediate_1"] = INTERMEDIATE_1_POS
        self.get_logger().info("✅ All TF positions collected successfully.")

        self.sequence = []
        
        # Define P2 position once for cleaner code
        p2_pos = self.tf_positions["intermediate_1"]

        # --- 1. Fertilizer Pick (Aruco 3) ---
        pos = self.tf_positions[f"{self.team_id}_fertiliser_can"]
        self.sequence.append({
            'pos': pos + FERTILIZER_PICK_OFFSET, 'orn': PICK_ORIENTATION, 'label': "Fertilizer Start",
            'action': "attach", 'model': FERTILIZER_MODEL
        })

        # --- 2. Fertilizer Drop (Aruco 6) ---
        pos = self.tf_positions[f"{self.team_id}_aruco_6"]
        self.sequence.append({
            'pos': pos + HOVER_OFFSET, 
            'orn': DROP_ORIENTATION,
            'label': "Fertilizer Drop",
            'action': "detach", 'model': FERTILIZER_MODEL
        })

        # --- 3. Intermediate ---
        pos = self.tf_positions["intermediate_1"]
        self.sequence.append({
            'pos': pos, 
            'orn': FRUIT_PICK_ORIENTATION, # <-- FIXED
            'label': "Intermediate 1",
            'action': "none"
        })

        # --- 4. Bad Fruit 1 -> Trash ---
        pos = self.tf_positions[f"{self.team_id}_bad_fruit_1"]
        self.sequence.append({
            'pos': pos + FRUIT_PICK_OFFSET,
            'orn': FRUIT_PICK_ORIENTATION,
            'label': "Bad Fruit 1",
            'action': "attach", 'model': BAD_FRUIT_MODEL
        })
        self.sequence.append({
            'pos': p2_pos, 'orn': INTERMEDIATE_P2_ORN,
            'label': "Intermediate P2 (After Pick 1)",
            'action': "none"
        })
        pos = self.tf_positions["trash_bin"]
        self.sequence.append({
            'pos': pos + HOVER_OFFSET, 'orn': DROP_ORIENTATION, 'label': "Trash Bin 1",
            'action': "detach", 'model': BAD_FRUIT_MODEL
        })
        
        # --- NEW STEP: Return to P2 before next fruit ---
        self.sequence.append({
            'pos': p2_pos, 'orn': INTERMEDIATE_P2_ORN,
            'label': "Intermediate P2 (Before Fruit 2)",
            'action': "none"
        })

        # --- 5. Bad Fruit 2 -> Trash ---
        pos = self.tf_positions[f"{self.team_id}_bad_fruit_2"]
        self.sequence.append({
            'pos': pos + FRUIT_PICK_OFFSET,
            'orn': FRUIT_PICK_ORIENTATION,
            'label': "Bad Fruit 2",
            'action': "attach", 'model': BAD_FRUIT_MODEL
        })
        # --- NEW STEP: Return to P2 after pick ---
        self.sequence.append({
            'pos': p2_pos, 'orn': INTERMEDIATE_P2_ORN,
            'label': "Intermediate P2 (After Pick 2)",
            'action': "none"
        })
        pos = self.tf_positions["trash_bin"]
        self.sequence.append({
            'pos': pos + HOVER_OFFSET, 'orn': DROP_ORIENTATION, 'label': "Trash Bin 2",
            'action': "detach", 'model': BAD_FRUIT_MODEL
        })
        
        # --- NEW STEP: Return to P2 before next fruit ---
        self.sequence.append({
            'pos': p2_pos, 'orn': INTERMEDIATE_P2_ORN,
            'label': "Intermediate P2 (Before Fruit 3)",
            'action': "none"
        })

        # --- 6. Bad Fruit 3 -> Trash ---
        pos = self.tf_positions[f"{self.team_id}_bad_fruit_3"]
        self.sequence.append({
            'pos': pos + FRUIT_PICK_OFFSET,
            'orn': FRUIT_PICK_ORIENTATION,
            'label': "Bad Fruit 3",
            'action': "attach", 'model': BAD_FRUIT_MODEL
        })
        # --- NEW STEP: Return to P2 after pick ---
        self.sequence.append({
            'pos': p2_pos, 'orn': INTERMEDIATE_P2_ORN,
            'label': "Intermediate P2 (After Pick 3)",
            'action': "none"
        })
        pos = self.tf_positions["trash_bin"]
        self.sequence.append({
            'pos': pos + HOVER_OFFSET, 'orn': DROP_ORIENTATION, 'label': "Trash Bin 3",
            'action': "detach", 'model': BAD_FRUIT_MODEL
        })
        # No P2 needed after the final drop

        self.get_logger().info(f"📍 Sequence loaded with {len(self.sequence)} waypoints.")
        self.state = "MOVE_SEQUENCE"
        self.current_index = 0

    def follow_sequence(self):
        if self.current_index >= len(self.sequence):
            self.get_logger().info("🎯 Full Path Execution Complete.")
            self.state = "DONE"
            return
        
        if self.service_call_in_progress:
            return

        waypoint = self.sequence[self.current_index]
        target_pos = waypoint['pos']
        target_orn = waypoint['orn']
        label = waypoint['label']

        reached = self.move_to_pose(target_pos, target_orn)

        if reached:
            if not self.reached_target:
                self.reached_target = True
                self.last_reach_time = time.time()
                self.get_logger().info(f"✅ Reached waypoint {self.current_index + 1}/{len(self.sequence)} → {label}")

            elif time.time() - self.last_reach_time >= WAIT_AT_WAYPOINT:
                action = waypoint.get('action', 'none')
                model = waypoint.get('model', None)

                if action != 'none' and model is not None:
                    self.call_gripper_service(action, model)
                else:
                    self.current_index += 1
                    self.reached_target = False
                    if self.current_index < len(self.sequence):
                        self.get_logger().info(f"➡️ Moving to next waypoint ({self.current_index + 1}/{len(self.sequence)})")
        else:
            self.reached_target = False


def main(args=None):
    rclpy.init(args=args)
    node = UR5PickPlace()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.stop()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()