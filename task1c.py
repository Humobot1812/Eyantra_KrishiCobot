# Team ID:          < eYRC#1732 >
# Theme:            Krishi coBot
# Author List:      < Harshit Kumar Saxena, Abhinav Goel, Akash Dhyani, Shirshendu Ranjana Tripathi >
# Filename:         task1c.py
# Functions:        get_end_effector_pose, control_loop, stop_robot, quaternion_multiply, main
# Global variables: BASE_FRAME, END_EFFECTOR_FRAME, LINEAR_KP, ANGULAR_KP, MAX_LINEAR_VEL, MAX_ANGULAR_VEL, POSITION_TOLERANCE, ORIENTATION_TOLERANCE, WAIT_TIME_S

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from tf2_ros import Buffer, TransformListener
import tf2_ros
import numpy as np
import time
import math

# BASE_FRAME: The frame of reference for the waypoints and the arm's base
BASE_FRAME = 'base_link'
# END_EFFECTOR_FRAME: The frame of the end-effector we want to control
END_EFFECTOR_FRAME = 'wrist_3_link'

# LINEAR_KP: Proportional gain for linear velocity control
LINEAR_KP = 0.8
# ANGULAR_KP: Proportional gain for angular velocity control
ANGULAR_KP = 0.5

# MAX_LINEAR_VEL: Maximum allowed linear velocity (m/s)
MAX_LINEAR_VEL = 12.0
# MAX_ANGULAR_VEL: Maximum allowed angular velocity (rad/s)
MAX_ANGULAR_VEL = 10.0

# POSITION_TOLERANCE: Distance tolerance to consider a waypoint reached (meters)
POSITION_TOLERANCE = 0.15
# ORIENTATION_TOLERANCE: Angular tolerance to consider a waypoint reached (radians)
ORIENTATION_TOLERANCE = 0.15

# WAIT_TIME_S: Time to wait at each waypoint before moving to the next (seconds)
WAIT_TIME_S = 1.0


class ArmServoingController(Node):
    """
    A ROS 2 Node to control the UR5 arm using end-effector servoing.
    """

    def __init__(self):
        """
        Purpose:
        ---
        Initializes the node, sets up waypoints, publisher, TF listener, and main control loop.

        Input Arguments:
        ---
        None

        Returns:
        ---
        None

        Example call:
        ---
        arm_controller = ArmServoingController()
        """
        super().__init__('arm_servoing_controller')

        # Waypoint Definitions: List of dicts with 'position' and 'orientation' quaternions
        self.waypoints = [
            {  # P1
                'position': np.array([-0.214, -0.532, 0.557]),
                'orientation': np.array([0.707, 0.028, 0.034, 0.707])
            },
            {  # Intermediate_point
                'position': np.array([0.150, 0, 0.600]),
                'orientation': np.array([0.029, 0.997, 0.045, 0.033])
            },
            {  # P2
                'position': np.array([-0.159, 0.501, 0.415]),
                'orientation': np.array([0.029, 0.997, 0.045, 0.033])
            },
            {  # P3
                'position': np.array([-0.806, 0.010, 0.182]),
                'orientation': np.array([-0.684, 0.726, 0.05, 0.008])
            }
            
        ]

        # Normalize orientation quaternions
        for wp in self.waypoints:
            wp['orientation'] /= np.linalg.norm(wp['orientation'])

        # State variables
        self.current_waypoint_index = 0   # Index of the current target waypoint
        self.waypoint_reached_time = None # Timestamp when a waypoint was reached
        self.is_waiting_at_waypoint = False
        self.all_waypoints_completed = False

        # Publisher for velocity commands
        self.twist_pub = self.create_publisher(Twist, '/delta_twist_cmds', 10)

        # TF2 listener to get end-effector pose
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Timer for main control loop at 10 Hz
        self.timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info("UR5 Arm Servoing Controller has been started.")
        self.get_logger().info(f"Moving to Waypoint {self.current_waypoint_index + 1}...")

    def get_end_effector_pose(self):
        """
        Purpose:
        ---
        Retrieves the current position and orientation of the end-effector relative to the base.

        Input Arguments:
        ---
        None

        Returns:
        ---
        `pos` :  [ numpy.ndarray ]
            3-element array for x, y, z position of the end-effector
        `ori` :  [ numpy.ndarray ]
            4-element quaternion [x, y, z, w] of the end-effector orientation
        or
        None, None if the transform is unavailable

        Example call:
        ---
        pos, ori = self.get_end_effector_pose()
        """
        try:
            # Look up the transform from base to end-effector
            transform = self.tf_buffer.lookup_transform(
                BASE_FRAME, END_EFFECTOR_FRAME, rclpy.time.Time()
            )
            # Extract translation
            pos = np.array([
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z
            ])
            # Extract rotation quaternion
            ori = np.array([
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w
            ])
            return pos, ori
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(
                f"Could not get transform from '{BASE_FRAME}' to '{END_EFFECTOR_FRAME}': {e}"
            )
            return None, None

    def control_loop(self):
        """
        Purpose:
        ---
        The main control loop executed at a fixed rate. Computes pose error to the
        target waypoint, applies proportional control, and publishes velocity commands.

        Input Arguments:
        ---
        None

        Returns:
        ---
        None

        Example call:
        ---
        Called automatically by ROS timer at 10 Hz
        """
        if self.all_waypoints_completed:
            return

        # Handle wait at a reached waypoint
        if self.is_waiting_at_waypoint:
            if time.time() - self.waypoint_reached_time >= WAIT_TIME_S:
                self.is_waiting_at_waypoint = False
                self.waypoint_reached_time = None
                self.current_waypoint_index += 1
                if self.current_waypoint_index < len(self.waypoints):
                    self.get_logger().info(
                        f"Moving to Waypoint {self.current_waypoint_index + 1}..."
                    )
                else:
                    self.get_logger().info("All waypoints have been reached successfully!")
                    self.all_waypoints_completed = True
                    self.stop_robot()
                    rclpy.shutdown()
            return

        # Get current pose
        current_pos, current_ori = self.get_end_effector_pose()
        if current_pos is None:
            return  # TF not ready

        target = self.waypoints[self.current_waypoint_index]
        target_pos = target['position']
        target_ori = target['orientation']

        # --- Error Calculation ---
        error_pos = target_pos - current_pos

        # Compute orientation error quaternion: q_err = q_target * q_current_conj
        current_conj = current_ori * np.array([-1, -1, -1, 1])
        error_quat = self.quaternion_multiply(target_ori, current_conj)

        # Axis-angle from error quaternion
        error_axis = error_quat[:3]
        angle_error = 2 * math.acos(error_quat[3])
        if angle_error > math.pi:
            angle_error -= 2 * math.pi
        error_ori = angle_error * (error_axis / (np.linalg.norm(error_axis) + 1e-6))

        # Check waypoint reach criteria
        pos_err_norm = np.linalg.norm(error_pos)
        orient_err_norm = abs(angle_error)
        if pos_err_norm < POSITION_TOLERANCE and orient_err_norm < ORIENTATION_TOLERANCE:
            self.get_logger().info(f"Waypoint {self.current_waypoint_index + 1} reached.")
            self.stop_robot()
            self.is_waiting_at_waypoint = True
            self.waypoint_reached_time = time.time()
            return

        # --- Proportional Control ---
        cmd_vel_linear = LINEAR_KP * error_pos
        cmd_vel_angular = ANGULAR_KP * error_ori

        # Velocity limiting
        lin_norm = np.linalg.norm(cmd_vel_linear)
        if lin_norm > MAX_LINEAR_VEL:
            cmd_vel_linear = (cmd_vel_linear / lin_norm) * MAX_LINEAR_VEL
        ang_norm = np.linalg.norm(cmd_vel_angular)
        if ang_norm > MAX_ANGULAR_VEL:
            cmd_vel_angular = (cmd_vel_angular / ang_norm) * MAX_ANGULAR_VEL

        # Publish Twist command
        twist_msg = Twist()
        twist_msg.linear.x = float(cmd_vel_linear[0])
        twist_msg.linear.y = float(cmd_vel_linear[1])
        twist_msg.linear.z = float(cmd_vel_linear[2])
        twist_msg.angular.x = float(cmd_vel_angular[0])
        twist_msg.angular.y = float(cmd_vel_angular[1])
        twist_msg.angular.z = float(cmd_vel_angular[2])
        self.twist_pub.publish(twist_msg)

    def stop_robot(self):
        """
        Purpose:
        ---
        Publishes a zero-velocity Twist message to stop the arm's motion.

        Input Arguments:
        ---
        None

        Returns:
        ---
        None

        Example call:
        ---
        self.stop_robot()
        """
        stop_msg = Twist()
        self.twist_pub.publish(stop_msg)
        self.get_logger().info("Stopping robot motion.")

    def quaternion_multiply(self, q1, q2):
        """
        Purpose:
        ---
        Multiplies two quaternions q1 and q2.

        Input Arguments:
        ---
        `q1` :  [ numpy.ndarray ]
            First quaternion [x, y, z, w]
        `q2` :  [ numpy.ndarray ]
            Second quaternion [x, y, z, w]

        Returns:
        ---
        `result` :  [ numpy.ndarray ]
            Resulting quaternion [x, y, z, w]

        Example call:
        ---
        result = self.quaternion_multiply(q_target, q_current_conj)
        """
        # Quaternion components
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        # Compute quaternion product
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        return np.array([x, y, z, w])


def main(args=None):
    """
    Purpose:
    ---
    Initializes and runs the ROS 2 node for UR5 arm servoing.

    Input Arguments:
    ---
    `args` :  [ list ]
        Command-line arguments passed to rclpy.init()

    Returns:
    ---
    None

    Example call:
    ---
    main()
    """
    rclpy.init(args=args)
    arm_controller = ArmServoingController()
    try:
        rclpy.spin(arm_controller)
    except KeyboardInterrupt:
        arm_controller.get_logger().info("Keyboard interrupt, shutting down.")
    finally:
        # Ensure robot is stopped and node is destroyed
        arm_controller.stop_robot()
        arm_controller.destroy_node()


if __name__ == '__main__':
    main()
