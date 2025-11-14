# Team ID:          < eYRC#1732 >
# Theme:            Krishi coBot
# Author List:      < Harshit Kumar Saxena, Abhinav Goel, Akash Dhyani, Shirshendu Ranjana Tripathi >
# Filename:         ebot_nav_task1A.py
# Functions:        odom_callback, scan_callback, clamp, angle_diff, get_lidar_range_at_angle, distance, loop, stop, main
# Global variables: waypoints, waypoints_exec

import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from tf_transformations import euler_from_quaternion


# waypoints: List of target x, y coordinates along with yaw angle for the robot to navigate.
waypoints = [
    [-1.53, -1.95, 1.57],  # P1
    [0.13, 1.24, 0.00],    # P2
    [0.38, -3.32, -1.57]   # P3
]
print(f"Had to reach these waypoints {waypoints}")

waypoints_exec = waypoints

# Adjust yaw angles slightly based on conditions to avoid exact zero or boundary values for smooth navigation.
for w in waypoints_exec:
    if w[2] < 0:
        w[2] = w[2] - 0.1
    elif w[2] > 0:
        w[2] = w[2] + 0.1
    else:
        a = waypoints_exec.index(w)
        if waypoints[a+1][2] < 0:
            w[2] += -0.1
        elif waypoints_exec[a+1][2] > 0:
            w[2] += 0.1
        else:
            w[2] = w[2]
waypoints_exec.append(waypoints_exec[-1])


class EbotNavigator(Node):
    def __init__(self):
        '''
        Purpose:
        ---
        Initializes the EbotNavigator node with ROS subscriptions, publishers,
        navigation parameters, and waypoint setup.

        Input Arguments:
        ---
        None

        Returns:
        ---
        None

        Example call:
        ---
        node = EbotNavigator()
        '''
        super().__init__('ebot_nav')
        global waypoints_exec
        self.waypoints = waypoints_exec

        # Position tolerance for waypoint reach (meters)
        self.pos_tol = 0.3
        # Yaw tolerance for angle alignment (radians)
        self.yaw_tol = math.radians(10)
        # Maximum linear velocity (m/s)
        self.max_lin_vel = 0.5
        # Maximum angular velocity (rad/s)
        self.max_ang_vel = 1.0
        # Linear velocity proportional gain
        self.K_lin = 0.8
        # Angular velocity proportional gain
        self.K_ang = 1.5
        # Side distance gain (unused in logic but initialized)
        self.K_side_dist = 2.0

        # Current waypoint index
        self.current_wp = 0
        # Navigation state: navigate_xy, navigate_yaw, or done
        self.state = 'navigate_xy'

        # Initial robot position and orientation (predefined starting location)
        self.x = -1.5339
        self.y = -6.6156
        self.yaw = 1.57
        self.scan = None

        # ROS subscriptions
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        # Publisher for robot velocity commands
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        # Timer for periodic navigation loop every 0.1 seconds
        self.create_timer(0.1, self.loop)

        self.get_logger().info('Ebot Navigator started with sequential yaw then position logic.')

    def odom_callback(self, msg):
        '''
        Purpose:
        ---
        Updates the robot's current position and orientation based on Odometry messages.

        Input Arguments:
        ---
        `msg` :  [ Odometry ]
            Odometry message containing position and orientation data

        Returns:
        ---
        None

        Example call:
        ---
        Called automatically when new /odom messages arrive
        '''
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _, _, self.yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])

    def scan_callback(self, msg):
        '''
        Purpose:
        ---
        Updates the latest LaserScan data for obstacle sensing.

        Input Arguments:
        ---
        `msg` :  [ LaserScan ]
            Laser scan message with range data

        Returns:
        ---
        None

        Example call:
        ---
        Called automatically when new /scan messages arrive
        '''
        self.scan = msg

    def clamp(self, v, vmin, vmax):
        '''
        Purpose:
        ---
        Clamps a value 'v' between minimum 'vmin' and maximum 'vmax'.

        Input Arguments:
        ---
        `v` :  [ float ]
            Value to clamp

        `vmin` :  [ float ]
            Minimum allowable value

        `vmax` :  [ float ]
            Maximum allowable value

        Returns:
        ---
        `v_clamped` :  [ float ]
            Value clamped to the range [vmin, vmax]

        Example call:
        ---
        val = self.clamp(2.5, 0, 1.0)
        '''
        return max(vmin, min(vmax, v))

    def angle_diff(self, a, b):
        '''
        Purpose:
        ---
        Computes the signed shortest difference between two angles 'a' and 'b'.

        Input Arguments:
        ---
        `a` :  [ float ]
            First angle in radians

        `b` :  [ float ]
            Second angle in radians

        Returns:
        ---
        `diff` :  [ float ]
            Angle difference within [-pi, pi]

        Example call:
        ---
        diff = self.angle_diff(current_yaw, target_yaw)
        '''
        diff = b - a
        while diff > math.pi:
            diff -= 2 * math.pi
        while diff < -math.pi:
            diff += 2 * math.pi
        return diff

    def get_lidar_range_at_angle(self, angle_deg):
        '''
        Purpose:
        ---
        Returns the LIDAR distance measurement at a specified angle in degrees.

        Input Arguments:
        ---
        `angle_deg` :  [ float ]
            Angle in degrees at which range is queried from LIDAR scan

        Returns:
        ---
        `range_val` :  [ float ]
            Distance reading at the specified angle, or infinity if invalid

        Example call:
        ---
        dist = self.get_lidar_range_at_angle(0)
        '''
        if not self.scan:
            return float('inf')
        angle_min = self.scan.angle_min
        angle_inc = self.scan.angle_increment
        n = len(self.scan.ranges)
        angle_rad = math.radians(angle_deg)
        idx = int(round((angle_rad - angle_min) / angle_inc)) % n
        r = self.scan.ranges[idx]
        return r if self.scan.range_min < r < self.scan.range_max else float('inf')

    def distance(self, p1, p2):
        '''
        Purpose:
        ---
        Computes Euclidean distance between two points in 2D.

        Input Arguments:
        ---
        `p1` :  [ tuple of float ]
            Coordinates of first point (x1, y1)

        `p2` :  [ tuple of float ]
            Coordinates of second point (x2, y2)

        Returns:
        ---
        `dist` :  [ float ]
            Euclidean distance between p1 and p2

        Example call:
        ---
        dist = self.distance((x1, y1), (x2, y2))
        '''
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def loop(self):
        '''
        Purpose:
        ---
        Main navigation control loop called periodically to move the robot through waypoints.
        Implements waypoint reaching with sequential position then yaw alignment and obstacle avoidance based on LIDAR.

        Input Arguments:
        ---
        None

        Returns:
        ---
        None

        Example call:
        ---
        Called automatically every 0.1 seconds by ROS timer
        '''
        if None in (self.x, self.y, self.yaw, self.scan):
            # Wait until all necessary data is available
            return
        if self.current_wp >= len(self.waypoints):
            if self.state != 'done':
                self.stop()
                self.state = 'done'
                self.get_logger().info("Mission Complete!")
            return

        tx, ty, tyaw = self.waypoints[self.current_wp]
        pos_err = self.distance((self.x, self.y), (tx, ty))
        yaw_err = self.angle_diff(self.yaw, tyaw)

        range_front = self.get_lidar_range_at_angle(0)
        range_left_80 = self.get_lidar_range_at_angle(90)
        range_right_80 = self.get_lidar_range_at_angle(-90)
        range_side = max(range_left_80, range_right_80)

        cmd = Twist()

        if range_front == float('inf') or range_front > range_side or range_front > 0.6:
            # Prioritize position navigation first then yaw alignment
            if self.state != 'navigate_xy' and self.state != 'done':
                self.get_logger().info("State: navigating position first, then yaw")
            if self.state == 'navigate_xy':
                if pos_err < self.pos_tol:
                    self.state = 'navigate_yaw'
                    cmd.linear.x = 0.0
                else:
                    heading = math.atan2(ty - self.y, tx - self.x)
                    ang_err = self.angle_diff(self.yaw, heading)
                    # Angular velocity control commented out in original code
                    # cmd.angular.z = self.clamp(self.K_ang * ang_err, -self.max_ang_vel, self.max_ang_vel)
                    cmd.linear.x = self.clamp(self.K_lin * pos_err, 0.0, self.max_lin_vel)

            elif self.state == 'navigate_yaw':
                if abs(yaw_err) < self.yaw_tol:
                    self.current_wp += 1
                    if self.current_wp >= len(self.waypoints):
                        self.state = 'done'
                    else:
                        self.state = 'navigate_xy'
                    cmd.linear.x = 0.0
                else:
                    cmd.angular.z = self.clamp(self.K_ang * yaw_err, -self.max_ang_vel, self.max_ang_vel)
                    cmd.linear.x = 0.0

        elif range_front < range_side and range_front < 0.6:
            # Obstacle detected ahead, prioritize yaw navigation
            if self.state != 'navigate_yaw' and self.state != 'done':
                self.get_logger().info("State: switching to yaw navigation because side path is wider")

            if self.state != 'navigate_yaw':
                self.state = 'navigate_yaw'

            if abs(yaw_err) >= self.yaw_tol:
                cmd.angular.z = self.clamp(self.K_ang * yaw_err, -self.max_ang_vel, self.max_ang_vel)
                cmd.linear.x = 0.0
            else:
                if self.state != 'navigate_xy':
                    self.state = 'navigate_xy'

                if pos_err < self.pos_tol:
                    self.current_wp += 1
                    self.state = 'navigate_xy' if self.current_wp < len(self.waypoints) else 'done'
                    cmd.linear.x = 0.0
                else:
                    heading = math.atan2(ty - self.y, tx - self.x)
                    ang_err = self.angle_diff(self.yaw, heading)
                    # Angular velocity control commented out in original code
                    # cmd.angular.z = self.clamp(self.K_ang * ang_err, -self.max_ang_vel, self.max_ang_vel)
                    cmd.linear.x = self.clamp(self.K_lin * pos_err, 0.0, self.max_lin_vel)

        # Publish velocity command
        self.cmd_pub.publish(cmd)

    def stop(self):
        '''
        Purpose:
        ---
        Stops the robot by publishing zero velocities.

        Input Arguments:
        ---
        None

        Returns:
        ---
        None

        Example call:
        ---
        self.stop()
        '''
        self.cmd_pub.publish(Twist())


def main():
    '''
    Purpose:
    ---
    Initializes the ROS2 node and starts the navigation event loop.

    Input Arguments:
    ---
    None

    Returns:
    ---
    None

    Example call:
    ---
    Called automatically when script is run as main program.
    '''
    rclpy.init()
    node = EbotNavigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
