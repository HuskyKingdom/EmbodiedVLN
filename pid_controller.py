#!/usr/bin/env python

from __future__ import print_function

import threading

import roslib; roslib.load_manifest('teleop_twist_keyboard')
import rospy

from geometry_msgs.msg import Twist
from geometry_msgs.msg import TwistStamped

import sys
from select import select
import signal

from sensor_msgs.msg import PointCloud2, LaserScan
from sensor_msgs import point_cloud2
import time

if sys.platform == 'win32':
    import msvcrt
else:
    import termios
    import tty

from nav_msgs.msg import Odometry

TwistMsg = Twist


class DistanceTracker:
    def __init__(self):
        self.last_position = None
        self.total_distance = 0.0
        rospy.Subscriber("/husky_velocity_controller/odom", Odometry, self.odom_callback)

    def odom_callback(self, msg):
        position = msg.pose.pose.position
        current_position = np.array([position.x, position.y, position.z])

        if self.last_position is not None:
            distance = np.linalg.norm(current_position - self.last_position)
            self.total_distance += distance

        self.last_position = current_position



class PublishThread(threading.Thread):
    def __init__(self, rate):
        super(PublishThread, self).__init__()
        self.publisher = rospy.Publisher('cmd_vel', TwistMsg, queue_size = 1)
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.th = 0.0
        self.speed = 0.0
        self.turn = 0.0
        self.condition = threading.Condition()
        self.done = False

        # Set timeout to None if rate is 0 (causes new_message to wait forever
        # for new data to publish)
        if rate != 0.0:
            self.timeout = 1.0 / rate
        else:
            self.timeout = None

        self.start()

    def wait_for_subscribers(self):
        i = 0
        while not rospy.is_shutdown() and self.publisher.get_num_connections() == 0:
            if i == 4:
                print("Waiting for subscriber to connect to {}".format(self.publisher.name))
            rospy.sleep(0.5)
            i += 1
            i = i % 5
        if rospy.is_shutdown():
            raise Exception("Got shutdown request before subscribers connected")

    def update(self,speed,angular):
        if abs(angular) >= 0.2:
            self.x = 0
        else:
            self.x = 1
        self.condition.acquire()
        # self.x = 1
        self.y = 0
        self.z = 0
        self.th = 0
        self.speed = speed
        self.turn = angular
        # Notify publish thread that we have a new message.
        self.condition.notify()
        self.condition.release()

    def stop(self):
        self.done = True
        self.update(0, 0)
        self.join()

    def run(self):
        
        twist_msg = TwistMsg()
        stamped = rospy.get_param("~stamped", False)
        twist_frame = rospy.get_param("~frame_id", '')

        if stamped:
            twist = twist_msg.twist
            twist_msg.header.stamp = rospy.Time.now()
            twist_msg.header.frame_id = twist_frame
        else:
            twist = twist_msg
        while not self.done:
            if stamped:
                twist_msg.header.stamp = rospy.Time.now()
            self.condition.acquire()
            # Wait for a new message or timeout.
            self.condition.wait(self.timeout)

            # Copy state into twist message.
            twist.linear.x = self.x * self.speed
            twist.linear.y = self.y * self.speed
            twist.linear.z = self.z * self.speed
            twist.angular.x = 0
            twist.angular.y = 0
            twist.angular.z = self.turn

            self.condition.release()

            # Publish.
            self.publisher.publish(twist_msg)

    

        # Publish stop message when thread exits.
        twist.linear.x = 0
        twist.linear.y = 0
        twist.linear.z = 0
        twist.angular.x = 0
        twist.angular.y = 0
        twist.angular.z = 0
        self.publisher.publish(twist_msg)











def getKey(settings, timeout):
    if sys.platform == 'win32':
        # getwch() returns a string on Windows
        key = msvcrt.getwch()
    else:
        tty.setraw(sys.stdin.fileno())
        # sys.stdin.read() returns a string on Linux
        rlist, _, _ = select([sys.stdin], [], [], timeout)
        if rlist:
            key = sys.stdin.read(1)
        else:
            key = ''
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

def saveTerminalSettings():
    if sys.platform == 'win32':
        return None
    return termios.tcgetattr(sys.stdin)

def restoreTerminalSettings(old_settings):
    if sys.platform == 'win32':
        return
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

def vels(speed, turn):
    return "currently:\tspeed %s\tturn %s " % (speed,turn)





def spherical_to_cartesian(s):

    R = s[0]
    omega = s[1]
    alpha = s[2]
    x = R * np.cos(omega) * np.sin(alpha)
    y = R * np.cos(alpha) * np.cos(omega)
    z = R * np.sign(alpha)
    return (x, y, z)



# IMPORTS ___________________________

# Standard
import numpy as np
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Road Making System")

    parser.add_argument(
        "--wheel_radius",
        default=0.165,
        type=float,
    )

    parser.add_argument(
        "--wheel_base",
        default=0.42,
        type=float,
    )

    parser.add_argument(
        "--agent_speed",
        default=0.1,
        type=float,
    )

    parser.add_argument(
        "--target_distance",
        default=1,
        type=float,
    )


    parser.add_argument(
        "--kp",
        default=3.05,
        type=float,
    )

    parser.add_argument(
        "--ki",
        default=0,
        type=float,
    )

    parser.add_argument(
        "--kd",
        default=1.2,
        type=float,
    )


    parser.add_argument(
        "--side",  # 0-left 1=right
        default=1,
        type=int,
    )


    args = parser.parse_args()

    return args


def euclidean(point1, point2):
    """
    Calculate the Euclidean distance between two 3D points.

    Parameters:
    - point1: Tuple[float, float, float], the first point (x1, y1, z1).
    - point2: Tuple[float, float, float], the second point (x2, y2, z2).

    Returns:
    - float, the Euclidean distance between the two points.
    """
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2 + (point1[2] - point2[2]) ** 2) ** 0.5




class Road_maker:


    def __init__(self,args):

        self.args = args

        self.walked_len = 0
        self.adjusting = True
        self.back = False
        self.pre_point = None

        # task params
        self.target_distance = args.target_distance  # meters
        self.integral_error = 0
        self.previous_error = 0
        self.ki = args.ki
        self.kp = args.kp
        self.kd = args.kd


        self.forward_speed = args.agent_speed

        
        signal.signal(signal.SIGINT, self.signal_handler)

        # agent movement publisher
        speed = rospy.get_param("~speed", args.agent_speed)
        turn = rospy.get_param("~turn", 1.0)
        speed_limit = rospy.get_param("~speed_limit", 1000)
        turn_limit = rospy.get_param("~turn_limit", 1000)
        repeat = rospy.get_param("~repeat_rate", 0.0)
        key_timeout = rospy.get_param("~key_timeout", 0.5)
        stamped = rospy.get_param("~stamped", False)
        twist_frame = rospy.get_param("~frame_id", '')

        if stamped:
            TwistMsg = TwistStamped

        self.pub_thread = PublishThread(repeat)
        self.pub_thread.wait_for_subscribers()
        self.pub_thread.update(0,0)

        self.distracker = DistanceTracker() 
        # lidar subscriber
        rospy.Subscriber('/scan', LaserScan, self.lidar_callback)
        rospy.spin()


    def signal_handler(self,signal, frame):
        print('You pressed Ctrl+C!')
        self.pub_thread.stop()
        restoreTerminalSettings(settings)
        sys.exit(0)


    def get_distance(self,point):
        origin = (0,0)
        return ((point[0] - origin[0]) ** 2 + (point[1] - origin[1]) ** 2) ** 0.5




    # side: 0 - left |  1 - right
    def get_distance(self,side,data):
        
        
        if side == 0: # left
            data = data[448:]
            
            # min data
            min_data = min(data)

        else: # right
            data = data[:448]
            min_data = min(data)
            
            
        return min_data

    
    def lidar_callback(self,data):

        assert isinstance(data, LaserScan)
        
        ranges = data.ranges
        # print(data.angle_min) # -pi
        # print(data.angle_max) # +pi
        # print(data.angle_increment) # 0.007
 
        distance = self.get_distance(args.side,ranges)

        
        self.step_action(distance)

        





    
    def get_walked_len(self):

        # get walked length
        print("Walked distance {}".format(self.distracker.total_distance))
        return 5



        
    def adjust(self,wall_distance):

        error = - (self.target_distance - wall_distance)

        self.integral_error += error
        derivative_error = error - self.previous_error

        angular_velocity = (self.kp * error + self.ki * self.integral_error + self.kd * derivative_error)

        ang_th = 1.2
       
        if angular_velocity >= ang_th:
            angular_velocity = ang_th

        
        if angular_velocity <= -ang_th:
            angular_velocity = -ang_th

        
        if self.args.side == 1:
            angular_velocity = -angular_velocity

        self.previous_error = error


        print("wall dis: %.2f ; P: %.2f ; angular vel: %.2f" % (wall_distance,error, angular_velocity))

        if not self.back:
            return self.forward_speed, angular_velocity
        else:
            angular_velocity = 0.9 * error + 0.1 * derivative_error
            return -self.forward_speed, -angular_velocity



    def step_action(self,distance):
        
        if distance >= 90:
            return
        forward, angular = self.adjust(distance)
        # publish to robot
        self.pub_thread.update(forward,angular)

        if self.get_walked_len() >= 13 and self.adjusting: # adjusting forward ends
            self.adjusting = False
            self.back = True
            agent_poses = []


        if self.back and self.get_walked_len() >= 6: # back to original
            self.back = False
            agent_poses = []

       
        






args = parse_arguments()
settings = saveTerminalSettings()

rospy.init_node('pid_controller')

# pid controlloer [listening to lidar scan]
app = Road_maker(args)

