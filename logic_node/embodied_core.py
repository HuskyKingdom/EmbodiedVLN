#!/usr/bin/env python3

from __future__ import print_function
import threading
import roslib; roslib.load_manifest('teleop_twist_keyboard')
import rospy
from geometry_msgs.msg import Twist
from geometry_msgs.msg import TwistStamped
import sys
from select import select
import signal
from sensor_msgs.msg import PointCloud2, LaserScan,Image
from sensor_msgs import point_cloud2
import time
if sys.platform == 'win32':
    import msvcrt
else:
    import termios
    import tty
from nav_msgs.msg import Odometry

# Embodied VLN
from rospy.numpy_msg import numpy_msg
import cv2
from cv_bridge import CvBridge, CvBridgeError

TwistMsg = Twist


moveBindings = {
        0:(1,0,0,0),
        1:(0,0,0,1),
        2:(0,0,0,-1),
        3:(-1,0,0,0),
        4:(0,0,0,0),
    }


class observation_monitor:

    def __init__(self):
        rgb_suber = rospy.Subscriber('/zed2i/zed_node/right/image_rect_color', numpy_msg(Image), self.rgb_callback)
        depth_suber = rospy.Subscriber('/zed2i/zed_node/depth/depth_registered', numpy_msg(Image), self.dep_callback)
        self.bridge = CvBridge()


    def rgb_callback(self,data):
        rgb_image = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
        # cv2.imshow('SORA-VLN ZED2i Camera | RGB', rgb_image)
        # cv2.waitKey(1)
        print("rgb received.")

    def dep_callback(self,data):

        try:

            depth_image = self.bridge.imgmsg_to_cv2(data, "32FC1")
            print("shape {}".format(depth_image.shape))


            max_depth = 5.0
            depth_image = np.nan_to_num(depth_image)  
            depth_image[depth_image > max_depth] = max_depth  

            depth_image_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
            depth_image_normalized = depth_image_normalized.astype(np.uint8)


            cv2.imshow('SORA-VLN ZED2i Camera | Depth', depth_image_normalized)
            cv2.waitKey(1)

        except CvBridgeError as e:
            print(e)

        # depth_image = np.frombuffer(data.data, dtype=np.float32).reshape(data.height, data.width)



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

    def update(self,action_index):
        self.condition.acquire()
        self.x = moveBindings[action_index][0]
        self.y = moveBindings[action_index][1]
        self.z = moveBindings[action_index][2]
        self.th = moveBindings[action_index][3]
        self.speed = 0.5
        self.turn = 1.0
        # Notify publish thread that we have a new message.
        self.condition.notify()
        self.condition.release()

    def stop(self):
        self.done = True
        self.update(4)
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










def saveTerminalSettings():
    if sys.platform == 'win32':
        return None
    return termios.tcgetattr(sys.stdin)

def restoreTerminalSettings(old_settings):
    if sys.platform == 'win32':
        return
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)



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




class Husky_controllor:


    def __init__(self,args):

        self.args = args


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
        self.pub_thread.update(4)

        # self.distracker = DistanceTracker() 

        # waiting for action command
        self.cml_action()


    def signal_handler(self,signal, frame):
        print('You pressed Ctrl+C!')
        self.pub_thread.stop()
        restoreTerminalSettings(settings)
        sys.exit(0)


    def step_action(self,action_index): # 0-forward 1-backward 2-left15 3-right-15 4-stop

        if action_index == -1 or action_index == 4: # invalid or stop action
            return
        # publish to robot
        self.pub_thread.update(action_index)
        rospy.sleep(1.0)

    def cml_action(self):

        end_flag = 0

        while end_flag != -1:
            action_index = input("Enter an action index to perform (-1 to exit):")
            self.step_action(int(action_index))
            end_flag = action_index
        




args = parse_arguments()
settings = saveTerminalSettings()

rospy.init_node('embodied_core')

# core components
observation_handle = observation_monitor() # zed img subscriber
husky_handle = Husky_controllor(args) # husky motion publisher

rospy.spin()


cv2.destroyAllWindows()