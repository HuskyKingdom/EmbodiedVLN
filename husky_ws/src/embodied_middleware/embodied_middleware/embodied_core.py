#!/usr/bin/env python

from __future__ import print_function
import threading
import sys
from select import select
import signal
import time
if sys.platform == 'win32':
    import msvcrt
else:
    import termios
    import tty

# Embodied VLN

import cv2
from cv_bridge import CvBridge, CvBridgeError

# # ros1
# from sensor_msgs.msg import PointCloud2, LaserScan,Image
# from sensor_msgs import point_cloud2
# import roslib; roslib.load_manifest('teleop_twist_keyboard')
# import rospy

# from rospy.numpy_msg import numpy_msg


# ros2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry


TwistMsg = Twist


moveBindings = {
        0:(1,0,0,0),
        1:(0,0,0,1),
        2:(0,0,0,-1),
        3:(-1,0,0,0),
        4:(0,0,0,0),
    }


class MiddleWare(Node): # sub to obs, pub to act.

    def __init__(self):
        
        super().__init__('middleware_node')
        self.bridge = CvBridge()

        self.rgb_suber = self.create_subscription(
            Image,
            '/zed2i/zed_node/left/image_rect_color',
            self.rgb_callback,
            10)
        
        self.depth_suber = self.create_subscription(
            Image,
            '/zed2i/zed_node/depth/depth_registered',
            self.dep_callback,
            10)
        
        self.publish_thread = PublishThread(self, rate=10)
        


    def rgb_callback(self,data):
        rgb_image = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
        cv2.imshow('SORA-VLN ZED2i Camera | RGB', rgb_image)
        cv2.waitKey(1)
        
      
    def dep_callback(self,data):

        pass

        # try:

        depth_image = self.bridge.imgmsg_to_cv2(data, "32FC1")
        


        max_depth = 5.0
        depth_image = np.nan_to_num(depth_image)  
        depth_image[depth_image > max_depth] = max_depth  

        depth_image_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
        depth_image_normalized = depth_image_normalized.astype(np.uint8)

        print("depth received.")

        # cv2.imshow('SORA-VLN ZED2i Camera | Depth', depth_image_normalized)
        # cv2.waitKey(1)

        # except CvBridgeError as e:
        #     print(e)

        # depth_image = np.frombuffer(data.data, dtype=np.float32).reshape(data.height, data.width)

    def send_command(self, action_index):

        if action_index == 4: # stop action
            return
        
        self.publish_thread.update(action_index)

    def stop(self):
        self.publish_thread.stop()

class DistanceTracker(Node):
    def __init__(self):
        self.last_position = None
        self.total_distance = 0.0
        self.rgb_suber = self.create_subscription(
            Odometry,
            '/husky_velocity_controller/odom',
            self.odom_callback,
            10)


    def odom_callback(self, msg):
        position = msg.pose.pose.position
        current_position = np.array([position.x, position.y, position.z])

        if self.last_position is not None:
            distance = np.linalg.norm(current_position - self.last_position)
            self.total_distance += distance

        self.last_position = current_position






class PublishThread(threading.Thread):
    def __init__(self, node, rate):
        super(PublishThread, self).__init__()

        self.node = node
        self.publisher = node.create_publisher(Twist, 'cmd_vel', 10)
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
        while self.publisher.get_subscription_count() == 0:
            if i == 4:
                self.node.get_logger().info(f"Waiting for subscriber to connect to {self.publisher.topic_name}")
            rclpy.spin_once(self.node, timeout_sec=0.5)
            i += 1
            i = i % 5
        if not rclpy.ok():
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
        
        twist = Twist()


        while not self.done:
            
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
            self.publisher.publish(twist)

    

        # Publish stop message when thread exits.
        twist.linear.x = 0
        twist.linear.y = 0
        twist.linear.z = 0
        twist.angular.x = 0
        twist.angular.y = 0
        twist.angular.z = 0
        self.publisher.publish(twist)










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
        



def main():

    args = parse_arguments()
    settings = saveTerminalSettings()

    rclpy.init(args=None)

    midware = MiddleWare() # middleware node
    rclpy.spin(midware)

    cv2.destroyAllWindows()