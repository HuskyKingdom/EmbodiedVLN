#!/usr/bin/env python3

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

from .utils.common import text_to_tensor,load_vocab,generate_random_obs


# import sys
# import types

# class MockHabitat:
#     def __getattr__(self, name):
#         return None

# sys.modules['habitat'] = MockHabitat()
# sys.modules['habitat.config'] = MockHabitat()
# sys.modules['habitat.config.default'] = MockHabitat()
# sys.modules['habitat.config.default.Config'] = MockHabitat()



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
from std_msgs.msg import Int8
import gym

TwistMsg = Twist

# [STOP, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT]

# moveBindings = {
#         0:(1,0,0,0),
#         1:(0,0,0,1),
#         2:(0,0,0,-1),
#         3:(-1,0,0,0),
#         4:(0,0,0,0),
#     }

moveBindings = {
        0:(0,0,0,0),
        1:(1,0,0,0),
        2:(0,0,0,1),
        3:(0,0,0,-1),
    }

class MiddleWare(Node): # sub to obs, pub to act.

    def __init__(self):
        
        super().__init__('middleware_node')
        self.bridge = CvBridge()

        self.rgb_suber = self.create_subscription(
            Image,
            '/zed/zed_node/left/image_rect_color',
            self.rgb_callback,
            10)
        
        self.depth_suber = self.create_subscription(
            Image,
            '/zed/zed_node/depth/depth_registered',
            self.dep_callback,
            10)
        
        self.publish_thread = PublishThread(self, rate=10)

        self.start_cml()

        # # obervation buffers
        # self.obs_space = gym.spaces.Dict({
        #     "instruction": gym.spaces.Box(low=0, high=255, shape=(1,), dtype=np.uint8),
        #     "rgb": spaces.Box(low=0, high=255, shape=(256, 256, 3), dtype=np.uint8),
        #     "depth": spaces.Box(low=0, high=1, shape=(256, 256, 1), dtype=np.float32),
        # })
        # self.obs_buffer = generate_random_obs(self.obs_space)
        
        # # tokenlize instruction
        # self.vocab = "/home/ros2-agv-essentials/deeplab_ws/src/logic_node/logic_node/data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz"
        # self.vocab = load_vocab(self.vocab)
        # inter_text = input("Enter a new textual instruction here:")
        # self.obs_buffer["instruction"] = text_to_tensor(inter_text,self.vocab)

        


    def action(self,data):
        self.send_command(data)
        
        
    def clear_buffers(self):

        self.obs_buffer = generate_random_obs(self.obs_space)

        inter_text = input("Enter a new textual instruction here:")
        self.obs_buffer["instruction"] = text_to_tensor(inter_text,self.vocab)

    
    def resample_random(self):

        self.obs_buffer = generate_random_obs(self.obs_space)




    def rgb_callback(self,data):
        rgb_image = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
        self.obs_buffer["rgb"] = rgb_image
        
      
    def dep_callback(self,data):

        depth_image = self.bridge.imgmsg_to_cv2(data, "32FC1")
        max_depth = 5.0
        depth_image = np.nan_to_num(depth_image)  
        depth_image[depth_image > max_depth] = max_depth  

        depth_image_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
        depth_image_normalized = depth_image_normalized.astype(np.uint8)
        self.obs_buffer["depth"] = depth_image_normalized

        # depth_image = np.frombuffer(data.data, dtype=np.float32).reshape(data.height, data.width)

    def send_command(self, action_index):

        # if action_index == 4: # stop action
        #     return
        self.publish_thread.update(action_index)

        time.sleep(1.0)

        self.publish_thread.update(0) # 'stop' action


    def stop(self):
        self.publish_thread.stop()

    def start_cml(self):
        inference_thread = threading.Thread(target=self.cml_action)
        inference_thread.start()
        
    def cml_action(self):

        end_flag = 0
        while end_flag != -1:
            action_index = input("Enter an action index to perform (-1 to exit):")
            self.send_command(int(action_index))
            end_flag = action_index

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
        self.publisher = node.create_publisher(Twist, 'a200_0000/cmd_vel', 10)
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
        self.speed = 0.25 # 0.25 m/s
        self.turn = 0.26 # 15 deg/s
        # Notify publish thread that we have a new message.
        self.condition.notify()
        self.condition.release()

    def stop(self):
        self.done = True
        self.update(0)
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
            twist.angular.x = 0.0
            twist.angular.y = 0.0
            twist.angular.z = self.th * self.turn

            self.condition.release()

            # Publish.
            self.publisher.publish(twist)

    

        # Publish stop message when thread exits.
        twist.linear.x = 0.0
        twist.linear.y = 0.0
        twist.linear.z = 0.0
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = 0.0
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







from gym import spaces

# logic 
from .policy.cma_policy import CMAPolicy
from .utils.common import batch_obs
import torch




class CORE_FUNC():

    def __init__(self):

        # init
        self.device = "cuda"

        # middleware
        self.middleware = MiddleWare()
        # self.observations = self.middleware.obs_buffer

        # action_space = 4

        # obs_space = gym.spaces.Dict({
        #     "instruction": gym.spaces.Box(low=0, high=255, shape=(1,), dtype=np.uint8),
        #     "rgb": spaces.Box(low=0, high=255, shape=(256, 256, 3), dtype=np.uint8),
        #     "depth": spaces.Box(low=0, high=1, shape=(256, 256, 1), dtype=np.float32),
        # })

        # # load policy
        # self.policy = CMAPolicy(obs_space,action_space)
        # self.policy.to("cuda")
        # ckpt_dict = torch.load("/home/ros2-agv-essentials/deeplab_ws/src/logic_node/logic_node/data/checkpoints/CMA_PM_DA_Aug.pth",map_location="cpu")
        # self.policy.load_state_dict(ckpt_dict["state_dict"])
        # self.policy.eval()

        # input("Start Inference?")

        # self.inference()

    

    def inference(self):
        
        # network reset       
        inf_action = -1

        rnn_states = torch.zeros(
            1,
            self.policy.net.num_recurrent_layers,
            512,
            device=self.device,
        )
        prev_actions = torch.zeros(
            1, 1, device=self.device, dtype=torch.long
        )
        not_done_masks = torch.zeros(
            1, 1, dtype=torch.uint8, device=self.device
        )

        # model inference____

        while inf_action != 0: # done is not called
            
            # get obs
            self.observations = self.middleware.obs_buffer
            batch = batch_obs(self.observations,"cuda")
            with torch.no_grad():
                actions, rnn_states = self.policy.act(
                    batch,
                    rnn_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=False,
                )

                print(f"Logits {self.policy.get_distributions(batch,rnn_states,prev_actions,not_done_masks)} | Action {actions[0].item()} Performed...")
                
                prev_actions.copy_(actions)
                output = actions[0].item()
                # send action
                self.middleware.send_command(output)

                

                inf_action = output


                self.middleware.resample_random() # rm, testing only

                




def main():

    print("Embodied Middleware Started...")
    args = parse_arguments()
    settings = saveTerminalSettings()

    rclpy.init(args=None)
    core = CORE_FUNC()

    rclpy.spin(core.middleware)
    print("spined ...")
    core.middleware.destroy_node()
    rclpy.spin.shutdown()


    


if __name__ == '__main__':
    main()