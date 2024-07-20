

<br />
<div align="center" id="readme-top">
  
  <h1 align="center">Docker image of running PID based wall follower on Husky</h1>

  <p align="center" >



[<img src="https://img.shields.io/badge/dockerhub-image-important.svg?logo=docker">](https://hub.docker.com/r/j3soon/ros-melodic-husky/tags)


This repo is based on [Husky Control Docker](https://github.com/j3soon/docker-ros-husky). Provides quick starting guide of using Husky A200 on ROS 1 Melodic with Velodyne Lidar sensor.  

The demonstrated program controls the robot to move smoothly N meters from the wall, you may customize the brain of the robot easily by went through the repo. [Customize guide](#customize) is also included for you to better understand the I/Os.


<br />
<a href="https://yuhang.topsoftint.com">Contact me at: <strong>me@yhscode.com</strong></a>

<a href="https://yhscode.com"><strong>View my full bio.</strong></a>
    <br />
    <br />
  </p>
</div>



## Prerequisites

Hardware:

- Husky base
- Power supply cable (for recharging the battery)
- USB cable
- Velodyne VLP-16 Lidar & Cable pack

We choose not to use the MINI-ITX computer, and control Husky directly through a Jetson board or laptop.

More information such as User Guide and Manual Installation steps can be found in [this post](https://j3soon.com/cheatsheets/clearpath-husky/).

## Installation

Clone the repo:

```
git clone https://github.com/HuskyKingdom/husky_road_making.git
cd husky_road_making
```

Installation of udev rules must be done on the host machine:

```sh
./setup_udev_rules.sh
```

You should see `done` if everything works correctly.

You need to reboot the host machine to make the udev rules take effect.

#

## Build Docker Images Locally

- On amd64 machine:

```sh
docker build -f Dockerfile -t j3soon/ros-melodic-husky:latest .
```

- On arm64 machine:

```sh
docker build -f Dockerfile.jetson -t j3soon/ros-melodic-husky:latest .
```

If you want to build an image that supports multiple architectures, please refer to the [build workflow](./.github/workflows/build.yaml).



## Running

### Running the container

Connect and power on the Husky.

Open terminal and run the following to start a container from local image:

```
docker-compose up -d
```

### Start Husky Core Nodes

Run the following to start all husky core nodes:

```
./docker-exec-bringup.sh
```

### Setting lidar
Power on your lidar and connect the ethernet cable to the laptop. Open a NEW terminal and run the following:

```
sudo ifconfig <port_name> 192.168.3.100
sudo route add 192.168.1.201 <port_name>
```
Replace `<port_name>` with the port name of your connected ethernet port. If you are not sure with this, you could check the name of ethernet ports by `ifconfig -a`.

Once finish setting up the ip configs, on the same terminal, open the runnning container in IT mode and run the lidar nodes:

```
docker exec -it ros-melodic-husky bash

roslaunch velodyne_pointcloud VLP16_points.launch
```
You can find more supports on lidar nodes in [Velodyne ROS](https://wiki.ros.org/velodyne).


### Running PID Control

Open a NEW terminal, and enter the running container:

```
docker exec -it ros-melodic-husky bash
```

Config the make file for PID package:

```
vim ~/catkin_ws/src/pid_controller/CMakeLists.txt
```
Add the following:

```
catkin_install_python(PROGRAMS scripts/pid_controller.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
```

Build the package and source the envrionment:

```
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

Running the pid node:

```
rosrun pid_controller pid_controller.py 
```


## Customize Guide
<a id="customize"></a>

### Inputs

**Lidar Scan** - The program subscribes to `/scan` topic, which was published by [velodyne_laserscan](https://wiki.ros.org/velodyne_laserscan), it contains the scaned environment depth information, you could also subscribe to `/velodyne_points` topic to obtain raw 3D points data for more fixible useage. Find out more details about raw 3D points [here](velodyne_pointcloud).

**Robot Odmetry** - The program demos a way of obtaining odmetry information from the robot, by subscribe to `/husky_velocity_controller/odom` topic, and then infer the total distance walked by the agent accordingly. Find out more about Husky control in [here](https://wiki.ros.org/husky_control).

### Outputs

**Robot Control** - The Husky robot is control by `geometry_msgs/Twist Message` like the mojority of the ROS robot, however the mechanical architecture of the Husky limits its DOF. In other words, it only supports moving in X axis and rotate along Z axis. The program provides a showcase of how to convert the partial movement order to full Twist form order. You can directly pass two simple float parameters (speed & angular) while calling PublishThread.update() to move the agent.

## Uninstall

Uninstallation of udev rules must be done on the host machine:

```sh
./remove_udev_rules.sh
```

You should see `done` if everything works correctly.

