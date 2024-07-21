

<br />
<div align="center" id="readme-top">
  
  <h1 align="center">Embodied VLN on Husky</h1>

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
- ZED2i Camera
- NVIDIA GPU and CUDA Installed

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


- On amd64 machine:

```sh
docker build -f Dockerfile -t j3soon/ros-melodic-husky:latest .
```

- On arm64 machine:

```sh
docker build -f Dockerfile.jetson -t j3soon/ros-melodic-husky:latest .
```

If you want to build an image that supports multiple architectures, please refer to the [build workflow](./.github/workflows/build.yaml).


## Installing ZED Camera Dependencies

Install the NVIDIA Container Toolkit following [this](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) instruction.



1. Enter the docker with cuda support:

```
docker run --gpus all -it j3soon/ros-melodic-husky bash
```


2. Download & Install ZED SKD in home directory:

```
chmod +x ZED_SDK_Ubuntu18_cuda11.1_v3.7.0.run
./ZED_SDK_Ubuntu18_cuda11.1_v3.7.0.run -- silent
```

3. Install the ROS Wrapper:

```
cd ~/catkin_ws/src
git clone --branch v3.8.x --recursive https://github.com/stereolabs/zed-ros-wrapper.git
cd ../
rosdep install --from-paths src --ignore-src -r -y
catkin_make -DCMAKE_BUILD_TYPE=Release
source ./devel/setup.bash
```



#


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

<!-- ### Setting lidar
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
You can find more supports on lidar nodes in [Velodyne ROS](https://wiki.ros.org/velodyne). -->


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




## Uninstall

Uninstallation of udev rules must be done on the host machine:

```sh
./remove_udev_rules.sh
```

You should see `done` if everything works correctly.

