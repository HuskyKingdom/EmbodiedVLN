

<br />
<div align="center" id="readme-top">
  
  <h1 align="center">Sim-to-Real Embodied VLN on Husky <br> In ROS-2 Humble</h1>

  <p align="center" >



[<img src="https://img.shields.io/badge/dockerhub-image-important.svg?logo=docker">](https://github.com/airvlab/Embodied_Husky)


This repo is based on [this repo](https://github.com/j3soon/ros2-essentials/).

We follow the same settings of the simulated VLN agent defined in VLN-CE, the inputs to the agent are egocentric RGBD images with a resolution of **256×256** and a horizontal field-of-view of 90 degree; the action space of the agent is move **forward 0.25m, turn-left or turn-right 15 degrees, or stop**.



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


- On amd64 & arm64 machine:

```sh
docker build -f Dockerfile -t yhs/embodiedvln_ros2:latest .
```


## Installing ZED Camera Dependencies

Install the NVIDIA Container Toolkit following [this](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) instruction.


Then install the ZED SDK and ZED ROS Wrappper, you need to install them with respect to your CUDA version and Linux version, note that in our implementation we use the following dependencies:

- [ZED_SDK_Ubuntu18_cuda12.1_v4.1.3.zstd.run](https://stereolabs.sfo2.cdn.digitaloceanspaces.com/zedsdk/4.1/ZED_SDK_Ubuntu22_cuda12.1_v4.1.3.zstd.run)
- [ZEDWrapper-tag-v4.0.8](https://github.com/stereolabs/zed-ros2-wrapper)



1. Install ZED udev rules on the host:

```
./zed_rules.sh
```

2. Enter the docker with cuda support, display and network support:

```
xhost +si:localuser:root

docker run --gpus all -it --network host -v /dev:/dev -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --privileged yhs/embodiedvln_ros2:latest:latest bash
```


3. Download & Install ZED SKD in `Home` directory:

```
cd /home && sudo curl -O https://stereolabs.sfo2.cdn.digitaloceanspaces.com/zedsdk/4.1/ZED_SDK_Ubuntu22_cuda12.1_v4.1.3.zstd.run

sudo chmod +x ZED_SDK_Ubuntu22_cuda12.1_v4.1.3.zstd.run

sudo ./ZED_SDK_Ubuntu22_cuda12.1_v4.1.3.zstd.run -- silent
```

4. With the corresponded verion of the CUDA Toolkit installed, creating CUDA symlink by the following:

```
sudo ln -s /usr/local/cuda-<Your Version> /usr/local/cuda
```

5. Install the ROS Wrapper:

```
cd /home/ros2-agv-essentials/husky_ws/src/ 

git clone  --recursive https://github.com/stereolabs/zed-ros2-wrapper.git
cd ..

sudo apt update

rosdep install --from-paths src --ignore-src -r -y

colcon build --symlink-install --cmake-args=-DCMAKE_BUILD_TYPE=Release --parallel-workers $(nproc)

echo source $(pwd)/install/local_setup.bash >> ~/.bashrc

source ~/.bashrc
```

5. Launch ROS Node:

```
ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zed2i
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


### Running Embodied Core Node

Open a NEW terminal, and enter the running container:

```
docker exec -it embodiedvln bash
```

Build the package and source the envrionment:

```
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```


Install the requirements:
```
pip install -r requirements.txt
```

Running the pid node:

```
rosrun embodied_vln embodied_core.py 
```




## Uninstall

Uninstallation of udev rules must be done on the host machine:

```sh
./remove_udev_rules.sh
```

You should see `done` if everything works correctly.



## Getting Start with Baselines

We examinate two baselines from VLN-CE paper, the seq-2-seq model, and the cross-attention model.

In order to run the logical components, we need another docker environment, building it by the following instructions:

```
cd logic_node
docker build -f Dockerfile -t yhs/logicnode:latest .
```

Enter the logic_node:

```
xhost +si:localuser:root

docker run --gpus all -it --network host -v /dev:/dev -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --privileged yhs/logicnode:latest bash
```

Then install torch according to your cuda version, for cuda 12.4, run the following:

```
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

### Seq-2-Seq

