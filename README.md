

<br />
<div align="center" id="readme-top">
  
  <h1 align="center">Sim-to-Real Embodied VLN on Husky <br> In ROS-2 Humble</h1>

  <p align="center" >





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


## Before Everything

Clone the repo:

```
git clone https://github.com/HuskyKingdom/EmbodiedVLN
cd EmbodiedVLN
```

Installation of udev rules must be done on the host machine:

```sh
./setup_udev_rules.sh
```

You should see `done` if everything works correctly.

You need to reboot the host machine to make the udev rules take effect.


## Building Docker Images

First you need to build embodied core docker, which contains husky controlling nodes and embodied middleware nodes.

```
cd husky_ws/docker
```

On amd64 & arm64 machine, build the image by the following:

```sh
docker build -f Dockerfile -t yhs/ros2_embodied_mid:latest .
```



Then install zed workspace docker, which contains zed core nodes.

```
cd zed_ws/docker
```

On amd64 & arm64 machine, build the image by the following:

```sh
docker build -f Dockerfile -t yhs/ros2_zed:latest .
```


Then install embodied logic workspace docker, which contains zed core nodes.

```
cd logic_ws/docker
```

On amd64 & arm64 machine, build the image by the following:

```sh
docker build -f Dockerfile -t yhs/ros2_logic:latest .
```




Brining up all containers:

```
cd <project folder>
docker-compose up -d
```

Note: Remember to call `docker-compose down` after you finished using them.


## ZED Core Docker



Enther the docker container and build the workspace:

```
docker exec -it ros2_zed bash

cd /home/ros2-agv-essentials/zed_ws
colcon build --symlink-install
```


Starting ZED nodes by:

```
ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zed2i
```



## Husky Core Docker



Enther the docker container and build the workspace:

```
docker exec -it ros2_embodied_mid bash

rosdep update
rosdep install --from-paths src --ignore-src --rosdistro humble -y
colcon build
```


Start up the husky control nodes first:

```
source ~/.bashrc
./script/husky-bringup.sh
```


## Embodied Logic Core Docker


Enther the docker container and build the workspace:

```
docker exec -it ros2_logic bash

rosdep update
rosdep install --from-paths src --ignore-src --rosdistro humble -y
colcon build
```


Installing requirements:

```
python3 -m pip install -r requirements.txt
```



Then start up the embodied middleware nodes by:

```
source ~/.bashrc
ros2 run logic_node embodied_core
```


## To Stop All Containers

```
docker stop $(docker ps -q) && docker rm $(docker ps -a -q)
```