#   Quick Start

## Lunch Husky Node

Connect and power on the robot, open a new terminal, run:

```
docker-compose up -d
```
Then,
```
./docker-exec-bringup.sh
```

## Lunch Lidar Node

Open a new terminal, run:

```
sudo ifconfig eth0 192.168.3.100
sudo route add 192.168.1.201 eth0
```

Then:

```
docker exec -it ros-melodic-husky bash
roslaunch velodyne_pointcloud VLP16_points.launch
```

## Keyboard Control

Open a new terminal, run:

```
./docker-exec-teleop.sh
```

## PID Wall Follow Control

Open a new terminal, run:

```
docker exec -it ros-melodic-husky bash
source devel/setup.bash
rosrun pid_controller pid_controller.py --side 0 --agent_speed 0.3 --target_distance 1
```

--side 0-left 1-right
--agent_speed 0.3
--target_distance 1