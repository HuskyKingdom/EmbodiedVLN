#!/bin/bash -e

# Add clearpath sources
# Ref: https://github.com/clearpathrobotics/clearpath_computer_installer/blob/9ba06561ec19c72dd79943c210bfee6b51dcf4dc/clearpath_computer_installer.sh#L152-L153
wget https://packages.clearpathrobotics.com/public.key -O - | sudo apt-key add -
sudo bash -c 'echo "deb https://packages.clearpathrobotics.com/stable/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/clearpath-latest.list'
sudo apt-get update

# Create custom global workspace
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
# `clearpath_computer_installer.sh` requires `ros-humble-clearpath-robot`.
# Install from source since package doesn't exist on arm64, error message:
#
#     E: Unable to locate package ros-humble-clearpath-robot
#
git clone https://github.com/clearpathrobotics/clearpath_robot -b 0.2.11
cd ..

# Remove unnecessary dependencies (`micros_ros_agent`, `sevcon_traction`, `umx_driver`, `valence_bms_driver`)
# Ref: https://github.com/clearpathrobotics/clearpath_robot/blob/40b1c40a7d229ede7a674cb0fb359fc83c754adb/.github/workflows/ci.yml#L15-L16

# Temporarily remove `micro_ros_agent` dependency
# Ref: https://github.com/clearpathrobotics/clearpath_robot/pull/19
# Ref: https://github.com/clearpathrobotics/clearpath_robot/pull/22
# I think this package isn't used anyway.
sed -i 's/<exec_depend>micro_ros_agent<\/exec_depend>/<!-- <exec_depend>micro_ros_agent<\/exec_depend> -->/' ~/ros2_ws/src/clearpath_robot/clearpath_generator_robot/package.xml

# Temporarily remove `sevcon_traction` dependency
# Ref: https://github.com/clearpathrobotics/clearpath_robot/commit/2ef406f30c074e578db6ea799f5c2714bce1c15d
# This package is only used for W200, so it isn't required for Husky.
# Ref: https://github.com/clearpathrobotics/clearpath_robot/blob/40b1c40a7d229ede7a674cb0fb359fc83c754adb/clearpath_generator_robot/clearpath_generator_robot/launch/generator.py#L220
# On the other hand, we don't have access to the source code anyway.
# Ref: https://github.com/clearpathrobotics/public-rosdistro/blob/master/humble/distribution.yaml#L262-L279
sed -i 's/<exec_depend>sevcon_traction<\/exec_depend>/<!-- <exec_depend>sevcon_traction<\/exec_depend> -->/' ~/ros2_ws/src/clearpath_robot/clearpath_generator_robot/package.xml

# Temporarily remove `umx_driver` dependency
# This package is only used for redshift and chrobotics, so it isn't required for our case.
# Ref: https://github.com/clearpathrobotics/clearpath_robot/blob/40b1c40a7d229ede7a674cb0fb359fc83c754adb/clearpath_sensors/launch/redshift_um7.launch.py#L69
# Ref: https://github.com/clearpathrobotics/clearpath_robot/blob/40b1c40a7d229ede7a674cb0fb359fc83c754adb/clearpath_sensors/launch/chrobotics_um6.launch.py#L69
sed -i 's/<exec_depend>umx_driver<\/exec_depend>/<!-- <exec_depend>umx_driver<\/exec_depend> -->/' ~/ros2_ws/src/clearpath_robot/clearpath_sensors/package.xml

# Temporarily remove `valence_bms_driver` dependency
# Ref: https://github.com/clearpathrobotics/clearpath_robot/pull/47
# This type of batteries are only used in W200, so they aren't required for Husky.
# Ref: https://github.com/clearpathrobotics/clearpath_robot/blob/40b1c40a7d229ede7a674cb0fb359fc83c754adb/clearpath_generator_robot/clearpath_generator_robot/launch/generator.py#L152-L156
# Ref: https://github.com/clearpathrobotics/clearpath_config/blob/996eb50d0b05c87b65b8ffddcddd33239abd422e/clearpath_config/platform/battery.py#L51-L54
sed -i 's/<exec_depend>valence_bms_driver<\/exec_depend>/<!-- <exec_depend>valence_bms_driver<\/exec_depend> -->/' ~/ros2_ws/src/clearpath_robot/clearpath_generator_robot/package.xml

# Continue building the workspace
cd ~/ros2_ws
rosdep update
rosdep install -i --from-path src --rosdistro humble -y
colcon build
