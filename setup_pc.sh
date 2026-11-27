#!/usr/bin/env bash
# One-time setup of a ground-station PC (Ubuntu 22.04 + ROS 2 Humble) for this workspace.
#   cd ~/Documents/fire_drone_ws && ./setup_pc.sh
set -e
cd "$(dirname "$0")"

if [ ! -f /opt/ros/humble/setup.bash ]; then
  echo "ROS 2 Humble not found in /opt/ros/humble. Install it first: https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html"
  exit 1
fi
source /opt/ros/humble/setup.bash

echo "== installing build tools and rosdep"
sudo apt-get update
sudo apt-get install -y python3-rosdep python3-colcon-common-extensions python3-pytest
[ -f /etc/ros/rosdep/sources.list.d/20-default.list ] || sudo rosdep init
rosdep update

echo "== installing the workspace's ROS dependencies (vision_msgs, foxglove_bridge, vrpn_mocap, ...)"
rosdep install --from-paths src --ignore-src -y

echo "== building"
colcon build --symlink-install

echo "== unit tests"
python3 -m pytest -q src/fire_mapper/test

cat <<EOF

Setup done. In EVERY new terminal run (or add to ~/.bashrc / ~/.zshrc):
    source /opt/ros/humble/setup.bash          # setup.zsh if you use zsh
    source $(pwd)/install/setup.bash           # setup.zsh if you use zsh

Foxglove app (if not installed):  sudo snap install foxglove-studio   (or the .deb from https://foxglove.dev/download)
Test without hardware: see README.md "Quick test with fake drones".
EOF
