# drone_tracker_quanser

Two Quanser QDrone 2. Drone 1 looks down with its D435, finds 3D-printed **fires** and **taps** with YOLO, and
the ground PC puts them on a map (OptiTrack frame). The confirmed targets are sent to Simulink for drone 2.
Everything is shown live in Foxglove. Details: [`src/fire_mapper/README.md`](src/fire_mapper/README.md).

```
host PC (Simulink + host/quarc_relay.py) --UDP 18010 poses--> ground PC (ROS 2: ground.launch.py) <--TCP 18020-- drone 1 (drone/fire_detector.py)
                                         <--UDP 18011 targets-- /targets/send                        --ws 8765--> Foxglove
```

## Set up a new ground PC (Ubuntu 22.04 + ROS 2 Humble)
```bash
git clone https://github.com/Josemiliowo/drone_tracker_quanser.git ~/Documents/fire_drone_ws
cd ~/Documents/fire_drone_ws
./setup_pc.sh                      # apt/rosdep dependencies, build, unit tests (asks for sudo)
```
In every new terminal (or add both lines to `~/.bashrc`):
```bash
source /opt/ros/humble/setup.bash
source ~/Documents/fire_drone_ws/install/setup.bash
```

## Quick test with fake drones (no hardware needed)
```bash
# terminal 1
ros2 launch fire_mapper ground.launch.py mocap_source:=tcp
# terminal 2
ros2 run fire_mapper fake_drone
```
Foxglove: **Open connection → Foxglove WebSocket → `ws://localhost:8765`**, then **Layouts → Import from file →**
`src/fire_mapper/foxglove/layout.json`. You should see:
- the orange drone 1 flying a circle at 1.5 m, and the cyan drone 2 parked at (-2, 0);
- the Image panel with a test image (`FAKE DRONE (test)`) and boxes;
- small dots under drone 1, then within about 30 s two **red cylinders** (fires at (1.0, 0.5) and (-0.6, -1.0)) and one
  **blue** cylinder (tap at (0.4, -1.3)).

Terminal 1 logs `tcp_pose_bridge: ... tracked RigidBody_Drone1 100%, RigidBody_Drone2 100%` and
`drone_link: drone 127.0.0.1: ~30 frames/s ... observations depth=...` and `target_map: CONFIRMED ...`.

Send the targets (prints the matrix for Simulink in MATLAB format in terminal 1):
```bash
ros2 service call /targets/send std_srvs/srv/Trigger
```
Clear the map: `ros2 service call /targets/clear std_srvs/srv/Trigger`. Unit tests: `python3 -m pytest src/fire_mapper/test`.

## Using it with the real drones on a different PC
The other machines send to the ground PC's IP, so when the ground PC changes, update:
1. `LAPTOP_IP` in `src/fire_mapper/host/quarc_relay.py` on the Simulink host PC (`ip addr` shows this PC's address).
2. `--laptop <this PC's IP>` when starting the detector on the drone:
   `python3 ~/fire_drone_ws/drone/fire_detector.py --laptop <IP> --engine ~/fire_drone_ws/models/best.engine`
3. If the PC has a firewall: `sudo ufw allow 18010/udp && sudo ufw allow 18020/tcp`.

Then `ros2 launch fire_mapper ground.launch.py mocap_source:=tcp` (flight, poses from Simulink), or
`mocap_server:=<Motive PC IP>` (bench, VRPN straight from Motive).
