# fire_mapper

Drone 1 (QDrone 2, D435 looking down) detects **fires** and **taps** with YOLO and places each
one in the OptiTrack `map` frame. The confirmed targets are re-expressed in **drone 2's start
frame** and published as goals. Foxglove shows everything live.

```
OptiTrack ─► /drone1/pose, /drone2/pose ─► mocap_tf_bridge ─► TF map→drone1/base_link, map→drone2/base_link
Jetson:  realsense ─► yolo_detector ─► detection_localizer ─► /drone1/target_observations (map)
         static TF drone1/base_link → camera_link (camera on the edge, pitch +90°)
Laptop:  target_map ─► /targets, /targets/markers, ~/fire_drone_targets.yaml
         target_dispatcher ─► /drone2/goals (PoseArray in drone2/odom), /drone2/goal_classes, TF map→drone2/odom
         foxglove_bridge :8765
```

## QDrone 2 setup (JetPack 4.6: Python 3.6, ROS Eloquent), used for the flights
The drone runs **plain Python, no ROS**: `drone/fire_detector.py` (RealSense → TensorRT YOLO11 → box + depth,
sent as JSON over TCP). All the ROS work happens on the laptop in `drone_link` (started by `ground.launch.py`):
camera → map with tf2, observations, images for Foxglove.
`launch/drone1.launch.py`, `yolo_detector`, `detection_localizer` and `docker/Dockerfile.jetson` are the
ROS-on-the-drone version. It needs Humble on the drone and is **not used** on the QDrone 2.

1. **Model (laptop, once):** `models/best.pt` is the teammate's YOLO11n (classes `fuego`/`toma`, renamed to
   `fire`/`tap`). `models/best.onnx` was exported with opset 12 for TensorRT 8.2:
   `cd models && ../.venv_yolo/bin/python -c "from ultralytics import YOLO; YOLO('best.pt').export(format='onnx', opset=12, imgsz=640, simplify=True)"`
2. **Copy to the drone (from a laptop terminal, not on the drone):**
   `rsync -av ~/Documents/fire_drone_ws/src/fire_mapper/drone ~/Documents/fire_drone_ws/models nvidia@192.168.2.16:~/fire_drone_ws/`
3. **TensorRT engine (on the drone, once, ~10–20 min):**
   `cd ~/fire_drone_ws/models && /usr/src/tensorrt/bin/trtexec --onnx=best.onnx --saveEngine=best.engine --fp16 --workspace=1024`
4. **Laptop:** `ros2 launch fire_mapper ground.launch.py mocap_source:=tcp` (or `vrpn` on the bench).
5. **Drone:** `python3 ~/fire_drone_ws/drone/fire_detector.py --laptop <laptop-ip> --engine ~/fire_drone_ws/models/best.engine`
   It prints `connected to laptop`, then the fps every 5 s. The laptop's `drone_link` log shows frames/s,
   observations (depth / plane / no-pose) and the clock offset.
6. **Foxglove:** the Image panel shows `/drone1/detections/image/compressed`. In the 3D panel, every raw observation is a small
   dot (`/drone1/observation_markers`, fades after 30 s) and every confirmed target is a cylinder (`/targets/markers`).

Test without the drone: `.venv_yolo/bin/python drone/fire_detector.py --laptop 127.0.0.1 --onnx ../../models/best.onnx --images <folder>`.
Camera offset and fallback heights for this path are in `config/params.yaml` under `drone_link`.

### Pose from Simulink during flights (reliable over WiFi)
`host/quarc_relay.py` runs on the host PC. It reads each pose from Simulink's QUARC Stream Server (:18005) as
**7 doubles: x, y, z, roll, pitch, yaw, isTracking** (Z up, X forward, Y left, radians) and sends it to the laptop as one UDP
packet (:18010). UDP never waits, so WiFi stalls can't block Simulink.
`tcp_pose_bridge` publishes two topics:
- `/mocap/RigidBody_Drone1/pose`: always (when the pose changes, otherwise repeated every 50 ms). It drives TF, so
  the drone and the `map` frame stay visible even when the drone sits still or tracking is lost.
- `/mocap/RigidBody_Drone1/pose_valid`: only while `isTracking = 1`. `drone_link` maps a detection only if one of these
  exists within `max_pose_gap` (0.1 s) of the image. Detections while OptiTrack has lost the drone are counted as
  `stale-pose` and skipped.
Every 5 s the log shows records/s, longest network gap, % tracked, how often tracking was lost, and the last values.

### Sending the targets to Simulink (drone 2)
`ros2 service call /targets/send std_srvs/srv/Trigger` (laptop) takes a snapshot of all confirmed fires and taps in
world (Simulink) coordinates. It sends them by UDP to `host/quarc_relay.py` (:18011), which hands them to Simulink as the
Stream Server's reply every cycle until the next send. The snapshot is re-sent every 0.5 s, so a lost packet doesn't matter.
Simulink's Stream Server must **receive 41 doubles**: `count = u(1); M = reshape(u(2:41), 10, 4)`, with rows
`[X Y Z Yaw]` and Yaw = 0. Rows after `count` repeat the last real target (never 0,0,0). `count = 0` means
nothing has been sent yet: don't fly. Row order is a nearest-neighbour tour from drone 2's start
(`/drone2/set_origin`), otherwise the order the targets were found. Max rows: `max_rows` in params.yaml and
`MAX_ROWS` in the relay (both 10). The snapshot is shown in Foxglove as `/targets/sent`.

## How a bounding box becomes a map point
1. `yolo_detector` produces a box, keeping the **image timestamp**.
2. `detection_localizer` takes the median depth over the central 50 % of the box (aligned depth, 0.3–3 m only).
3. It deprojects the box centre with the colour intrinsics, giving a point in `camera_color_optical_frame`.
4. tf2 transforms the point to `map` at the image time:
   `map →(mocap) drone1/base_link →(measured static) camera_link →(realsense) optical`.
5. If there's no valid depth (too high, or a reflective surface), it intersects the pixel ray with the plane
   z = `fallback_plane_z` instead. The camera looks down and the floor is z=0 in mocap, so this works well.
6. `target_map` merges observations of the same class that fall within 25 cm of each other. A target
   is **confirmed** after 5 hits, which filters out YOLO false positives.

## Setup checklist
1. **Motive**: **don't change its settings**, because Quanser's Simulink stack depends on them. Ours streams **Y-up, multicast**.
   `mocap_tf_bridge` converts Y-up to ROS Z-up (`x_map = x_m, y_map = -z_m, z_map = y_m`), so `map` x is
   Motive's x, and `map` z points up.
   Rigid bodies: `RigidBody_Drone1` → `drone1`, `RigidBody_Drone2` → `drone2` (`mocap_names` in params.yaml).
   The two drones' marker layouts must differ, or Motive can swap them.
2. **Mocap driver** (laptop): `sudo apt install ros-humble-vrpn-mocap`. In Motive's streaming settings, enable
   **VRPN** (port 3883). This adds a stream; the NatNet one Quanser uses is unchanged. Start it with
   `ros2 launch fire_mapper ground.launch.py mocap_server:=<Motive PC IP>`. It publishes `/vrpn_mocap/RigidBody_Drone1/pose`.
   **Lift test**: raise the drone. `ros2 run tf2_ros tf2_echo map drone1/base_link` must show **z** growing.
   If y grows instead, set `up_axis: 'z'` (some Motive versions always stream VRPN as Z-up).
   **Nose check**: put drone 1 on the floor with its nose pointing along Motive's +X axis (see the calibration square).
   `ros2 run tf2_ros tf2_echo map drone1/base_link` should then show yaw ≈ 0. If it shows another angle,
   enter that angle **with the opposite sign** in `body_yaw_offsets_deg`.
3. **Camera offset**: measure from the rigid-body pivot to the D435 (centre of the front glass), in body
   axes (x forward, y left, z up). Pass it as `cam_x/cam_y/cam_z` to `drone1.launch.py`.
   The D435's USB port points to the drone's **right**, which matches `cam_yaw:=0`. Check: an object in front of the drone
   should appear at the top of the image, and an object on the drone's left at the left of the image.
4. **Time sync** (important: image stamps come from the Jetson clock and TF comes from the laptop clock):
   - Laptop `/etc/chrony/chrony.conf`: add `allow 192.168.0.0/16` and `local stratum 8`.
   - Jetson `/etc/chrony/chrony.conf`: add `server <laptop-ip> iburst prefer`.
   - Check on the Jetson with `chronyc tracking`. The offset should be < 5 ms.
5. **Same DDS domain** on both machines: `export ROS_DOMAIN_ID=42` (any number, but the same on both).
6. **Laptop deps**: `sudo apt install ros-humble-vision-msgs ros-humble-foxglove-bridge`
7. **Jetson**: `docker/Dockerfile.jetson` (ROS 2 Humble, librealsense, realsense-ros, ultralytics).
   Export the teammate's model **on the Jetson**: `yolo export model=best.pt format=engine half=True imgsz=640`.
   Class names in the model must be `fire` and `tap`, or change `classes` in params.yaml.

## Run
```bash
# laptop
ros2 launch fire_mapper ground.launch.py
# Jetson (in the container)
ros2 launch fire_mapper drone1.launch.py model:=/models/best.engine cam_x:=0.20 cam_y:=0.0 cam_z:=-0.05
```
Foxglove: open a connection to `ws://<laptop-ip>:8765` and import `foxglove/layout.json`.
The 3D panel shows `/drone1/cloud` with a 60 s **decay time**, so the floor map builds up as
the drone flies. Fires are drawn as red cylinders and taps as blue ones.
If the layout import misses a setting, open `/drone1/cloud` in the 3D panel and set Decay time by hand.

### Drone 2 origin (the "origin vector")
Pick one way to set drone 2's start pose:
- `ros2 service call /drone2/set_origin std_srvs/srv/Trigger` takes a snapshot of drone 2's mocap pose. Do this just before take-off.
- Foxglove 3D panel → **Publish → Pose estimate** → click and drag where drone 2 starts (published to `/drone2/initialpose`).
  RViz "2D Pose Estimate" works the same way if you remap it to `/drone2/initialpose`.
- `auto_origin: true` in params.yaml uses the first mocap pose it receives.

The goals then appear on `/drone2/goals` (PoseArray in `drone2/odom`, nearest-neighbour order,
z = `goal_altitude`). `/drone2/goal_classes` holds the matching class list. Goals are re-published whenever
the target map changes. Clear the map with `ros2 service call /targets/clear std_srvs/srv/Trigger`.

## Tests
`python3 -m pytest src/fire_mapper/test` checks the whole camera→map chain for a downward camera
on the drone edge, the depth/ray-plane agreement, and the map→drone2 transform.

## Bench checks before flying
1. `ros2 run tf2_ros tf2_echo map camera_color_optical_frame`: z should point down (−z in map), and the position
   should match a tape measurement.
2. Carry drone 1 by hand (motors off) ~1–1.5 m above a printed fire at a known position. The target should be
   within 10 cm of it. Repeat at yaw 0/90/180°. If the error rotates with yaw, the camera offset is wrong.
3. Hold it above 3 m: the `plane` fallback should stay within ~15 cm (the localizer logs depth/plane counts).
4. Put drone 2 at a known pose, call `/drone2/set_origin`, and check one goal by hand.
5. Record flights without raw images:
   `ros2 bag record /tf /tf_static /drone1/detections /drone1/target_observations /targets /drone1/cloud`
