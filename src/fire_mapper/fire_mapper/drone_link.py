"""Laptop side of drone 1's detector (drone/fire_detector.py runs on the QDrone 2 without ROS).

Listens on TCP for newline-delimited JSON from the drone:
  {"type":"det",  "t_cap", "t_send", "K":[fx,fy,cx,cy], "ext":[x,y,z], "dets":[{cls,score,u,v,bw,bh,z}]}
  {"type":"jpeg", "t_cap", "t_send", "data": base64 JPEG with the boxes drawn}
and publishes
  /drone1/detections/image/compressed  sensor_msgs/CompressedImage  (Foxglove Image panel)
  /drone1/detections                    vision_msgs/Detection2DArray
  /drone1/target_observations           vision_msgs/Detection3DArray in map  -> target_map
  /drone1/observation_markers           MarkerArray, one dot per raw observation (fades after a while)
plus the static TFs  drone1/base_link -> camera_link -> camera_color_optical_frame.

Box -> map: deproject the box centre with its depth (or intersect the pixel ray with the plane at
the class height if the drone had no valid depth), then tf2 to map at the capture time.
Capture time -> laptop clock: offset = min(t_recv - t_send) over the last messages (no chrony needed;
the error is the smallest one-way network delay, a few ms).
"""
import base64
import bisect
import json
import socket
import threading
import time
from collections import deque

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from builtin_interfaces.msg import Time as TimeMsg
from geometry_msgs.msg import PoseStamped, TransformStamped
from sensor_msgs.msg import CompressedImage
from vision_msgs.msg import (Detection2D, Detection2DArray, Detection3D, Detection3DArray,
                             ObjectHypothesisWithPose)
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import Buffer, TransformListener, TransformException, StaticTransformBroadcaster

from fire_mapper.geometry import make_T, deproject, ray_plane, rot_to_quat, quat_to_rot

OPT_IN_LINK = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)  # optical axes in camera_link axes
COLORS = {'fire': (1.0, 0.3, 0.1), 'tap': (0.2, 0.5, 1.0)}


def to_msg_time(t):
    sec = int(t)
    return TimeMsg(sec=sec, nanosec=int((t - sec) * 1e9))


class DroneLink(Node):
    def __init__(self):
        super().__init__('drone_link')
        self.declare_parameter('port', 18020)
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'drone1/base_link')
        self.declare_parameter('cam_xyz', [0.2634, 0.0, -0.0319])   # measured, drone body axes
        self.declare_parameter('cam_rpy', [0.0, 1.5707963, 0.0])    # pitch +90 deg = looking down
        self.declare_parameter('fallback_plane_z', 0.05)
        self.declare_parameter('fallback_classes', ['fire', 'tap'])
        self.declare_parameter('fallback_heights', [0.08, 0.12])
        self.declare_parameter('tf_timeout', 0.15)
        self.declare_parameter('dot_lifetime', 30.0)                # s a raw-observation dot stays
        # only place an observation if a REAL mocap pose exists this close to the image time;
        # otherwise tf2 would interpolate across a gap in the pose stream and invent a position
        self.declare_parameter('pose_topic', '/drone1/mocap_pose')
        self.declare_parameter('max_pose_gap', 0.1)                 # s

        g = lambda n: self.get_parameter(n).value
        self.map_frame, self.base_frame = g('map_frame'), g('base_frame')
        self.cam_xyz, self.cam_rpy = list(g('cam_xyz')), list(g('cam_rpy'))
        self.plane_z = g('fallback_plane_z')
        self.plane_by_class = dict(zip(g('fallback_classes'), g('fallback_heights')))
        self.tf_timeout = Duration(seconds=g('tf_timeout'))
        self.dot_life = g('dot_lifetime')
        self.max_pose_gap = g('max_pose_gap')
        self.pose_times = deque(maxlen=3000)                        # ~10-30 s of mocap stamps
        self.create_subscription(PoseStamped, g('pose_topic'),
                                 lambda m: self.pose_times.append(Time.from_msg(m.header.stamp).nanoseconds * 1e-9), 50)

        self.tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.static_br = StaticTransformBroadcaster(self)
        self.ext = None
        self.publish_static_tf([0.0, 0.0, 0.0])

        self.pub_img = self.create_publisher(CompressedImage, '/drone1/detections/image/compressed', 2)
        self.pub_2d = self.create_publisher(Detection2DArray, '/drone1/detections', 10)
        self.pub_obs = self.create_publisher(Detection3DArray, '/drone1/target_observations', 10)
        self.pub_dots = self.create_publisher(MarkerArray, '/drone1/observation_markers', 10)

        self.offsets = deque(maxlen=300)
        self.stats = {'frames': 0, 'jpeg': 0, 'depth': 0, 'plane': 0, 'tf_fail': 0, 'stale': 0}
        self.connected_from, self.dot_id = None, 0
        threading.Thread(target=self.serve, args=(g('port'),), daemon=True).start()
        self.create_timer(5.0, self.report)
        self.get_logger().info(f'listening on TCP :{g("port")} for drone/fire_detector.py')

    # ------------------------------------------------------------------ TF
    def publish_static_tf(self, ext):
        """base_link -> camera_link (measured mount) and camera_link -> color optical (D435 extrinsics)."""
        r, p, y = self.cam_rpy
        cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
        R = (np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]) @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
             @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]]))
        t1 = self.make_tf(self.base_frame, 'camera_link', self.cam_xyz, rot_to_quat(R))
        t_link = OPT_IN_LINK @ np.asarray(ext, float)       # color camera origin in camera_link axes
        t2 = self.make_tf('camera_link', 'camera_color_optical_frame', t_link, rot_to_quat(OPT_IN_LINK))
        self.static_br.sendTransform([t1, t2])

    def make_tf(self, parent, child, xyz, q):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id, t.child_frame_id = parent, child
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = map(float, xyz)
        (t.transform.rotation.x, t.transform.rotation.y,
         t.transform.rotation.z, t.transform.rotation.w) = map(float, q)
        return t

    def lookup(self, stamp):
        tf = self.tf_buffer.lookup_transform(self.map_frame, 'camera_color_optical_frame',
                                             Time.from_msg(stamp), self.tf_timeout)
        tr, q = tf.transform.translation, tf.transform.rotation
        return make_T([tr.x, tr.y, tr.z], [q.x, q.y, q.z, q.w])

    # ------------------------------------------------------------------ network
    def serve(self, port):
        while rclpy.ok():
            try:
                srv = socket.create_server(('0.0.0.0', port))
                break
            except OSError as e:   # e.g. another drone_link still running: keep trying, say why
                self.get_logger().error(f'cannot listen on TCP :{port} ({e}); retrying',
                                        throttle_duration_sec=5.0)
                time.sleep(1.0)
        else:
            return
        srv.settimeout(1.0)
        while rclpy.ok():
            try:
                conn, addr = srv.accept()
            except socket.timeout:
                continue
            self.connected_from = addr[0]
            self.offsets.clear()
            self.get_logger().info(f'drone connected from {addr[0]}')
            buf = b''
            conn.settimeout(2.0)
            with conn:
                while rclpy.ok():
                    try:
                        chunk = conn.recv(262144)
                    except socket.timeout:
                        continue
                    except OSError:
                        break
                    if not chunk:
                        break
                    buf += chunk
                    *lines, buf = buf.split(b'\n')
                    t_recv = time.time()
                    for line in lines:
                        try:
                            self.on_message(json.loads(line), t_recv)
                        except (ValueError, KeyError) as e:
                            self.get_logger().warn(f'bad message: {e}', throttle_duration_sec=5.0)
            self.get_logger().warn('drone disconnected')
            self.connected_from = None

    def stamp_of(self, m, t_recv):
        self.offsets.append(t_recv - m['t_send'])
        return to_msg_time(m['t_cap'] + min(self.offsets))

    def on_message(self, m, t_recv):
        stamp = self.stamp_of(m, t_recv)
        if m['type'] == 'jpeg':
            img = CompressedImage()
            img.header.stamp, img.header.frame_id = stamp, 'camera_color_optical_frame'
            img.format = 'jpeg'
            img.data = base64.b64decode(m['data'])
            self.pub_img.publish(img)
            self.stats['jpeg'] += 1
        elif m['type'] == 'det':
            self.stats['frames'] += 1
            if m.get('ext') and m['ext'] != self.ext:
                self.ext = m['ext']
                self.publish_static_tf(self.ext)
                self.get_logger().info(f'D435 color camera offset from depth camera: {self.ext} m')
            self.on_dets(m, stamp)

    # ------------------------------------------------------------------ detections
    def on_dets(self, m, stamp):
        fx, fy, cx, cy = m['K']
        a2 = Detection2DArray()
        a2.header.stamp, a2.header.frame_id = stamp, 'camera_color_optical_frame'
        for d in m['dets']:
            k = Detection2D()
            k.header = a2.header
            k.bbox.center.position.x, k.bbox.center.position.y = float(d['u']), float(d['v'])
            k.bbox.size_x, k.bbox.size_y = float(d['bw']), float(d['bh'])
            h = ObjectHypothesisWithPose()
            h.hypothesis.class_id, h.hypothesis.score = d['cls'], float(d['score'])
            k.results.append(h)
            a2.detections.append(k)
        self.pub_2d.publish(a2)
        if not m['dets']:
            return
        if not self.pose_fresh(Time.from_msg(stamp).nanoseconds * 1e-9):
            self.stats['stale'] += len(m['dets'])
            self.get_logger().warn('no tracked mocap pose near the image time (OptiTrack lost the drone, '
                                   'or the pose stream stopped) - detections skipped, not mapped',
                                   throttle_duration_sec=3.0)
            return
        try:
            T = self.lookup(stamp)
        except TransformException as e:
            self.stats['tf_fail'] += len(m['dets'])
            self.get_logger().warn(f'no drone pose at image time ({e})', throttle_duration_sec=3.0)
            return

        a3 = Detection3DArray()
        a3.header.stamp, a3.header.frame_id = stamp, self.map_frame
        dots = MarkerArray()
        for d in m['dets']:
            u, v, z = d['u'], d['v'], d['z']
            if z is not None:
                p, src = (T @ np.append(deproject(u, v, z, fx, fy, cx, cy), 1.0))[:3], 'depth'
            else:
                p = ray_plane(T, u, v, fx, fy, cx, cy, self.plane_by_class.get(d['cls'], self.plane_z))
                src = 'plane'
                if p is None:
                    continue
            self.stats[src] += 1
            o = Detection3D()
            o.header = a3.header
            o.id = src
            o.bbox.center.position.x, o.bbox.center.position.y, o.bbox.center.position.z = map(float, p)
            o.bbox.center.orientation.w = 1.0
            h = ObjectHypothesisWithPose()
            h.hypothesis.class_id, h.hypothesis.score = d['cls'], float(d['score'])
            h.pose.pose = o.bbox.center
            o.results.append(h)
            a3.detections.append(o)
            dots.markers.append(self.dot(stamp, p, d['cls']))
        if a3.detections:
            self.pub_obs.publish(a3)
            self.pub_dots.publish(dots)

    def pose_fresh(self, t):
        """True if real mocap poses bracket t closely (or the newest one is close and t is just after it)."""
        times = list(self.pose_times)
        if not times:
            return False
        i = bisect.bisect_right(times, t)
        before = times[i - 1] if i > 0 else None
        after = times[i] if i < len(times) else None
        if before is None or t - before > self.max_pose_gap:
            return False
        return after is None or after - t <= self.max_pose_gap

    def dot(self, stamp, p, cls):
        k = Marker()
        k.header.stamp, k.header.frame_id = stamp, self.map_frame
        k.ns, k.id, k.type, k.action = 'observations', self.dot_id % 100000, Marker.SPHERE, Marker.ADD
        self.dot_id += 1
        k.pose.position.x, k.pose.position.y, k.pose.position.z = map(float, p)
        k.pose.orientation.w = 1.0
        k.scale.x = k.scale.y = k.scale.z = 0.03
        k.color.r, k.color.g, k.color.b = COLORS.get(cls, (0.8, 0.8, 0.8))
        k.color.a = 0.6
        k.lifetime = Duration(seconds=self.dot_life).to_msg()
        return k

    def report(self):
        s = self.stats
        off = f'{min(self.offsets) * 1000:.0f} ms' if self.offsets else '-'
        state = f'drone {self.connected_from}' if self.connected_from else 'waiting for drone'
        self.get_logger().info(f'{state}: {s["frames"] / 5:.1f} frames/s, {s["jpeg"] / 5:.1f} images/s, '
                               f'observations depth={s["depth"]} plane={s["plane"]} no-pose={s["tf_fail"]} stale-pose={s["stale"]}, '
                               f'clock offset {off}')
        for k in s:
            s[k] = 0


def main():
    rclpy.init()
    rclpy.spin(DroneLink())


if __name__ == '__main__':
    main()
