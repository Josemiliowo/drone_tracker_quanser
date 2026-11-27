"""Express the confirmed targets in drone 2's own frame (drone2/odom) and publish them as goals.

drone2/odom = drone 2's start pose in map (yaw only, level). It is set by either:
  a) ros2 service call /drone2/set_origin std_srvs/srv/Trigger   -> snapshot of drone 2's mocap pose
  b) a pose on /drone2/initialpose (Foxglove "Publish pose estimate" / RViz "2D Pose Estimate")
     = the manual "origin vector"
  c) auto_origin:=true -> first mocap pose received
Outputs (latched):
  /drone2/goals         geometry_msgs/PoseArray in drone2/odom, greedy nearest-neighbour order
  /drone2/goal_classes  std_msgs/String, JSON list [{"id","class"}] in the same order
  TF map -> drone2/odom (static)

Sending the targets to Simulink (drone 2's controller):
  ros2 service call /targets/send std_srvs/srv/Trigger
takes a SNAPSHOT of the confirmed fires/taps in world (map = Simulink) coordinates and sends it by UDP
to host/quarc_relay.py, which hands it to Simulink through the QUARC Stream Server reply, every cycle.
The snapshot is re-sent every `resend_period` s (UDP may drop a packet) until the next /targets/send.
Packet = 1 + 4*max_rows doubles: [count; X(1..N); Y(1..N); Z(1..N) = send_z; Yaw(1..N) = 0]  (column-major, so in
Simulink: count = u(1); M = reshape(u(2:end), N, 4) gives rows [X Y Z Yaw]).
Unused rows repeat the LAST real target, so a drone that walks all N rows just stays there.
Rows are a nearest-neighbour tour from drone 2's start (origin if set), else in detection order.
/targets/sent (PoseArray in map, latched) shows the snapshot in Foxglove.
"""
import json
import socket
import struct

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, qos_profile_sensor_data
from geometry_msgs.msg import PoseArray, Pose, PoseStamped, PoseWithCovarianceStamped, TransformStamped
from std_msgs.msg import String
from std_srvs.srv import Trigger
from vision_msgs.msg import Detection3DArray
from tf2_ros import StaticTransformBroadcaster

from fire_mapper.geometry import make_T, yaw_only, rot_to_quat


def nn_tour(points, start=None):
    """Indices of `points` (Nx2+) in greedy nearest-neighbour order from `start` (xy); None = keep order."""
    if start is None:
        return list(range(len(points)))
    order, cur, left = [], np.asarray(start[:2], float), list(range(len(points)))
    while left:
        i = min(left, key=lambda k: np.linalg.norm(np.asarray(points[k][:2]) - cur))
        order.append(i)
        left.remove(i)
        cur = np.asarray(points[i][:2], float)
    return order


def pack_for_simulink(rows, max_rows):
    """rows: list of (x, y, z, yaw). -> list of 1 + 4*max_rows floats, column-major, padded with the last row."""
    rows = [tuple(map(float, r)) for r in rows[:max_rows]]
    count = len(rows)
    if rows:
        rows = rows + [rows[-1]] * (max_rows - count)      # pad by repeating the last real target
    else:
        rows = [(0.0, 0.0, 0.0, 0.0)] * max_rows          # nothing found: count = 0, all zeros
    cols = [v for c in range(4) for v in (r[c] for r in rows)]
    return [float(count)] + cols


def matlab_matrix(vals, max_rows):
    """Packed Simulink vector -> 'M = [x1,y1,z1,t1;...]' exactly as reshape(u(2:end), N, 4) gives it."""
    cols = [vals[1 + c * max_rows:1 + (c + 1) * max_rows] for c in range(4)]
    rows = ['%.3f,%.3f,%.3f,%.3f' % tuple(cols[c][i] for c in range(4)) for i in range(max_rows)]
    return 'count = %d;\nM = [%s];' % (int(vals[0]), ';'.join(rows))


class TargetDispatcher(Node):
    def __init__(self):
        super().__init__('target_dispatcher')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('odom_frame', 'drone2/odom')
        self.declare_parameter('drone2_pose_topic', '/drone2/mocap_pose')
        self.declare_parameter('auto_origin', False)
        self.declare_parameter('goal_altitude', 0.0)  # >0: replace goal z with this height (drone2/odom)
        self.declare_parameter('classes', ['fire', 'tap'])  # which classes to send
        self.declare_parameter('simulink_ip', '192.168.2.11')   # host PC running quarc_relay.py
        self.declare_parameter('simulink_port', 18011)
        self.declare_parameter('max_rows', 10)                  # must match Simulink's receive size
        self.declare_parameter('resend_period', 0.5)            # s
        self.declare_parameter('send_z', 0.5)                   # m; Z sent to Simulink for every target (flight height)

        g = lambda n: self.get_parameter(n).value
        self.map_frame, self.odom_frame = g('map_frame'), g('odom_frame')
        self.auto_origin, self.alt = g('auto_origin'), g('goal_altitude')
        self.classes = set(g('classes'))
        self.T_map_odom = None
        self.last_d2 = None
        self.targets = None

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_goals = self.create_publisher(PoseArray, '/drone2/goals', latched)
        self.pub_cls = self.create_publisher(String, '/drone2/goal_classes', latched)
        self.static_br = StaticTransformBroadcaster(self)
        self.create_subscription(PoseStamped, g('drone2_pose_topic'), self.on_d2_pose, qos_profile_sensor_data)
        self.create_subscription(PoseWithCovarianceStamped, '/drone2/initialpose', self.on_initialpose, 10)
        self.create_subscription(Detection3DArray, '/targets', self.on_targets, latched)
        self.create_service(Trigger, '/drone2/set_origin', self.on_set_origin)

        self.sim_addr = (g('simulink_ip'), int(g('simulink_port')))
        self.max_rows = int(g('max_rows'))
        self.send_z = float(g('send_z'))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        self.packet = None                                       # current snapshot (bytes)
        self.pub_sent = self.create_publisher(PoseArray, '/targets/sent', latched)
        self.create_service(Trigger, '/targets/send', self.on_send)
        self.create_timer(g('resend_period'), self.resend)

    @staticmethod
    def pose_to_T(p):
        return make_T([p.position.x, p.position.y, p.position.z],
                      [p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w])

    def on_d2_pose(self, msg: PoseStamped):
        self.last_d2 = msg.pose
        if self.auto_origin and self.T_map_odom is None:
            self.set_origin(self.pose_to_T(msg.pose), 'auto (first mocap pose)')

    def on_set_origin(self, req, resp):
        if self.last_d2 is None:
            resp.success, resp.message = False, 'no drone2 mocap pose received yet'
        else:
            self.set_origin(self.pose_to_T(self.last_d2), 'mocap snapshot')
            resp.success, resp.message = True, 'origin set from drone2 mocap pose'
        return resp

    def on_initialpose(self, msg: PoseWithCovarianceStamped):
        if msg.header.frame_id and msg.header.frame_id != self.map_frame:
            self.get_logger().warn(f'initialpose in {msg.header.frame_id}, expected {self.map_frame}; ignored')
            return
        self.set_origin(self.pose_to_T(msg.pose.pose), 'manual initialpose')

    def set_origin(self, T, source):
        self.T_map_odom = yaw_only(T)
        x, y, z = self.T_map_odom[:3, 3]
        yaw = np.degrees(np.arctan2(self.T_map_odom[1, 0], self.T_map_odom[0, 0]))
        self.get_logger().info(f'drone2 origin ({source}): x={x:.2f} y={y:.2f} z={z:.2f} yaw={yaw:.1f} deg')
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id, t.child_frame_id = self.map_frame, self.odom_frame
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = float(x), float(y), float(z)
        q = rot_to_quat(self.T_map_odom[:3, :3])
        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = map(float, q)
        self.static_br.sendTransform(t)
        self.dispatch()

    def on_targets(self, msg: Detection3DArray):
        self.targets = [(d.id, d.results[0].hypothesis.class_id,
                         np.array([d.bbox.center.position.x, d.bbox.center.position.y, d.bbox.center.position.z]))
                        for d in msg.detections if d.results[0].hypothesis.class_id in self.classes]
        self.dispatch()

    def on_send(self, req, resp):
        targets = sorted(self.targets or [], key=lambda t: int(t[0]) if str(t[0]).isdigit() else 0)
        if self.T_map_odom is not None:
            start = self.T_map_odom[:2, 3]
        elif self.last_d2 is not None:
            start = (self.last_d2.position.x, self.last_d2.position.y)
        else:
            start = None
        order = [targets[i] for i in nn_tour([t[2] for t in targets], start)]
        if len(order) > self.max_rows:
            self.get_logger().warn(f'{len(order)} targets but Simulink takes {self.max_rows}: sending the first '
                                   f'{self.max_rows} of the tour')
        rows = [(p[0], p[1], self.send_z, 0.0) for _, _, p in order]   # only X, Y matter; Z fixed, Yaw 0
        vals = pack_for_simulink(rows, self.max_rows)
        self.packet = struct.pack('<%dd' % len(vals), *vals)
        self.resend()

        pa = PoseArray()
        pa.header.stamp, pa.header.frame_id = self.get_clock().now().to_msg(), self.map_frame
        for x, y, z, _ in rows[:self.max_rows]:
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = float(x), float(y), float(z)
            pose.orientation.w = 1.0
            pa.poses.append(pose)
        self.pub_sent.publish(pa)

        listing = ', '.join(f'{cls}#{tid}=({p[0]:.2f},{p[1]:.2f},{self.send_z:.2f})' for tid, cls, p in order[:self.max_rows])
        msg = (f'sent {min(len(order), self.max_rows)} targets to {self.sim_addr[0]}:{self.sim_addr[1]} '
               f'(tour from {"drone 2 start" if start is not None else "detection order"}): {listing or "none"}')
        matrix = matlab_matrix(vals, self.max_rows)               # what Simulink receives, MATLAB syntax
        self.get_logger().info(msg + '\n' + matrix)
        resp.success, resp.message = True, msg + '\n' + matrix
        return resp

    def resend(self):
        if self.packet is None:
            return
        try:
            self.sock.sendto(self.packet, self.sim_addr)
        except OSError as e:
            self.get_logger().warn(f'could not send targets to {self.sim_addr}: {e}', throttle_duration_sec=5.0)

    def dispatch(self):
        if self.T_map_odom is None or self.targets is None:
            return
        T_odom_map = np.linalg.inv(self.T_map_odom)
        pts = [(tid, cls, (T_odom_map @ np.append(p, 1.0))[:3]) for tid, cls, p in self.targets]

        # greedy nearest-neighbour tour starting at drone 2's origin
        order, cur, left = [], np.zeros(2), list(pts)
        while left:
            i = min(range(len(left)), key=lambda k: np.linalg.norm(left[k][2][:2] - cur))
            order.append(left.pop(i))
            cur = order[-1][2][:2]

        pa = PoseArray()
        pa.header.stamp = self.get_clock().now().to_msg()
        pa.header.frame_id = self.odom_frame
        for _, _, p in order:
            pose = Pose()
            pose.position.x, pose.position.y = float(p[0]), float(p[1])
            pose.position.z = float(self.alt if self.alt > 0 else p[2])
            pose.orientation.w = 1.0
            pa.poses.append(pose)
        self.pub_goals.publish(pa)
        self.pub_cls.publish(String(data=json.dumps([{'id': tid, 'class': cls} for tid, cls, _ in order])))


def main():
    rclpy.init()
    rclpy.spin(TargetDispatcher())


if __name__ == '__main__':
    main()
