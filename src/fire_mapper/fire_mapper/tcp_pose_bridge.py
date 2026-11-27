"""Receive mocap poses as raw bytes over TCP (from Simulink/QUARC or a Python relay) and
publish them as geometry_msgs/PoseStamped on /mocap/<body>/pose, just like vrpn_mocap does.

Each TCP "record" is a fixed number of numbers, back to back, no framing:
    for each body in `bodies`: the values listed in `fields`
`fields` names (any order, per body):
    x y z                 position (m)
    qx qy qz qw           quaternion            -- or --
    roll pitch yaw        Euler angles, ZYX (yaw-pitch-roll), `angles_in_degrees` selects units
    tracked               OptiTrack isTracking flag (> 0.5 = tracked). Without it every pose counts as tracked.
    skip                  ignore this value (padding, timestamps, flags, ...)
    a leading '-' negates the value, e.g. '-z' or '-yaw' (axis/sign conventions differ)
Values are `dtype` ('f8' = double, 'f4' = single) with `byte_order` ('<' little, '>' big).
QUARC stream blocks send native order (little endian on Windows/x86 and the Jetson).

mode 'udp':    listen for UDP datagrams on `port`, one record per datagram (recommended over WiFi:
              the sender never blocks, a lost packet only loses that one pose).
mode 'server': listen on `port`, the sender connects (QUARC Stream Client URI tcpip://<laptop-ip>:<port>).
mode 'client': connect to `host`:`port` (the sender is the server, e.g. a QUARC Stream Server).
reply_count: number of zero doubles sent back per record (only if the sender waits for it); 0 = never reply.
Stamps with the laptop's ROS time on receipt.

Two topics per body:
  /mocap/<body>/pose        for display/TF: published when the pose changes, otherwise the last pose is
                            repeated every `keepalive` s, so the drone and the map frame always exist.
  /mocap/<body>/pose_valid  the same, but ONLY while tracked. drone_link uses it to decide whether a
                            detection can be mapped (no tracking = Simulink holds an old pose).
"""
import math
import socket
import struct
import threading
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped



def euler_zyx_to_quat(roll, pitch, yaw):
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy)


def decode_record(values, bodies, fields, degrees=False):
    """values: flat tuple of numbers -> {body: (pos(3), quat xyzw(4), tracked bool)}. Pure function (tested)."""
    out, n = {}, len(fields)
    for i, body in enumerate(bodies):
        d = {}
        for name, v in zip(fields, values[i * n:(i + 1) * n]):
            if name.startswith('-'):
                name, v = name[1:], -v
            if name != 'skip':
                d[name] = float(v)
        pos = (d.get('x', 0.0), d.get('y', 0.0), d.get('z', 0.0))
        if 'qw' in d:
            q = (d.get('qx', 0.0), d.get('qy', 0.0), d.get('qz', 0.0), d['qw'])
        else:
            k = math.pi / 180.0 if degrees else 1.0
            q = euler_zyx_to_quat(d.get('roll', 0.0) * k, d.get('pitch', 0.0) * k, d.get('yaw', 0.0) * k)
        out[body] = (pos, q, d.get('tracked', 1.0) > 0.5)
    return out


class TcpPoseBridge(Node):
    def __init__(self):
        super().__init__('tcp_pose_bridge')
        self.declare_parameter('mode', 'server')
        self.declare_parameter('host', '0.0.0.0')
        self.declare_parameter('port', 18000)
        self.declare_parameter('bodies', ['RigidBody'])
        self.declare_parameter('fields', ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])
        self.declare_parameter('dtype', 'f8')
        self.declare_parameter('byte_order', '<')
        self.declare_parameter('angles_in_degrees', False)
        self.declare_parameter('topic_fmt', '/mocap/{body}/pose')
        self.declare_parameter('frame_id', 'mocap')
        self.declare_parameter('reply_count', 0)  # 0 = receive only, never send anything back
        # Simulink sends ~1000 Hz, OptiTrack updates ~100 Hz: publish when the pose changes, else repeat
        # it every `keepalive` s (keeps the drone + map frame alive when it sits still)
        self.declare_parameter('keepalive', 0.05)
        self.declare_parameter('valid_topic_fmt', '/mocap/{body}/pose_valid')
        g = lambda n: self.get_parameter(n).value
        self.mode, self.host, self.port = g('mode'), g('host'), g('port')
        self.bodies, self.fields, self.deg = g('bodies'), g('fields'), g('angles_in_degrees')
        self.frame_id = g('frame_id')
        code = {'f8': 'd', 'f4': 'f'}[g('dtype')]
        count = len(self.bodies) * len(self.fields)
        self.fmt = struct.Struct(f"{g('byte_order')}{count}{code}")
        n_reply = int(g('reply_count'))
        self.reply = struct.pack(f"{g('byte_order')}{n_reply}d", *([0.0] * n_reply)) if n_reply > 0 else b''
        self.pubs = {b: self.create_publisher(PoseStamped, g('topic_fmt').format(body=b), 10) for b in self.bodies}
        self.n_rx, self.connected = 0, False
        self.last_rx, self.max_gap = None, 0.0
        self.keepalive = g('keepalive')
        self.has_flag = 'tracked' in [f.lstrip('-') for f in self.fields]
        self.pubs_valid = {b: self.create_publisher(PoseStamped, g('valid_topic_fmt').format(body=b), 10)
                           for b in self.bodies}
        self.last_vals, self.last_pub = {}, {}
        self.was_tracked, self.untracked_since = {}, {}
        self.n_pub = self.n_lost = 0
        self.n_tracked, self.n_untracked = {}, {}           # per body
        self.max_lost = 0.0
        self.raw_values, self.last_raw = None, None
        self.get_logger().info(f'{self.mode} {self.host}:{self.port}, record = {self.fmt.size} bytes '
                               f'({len(self.bodies)} bodies x {self.fields}, {g("dtype")} {g("byte_order")})')
        threading.Thread(target=self.loop, daemon=True).start()
        self.create_timer(5.0, self.report)

    def report(self):
        if self.mode == 'udp':
            fresh = self.last_rx is not None and time.monotonic() - self.last_rx < 1.0
            state = 'receiving' if fresh else 'waiting for UDP packets'
        else:
            state = 'connected' if self.connected else 'waiting for connection'
        parts, never = [], []
        for b in self.bodies:
            t, u = self.n_tracked.get(b, 0), self.n_untracked.get(b, 0)
            if t + u:
                parts.append(f'{b} {100.0 * t / (t + u):.0f}%')
                if t == 0:
                    never.append(b)
            else:
                parts.append(f'{b} no data')
        tracked = ', '.join(parts) + ('' if self.has_flag else ' (no tracked field)')
        msg = (f'{state}, {self.n_rx / 5.0:.0f} records/s, longest gap {self.max_gap * 1000:.0f} ms | '
               f'published {self.n_pub / 5.0:.0f}/s, tracked {tracked}, tracking lost {self.n_lost}x '
               f'(longest {self.max_lost * 1000:.0f} ms) | last {self.fields}: {self.last_raw}')
        if never:
            note = (' Drone 1 untracked: no detections are mapped until tracking returns.'
                    if self.bodies[0] in never else '')
            self.get_logger().warn(msg + f'\n  -> OptiTrack is NOT tracking {", ".join(never)} '
                                   f'(shown at its last pose).{note}')
        elif self.max_gap > 0.2 or self.n_lost:   # separate calls: rclpy forbids changing severity at one call site
            self.get_logger().warn(msg)
        else:
            self.get_logger().info(msg)
        self.n_rx, self.max_gap, self.n_pub = 0, 0.0, 0
        self.n_tracked, self.n_untracked = {}, {}
        self.n_lost, self.max_lost = 0, 0.0

    def mark_rx(self, n):
        now = time.monotonic()
        if self.last_rx is not None:
            self.max_gap = max(self.max_gap, now - self.last_rx)
        self.last_rx = now
        self.n_rx += n

    def loop_udp(self):
        size = self.fmt.size
        body_size = size // len(self.bodies)              # bytes of one drone (len(fields) values)
        code = self.fmt.format[-1]
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 16)  # small buffer: never queue old poses
        sock.bind((self.host, self.port))
        sock.settimeout(1.0)
        bad, warned_partial = 0, False
        while rclpy.ok():
            try:
                data, addr = sock.recvfrom(65536)
            except socket.timeout:
                continue
            if len(data) == size:                          # one datagram = one record with all drones
                values, bodies = self.fmt.unpack(data), self.bodies
            elif len(data) % body_size == 0 and len(data) < size:
                # fewer drones than configured (e.g. Simulink still sends only drone 1): decode those
                m = len(data) // body_size
                values = struct.unpack(f'{self.fmt.format[0]}{m * len(self.fields)}{code}', data)
                bodies = self.bodies[:m]
                if not warned_partial:
                    self.get_logger().warn(f'packets carry {m} of {len(self.bodies)} drones ({len(data)} bytes): '
                                           f'publishing only {bodies}')
                    warned_partial = True
            else:
                bad += 1
                self.get_logger().warn(f'datagram of {len(data)} bytes from {addr[0]}, expected {size} '
                                       f'({len(self.bodies)} drones x {len(self.fields)} doubles) ({bad} so far)',
                                       throttle_duration_sec=5.0)
                continue
            self.mark_rx(1)
            self.raw_values = values
            self.publish(decode_record(values, bodies, self.fields, self.deg))

    def loop(self):
        if self.mode == 'udp':
            return self.loop_udp()
        while rclpy.ok():
            try:
                if self.mode == 'server':
                    with socket.create_server((self.host, self.port), reuse_port=False) as srv:
                        srv.settimeout(1.0)
                        while rclpy.ok():
                            try:
                                conn, addr = srv.accept()
                            except socket.timeout:
                                continue
                            self.get_logger().info(f'sender connected from {addr[0]}:{addr[1]}')
                            self.read(conn)
                else:
                    conn = socket.create_connection((self.host, self.port), timeout=2.0)
                    self.get_logger().info(f'connected to {self.host}:{self.port}')
                    self.read(conn)
            except OSError as e:
                self.get_logger().warn(f'socket: {e}; retrying in 1 s', throttle_duration_sec=5.0)
            self.connected = False
            time.sleep(1.0)

    def read(self, conn):
        self.connected = True
        conn.settimeout(2.0)
        buf = b''
        with conn:
            while rclpy.ok():
                try:
                    chunk = conn.recv(65536)
                except socket.timeout:
                    continue
                if not chunk:
                    self.get_logger().warn('sender disconnected')
                    return
                buf += chunk
                size = self.fmt.size
                if len(buf) < size:
                    continue
                # keep only the newest complete record (stale poses are useless)
                n = len(buf) // size
                rec = buf[(n - 1) * size:n * size]
                buf = buf[n * size:]
                self.mark_rx(n)
                if self.reply:
                    conn.sendall(self.reply * n)  # one answer per record keeps the sender in lockstep
                self.raw_values = self.fmt.unpack(rec)
                self.publish(decode_record(self.raw_values, self.bodies, self.fields, self.deg))

    def publish(self, poses):
        if self.raw_values:
            self.last_raw = [round(v, 4) for v in self.raw_values]
        now = time.monotonic()
        stamp = None
        for body, (p, q, tracked) in poses.items():
            # tracking statistics
            if tracked:
                self.n_tracked[body] = self.n_tracked.get(body, 0) + 1
                if self.was_tracked.get(body) is False:          # tracking came back
                    lost = now - self.untracked_since[body]
                    self.max_lost = max(self.max_lost, lost)
            else:
                self.n_untracked[body] = self.n_untracked.get(body, 0) + 1
                if self.was_tracked.get(body, True):             # tracking just got lost
                    self.untracked_since[body] = now
                    self.n_lost += 1
            self.was_tracked[body] = tracked

            vals = (*p, *q, tracked)
            changed = vals != self.last_vals.get(body)
            due = now - self.last_pub.get(body, 0.0) >= self.keepalive
            if not (changed or due):
                continue
            self.last_vals[body], self.last_pub[body] = vals, now
            if stamp is None:
                stamp = self.get_clock().now().to_msg()
            m = PoseStamped()
            m.header.stamp, m.header.frame_id = stamp, self.frame_id
            m.pose.position.x, m.pose.position.y, m.pose.position.z = p
            m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z, m.pose.orientation.w = q
            self.pubs[body].publish(m)                           # display / TF: always
            if tracked:
                self.pubs_valid[body].publish(m)                 # mapping: only while tracked
            self.n_pub += 1


def main():
    rclpy.init()
    rclpy.spin(TcpPoseBridge())


if __name__ == '__main__':
    main()
